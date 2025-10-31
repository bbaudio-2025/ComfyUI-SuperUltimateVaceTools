import torch
import comfy
from .nag.sample import sample_with_nag #from https://github.com/ChenDarYen/ComfyUI-NAG
import nodes
import node_helpers
import latent_preview
from comfy.comfy_types import IO
from PIL import Image, ImageOps, ImageFilter
import numpy as np

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def img_whiten_pil(img, ratio):
    white = Image.new('RGB', img.size, (255, 255, 255))
    mixed = Image.blend(img, white, ratio)
    return mixed

def img_greyscale_pil(img, saturation):
    greyscaled = ImageOps.grayscale(img).convert('RGB')
    mixed = Image.blend(greyscaled, img, saturation)
    return mixed

def img_blur_pil(img, radius):
    blured = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return blured

def img_contour_pil(img):
    contoured = img.filter(ImageFilter.CONTOUR)
    return contoured

def color2mask_pil(img, tolerance):
    def color_variance(color1, color2):
        red_vs = (color1[0] - color2[0]) ** 2
        green_vs = (color1[1] - color2[1]) ** 2
        blue_vs = (color1[2] - color2[2]) ** 2
        variance = (red_vs + green_vs + blue_vs) ** 0.5
        variance_unified = variance / (255 * 3 ** 0.5)
        return variance_unified
    data = img.getdata()
    new_data = []
    for item in data:
        if color_variance(item[:3], (255, 255, 255)) < tolerance:
            new_data.append((255, 255, 255))
        else:
            new_data.append((0, 0, 0))
    img.putdata(new_data)
    return img

def emptyimage(width, height, batch_size=1, color=(0,0,0)):
    r = torch.full([batch_size, height, width, 1], color[0] / 255, dtype=torch.float32, device="cpu")
    g = torch.full([batch_size, height, width, 1], color[1] / 255, dtype=torch.float32, device="cpu")
    b = torch.full([batch_size, height, width, 1], color[2] / 255, dtype=torch.float32, device="cpu")
    return torch.cat((r, g, b), dim=-1)

def imagecrop(image, width, height, x, y):
    x = min(x, image.shape[2] - 1)
    y = min(y, image.shape[1] - 1)
    to_x = width + x
    to_y = height + y
    img = image[:,y:to_y, x:to_x, :]
    return img

def feather(mask, left=0, top=0, right=0, bottom=0):
    # from comfyui
    output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()

    left = min(left, output.shape[-1])
    right = min(right, output.shape[-1])
    top = min(top, output.shape[-2])
    bottom = min(bottom, output.shape[-2])

    for x in range(left):
        feather_rate = (x + 1.0) / left
        output[:, :, x] *= feather_rate

    for x in range(right):
        feather_rate = (x + 1) / right
        output[:, :, -x] *= feather_rate

    for y in range(top):
        feather_rate = (y + 1) / top
        output[:, y, :] *= feather_rate

    for y in range(bottom):
        feather_rate = (y + 1) / bottom
        output[:, -y, :] *= feather_rate

    return output

def repeat_tensor(tensor, batch, dim=0):
    repeat_list = []
    for n in range(batch):
        repeat_list.append(tensor)
    result = torch.cat(repeat_list, dim=dim)
    return result

def imgcomposite(destination, source, x, y, mask):
    des_copy = destination.clone()
    des_crop = des_copy[:, y:(source.shape[1] + y), x:(source.shape[2] + x), :]
    composed_area = des_crop * (1 - mask.unsqueeze(-1)) + source * mask.unsqueeze(-1)
    des_copy[:, y:(source.shape[1]+y), x:(source.shape[2]+x), :] = composed_area
    return des_copy

def maskasemble(batchsize, width, height, value_bg, value_fg, left, top, right, bottom):
    output = torch.full((batchsize, height, width), value_bg, dtype=torch.float32, device="cpu")
    output[:, 0:top, :] = value_fg
    output[:, (height - bottom):height, :] = value_fg
    output[:, :, 0:left] = value_fg
    output[:, :, (width - right):width] = value_fg
    return output

def add_noise(image, noise_aug_strength, seed):
    # from KJNODES
    torch.manual_seed(seed)
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * noise_aug_strength
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image_out = image + image_noise
    return image_out

def spatialistgen(width_upscale, height_upscale, width, height, spatial_multiplier=16):
    if width >= width_upscale or height >= height_upscale:
        raise ValueError("spatialistgen: 放大尺寸应该大于生成尺寸\ndimension_upscale should be large than dimension")
    width = width // spatial_multiplier * spatial_multiplier
    height = height // spatial_multiplier * spatial_multiplier
    num_tile_x =  width_upscale // width + 1
    num_tile_y = height_upscale // height + 1
    pad_x = (num_tile_x * width - width_upscale) // (num_tile_x - 1)
    pad_x_res = (num_tile_x * width - width_upscale) % (num_tile_x - 1)
    pad_y = (num_tile_y * height - height_upscale) // (num_tile_y - 1)
    pad_y_res = (num_tile_y * height - height_upscale) % (num_tile_y - 1)

    croparea_list = []
    for i in range(num_tile_y):
        for j in range(num_tile_x):
            croparea_list.append({
                'width_crop': width,
                'height_crop': height,
                'offset_x': j * width - (pad_x * j + pad_x_res if j == num_tile_x - 1 else pad_x * j),
                'offset_y': i * height - (pad_y * i + pad_y_res if i == num_tile_y - 1 else pad_y * i),
                'mask_left': 0 if j == 0 else pad_x + pad_x_res if j == num_tile_x - 1 else pad_x,
                'mask_right': 0,
                'mask_top': 0 if i == 0 else pad_y + pad_y_res if j == num_tile_y - 1 else pad_y,
                'mask_bottom': 0,
                'feather_left': 0 if j == 0 else pad_x + pad_x_res if j == num_tile_x - 1 else pad_x,
                'feather_right': 0,
                'feather_top': 0 if i == 0 else pad_y + pad_y_res if i == num_tile_y - 1 else pad_y,
                'feather_bottom': 0,
            })
    return croparea_list

def temporalistgen(num_total_frame, length, num_crossfade, num_loopback_crossfade, temporal_multiplier=4):
    if num_total_frame < length:
        raise ValueError("temporalistgen: 视频帧数应该大于或等于length\nframe count of input video should be larger than or equal to length")
    res_frame = num_total_frame
    slice_list = []
    start_index = 0
    while start_index + length < num_total_frame:
        slice_list.append({
            'start_index': start_index,
            'length': length,
            'num_crossfade': num_crossfade,
            'flag_final_slice': False,
        })
        start_index = start_index + length - num_crossfade

    res_frame = num_total_frame - start_index
    if (res_frame - 1) % temporal_multiplier != 0:
        res_frame_padded = res_frame + temporal_multiplier - (res_frame - 1) % temporal_multiplier
    else:
        res_frame_padded = res_frame
    num_crossfade_end = res_frame_padded - res_frame + num_crossfade
    if num_loopback_crossfade > num_crossfade_end:
        raise ValueError("temporalistgen: num_loopback_crossfade过大，尝试减小\nnum_loopback_crossfade is too large, try to decrease it")
    slice_list.append({
        'start_index': num_total_frame - res_frame_padded,
        'length': res_frame_padded,
        'num_crossfade': num_crossfade_end,
        'flag_final_slice': True,
    })
    return slice_list

def crossfadevideos(video1, video2, num_corssfade_frame):
    if video1.ndim != video2.ndim:
        raise ValueError("crossfadevideos: 拼接图片类型不一致\nImageType Mismatch")
    if video1[[0],].shape != video2[[0],].shape:
        raise ValueError("crossfadevideos: 拼接图片尺寸不一致\nImageSize Mismatch")
    if num_corssfade_frame > video1.shape[0] or num_corssfade_frame > video2.shape[0]:
        raise ValueError("crossfadevideos: 拼接图片数目应大于过渡数目\nVideoLength should be longer than CrossLength")
    video_slice1 = video1[:-num_corssfade_frame]
    video_slice2 = video1[-num_corssfade_frame:]
    video_slice3 = video2[:num_corssfade_frame]
    video_slice4 = video2[num_corssfade_frame:]
    alpha_list = []
    count = num_corssfade_frame + 1
    while count > 1:
        alpha_list.append((count - 1) / (num_corssfade_frame + 1))
        count -= 1
    alpha_list.reverse()
    blend_list = []
    index = 0
    for alpha in alpha_list:
        mixed = video_slice2[[index],] * (1 - alpha) + video_slice3[[index],] * alpha
        blend_list.append(mixed)
        index += 1
    blended_slice = torch.cat(blend_list, dim=0)
    result = torch.cat((video_slice1, blended_slice, video_slice4), dim=0)
    return result

def vace_cond_execute(positive, negative, vae, width, height, length, batch_size, strength, input_video=None, control_video=None, control_masks=None, reference_image=None, latent_strength_list=None):
    #from comfyui, modified
    latent_length = ((length - 1) // 4) + 1
    if control_video is not None:
        control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if control_video.shape[0] < length:
            control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
    else:
        control_video = torch.ones((length, height, width, 3)) * 0.5

    if reference_image is not None:
        reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        reference_image = vae.encode(reference_image[:, :, :, :3])
        reference_image_vaed = reference_image.clone()
        reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

    if control_masks is None:
        mask = torch.ones((length, height, width, 1))
    else:
        mask = control_masks
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
        if mask.shape[0] < length:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

    control_video = control_video - 0.5
    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive, reactive), dim=1)

    if latent_strength_list is not None:
        for i in range(len(latent_strength_list)):
            control_video_latent[:, :, [i],].mul_(latent_strength_list[i])
    if reference_image is not None:
        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)
    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

    trim_latent = 0
    if reference_image is not None:
        mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
        mask = torch.cat((mask_pad, mask), dim=1)
        latent_length += reference_image.shape[2]
        trim_latent = reference_image.shape[2]

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

    if input_video is not None:
        latent = vae.encode(input_video[:,:,:,:3])
    else:
        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    if reference_image is not None and input_video is not None:
        latent = torch.cat((reference_image_vaed, latent), dim=2)
    out_latent = {}
    out_latent["samples"] = latent
    return {'positive': positive, 'negative': negative, 'out_latent': out_latent, 'trim_latent': trim_latent}

def trim_video_latent_execute(samples, trim_amount):
    #from comfyui
    samples_out = samples.copy()
    s1 = samples["samples"]
    samples_out["samples"] = s1[:, :, trim_amount:]
    return samples_out

def vae_decode(vae, samples):
    #from comfyui
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images

def colormatch(image_ref, image_target, method='mkl', strength=1.0):
    # from KJNODES
    try:
        from color_matcher import ColorMatcher
    except:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    cm = ColorMatcher()
    image_ref = image_ref.cpu()
    image_target = image_target.cpu()
    batch_size = image_target.size(0)
    out = []
    images_target = image_target.squeeze()
    images_ref = image_ref.squeeze()

    image_ref_np = images_ref.numpy()
    images_target_np = images_target.numpy()

    if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

    for i in range(batch_size):
        image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
        image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
        try:
            image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
        except BaseException as e:
            print(f"Error occurred during transfer: {e}")
            break
        # Apply the strength multiplier
        image_result = image_target_np + strength * (image_result - image_target_np)
        out.append(torch.from_numpy(image_result))
        
    out = torch.stack(out, dim=0).to(torch.float32)
    out.clamp_(0, 1)
    return out

class UltimateVideoUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Only VACE models are supported"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "input_video": ("IMAGE", ),
                "width_upscale": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height_upscale": ("INT", {"default": 720, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "pad_mask_limit": ("INT", {"default": 32, "min": 8, "max": 512, "step": 8}),
                "crossfade_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "loopback_crossfade": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "crop_ref": ("BOOLEAN", {"default": False}),
                "ref_as_init_frame": ("BOOLEAN", {"default": False}),
                "color_match": ("BOOLEAN", {"default": False}),
                "color_ref": (["input_video", "reference_image"], {"default": "input_video"}),
                "noise_aug": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step":0.001, "round": 0.001, }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "croparea_list": ("LIST", ),
                "reference_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
                "nag_params": ("NAGParamtersSetting", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("video", )
    OUTPUT_TOOLTIPS = ("Upscaled Video",)
    FUNCTION = "upscale_video"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = """
视频分块放大|Upscale video by splitting into tiled areas
by bbaudio
联系方式
QQ：1953761458
Email：1953761458@qq.com
QQ群：948626609
"""
    
    def upscale_video(self, model, width_upscale, height_upscale, width, height, length, pad_mask_limit, crossfade_frame, loopback_crossfade, 
                      crop_ref, ref_as_init_frame, color_match, color_ref, noise_aug, input_video, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, vae, 
                      croparea_list=None, reference_image=None, control_video=None, nag_params=None):
        if control_video is not None and control_video.shape[0] != input_video.shape[0]:
            raise ValueError("控制视频帧数与输入视频帧数应当一致\nFrame count of ControlVideo and InputVideo should be the same")
        if loopback_crossfade > 0:
            cross_slice = input_video[:loopback_crossfade].clone()
            input_video = torch.cat((input_video, cross_slice), dim=0)
            if control_video is not None:
                control_cross_slice = control_video[:loopback_crossfade].clone()
                control_video = torch.cat((control_video, control_cross_slice), dim=0)
        total_frame = input_video.shape[0]
        if total_frame > length and crossfade_frame == 0:
            raise ValueError("视频帧数大于length，需要设置crossfade_frame以启用时间分割\nFrame count of input video is larger than length, need set a proper value for crossfade_frame to enable temporal tiling")        
        strength = 1 # VACE Strength
        temporalist = temporalistgen(total_frame, length, crossfade_frame, loopback_crossfade)
        upscaled_videos_list = []
        turn_index = 0
        for turn in temporalist:
            start_index = turn['start_index']
            length_n = turn['length']
            cross_fade = turn['num_crossfade']
            upscaled_video = comfy.utils.common_upscale(input_video[start_index:(start_index + length_n)].movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
            if control_video is not None:
                up_scaled_control = comfy.utils.common_upscale(control_video[start_index:(start_index + length_n)].movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
            upscaled_video = add_noise(upscaled_video, noise_aug, seed)
            if turn_index > 0 and cross_fade > 0:
                upscaled_video[:cross_fade] = upscaled_videos_list[turn_index-1][-cross_fade:].clone()
            if croparea_list is None:
                croparea_list = spatialistgen(width_upscale, height_upscale, width, height)
            result_video = torch.full((length_n, height_upscale, width_upscale, 3), 0.5, device='cpu')
            index = 0
            for item in croparea_list:
                width_crop_n = item['width_crop']
                height_crop_n = item['height_crop']
                offset_x_n = item['offset_x']
                offset_y_n = item['offset_y']
                video = imagecrop(upscaled_video, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                mask_ctl = maskasemble(1, width_crop_n, height_crop_n, 1, 0, 
                                    min(item['mask_left'], pad_mask_limit), 
                                    min(item['mask_top'], pad_mask_limit), 
                                    min(item['mask_right'], pad_mask_limit), 
                                    min(item['mask_bottom'], pad_mask_limit))
                mask_ctl = repeat_tensor(mask_ctl, length_n)
                if turn_index > 0 and cross_fade > 0:
                    mask_ctl[:cross_fade] = torch.full((cross_fade, height_crop_n, width_crop_n,), 0.0, device='cpu')
                crop_gen = imagecrop(result_video, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                if control_video is not None:
                    crop_ctl = imagecrop(up_scaled_control, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls = imgcomposite(crop_ctl, crop_gen, 0, 0, 1-mask_ctl) if index != 0 else crop_ctl
                else:
                    controls = crop_gen[:length_n] if index != 0 else torch.full((length_n, height_crop_n, width_crop_n, 3), 0.5, device='cpu')
                if reference_image is not None:
                    reference_image = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
                    if crop_ref is True:
                        refimg = imagecrop(reference_image, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    else:
                        refimg = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width_crop_n, height_crop_n, "bilinear", "center").movedim(1, -1)
                    if ref_as_init_frame is True and turn_index == 0:
                        init_ref = imagecrop(reference_image, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                        controls[:1,:,:,:] = init_ref
                        mask_ctl[:1,:,:] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                else:
                    refimg = None
                if turn_index > 0 and cross_fade > 0:
                    init_ctl = imagecrop(upscaled_videos_list[turn_index-1][-cross_fade:].clone(), width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls[:cross_fade] = init_ctl
                    mask_ctl[:cross_fade] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                if turn['flag_final_slice'] is True and loopback_crossfade > 0:
                    end_ctl = imagecrop(upscaled_videos_list[0][:loopback_crossfade].clone(), width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls[-loopback_crossfade:] = end_ctl
                    mask_ctl[-loopback_crossfade:] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                conditions = vace_cond_execute(positive, negative, vae, width, height, length_n, 1, strength, input_video=video, control_video=controls, control_masks=mask_ctl, reference_image=refimg, latent_strength_list=None)
                nag_parameters = nag_params
                if nag_parameters is None:
                    sample = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, conditions['positive'], conditions['negative'], conditions['out_latent'], 
                                                        denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False)[0]
                else:
                    latent_image = conditions['out_latent']["samples"]
                    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
                    noise = comfy.sample.prepare_noise(latent_image, seed, None)
                    callback = latent_preview.prepare_callback(model, steps)
                    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                    nag_scale = nag_parameters['nag_scale']
                    nag_tau = nag_parameters['nag_tau']
                    nag_alpha = nag_parameters['nag_alpha']
                    nag_sigma_end = nag_parameters['nag_sigma_end']
                    nag_negative = conditions['negative']
                    nag_sample_out = sample_with_nag(model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, conditions['positive'], conditions['negative'], 
                                            nag_negative, latent_image, denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, 
                                            noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
                    sample = {"samples": nag_sample_out}
                trimmed_sample = trim_video_latent_execute(sample, conditions['trim_latent'])
                sampled_video = vae_decode(vae, trimmed_sample)
                mask_feather = feather(torch.full((1, height_crop_n, width_crop_n), 1.0, device='cpu'), item['feather_left'], item['feather_top'], item['feather_right'], item['feather_bottom'])
                mask_feather = repeat_tensor(mask_feather, length_n)
                result_video = imgcomposite(result_video, sampled_video, offset_x_n, offset_y_n, mask_feather)
                index += 1
                total_tile = len(temporalist) * len(croparea_list)
                print('第', turn_index + 1, '部分视频第', index, '块生成完成；整体完成', 100*(turn_index * len(croparea_list) + index)/total_tile, '%')
            upscaled_videos_list.append(result_video)
            turn_index += 1
        result_video = upscaled_videos_list.pop(0)
        index = 0
        while index < len(upscaled_videos_list):
            cross_fade = temporalist[index + 1]['num_crossfade']
            result_video = crossfadevideos(result_video, upscaled_videos_list[index], cross_fade)
            index += 1
        if loopback_crossfade > 0:
            crossed_start = crossfadevideos(result_video[-loopback_crossfade:], result_video[:loopback_crossfade], loopback_crossfade)
            result_video[:loopback_crossfade] = crossed_start
            result_video = result_video[:(total_frame - loopback_crossfade)]
        if color_match is True:
            matched_list = []
            for i in range(result_video.shape[0]):
                if color_ref == 'reference_image' and reference_image is not None:
                    ref = reference_image
                else:
                    ref = input_video[[i],] 
                matched = colormatch(ref, result_video[[i],])
                matched_list.append(matched)
            result_video = torch.cat(matched_list, dim=0)
        return (result_video, )

class CustomCropArea:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width_upscale": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height_upscale": ("INT", {"default": 720, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "presets": (["'H' for wide screen", "'三' for long narrow screen"], {
                    "default": "'H' for wide screen",
                    "tooltip": "预设分割方案\nPresets of cropping plan"
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "custom_croplist_gen"
    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = "Use preset of cropping plan"
    def custom_croplist_gen(self, width_upscale, height_upscale, presets):
        if presets == "'H' for wide screen":
            if height_upscale%16 != 0:
                raise ValueError("‘H’方案下放大高度必须为16的倍数\n'H' plan requires height_upscale to be multiplier of 16")
            result = [
                {
                'width_crop': width_upscale//2 + 16 - width_upscale//2%16 + 32,
                'height_crop': height_upscale//2 + 16 - height_upscale//2%16 + 16,
                'offset_x': width_upscale//4 - 32,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//2 + 16 - width_upscale//2%16 + 32,
                'height_crop': height_upscale//2 + 16 - height_upscale//2%16 + 16,
                'offset_x': width_upscale//4 - 32,
                'offset_y': height_upscale - (height_upscale//2 - height_upscale//2%16 +32),
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 32,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 32,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//4 + 16 - width_upscale//2%16 + 16,
                'height_crop': height_upscale,
                'offset_x': 0,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 32,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 32,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//4 + 16 - width_upscale//2%16 + 16,
                'height_crop': height_upscale,
                'offset_x': width_upscale- (width_upscale//4 + 16 - width_upscale//2%16 + 16),
                'offset_y': 0,
                'mask_left': 32,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 32,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
            ]
        elif presets == "'三' for long narrow screen":
            if width_upscale%16 != 0:
                raise ValueError("‘三’方案下放大宽度必须为16的倍数\n'三' plan requires height_upscale to be multiplier of 16")
            pad = 64
            h_i = (height_upscale + 3 * pad)//4
            h_res = h_i % 16
            h = h_i - h_res
            result = [
                {
                'width_crop': width_upscale,
                'height_crop': h,
                'offset_x': 0,
                'offset_y': h - pad,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale,
                'height_crop': h,
                'offset_x': 0,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': pad,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': pad,
                },
                {
                'width_crop': width_upscale,
                'height_crop':h ,
                'offset_x': 0,
                'offset_y': 2 * h - 2 * pad,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': pad,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': pad,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale,
                'height_crop': height_upscale - (3 * h - 3 * pad) - (height_upscale - 3 * h + 3 * pad)%16 + 16,
                'offset_x': 0,
                'offset_y': 3 * h - 3 * pad + (height_upscale - 3 * h + 3 * pad)%16 - 16,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': pad,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': pad,
                'feather_bottom': 0,
                },
            ]
        return (result,)

class RegionalBatchPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "prompt_list": ("STRING", ),
                "croparea_list": ("LIST", ),
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("croparea_list", )
    FUNCTION = "func"
    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = "batch conditioning prompt list"
    def func(self, clip, prompt_list, croparea_list):
        if len(prompt_list) != len(croparea_list):
            raise ValueError("提示词队列长度与切割队列长度不一致，检查节点连接是否正确\nLength of prompt_list is not same as croparea_list, check nodes connection")
        index = 0
        for prompt in prompt_list:
            tokens = clip.tokenize(prompt)
            croparea_list[index]['cond_p'] = clip.encode_from_tokens_scheduled(tokens)
            index += 1

        return (croparea_list, )

# VaceLongVideo
def sort_list(vace_control_list):
    n = len(vace_control_list)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if vace_control_list[j]['frame_position'] > vace_control_list[j + 1]['frame_position']:
                vace_control_list[j], vace_control_list[j + 1] = vace_control_list[j + 1], vace_control_list[j]
                swapped = True
        if not swapped:
            break
    return vace_control_list

def check_overlap(vace_control_list):
    n = len(vace_control_list)
    for i in range(n):
        for j in range(i + 1, n):
            if vace_control_list[i]['frame_position'] == vace_control_list[j]['frame_position']:
                raise ValueError("控制帧位置重复，检查控制帧位置设置\nidentical frame_position detected, check the frame_position setting")
            elif vace_control_list[j]['frame_position'] <= vace_control_list[i]['control_end_index'] and vace_control_list[j]['control_end_index'] >= vace_control_list[i]['frame_position']:
                raise ValueError("控制帧区域重叠，检查控制帧位置设置\nsome control frames overlapped, check the frame_position setting")
    return None

class VaceLongVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Only VACE models are supported"}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "loopback_crossfade": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "vace_prompt_list": ("PROMPTLIST", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "vace_control_list": ("CONTROLIMAGELIST", ),
                "nag_params": ("NAGParamtersSetting", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_TYPES = ("IMAGE", "LIST", "LIST", "IMAGE", )
    OUTPUT_TOOLTIPS = ("Generated Video",)
    FUNCTION = "long_video"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = """
VACE长视频拼接|Long video by concating multiple parts
by bbaudio
联系方式
QQ：1953761458
Email：1953761458@qq.com
QQ群：948626609
"""
    def long_video(self, model, clip, width, height, loopback_crossfade, vace_prompt_list, seed, steps, cfg, sampler_name, scheduler, 
                   denoise, vae, vace_control_list=None, nag_params=None):
        # check prompt list
        if len(vace_prompt_list) == 1:
            vace_prompt_list[0]['init_crossfade_frame'] = 0
        # get total frame
        total_frame = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            init_crossfade_frame = item['init_crossfade_frame']
            total_frame += num_frame - init_crossfade_frame
            if loopback_crossfade > num_frame:
                raise ValueError("循环过渡帧数目不能超过生成长度\nloopback_crossfade can not be larger than length of generation")
        # deal with control list
        control_video = torch.full((total_frame, height, width, 3), 0.5, device='cpu')
        control_mask = torch.full((total_frame, height, width), 1.0, device='cpu')
        if vace_control_list is not None:
            check_overlap(vace_control_list)
            vace_control_list = sort_list(vace_control_list)
            index_final = vace_control_list[-1]['control_end_index']
            if index_final >= total_frame:
                raise ValueError("控制帧长度超过生成长度\nLength of control image exceeds length of generation")
            for item in vace_control_list:
                index_start = item['frame_position']
                index_end = item['control_end_index']
                control_images = comfy.utils.common_upscale(item['control_image'].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                control_images = repeat_tensor(control_images, item['repeat'])
                custom_mask = item['custom_mask']
                if custom_mask is not None:
                    if custom_mask.shape[1] != height or custom_mask.shape[2] != width:
                        custom_mask = comfy.utils.common_upscale(custom_mask.unsqueeze(1), width, height, "bilinear", "else").squeeze(1)
                control_video[index_start:index_end + 1] = control_images[:index_end + 1 - index_start]
                if custom_mask is None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = torch.full((index_end + 1 - index_start, height, width), 0.0, device='cpu')
                elif custom_mask is not None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = custom_mask
        # deal with prompt list
        sampled = []
        # debug_control = []
        # debug_mask = []
        # debug_crossframes = {}
        vace_prompt_list[-1]['flag_end'] = True
        processed_frame_count = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            positive_prompt = item['prompt_p']
            negative_prompt = item['prompt_n']
            init_crossfade_frame = item['init_crossfade_frame']
            whiten_list = item['whiten_list']
            saturation_list = item['saturation_list']
            blur_list = item['blur_list']
            contour_list = item['contour_list']
            mask_value_list = item['mask_value_list']
            latent_strength_list = item['latent_strength_list']
            colormatch_strength_list = item['colormatch_strength_list']
            ref_image = item['ref_image']
            vace_strength = item['vace_strength']
            if item['model_override'] is not None:
                model = item['model_override']
            if item['seed_override'] != 0:
                seed = item['seed_override']
            # control
            if processed_frame_count == 0:
                controls = control_video[:num_frame].clone()
                mask_ctl = control_mask[:num_frame].clone()
            else:
                controls = control_video[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                mask_ctl = control_mask[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                for i in range(init_crossfade_frame):
                    crossfade_previous_frames = sampled[-1][[i-init_crossfade_frame],]
                    refined_control = tensor2pil(crossfade_previous_frames)
                    if saturation_list[i] < 0.999:
                        refined_control = img_greyscale_pil(refined_control, saturation_list[i])
                    if whiten_list[i] > 0.001:
                        refined_control = img_whiten_pil(refined_control, whiten_list[i])
                    if blur_list[i] > 0.001:
                        refined_control = img_blur_pil(refined_control, blur_list[i])
                    if contour_list[i] > 0.001:
                        contoured = img_contour_pil(tensor2pil(crossfade_previous_frames))
                        mask_contour = color2mask_pil(contoured, 1 - contour_list[i])
                        mask_contour = pil2tensor(mask_contour)[:, :, :, 0]
                        refined_control = imgcomposite(pil2tensor(refined_control), pil2tensor(contoured), 0, 0, 1 - mask_contour)
                    else:
                        refined_control = pil2tensor(refined_control)
                    controls[[i],] = refined_control
                    mask_ctl[[i],] = torch.full((1, height, width), mask_value_list[i], device='cpu')
            if item['flag_end'] is True and loopback_crossfade > 0:
                controls[-loopback_crossfade:] = sampled[0][:loopback_crossfade].clone()
                mask_ctl[-loopback_crossfade:] = torch.full((loopback_crossfade, height, width), 0.0, device='cpu')
            p_tokens = clip.tokenize(positive_prompt)
            n_tokens = clip.tokenize(negative_prompt)
            cond_p =  clip.encode_from_tokens_scheduled(p_tokens)
            cond_n = clip.encode_from_tokens_scheduled(n_tokens)
            conditions = vace_cond_execute(cond_p, cond_n, vae, width, height, num_frame, 1, vace_strength, control_video=controls, control_masks=mask_ctl, reference_image=ref_image, latent_strength_list=latent_strength_list)
            nag_parameters = nag_params
            if nag_parameters is None:
                sample = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, conditions['positive'], conditions['negative'], conditions['out_latent'], 
                                                    denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False)[0]
            else:
                latent_image = conditions['out_latent']["samples"]
                latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
                noise = comfy.sample.prepare_noise(latent_image, seed, None)
                callback = latent_preview.prepare_callback(model, steps)
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                nag_scale = nag_parameters['nag_scale']
                nag_tau = nag_parameters['nag_tau']
                nag_alpha = nag_parameters['nag_alpha']
                nag_sigma_end = nag_parameters['nag_sigma_end']
                nag_negative = conditions['negative']
                nag_sample_out = sample_with_nag(model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, conditions['positive'], conditions['negative'], 
                                           nag_negative, latent_image, denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, 
                                           noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
                sample = {"samples": nag_sample_out}
            trimmed_sample = trim_video_latent_execute(sample, conditions['trim_latent'])
            sample_result = vae_decode(vae, trimmed_sample)
            if processed_frame_count > 0 and colormatch_strength_list[i] > 0.001:
                image_ref = sampled[-1][-1:]
                for i in range(init_crossfade_frame):
                    sample_result[[i],] = colormatch(image_ref, sample_result[[i],], strength=colormatch_strength_list[i])
            processed_frame_count += num_frame - init_crossfade_frame
            # debug_control.append(controls)
            # debug_mask.append(mask_ctl)
            # debug_crossframes['colormatched'] = sample_result[:init_crossfade_frame+5]
            sampled.append(sample_result)
        result_video = sampled.pop(0)
        index = 0
        while index < len(sampled):
            cross_fade = vace_prompt_list[index + 1]['init_crossfade_frame']
            result_video = crossfadevideos(result_video, sampled[index], cross_fade)
            index += 1
        if loopback_crossfade > 0:
            crossed_start = crossfadevideos(result_video[-loopback_crossfade:], result_video[:loopback_crossfade], loopback_crossfade)
            result_video[:loopback_crossfade] = crossed_start
            result_video = result_video[:(total_frame - loopback_crossfade)]
        return (result_video, )
        # return (result_video, debug_control, debug_mask, debug_crossframes)

class VaceFunLongVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_h": ("MODEL", {"tooltip": "high noise model"}),
                "model_l": ("MODEL", {"tooltip": "low noise model"}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "loopback_crossfade": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "vace_prompt_list": ("PROMPTLIST", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, }),
                "steps_h": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg_h": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_h": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler_h": (comfy.samplers.KSampler.SCHEDULERS, ),
                "steps_l": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg_l": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_l": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler_l": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "vace_control_list": ("CONTROLIMAGELIST", ),
                "nag_params": ("NAGParamtersSetting", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_TYPES = ("IMAGE", "LIST", "LIST", "IMAGE", )
    OUTPUT_TOOLTIPS = ("Generated Video",)
    FUNCTION = "long_video_vace_fun"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = """
VACE长视频拼接|Long video by concating multiple parts
by bbaudio
联系方式
QQ：1953761458
Email：1953761458@qq.com
QQ群：948626609
"""
    def long_video_vace_fun(self, model_h, model_l, clip, vae, width, height, loopback_crossfade, vace_prompt_list, seed, steps_h, cfg_h, sampler_h, scheduler_h, steps_l, cfg_l, sampler_l, scheduler_l, 
                   vace_control_list=None, nag_params=None):
        # check prompt list
        if len(vace_prompt_list) == 1:
            vace_prompt_list[0]['init_crossfade_frame'] = 0
        # get total frame
        total_frame = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            init_crossfade_frame = item['init_crossfade_frame']
            total_frame += num_frame - init_crossfade_frame
            if loopback_crossfade > num_frame:
                raise ValueError("循环过渡帧数目不能超过生成长度\nloopback_crossfade can not be larger than length of generation")
        # deal with control list
        control_video = torch.full((total_frame, height, width, 3), 0.5, device='cpu')
        control_mask = torch.full((total_frame, height, width), 1.0, device='cpu')
        if vace_control_list is not None:
            check_overlap(vace_control_list)
            vace_control_list = sort_list(vace_control_list)
            index_final = vace_control_list[-1]['control_end_index']
            if index_final >= total_frame:
                raise ValueError("控制帧长度超过生成长度\nLength of control image exceeds length of generation")
            for item in vace_control_list:
                index_start = item['frame_position']
                index_end = item['control_end_index']
                control_images = comfy.utils.common_upscale(item['control_image'].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                control_images = repeat_tensor(control_images, item['repeat'])
                custom_mask = item['custom_mask']
                if custom_mask is not None:
                    if custom_mask.shape[1] != height or custom_mask.shape[2] != width:
                        custom_mask = comfy.utils.common_upscale(custom_mask.unsqueeze(1), width, height, "bilinear", "else").squeeze(1)
                control_video[index_start:index_end + 1] = control_images[:index_end + 1 - index_start]
                if custom_mask is None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = torch.full((index_end + 1 - index_start, height, width), 0.0, device='cpu')
                elif custom_mask is not None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = custom_mask
        # deal with prompt list
        sampled = []
        # debug_control = []
        # debug_mask = []
        # debug_crossframes = {}
        vace_prompt_list[-1]['flag_end'] = True
        processed_frame_count = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            positive_prompt = item['prompt_p']
            negative_prompt = item['prompt_n']
            init_crossfade_frame = item['init_crossfade_frame']
            whiten_list = item['whiten_list']
            saturation_list = item['saturation_list']
            blur_list = item['blur_list']
            contour_list = item['contour_list']
            mask_value_list = item['mask_value_list']
            latent_strength_list = item['latent_strength_list']
            colormatch_strength_list = item['colormatch_strength_list']
            ref_image = item['ref_image']
            vace_strength = item['vace_strength']
            # if item['model_override'] is not None:
            #     model = item['model_override']
            if item['seed_override'] != 0:
                seed = item['seed_override']
            # control
            if processed_frame_count == 0:
                controls = control_video[:num_frame].clone()
                mask_ctl = control_mask[:num_frame].clone()
            else:
                controls = control_video[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                mask_ctl = control_mask[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                for i in range(init_crossfade_frame):
                    crossfade_previous_frames = sampled[-1][[i-init_crossfade_frame],]
                    refined_control = tensor2pil(crossfade_previous_frames)
                    if saturation_list[i] < 0.999:
                        refined_control = img_greyscale_pil(refined_control, saturation_list[i])
                    if whiten_list[i] > 0.001:
                        refined_control = img_whiten_pil(refined_control, whiten_list[i])
                    if blur_list[i] > 0.001:
                        refined_control = img_blur_pil(refined_control, blur_list[i])
                    if contour_list[i] > 0.001:
                        contoured = img_contour_pil(tensor2pil(crossfade_previous_frames))
                        mask_contour = color2mask_pil(contoured, 1 - contour_list[i])
                        mask_contour = pil2tensor(mask_contour)[:, :, :, 0]
                        refined_control = imgcomposite(pil2tensor(refined_control), pil2tensor(contoured), 0, 0, 1 - mask_contour)
                    else:
                        refined_control = pil2tensor(refined_control)
                    controls[[i],] = refined_control
                    mask_ctl[[i],] = torch.full((1, height, width), mask_value_list[i], device='cpu')
            if item['flag_end'] is True and loopback_crossfade > 0:
                controls[-loopback_crossfade:] = sampled[0][:loopback_crossfade].clone()
                mask_ctl[-loopback_crossfade:] = torch.full((loopback_crossfade, height, width), 0.0, device='cpu')
            p_tokens = clip.tokenize(positive_prompt)
            n_tokens = clip.tokenize(negative_prompt)
            cond_p =  clip.encode_from_tokens_scheduled(p_tokens)
            cond_n = clip.encode_from_tokens_scheduled(n_tokens)
            conditions = vace_cond_execute(cond_p, cond_n, vae, width, height, num_frame, 1, vace_strength, control_video=controls, control_masks=mask_ctl, reference_image=ref_image, latent_strength_list=latent_strength_list)
            nag_parameters = nag_params
            steps = steps_h + steps_l
            if nag_parameters is None:
                sample_h = nodes.common_ksampler(model_h, seed, steps, cfg_h, sampler_h, scheduler_h, conditions['positive'], conditions['negative'], conditions['out_latent'], 
                                                    denoise=1.0, disable_noise=False, start_step=0, last_step=steps_h, force_full_denoise=False)[0]
                sample_l = nodes.common_ksampler(model_l, seed, steps, cfg_l, sampler_l, scheduler_l, conditions['positive'], conditions['negative'], sample_h, 
                                                    denoise=1.0, disable_noise=True, start_step=steps_h, last_step=10000, force_full_denoise=True)[0]
            else:
                latent_image = conditions['out_latent']["samples"]
                latent_image = comfy.sample.fix_empty_latent_channels(model_h, latent_image)
                noise_h = comfy.sample.prepare_noise(latent_image, seed, None)
                noise_l = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
                callback_h = latent_preview.prepare_callback(model_h, steps)
                callback_l = latent_preview.prepare_callback(model_l, steps)
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                nag_scale = nag_parameters['nag_scale']
                nag_tau = nag_parameters['nag_tau']
                nag_alpha = nag_parameters['nag_alpha']
                nag_sigma_end = nag_parameters['nag_sigma_end']
                nag_negative = conditions['negative']
                nag_sample_out_h = sample_with_nag(model_h, noise_h, steps, cfg_h, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_h, scheduler_h, conditions['positive'], conditions['negative'], 
                                           nag_negative, latent_image, denoise=1.0, disable_noise=False, start_step=0, last_step=steps_h, force_full_denoise=False, 
                                           noise_mask=None, callback=callback_h, disable_pbar=disable_pbar, seed=seed)
                nag_sample_out_l = sample_with_nag(model_l, noise_l, steps, cfg_l, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_l, scheduler_l, conditions['positive'], conditions['negative'], 
                                           nag_negative, nag_sample_out_h, denoise=1.0, disable_noise=True, start_step=steps_h, last_step=10000, force_full_denoise=True, 
                                           noise_mask=None, callback=callback_l, disable_pbar=disable_pbar, seed=seed)
                sample_l = {"samples": nag_sample_out_l}
            trimmed_sample = trim_video_latent_execute(sample_l, conditions['trim_latent'])
            sample_result = vae_decode(vae, trimmed_sample)
            if processed_frame_count > 0 and colormatch_strength_list[i] > 0.001:
                image_ref = sampled[-1][-1:]
                for i in range(init_crossfade_frame):
                    sample_result[[i],] = colormatch(image_ref, sample_result[[i],], strength=colormatch_strength_list[i])
            processed_frame_count += num_frame - init_crossfade_frame
            # debug_control.append(controls)
            # debug_mask.append(mask_ctl)
            # debug_crossframes['colormatched'] = sample_result[:init_crossfade_frame+5]
            sampled.append(sample_result)
        result_video = sampled.pop(0)
        index = 0
        while index < len(sampled):
            cross_fade = vace_prompt_list[index + 1]['init_crossfade_frame']
            result_video = crossfadevideos(result_video, sampled[index], cross_fade)
            index += 1
        if loopback_crossfade > 0:
            crossed_start = crossfadevideos(result_video[-loopback_crossfade:], result_video[:loopback_crossfade], loopback_crossfade)
            result_video[:loopback_crossfade] = crossed_start
            result_video = result_video[:(total_frame - loopback_crossfade)]
        return (result_video, )
        # return (result_video, debug_control, debug_mask, debug_crossframes)

class VACEControlImageCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_image": ("IMAGE", ),
                "frame_position": ("INT", {"default": 0, "min": 0, "max": 65535, "step": 1}),
                "masked": ("BOOLEAN", {"default": False}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 65535, "step": 1}),
            },
            "optional": {
                "custom_mask": ("MASK", ),
                "previous_control": ("CONTROLIMAGELIST", ),
            }
        }

    RETURN_TYPES = ("CONTROLIMAGELIST", )
    RETURN_NAMES = ("vace_control_list", )
    FUNCTION = "combine_controls"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def combine_controls(self, control_image, frame_position, masked, repeat, custom_mask=None, previous_control=None):
        control_list = []
        if previous_control is not None:
            control_list.extend(previous_control)
        control_end_index = frame_position + control_image.shape[0] - 1 + (repeat - 1)
        control_list.append({
            'frame_position': frame_position,
            'control_end_index': control_end_index,
            'control_image': control_image,
            'custom_mask': custom_mask,
            'masked': masked,
            'repeat': repeat,
        })
        return (control_list, )

class VACEPromptCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_frame": ("INT", {"default": 81, "min": 5, "max": 65535, "step": 4}),
                "init_crossfade_frame": ("INT", {"default": 3, "min": 0, "max": 65535, "step": 1}),
                "refine_init": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "model_override": ("MODEL", {"tooltip": "Use this model in this generation, no use for fun vace"}),
                "ref_image": ("IMAGE", ),
                "custom_refine": ("REFINELIST", ),
                "previous_prompt": ("PROMPTLIST", ),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "will be ignored when set to 0"}),
            }
        }

    RETURN_TYPES = ("PROMPTLIST", )
    RETURN_NAMES = ("vace_prompt_list", )
    FUNCTION = "combine_prompt"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def combine_prompt(self, positive_prompt, negative_prompt, num_frame, init_crossfade_frame, refine_init, vace_strength, model_override=None, ref_image=None, custom_refine=None, previous_prompt=None, seed_override=None):
        if init_crossfade_frame > num_frame:
            raise ValueError("过渡帧数目不能大于总帧数\ninit_crossfade_frame can not be larger than num_frame")
        prompt_list = []
        if custom_refine is not None:
            whiten_list = []
            saturation_list = []
            blur_list = []
            contour_list = []
            mask_value_list = []
            latent_strength_list = []
            colormatch_strength_list = []
            for i in range(init_crossfade_frame):
                whiten = custom_refine['whiten_list'][i] if len(custom_refine['whiten_list']) > i else custom_refine['whiten_list'][-1]
                saturation = custom_refine['saturation_list'][i] if len(custom_refine['saturation_list']) > i else custom_refine['saturation_list'][-1]
                blur = custom_refine['blur_list'][i] if len(custom_refine['blur_list']) > i else custom_refine['blur_list'][-1]
                contour = custom_refine['contour_list'][i] if len(custom_refine['contour_list']) > i else custom_refine['contour_list'][-1]
                mask_value = custom_refine['mask_value_list'][i] if len(custom_refine['mask_value_list']) > i else custom_refine['mask_value_list'][-1]
                latent_strength = custom_refine['latent_strength_list'][i] if len(custom_refine['latent_strength_list']) > i else None
                colormatch_strength = custom_refine['colormatch_strength_list'][i] if len(custom_refine['colormatch_strength_list']) > i else custom_refine['colormatch_strength_list'][-1]
                whiten_list.append(whiten)
                saturation_list.append(saturation)
                blur_list.append(blur)
                contour_list.append(contour)
                mask_value_list.append(mask_value)
                colormatch_strength_list.append(colormatch_strength)
                if latent_strength is not None:
                    latent_strength_list.append(latent_strength)
        if previous_prompt is not None:
            prompt_list.extend(previous_prompt)
        prompt_list.append({
            'model_override': model_override,
            'seed_override': seed_override,
            'prompt_p': positive_prompt,
            'prompt_n': negative_prompt,
            'num_frame': num_frame,
            'init_crossfade_frame': init_crossfade_frame,
            'vace_strength': vace_strength,
            'whiten_list': [refine_init] * init_crossfade_frame if custom_refine is None else whiten_list,
            'saturation_list': [1.0] * init_crossfade_frame if custom_refine is None else saturation_list,
            'blur_list': [0.0] * init_crossfade_frame if custom_refine is None else blur_list,
            'contour_list': [0.0] * init_crossfade_frame if custom_refine is None else contour_list,
            'latent_strength_list': [1.0] * (init_crossfade_frame//4) if custom_refine is None else latent_strength_list,
            'mask_value_list': [1.0] * init_crossfade_frame if custom_refine is None else mask_value_list,
            'colormatch_strength_list': [0.0] * init_crossfade_frame if custom_refine is None else colormatch_strength_list,
            'ref_image': ref_image,
            "flag_end": False
        })
        return (prompt_list, )

def str2float(input_list):
    out = []
    for item in input_list:
        try:
            out.append(float(item))
        except :
            pass
    return out

def str2int(input_list):
    out = []
    for item in input_list:
        try:
            out.append(float(item))
        except :
            pass
    return out

class CustomRefineOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "whiten_list": ("STRING", {"default": '0.2, 0.2, 0.2'}),
                "saturation_list": ("STRING", {"default": '0.8, 0.8, 0.8'}),
                "blur_list": ("STRING", {"default": '0, 0, 0'}),
                "contour_list": ("STRING", {"default": '0, 0, 0'}),
                "mask_value_list": ("STRING", {"default": '1.0, 1.0, 1.0'}),
                "latent_strength_list": ("STRING", {"default": '1.0'}),
                "colormatch_strength_list": ("STRING", {"default": '0.0, 0.0, 0.0'}),
            },
        }

    RETURN_TYPES = ("REFINELIST", )
    RETURN_NAMES = ("custom_refine_list", )
    FUNCTION = "make_refine_list"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def make_refine_list(self, whiten_list, saturation_list, blur_list, contour_list, mask_value_list, latent_strength_list, colormatch_strength_list):
        whiten = whiten_list.split(',')
        saturation = saturation_list.split(',')
        blur = blur_list.split(',')
        contour = contour_list.split(',')
        mask_value = mask_value_list.split(',')
        latent_strength = latent_strength_list.split(',')
        cm_strength = colormatch_strength_list.split(',')
        whiten_list = str2float(whiten)
        whiten_list = whiten_list if len(whiten_list) > 0 else [0.2]
        saturation_list = str2float(saturation)
        saturation_list = saturation_list if len(saturation_list) > 0 else [0.8]
        blur_list = str2float(blur)
        blur_list = blur_list if len(blur_list) > 0 else [0]
        contour_list = str2float(contour)
        contour_list = contour_list if len(contour_list) > 0 else [0]
        latent_strength_list = str2float(latent_strength)
        latent_strength_list = latent_strength_list if len(latent_strength_list) > 0 else [1.0]
        mask_value_list = str2float(mask_value)
        mask_value_list = mask_value_list if len(mask_value_list) > 0 else [1.0]
        colormatch_strength_list = str2float(cm_strength)
        colormatch_strength_list = colormatch_strength_list if len(colormatch_strength_list) > 0 else [0.0]
        return ({
            'whiten_list': whiten_list,
            'saturation_list': saturation_list,
            'blur_list': blur_list,
            'contour_list': contour_list,
            'mask_value_list': mask_value_list,
            'latent_strength_list': latent_strength_list,
            'colormatch_strength_list': colormatch_strength_list,
        }, )

class VACEPromptCheckTotalFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_list": ("PROMPTLIST", ),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("total_frame", )
    FUNCTION = "check_total_frame"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def check_total_frame(self, prompt_list):
        total_frame = 0
        for item in prompt_list:
            total_frame += item['num_frame'] - item['init_crossfade_frame']
        return (total_frame, )

class NAGParamtersSetting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nag_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 100, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
            },
        }

    RETURN_TYPES = ("NAGParamtersSetting", )
    RETURN_NAMES = ("nag_params", )
    FUNCTION = "set_nag_parames"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def set_nag_parames(self, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        result = {
            'nag_scale': nag_scale,
            'nag_tau': nag_tau,
            'nag_alpha': nag_alpha,
            'nag_sigma_end': nag_sigma_end
        }
        return (result, )

class RefineTest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "whiten": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contour": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("TEST_IMG", )
    FUNCTION = "test_function"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def test_function(self, image, whiten, blur, saturation, contour):
        refined_control = tensor2pil(image)
        refined_control = img_greyscale_pil(refined_control, saturation)
        refined_control = img_whiten_pil(refined_control, whiten)
        refined_control = img_blur_pil(refined_control, blur)
        contoured = img_contour_pil(tensor2pil(image))
        if contour > 0.001:
            mask_contour = color2mask_pil(contoured, 1-contour)
            mask_contour = pil2tensor(mask_contour)[:, :, :, 0]
            refined_control = imgcomposite(pil2tensor(refined_control), pil2tensor(contoured), 0, 0, 1-mask_contour)
        else:
            refined_control = pil2tensor(refined_control)
            mask_contour = None
        return (refined_control, )

NODE_CLASS_MAPPINGS = {
    "SuperUltimateVACEUpscale": UltimateVideoUpscaler,
    "CustomCropArea": CustomCropArea,
    "RegionalBatchPrompt": RegionalBatchPrompt,
    "VACEControlImageCombine": VACEControlImageCombine,
    "VACEPromptCombine": VACEPromptCombine,
    "VaceLongVideo": VaceLongVideo,
    "VaceFunLongVideo": VaceFunLongVideo,
    "VACEPromptCheckTotalFrame": VACEPromptCheckTotalFrame,
    "CustomRefineOption": CustomRefineOption,
    "NAGParamtersSetting": NAGParamtersSetting,
    "RefineTest": RefineTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperUltimateVACEUpscale": "SuperUltimate VACE Upscale",
    "CustomCropArea": "Custom Crop Area",
    "RegionalBatchPrompt": "Batch Prompt Crop Area",
    "VACEControlImageCombine": "VACE Control Image Combine",
    "VACEPromptCombine": "VACE Prompt Combine",
    "VaceLongVideo": "SuperUltimate VACE Long Video",
    "VaceFunLongVideo": "SuperUltimate VACEFUN Long Video",
    "VACEPromptCheckTotalFrame": "Check Total Frame",
    "CustomRefineOption": "Custom Refine Option",
    "NAGParamtersSetting": "NAG Paramters Setting",
    "RefineTest": "RefineTest",
}
