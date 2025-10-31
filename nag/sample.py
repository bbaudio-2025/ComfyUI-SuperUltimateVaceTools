import comfy
from .samplers import KSamplerWithNAG
from .samplers import sample_with_nag as samplers_sample_with_nag


def sample_with_nag(
        model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive, negative, nag_negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None, **kwargs):
    sampler = KSamplerWithNAG(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(
        noise, positive, negative, nag_negative,
        cfg=cfg, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, nag_sigma_end=nag_sigma_end,
        latent_image=latent_image,
        start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
        denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed,
        latent_shapes=latent_shapes, **kwargs,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples


def sample_custom_with_nag(
        model, noise, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler, sigmas, positive, negative, nag_negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None, **kwargs):
    samples = samplers_sample_with_nag(
        model, noise, positive, negative, nag_negative,
        cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
        model.load_device, sampler, sigmas,
        model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
        latent_shapes=latent_shapes, **kwargs,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples
