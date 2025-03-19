import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-jsd-guidance")
class StableDiffusionJSDGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        enable_vae_slicing: bool = False
        enable_vae_tiling: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[Any] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        trainer_max_steps: int = 10000

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4
        n_ddim_steps: int = 50

        inversion_guidance_scale: float = -7.5
        inversion_n_steps: int = 10
        inversion_eta: float = 0.3
        k: float = 0.3
        warm_step: int = 2000
        analysis: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                threestudio.warn("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
        
        if self.cfg.enable_vae_slicing:
            self.pipe.enable_vae_slicing()
        
        if self.cfg.enable_vae_tiling:
            self.pipe.enable_vae_tiling()

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device)
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps, device=self.device)

        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.inverse_scheduler.set_timesteps(self.cfg.inversion_n_steps, device=self.device)
        self.inverse_scheduler.alphas_cumprod = (self.inverse_scheduler.alphas_cumprod.to(device=self.device))

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.amp.autocast(enabled=False, device_type='cuda')
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.amp.autocast(enabled=False, device_type='cuda')
    def forward_unet(self, latents: Float[Tensor, "..."], t: Float[Tensor, "..."], encoder_hidden_states: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.amp.autocast(enabled=False, device_type='cuda')
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.amp.autocast(enabled=False, device_type='cuda')
    def decode_latents(self, latents: Float[Tensor, "B 4 H W"], latent_height: int = 64, latent_width: int = 64) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.amp.autocast(enabled=False, device_type='cuda')
    @torch.no_grad()
    def get_noise_pred(self, latents_noisy: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"], azimuth: Float[Tensor, "B"], camera_distances: Float[Tensor, "B"], guidance_scale: float = 1.0,
        text_embeddings: Optional[Float[Tensor, "..."]] = None,
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            text_embeddings, neg_guidance_weights = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 4),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1).to(
                    e_i_neg.device
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + guidance_scale * (e_pos + accum_grad)
        else:
            neg_guidance_weights = None

            if text_embeddings is None:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation,
                    azimuth,
                    camera_distances,
                    self.cfg.view_dependent_prompting,
                )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings
                )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred, neg_guidance_weights, text_embeddings

    def ddim_inversion_step(self, model_output: torch.FloatTensor, timestep: int, prev_timestep: int, sample: torch.FloatTensor) -> torch.FloatTensor:
        # 1. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = (
            self.inverse_scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.inverse_scheduler.initial_alpha_cumprod
        )
        alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.inverse_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.inverse_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.inverse_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.inverse_scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 3. Clip or threshold "predicted x_0"
        if self.inverse_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.inverse_scheduler.config.clip_sample_range,
                self.inverse_scheduler.config.clip_sample_range,
            )
        # 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        # 6. Add noise to the sample
        variance = self.scheduler._get_variance(prev_timestep, timestep) ** (0.5)
        prev_sample += self.cfg.inversion_eta * torch.randn_like(prev_sample) * variance

        return prev_sample

    @torch.no_grad()
    def get_inversion_timesteps(self, final_t):
        n_training_steps = self.inverse_scheduler.config.num_train_timesteps

        ratio = self.cfg.inversion_n_steps // n_training_steps
        inverse_ratio = n_training_steps // self.cfg.inversion_n_steps
        
        inverse_timesteps: torch.Tensor = torch.arange(0, self.cfg.inversion_n_steps) * ratio
        inverse_timesteps = inverse_timesteps.round().to(device=self.device, dtype=torch.long)
    
        inverse_timesteps = inverse_timesteps[inverse_timesteps < int(final_t)]
        inverse_timesteps = torch.cat([inverse_timesteps[0] - ratio, inverse_timesteps])

        delta_t = int(random.random() * inverse_ratio)
        last_t = torch.tensor([min(int(final_t) + delta_t, n_training_steps - 1)]).to(device=self.device)
        inverse_timesteps = torch.cat([inverse_timesteps, last_t])
        return inverse_timesteps

    @torch.no_grad()
    def inversion(self, latents, final_t, prompt_utils, elevation, azimuth, camera_distances):
        latents = latents.clone()

        inverse_timesteps = self.get_inversion_timesteps(final_t)
        for t, next_t in zip(inverse_timesteps[:-1], inverse_timesteps[1:]):
            noise_pred = self.get_noise_pred(latents, t, prompt_utils, elevation, azimuth, camera_distances,
                guidance_scale=self.cfg.inversion_guidance_scale,
            )

            if t >= 0:
                alpha_cumprod_t = self.inverse_scheduler.alphas_cumprod[t]
            else:
                alpha_cumprod_t = self.inverse_scheduler.initial_alpha_cumprod
            
            alpha_cumprod_t_next = self.inverse_scheduler.alphas_cumprod[next_t]

            beta_prod_t = 1 - alpha_cumprod_t

            x_0_pred = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_cumprod_t ** (0.5)

            if self.inverse_scheduler.config.clip_sample:
                x_0_pred = x_0_pred.clamp(
                    -self.inverse_scheduler.config.clip_sample_range,
                    self.inverse_scheduler.config.clip_sample_range,
                )
            
            x_t_pred = (1 - alpha_cumprod_t_next) ** (0.5) * noise_pred

            latents = alpha_cumprod_t_next ** (0.5) * x_0_pred + x_t_pred

            sigma_t = self.scheduler._get_variance(next_t, t) ** (0.5)
            latents += self.cfg.inversion_eta * torch.randn_like(latents) * sigma_t

        return latents

    def sample_origin(self, original_samples, noise_pred, t):
        return self.scheduler.step(noise_pred, t[0], original_samples, return_dict=True)["pred_original_sample"]

    @torch.no_grad()
    def compute_grad_jsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        global_step: int
    ):
        latents_noisy = self.inversion(latents, t, prompt_utils, elevation, azimuth, camera_distances)

        noise_pred = self.get_noise_pred(
            latents_noisy=latents_noisy, 
            t=t, 
            prompt_utils=prompt_utils, 
            elevation=elevation,
            azimuth=azimuth, 
            camera_distances=camera_distances,
            guidance_scale=self.cfg.guidance_scale,
        )

        latents_denoised_high_density = self.sample_origin(latents_noisy, noise_pred, t).detach()

        if self.cfg.k > 0 and global_step >= self.cfg.warm_step:
            latents_denoised_high_density_noisy = self.scheduler.add_noise(
                latents_denoised_high_density, 
                torch.randn_like(latents_denoised_high_density), 
                t
            )
            noise_pred_lower_density = self.get_noise_pred(
                latents_denoised_high_density_noisy, 
                t, 
                prompt_utils,
                elevation, 
                azimuth, 
                camera_distances, 
                guidance_scale=self.cfg.guidance_scale
            )
            latents_denoised_noisy_denoised = self.sample_origin(
                latents_denoised_high_density_noisy, 
                noise_pred_lower_density, 
                t
            ).detach()

        return latents_denoised_high_density, latents_noisy, latents_denoised_noisy_denoised

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        global_step: int,
        rgb_as_latents=False,
        guidance_eval=False,
        test_info=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)

        latents = rgb if rgb_as_latents else self.encode_images(rgb_BCHW_512)

        t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device)

        high_density_target, latents_noisy, low_density_target = self.compute_grad_jsd(
            latents, t, prompt_utils, elevation, azimuth, camera_distances, global_step
        )

        loss_high_density = 0.5 * F.mse_loss(latents, high_density_target.detach(), reduction="mean") / batch_size

        if self.cfg.k > 0 and global_step >= self.cfg.warm_step:
            loss_low_density = 0.5 * F.mse_loss(latents, low_density_target.detach(), reduction="mean") / batch_size

            loss_jsd = loss_high_density + self.cfg.k * loss_low_density

            guidance_out = {
                "loss_jsd": loss_jsd,
                "lhd": loss_high_density if self.cfg.analysis else None,
                "lld": loss_low_density if self.cfg.analysis else None,
                "hd_grad_norm": (latents - high_density_target).norm() if self.cfg.analysis else None,
                "ld_grad_norm": (latents - low_density_target).norm() if self.cfg.analysis else None,
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

            if test_info and self.cfg.analysis:
                guidance_out["hd_target"] = self.decode_latents(high_density_target)[0].permute(1, 2, 0)
                guidance_out["hd_target_latent"] = high_density_target
                guidance_out["ld_target"] = self.decode_latents(low_density_target)[0].permute(1, 2, 0)
                guidance_out["ld_target_latent"] = low_density_target
                guidance_out["noisy_img"] = self.decode_latents(latents_noisy)[0].permute(1, 2, 0)
                return guidance_out
        else:
            loss_jsd = loss_high_density

            guidance_out = {
                "loss_jsd": loss_jsd,
                "hd_grad_norm": (latents - high_density_target).norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

            if test_info and self.cfg.analysis:
                guidance_out["hd_target"] = self.decode_latents(high_density_target)[0].permute(1, 2, 0)
                guidance_out["hd_target_latent"] = high_density_target
                guidance_out["noisy_img"] = self.decode_latents(latents_noisy)[0].permute(1, 2, 0)
                return guidance_out

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        percentage = float(global_step) / self.cfg.trainer_max_steps
        if type(self.cfg.max_step_percent) not in [float, int]:
            max_step_percent = self.cfg.max_step_percent[1]
        else:
            max_step_percent = self.cfg.max_step_percent
        curr_percent = (
            max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
        ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
        self.set_min_max_steps(min_step_percent=curr_percent, max_step_percent=curr_percent)