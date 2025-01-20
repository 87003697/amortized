from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
)

from torchvision.transforms import functional as F


import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup
from threestudio.utils.typing import *

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from extern.era3D.pipeline.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline

from torch.autograd import Variable, grad as torch_grad
from threestudio.utils.ops import SpecifyGradient



@threestudio.register("Era3D-asynchronous-score-distillation-guidance")
class Era3DAsynchronousScoreDistillationGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        
        n_view: int = 6
        half_precision_weights: bool=True

        # the following is specific to era3d
        era3d_model_name_or_path: str='pretrained/Era3D-512-6view'
        era3d_embeddings_dir: str='extern/era3D/data/fixed_prompt_embeds_6view'

        era3d_guidance_scale: float=3.0

        era3d_min_step_percent: Optional[float]=0.02
        era3d_max_step_percent: Optional[float]=0.98

        era3d_image_size: int = 512

        era3d_weight: float = 1.
        era3d_color_weight: float = 1.
        era3d_normal_weight: float = 1.

        era3d_weighting_strategy: str = "dmd"  

        # the following is specific to asynchrounous score distillation
        era3d_plus_random: bool = True
        era3d_plus_ratio: float = 0.1


        # strategy to save memory
        gradient_checkpoint: bool = False
        auto_grad: bool = False

        eps: float = 0.01


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Era3D ...")


        self.weight_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            self.cfg.era3d_model_name_or_path, 
            torch_dtype=self.weight_dtype
        )
        try:
            self.normal_text_embeds = torch.load(f'{self.cfg.era3d_embeddings_dir}/normal_embeds.pt').to(self.device)
            self.color_text_embeds = torch.load(f'{self.cfg.era3d_embeddings_dir}/clr_embeds.pt').to(self.device)
        except:
            raise ValueError(f"Embeddings not found in {self.cfg.era3d_embeddings_dir}")
        del pipe.text_encoder
        cleanup()

        # Create the model
        self.era3d_vae = pipe.vae.eval().to(self.device)
        self.era3d_unet = pipe.unet.eval().to(self.device)

        for p in self.era3d_vae.parameters():
            p.requires_grad_(False)
        for p in self.era3d_unet.parameters():
            p.requires_grad_(False)

        if self.cfg.gradient_checkpoint:
            self.era3d_unet.enable_gradient_checkpointing()
            self.era3d_vae.enable_gradient_checkpointing()
        self.era3d_unet.enable_xformers_memory_efficient_attention()

        self.era3d_shceduler = DDPMScheduler.from_pretrained(
                self.cfg.era3d_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weight_dtype,
            )

        self.alphas = self.era3d_shceduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = None
        self.num_train_timesteps = 1000

        assert self.cfg.n_view in [4, 6], "n_view must be 4 or 6"      

    def get_t_plus(
        self, 
        t: Float[Tensor, "B"],
        module: str # "rd" or "mv"
    ):

        
        # determine the attributes that differ between rd and MV
        if module == "era3d":
            plus_random = self.cfg.era3d_plus_random
            plus_ratio = self.cfg.era3d_plus_ratio
            min_step = self.era3d_min_step
            max_step = self.era3d_max_step
        else:
            raise ValueError(f"Invalid module: {module}")

        # determine the timestamp for the second diffusion model
        t_delta = plus_ratio * (t - min_step)

        # clamp t_delta to the range [0, T_max - t], added in the revision
        t_delta = torch.clamp(
            t_delta,
            torch.zeros_like(t), 
            self.num_train_timesteps - t - 1,
        )

        # add the offset
        if plus_random:
             t_delta = (t_delta * torch.rand(*t.shape,device = self.device)).to(torch.long)
        else:
             t_delta =  t_delta.to(torch.long)
        t_plus = t + t_delta

        # double check the range in [1, 999]
        t_plus = torch.clamp(
            t_plus,
            1, # T_min = 1
            max = self.num_train_timesteps - 1, # T_max = 999
        )
        return t_plus


    def era3d_get_latents(
        self,
        rgb_BCWH: Float[Tensor, "B C H W"],
        normal_BCWH: Float[Tensor, "B C H W"],
        rgb_as_latents: bool = False,
        rgb_2nd_BCWH: Optional[Float[Tensor, "B C H W"]] = None,
        normal_2nd_BCWH: Optional[Float[Tensor, "B C H W"]] = None,
    ) -> Float[Tensor, "B N"]:

        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd_BCWH is not None else False
        
        size =  self.cfg.era3d_image_size // 8 if rgb_as_latents \
            else self.cfg.era3d_image_size
        
        # resize the input images from the 1st rendering
        if not is_dual:
            latents = torch.cat(
                [
                    F.interpolate(
                        normal_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        rgb_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim = 0,
            )
        else:
            # normal 1st, then normal 2nd, then color 1st, then color 2nd
            latents = torch.cat(
                [
                    F.interpolate(
                        normal_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        normal_2nd_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        rgb_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        rgb_2nd_BCWH,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim = 0,
            )
        
        if not rgb_as_latents:
            # convert the images to latents
            input_dtype = latents.dtype
            latents = latents * 2.0 - 1.0
            posterior = self.era3d_vae.encode(
                    latents.to(self.weight_dtype)
                ).latent_dist
            latents = posterior.sample() * self.era3d_vae.config.scaling_factor
            latents = latents.to(input_dtype)

        return latents


    def _era3d_noise_pred(
        self,
        era3d_latents: Float[Tensor, "B C H W"],
        era3d_noise: Float[Tensor, "B C H W"],
        image_embeddings_cond: Float[Tensor, "B C"],
        image_embeddings_uncond: Float[Tensor, "B C"],
        image_latents_cond: Float[Tensor, "B C H W"],
        image_latents_uncond: Float[Tensor, "B C H W"],
        t: Float[Tensor, "B"],
        t_plus: Optional[Float[Tensor, "B"]] = None,
        is_dual: bool = False,
    ) -> Tuple[Float[Tensor, "B C H W"], Float[Tensor, "B C H W"], Float[Tensor, "B C H W"]]:

        # determin is asynchronous timesteps are enabled
        use_t_plus = True if t_plus is not None else False

        # prepare image embeddings ################################################################################################
        image_embeddings = [
            image_embeddings_cond,
            image_embeddings_uncond,
        ]
        if use_t_plus:
            image_embeddings += [
                image_embeddings_cond,
            ]
        image_embeddings = torch.cat(
            image_embeddings * 2, # 2 for both normal and color
            dim=0,
        )

        # prepare text embeddings ################################################################################################
        num_repeats = 3 if use_t_plus else 2
        if is_dual:
            num_repeats *= 2
        text_embeddings = torch.cat(
            [
                self.normal_text_embeds,
            ] * num_repeats + \
            [
                self.color_text_embeds,
            ] * num_repeats,
            dim=0,
        )

        # prepare noisy latents ################################################################################################
        latents_noisy = self.era3d_shceduler.add_noise(
            era3d_latents,
            era3d_noise,
            t,
        )
        latents_noisy_normal, latents_noisy_color = torch.chunk(
            latents_noisy,
            2,
            dim=0,
        )

        if use_t_plus:
            latents_noisy_second = self.era3d_shceduler.add_noise(
                era3d_latents,
                era3d_noise,
                t_plus,
            )
            latents_noisy_normal_second, latents_noisy_color_second = torch.chunk(
                latents_noisy_second,
                2,
                dim=0,
            )

        latent_model_input = []
        latent_model_input += [
            torch.cat(
                [
                    latents_noisy_normal,
                    image_latents_cond,
                ],
                dim=1,
            ), # normal-t/cond
            torch.cat(
                [
                    latents_noisy_normal,
                    image_latents_uncond,
                ],
                dim=1,
            ), # normal-t/uncond
        ]
        if use_t_plus:
            latent_model_input += [
                torch.cat(
                    [
                        latents_noisy_normal_second,
                        image_latents_cond,
                    ],
                    dim=1,
                ), # normal-t_plus/cond
            ]
        latent_model_input += [
            torch.cat(
                [
                    latents_noisy_color,
                    image_latents_cond,
                ],
                dim=1,
            ), # color-t/cond
            torch.cat(
                [
                    latents_noisy_color,
                    image_latents_uncond,
                ],
                dim=1,
            ), # color-t/uncond
        ]
        if use_t_plus:
            latent_model_input += [
                torch.cat(
                    [
                        latents_noisy_color_second,
                        image_latents_cond,
                    ],
                    dim=1,
                ), # color-t_plus/cond
            ]
        latent_model_input = torch.cat(
            latent_model_input,
            dim=0,
        )

        # prepare time steps ################################################################################################
        t_half, _ = torch.chunk(
            t,
            2,
            dim=0,
        ) # the 1st half equals to the 2nd half
        t_expanded = []
        t_expanded += [
            t_half, # normal-t/cond
            t_half, # normal-t/uncond
        ]
        if use_t_plus:
            t_plus_half, _ = torch.chunk(
                t_plus,
                2,
                dim=0,
            ) # the 1st half equals to the 2nd half
            t_expanded += [
                t_plus_half, # normal-t_plus/cond
            ]

        t_expanded += [
            t_half, # color-t/cond
            t_half, # color-t/uncond
        ]
        if use_t_plus:
            t_expanded += [
                t_plus_half, # color-t_plus/cond
            ]
        t_expanded = torch.cat(
            t_expanded,
            dim=0,
        )

        # forward pass ################################################################################################
        noise_pred = self.era3d_unet(
            latent_model_input.to(self.weight_dtype),
            t_expanded,
            encoder_hidden_states=text_embeddings, # already in the weight_dtype
            class_labels=image_embeddings.to(self.weight_dtype),
            return_dict=False,
        )[0].to(era3d_latents.dtype)

        if use_t_plus:
            normal_noise_pred_cond, \
                normal_noise_pred_uncond, \
                    normal_noise_pred_cond_2nd, \
                        color_noise_pred_cond, \
                            color_noise_pred_uncond, \
                                color_noise_pred_cond_2nd = torch.chunk(noise_pred, 6, dim=0)
        else:
            normal_noise_pred_cond, \
                normal_noise_pred_uncond, \
                    color_noise_pred_cond, \
                        color_noise_pred_uncond = torch.chunk(noise_pred, 4, dim=0)
            normal_noise_pred_cond_2nd = normal_noise_pred_cond
            color_noise_pred_cond_2nd = color_noise_pred_cond

        return normal_noise_pred_cond, \
                normal_noise_pred_uncond, \
                    normal_noise_pred_cond_2nd, \
                        color_noise_pred_cond, \
                            color_noise_pred_uncond, \
                                color_noise_pred_cond_2nd



    def _era3d_grad_asd(
        self,
        rgb: Float[Tensor, "N_view H W C"],
        normal: Float[Tensor, "N_view H W C"],
        prompt_utils: PromptProcessorOutput,
        azimuth: Float[Tensor, "N_view"],
        rgb_as_latents: bool = False,
        rgb_2nd: Optional[Float[Tensor, "N_view H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "N_view H W C"]] = None,
        **kwargs,
    ):

        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd is not None else False
        
        view_batch_size = rgb.shape[0]
        img_batch_size = rgb.shape[0] + normal.shape[0] 

        # special case for dual rendering
        if is_dual:
            img_batch_size *= 2

        ################################################################################################
        # prepare latents
        # normal first, then color
        era3d_latents = self.era3d_get_latents(
            rgb_BCWH=rgb.permute(0, 3, 1, 2),
            normal_BCWH=normal.permute(0, 3, 1, 2),
            rgb_as_latents=rgb_as_latents,
            rgb_2nd_BCWH=rgb_2nd.permute(0, 3, 1, 2) \
                    if is_dual else None,
            normal_2nd_BCWH=normal_2nd.permute(0, 3, 1, 2) \
                    if is_dual else None,
        )

        # prepare noisy input
        era3d_noise = torch.randn_like(era3d_latents)

        # prepare conditions
        condict = prompt_utils.get_image_encodings()
        image_latents: Float[Tensor, "1 C H W"] = condict["image_latents"]
        image_embeddings: Float[Tensor, "1 C"] = condict["image_embeddings_global"]

        condition_batch_size = image_embeddings.shape[0]

        # repeat the text embeddings w.r.t. the number of views
        """
            assume n_view = 4
            ->
            [
                render_1, normal, view_1, cond,
                ...
                render_1, normal, view_4, cond,
                render_1, normal, view_1, uncond,
                ...
                render_1, normal, view_4, uncond,
                render_2, normal, view_1, cond,
                ...
                render_2, normal, view_4, cond,
                render_2, normal, view_1, uncond,
                ...
                render_2, normal, view_4, uncond,
                render_1, color, view_1, cond,
                ...
                render_1, color, view_4, cond,
                render_1, color, view_1, uncond,
                ...
                render_1, color, view_4, uncond,
                render_2, color, view_1, cond,
                ...
                render_2, color, view_4, cond,
                render_2, color, view_1, uncond,
                ...
                render_2, color, view_4, uncond,
            ]
        """        
        # N_view, c <-- 1, c
        image_embeddings_cond = image_embeddings.repeat_interleave(self.cfg.n_view, dim=0)
        image_embeddings_uncond = torch.zeros_like(image_embeddings_cond)
        
        image_latents_cond = image_latents.repeat_interleave(self.cfg.n_view, dim=0)
        image_latents_uncond = torch.zeros_like(image_latents_cond)

        if is_dual:
            image_embeddings_cond = image_embeddings_cond.repeat(2, 1)
            image_embeddings_uncond = image_embeddings_uncond.repeat(2, 1)

            image_latents_cond = image_latents_cond.repeat(2, 1, 1, 1)
            image_latents_uncond = image_latents_uncond.repeat(2, 1, 1, 1)

        with torch.no_grad():
            min_t, max_t = self.era3d_min_step, self.era3d_max_step
            _t = torch.randint(
                min_t, 
                max_t,
                [condition_batch_size if not is_dual else condition_batch_size * 2],
                dtype=torch.long,
                device=self.device,
            )
            # determine to use asynchrounous timesteps
            use_t_plus = True if self.cfg.era3d_plus_ratio > 0 else False
            if use_t_plus:
                _t_plus = self.get_t_plus(_t, module="era3d")

                # keep consistent with the number of views
                t = _t.repeat_interleave(self.cfg.n_view).repeat(2) # normal and color
                t_plus  = _t_plus.repeat_interleave(self.cfg.n_view).repeat(2) # normal and color

            else:
                # keep consistent with the number of views
                t = _t.repeat_interleave(self.cfg.n_view).repeat(2) # normal and color
                t_plus = None

            # perform noise prediction
            normal_noise_pred_cond, \
                normal_noise_pred_uncond, \
                    normal_noise_pred_2nd, \
                        color_noise_pred_cond, \
                            color_noise_pred_uncond, \
                                color_noise_pred_2nd  = self._era3d_noise_pred(
                                                                era3d_latents,
                                                                era3d_noise,
                                                                image_embeddings_cond,
                                                                image_embeddings_uncond,
                                                                image_latents_cond,
                                                                image_latents_uncond,
                                                                t,
                                                                t_plus = t_plus,
                                                                is_dual = is_dual,
                                                            )
            normal_noise_pred_1st = normal_noise_pred_uncond + self.cfg.era3d_guidance_scale * (
                normal_noise_pred_cond - normal_noise_pred_uncond
            )
            color_noise_pred_1st = color_noise_pred_uncond + self.cfg.era3d_guidance_scale * (
                color_noise_pred_cond - color_noise_pred_uncond
            )

        normal_era3d_latents, color_era3d_latents = torch.chunk(
            era3d_latents,
            2,
            dim=0,
        )
            
        if self.cfg.era3d_weighting_strategy == "dmd":
            with torch.no_grad():
                t, _ = torch.chunk( # the timestep for the normal and color are the same
                    t,
                    2,
                    dim=0,
                )
                alpha = (
                    self.alphas[t] ** 0.5
                ).view(-1, 1, 1, 1)
                sigma = (
                    (
                        1 - self.alphas[t]
                    ) ** 0.5
                ).view(-1, 1, 1, 1)

                # the gradient of the normal
                normal_latents_first = (
                    normal_era3d_latents - sigma * normal_noise_pred_1st
                ) / alpha
                normal_latents_second = (
                    normal_era3d_latents - sigma * normal_noise_pred_2nd
                ) / alpha
                w = torch.abs(
                    normal_era3d_latents - normal_latents_first
                ).mean(
                    dim=(1, 2, 3), 
                    keepdim=True
                )
                normal_grad = (normal_latents_second - normal_latents_first) / (w + self.cfg.eps)

                # the gradient of the color
                color_latents_first = (
                    color_era3d_latents - sigma * color_noise_pred_1st
                ) / alpha
                color_latents_second = (
                    color_era3d_latents - sigma * color_noise_pred_2nd
                ) / alpha
                w = torch.abs(
                    color_era3d_latents - color_latents_first
                ).mean(
                    dim=(1, 2, 3), 
                    keepdim=True
                )
                color_grad = (color_latents_second - color_latents_first) / (w + self.cfg.eps)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.era3d_weighting_strategy}")

        normal_grad = torch.nan_to_num(normal_grad)
        color_grad = torch.nan_to_num(color_grad)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target_normal = (normal_era3d_latents - normal_grad).detach()
        target_color = (color_era3d_latents - color_grad).detach()

        if not is_dual:
            loss_asd_normal = 0.5 * F.mse_loss(
                normal_era3d_latents,
                target_normal,
                reduction="sum",
            )
            loss_asd_color = 0.5 * F.mse_loss(
                color_era3d_latents,
                target_color,
                reduction="sum",
            )
            return loss_asd_normal, normal_grad.norm(), loss_asd_color, color_grad.norm()
        else:
            loss_asd_normal = torch.stack(
                [
                    0.5 * F.mse_loss(
                        normal_era3d_latents[:view_batch_size],
                        target_normal[:view_batch_size],
                        reduction="sum",
                    ),
                    0.5 * F.mse_loss(
                        color_era3d_latents[view_batch_size:],
                        target_color[view_batch_size:],
                        reduction="sum",
                    ),
                ]
            )
            normal_grad_norm = torch.stack(
                [
                    normal_grad[:view_batch_size].norm(),
                    normal_grad[view_batch_size:].norm(),
                ]
            )
            
            loss_asd_color = torch.stack(
                [
                    0.5 * F.mse_loss(
                        normal_era3d_latents[:view_batch_size],
                        target_normal[:view_batch_size],
                        reduction="sum",
                    ),
                    0.5 * F.mse_loss(
                        color_era3d_latents[view_batch_size:],
                        target_color[view_batch_size:],
                        reduction="sum",
                    ),
                ]
            )

            color_grad_norm = torch.stack(
                [
                    color_grad[:view_batch_size].norm(),
                    color_grad[view_batch_size:].norm(),
                ]
            )
            return loss_asd_normal, normal_grad_norm, loss_asd_color, color_grad_norm



    def era3d_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        azimuth: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        **kwargs,
    ):
        if self.cfg.auto_grad:
            rgb_var = Variable(rgb, requires_grad=True)
            normal_var = Variable(normal, requires_grad=True)
            
            if rgb_2nd is not None:
                rgb_2nd_var = Variable(rgb_2nd, requires_grad=True)
                normal_2nd_var = Variable(normal_2nd, requires_grad=True)

                normal_loss_era3d, normal_grad_norm_era3d, \
                    color_loss_era3d, color_grad_norm_era3d \
                = self._era3d_grad_asd(
                        rgb_var, 
                        normal_var,
                        prompt_utils,
                        azimuth,
                        rgb_as_latents = rgb_as_latents,
                        rgb_2nd = rgb_2nd_var,
                        normal_2nd = normal_2nd_var,
                        **kwargs,
                    )
                
                grad_normal, grad_normal_2nd, \
                    grad_color, grad_color_2nd \
                = torch_grad(
                    (normal_loss_era3d + color_loss_era3d).sum(),
                    ([normal_var, normal_2nd_var, rgb_var, rgb_2nd_var]),
                )

                normal_loss_era3d = torch.cat(
                    [
                        SpecifyGradient.apply(normal, grad_normal),
                        SpecifyGradient.apply(normal_2nd, grad_normal_2nd),
                    ],
                    dim=0,
                )
                color_loss_era3d = torch.cat(
                    [
                        SpecifyGradient.apply(rgb, grad_color),
                        SpecifyGradient.apply(rgb_2nd, grad_color_2nd),
                    ],
                    dim=0,
                )
                return normal_loss_era3d, normal_grad_norm_era3d, color_loss_era3d, color_grad_norm_era3d

            else:
                normal_loss_era3d, normal_grad_norm_era3d, \
                    color_loss_era3d, color_grad_norm_era3d \
                = self._era3d_grad_asd(
                        rgb_var, 
                        normal_var,
                        prompt_utils,
                        azimuth,
                        rgb_as_latents = rgb_as_latents,
                        **kwargs,
                    )
                
                grad_normal, grad_color = torch_grad(
                    (normal_loss_era3d + color_loss_era3d).sum(),
                    (normal_var, rgb_var),
                )

                normal_loss_era3d = SpecifyGradient.apply(normal, grad_normal)
                color_loss_era3d = SpecifyGradient.apply(rgb, grad_color)

                return normal_loss_era3d, normal_grad_norm_era3d, color_loss_era3d, color_grad_norm_era3d

        else:
            return self._era3d_grad_asd(
                rgb,
                normal,
                prompt_utils,
                azimuth,
                rgb_as_latents=rgb_as_latents,
                rgb_2nd=rgb_2nd,
                normal_2nd=normal_2nd,
                **kwargs,
            )



    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_distances_relative: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        depth_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        **kwargs,
    ):
        """
            # illustration of the concatenated rgb and rgb_2nd, assume n_view = 4
            # rgb: Float[Tensor, "B H W C"]
            [
                render_1st, prompt_1, view_1,
                ...
                render_1st, prompt_1, view_4,
                render_1st, prompt_2, view_1,
                ...
                render_1st, prompt_2, view_4,
                render_2nd, prompt_1, view_1,
                ...
                render_2nd, prompt_1, view_4,
                render_2nd, prompt_2, view_1,
                ...
                render_2nd, prompt_2, view_4,
            ]
        """
        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd is not None else False

        ################################################################################################
        # the following is specific to Era3D
        if self.era3d_weight > 0:
            normal_loss_era3d, normal_grad_norm_era3d, \
                color_loss_era3d, color_grad_norm_era3d \
            = self.era3d_grad_asd(
                    rgb, 
                    normal,
                    prompt_utils,
                    azimuth,
                    rgb_as_latents = rgb_as_latents,
                    rgb_2nd = rgb_2nd,
                    normal_2nd = normal_2nd,
                    **kwargs,
                )
        else:
            normal_loss_era3d = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0], 
                device=self.device
            )
            normal_grad_norm_era3d = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0], 
                device=self.device
            )
            color_loss_era3d = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0], 
                device=self.device
            )
            color_grad_norm_era3d = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0], 
                device=self.device
            )

        # return the loss and grad
        if not is_dual:
            return {
                "loss_asd": self.era3d_weight * (
                    self.cfg.era3d_normal_weight * normal_loss_era3d + \
                    self.cfg.era3d_color_weight * color_loss_era3d
                ),
                "grad_asd": self.era3d_weight * (
                    self.cfg.era3d_normal_weight * normal_grad_norm_era3d + \
                    self.cfg.era3d_color_weight * color_grad_norm_era3d
                )
            }
        else:
            # return the loss and grad_norm for the 1st renderings
            loss = 0
            grad_norm = 0

            loss += self.era3d_weight * (
                self.cfg.era3d_normal_weight * normal_loss_era3d[0] + \
                self.cfg.era3d_color_weight * color_loss_era3d[0]
            )
            grad_norm += self.era3d_weight * (
                self.cfg.era3d_normal_weight * normal_grad_norm_era3d[0] + \
                self.cfg.era3d_color_weight * color_grad_norm_era3d[0]
            )

            guidance_1st = {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,
            }

            # return the loss and grad_norm for the 2nd renderings
            loss = 0
            grad_norm = 0

            loss += self.era3d_weight * (
                self.cfg.era3d_normal_weight * normal_loss_era3d[1] + \
                self.cfg.era3d_color_weight * color_loss_era3d[1]
            )
            grad_norm += self.era3d_weight * (
                self.cfg.era3d_normal_weight * normal_grad_norm_era3d[1] + \
                self.cfg.era3d_color_weight * color_grad_norm_era3d[1]
            )

            guidance_2nd = {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,
            }
            return guidance_1st, guidance_2nd
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        
        # update the weights and min/max step for each module ################################################################################################
        self.era3d_weight = C(self.cfg.era3d_weight, epoch, global_step)
        self.era3d_min_step = int(self.num_train_timesteps * C(self.cfg.era3d_min_step_percent, epoch, global_step))
        self.era3d_max_step = int(self.num_train_timesteps * C(self.cfg.era3d_max_step_percent, epoch, global_step))



