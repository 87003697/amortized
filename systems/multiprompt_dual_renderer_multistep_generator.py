import os
import shutil
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_rank, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from functools import partial

from tqdm import tqdm
from threestudio.utils.misc import barrier
from threestudio.models.mesh import Mesh

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

def sample_timesteps(
    all_timesteps: List,
    num_parts: int,
    batch_size: int = 1,
):
    # separate the timestep into num_parts_training parts
    timesteps = []
    for i in range(num_parts):
        length_timestep = len(all_timesteps) // num_parts
        timestep = all_timesteps[
            i * length_timestep : (i + 1) * length_timestep
        ]
        # sample only one from the timestep
        idx = torch.randint(0, len(timestep), (batch_size,))
        timesteps.append(timestep[idx])
    return timesteps

@threestudio.register("multiprompt-dual-renderer-multistep-generator-system")
class MultipromptDualRendererMultiStepGeneratorSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):

        # validation related
        visualize_samples: bool = False

        # renderering related
        rgb_as_latents: bool = False

        # initialization related
        initialize_shape: bool = True

        # if the guidance requires training
        train_guidance: bool = False

        # added another renderer
        renderer_2nd_type: str = ""
        renderer_2nd: dict = field(default_factory=dict)

        # parallelly compute the guidance
        parallel_guidance: bool = False

        # scheduler path
        scheduler_dir: str = "pretrained/stable-diffusion-2-1-base"

        # the followings are related to the multi-step diffusion
        pure_zeros_training: bool = False
        num_parts_training: int = 4

        num_steps_training: int = 50
        num_steps_sampling: int = 50

        classifier_guidance_scale: float = 1.0
        
        sample_scheduler: str = "ddpm" #any of "ddpm", "ddim"
        noise_scheduler: str = "ddim"


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        # set up the second renderer
        self.renderer_2nd = threestudio.find(self.cfg.renderer_2nd_type)(
            self.cfg.renderer_2nd,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Sampler for training
        self.noise_scheduler = self._configure_scheduler(self.cfg.noise_scheduler)
        self.is_training_odd = True if self.cfg.noise_scheduler == "ddpm" else False

        # Sampler for inference
        self.sample_scheduler = self._configure_scheduler(self.cfg.sample_scheduler)

        # This property activates manual optimization.
        self.automatic_optimization = False 


    def _configure_scheduler(self, scheduler: str):
        assert scheduler in ["ddpm", "ddim", "dpm"]
        if scheduler == "ddpm":
            return DDPMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "ddim":
            return DDIMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "dpm":
            return DPMSolverMultistepScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )


    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training

        if not self.cfg.train_guidance: # if the guidance does not require training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize SDF
        if self.cfg.initialize_shape:
            # info
            if get_device() == "cuda_0": # only report from one process
                threestudio.info("Initializing shape...")
            
            # check if attribute exists
            if not hasattr(self.geometry, "initialize_shape"):
                threestudio.info("Geometry does not have initialize_shape method. skip.")
            else:
                self.geometry.initialize_shape()

    def forward_rendering(
        self,
        batch: Dict[str, Any],
    ):

        render_out = self.renderer(**batch, )
        render_out_2nd = self.renderer_2nd(**batch, )

        # decode the rgb as latents only in testing and validation
        if self.cfg.rgb_as_latents and not self.training: 
            # get the rgb
            if "comp_rgb" not in render_out:
                raise ValueError(
                    "comp_rgb is required for rgb_as_latents, no comp_rgb is found in the output."
                )
            else:
                out_image = render_out["comp_rgb"]
                out_image = self.guidance.decode_latents(
                    out_image.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1) 
                render_out['decoded_rgb'] = out_image

                out_image_2nd = render_out_2nd["comp_rgb"]
                out_image_2nd = self.guidance.decode_latents(
                    out_image_2nd.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
                render_out_2nd['decoded_rgb'] = out_image_2nd

        return {
            **render_out,
        }, {
            **render_out_2nd,
        }
        
    def compute_guidance_n_loss(
        self,
        out: Dict[str, Any],
        out_2nd: Dict[str, Any],
        idx: int,
        **batch,
    ) -> Dict[str, Any]:
        # guidance for the first renderer
        guidance_inp = out["comp_rgb"]

        # guidance for the second renderer
        guidance_inp_2nd = out_2nd["comp_rgb"]

        if not self.cfg.parallel_guidance:
            # the guidance is computed in two steps
            guidance_out = self.guidance(
                guidance_inp, 
                # batch["prompt_utils"], 
                **batch, 
                rgb_as_latents=self.cfg.rgb_as_latents,
            )

            guidance_out_2nd = self.guidance(
                guidance_inp_2nd, 
                # batch["prompt_utils"],
                **batch, 
                rgb_as_latents=self.cfg.rgb_as_latents,
            )
        else:
            # the guidance is computed in parallel
            guidance_out, guidance_out_2nd = self.guidance(
                guidance_inp,
                # batch["prompt_utils"],
                **batch,
                rgb_as_latents=self.cfg.rgb_as_latents,
                rgb_2nd = guidance_inp_2nd,
            )

        loss = self._compute_loss(guidance_out, out, renderer="1st", step = idx, **batch)
        loss_2nd = self._compute_loss(guidance_out_2nd, out_2nd, renderer="2nd", step = idx, **batch)

        return {
            "loss": loss["loss"] + loss_2nd["loss"]
        }

    def diffusion_reverse(
        self,
        batch: Dict[str, Any],
    ):

        prompt_utils = batch["prompt_utils"]
        if "prompt_target" in batch:
           raise NotImplementedError
        else:
            # more general case
            text_embed_cond = prompt_utils.get_global_text_embeddings()
            text_embed_uncond = prompt_utils.get_uncond_text_embeddings()
        
        text_embed = torch.cat(
            [
                text_embed_cond,
                text_embed_uncond,
            ],
            dim=0,
        )

        self.sample_scheduler.set_timesteps(self.cfg.num_steps_sampling)
        timesteps = self.sample_scheduler.timesteps

        latents = batch.pop("noise")

        for i, t in enumerate(timesteps):

            # prepare inputs
            noisy_latent_input = self.sample_scheduler.scale_model_input(
                latents, 
                t
            )
            noisy_latent_input = torch.cat([noisy_latent_input] * 2, dim=0)

            # predict the noise added
            noise_pred = self.geometry.denoise(
                noisy_input = noisy_latent_input,
                text_embed = text_embed, # TODO: text_embed might be null
                timestep = t.to(self.device),
            )

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.classifier_guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.sample_scheduler.step(noise_pred, t, latents).prev_sample


        # decode the latent to 3D representation
        space_cache = self.geometry.decode(
            latents = latents,
        )

        return space_cache

            


    def training_step(
        self,
        batch_list: List[Dict[str, Any]],
        batch_idx
    ):
        """
            Diffusion Forward Process
            but supervised by the 2D guidance
        """
        latent = batch_list[0]["noise"]
        batch_size = batch_list[0]["prompt_utils"].get_global_text_embeddings().shape[0] # sort of hacky

        self.noise_scheduler.set_timesteps(self.cfg.num_steps_training)
        timesteps = sample_timesteps(
            self.noise_scheduler.timesteps,
            num_parts = self.cfg.num_parts_training,
            batch_size=batch_size,
        )

        # zero the gradients
        opt = self.optimizers()
        opt.zero_grad()
    
        for i, (t, batch) in enumerate(zip(timesteps, batch_list)):

            # prepare the text embeddings as input
            prompt_utils = batch["prompt_utils"]
            if "prompt_target" in batch:
                raise NotImplementedError
            else:
                # more general case
                batch["text_embed"] = prompt_utils.get_global_text_embeddings()
                batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
        
            # add noise to the latent
            noise = torch.randn_like(latent) if not self.cfg.pure_zeros_training else torch.zeros_like(latent)
            noisy_latent = self.noise_scheduler.add_noise(
                latent,
                noise,
                t,
            )
 
            # prepare the text embeddings as input
            text_embed = batch["text_embed"]
            # under 10% of the time, the text embed is set to None
            if torch.rand(1) < 0.1:
                text_embed = prompt_utils.get_uncond_text_embeddings()

            # predict the noise added
            noise_pred = self.geometry.denoise(
                noisy_input = noisy_latent,
                text_embed = text_embed, # TODO: text_embed might be null
                timestep = t.to(self.device),
            )


            # convert epsilon into x0
            alphas: Float[Tensor, "..."] = self.noise_scheduler.alphas_cumprod.to(
                self.device
            )
            alpha = (alphas[t] ** 0.5).view(-1, 1, 1, 1).to(self.device)
            sigma = ((1 - alphas[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
            denoised_latents = (
                noisy_latent - sigma * noise_pred
            ) / alpha

            # decode the latent to 3D representation
            batch["space_cache"] = self.geometry.decode(
                latents = denoised_latents,
            )

            # render the image and compute the gradients
            out, out_2nd = self.forward_rendering(batch)
            loss = self.compute_guidance_n_loss(
                out, out_2nd, idx = i, **batch
            )

            # store gradients
            self.manual_backward(loss["loss"] / self.cfg.num_parts_training)

            # prepare for the next iteration
            latent = denoised_latents.detach()
            
        # update the weights
        opt.step()

    def validation_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        batch["space_cache"]  = self.diffusion_reverse(batch)
        out, out_2nd = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving val images"):
            self._save_image_grid(batch, batch_idx, out, phase="val", render="1st")
            self._save_image_grid(batch, batch_idx, out_2nd, phase="val", render="2nd")
                
        if self.cfg.visualize_samples:
            raise NotImplementedError

    def test_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        batch["space_cache"] = self.diffusion_reverse(batch)
        out, out_2nd = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
            self._save_image_grid(batch, batch_idx, out, phase="test", render="1st")
            self._save_image_grid(batch, batch_idx, out_2nd, phase="test", render="2nd")

    def _compute_loss(
        self,
        guidance_out: Dict[str, Any],
        out: Dict[str, Any],
        renderer: str = "1st",
        step: int = 0,
        **batch,
    ):
        
        assert renderer in ["1st", "2nd"]

        loss = 0.0
        for name, value in guidance_out.items():
            if renderer == "1st":
                self.log(f"train/{name}_{step}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            else:
                self.log(f"train/{name}_2nd_{step}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_") + "_2nd"])

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_orient) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_orient_2nd) > 0):
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            if renderer == "1st":
                self.log(f"train/loss_orient_{step}", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)
            else:
                self.log(f"train/loss_orient_2nd_{step}", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient_2nd)

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_sparsity) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_sparsity_2nd) > 0):
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            if renderer == "1st":
                self.log(f"train/loss_sparsity_{step}", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            else:
                self.log(f"train/loss_sparsity_2nd_{step}", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity_2nd)


        if (renderer == "1st" and self.C(self.cfg.loss.lambda_opaque) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_opaque_2nd) > 0):
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            if renderer == "1st":
                self.log(f"train/loss_opaque_{step}", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
            else:
                self.log(f"train/loss_opaque_2nd_{step}", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque_2nd)

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_z_variance) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_z_variance_2nd) > 0):
            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if 'z_variance' not in out:
                raise ValueError(
                    "z_variance is required for z_variance loss, no z_variance is found in the output."
                )
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            if renderer == "1st":
                self.log(f"train/loss_z_variance_{step}", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            else:
                self.log(f"train/loss_z_variance_2nd_{step}", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance_2nd)

        # sdf loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_eikonal) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_eikonal_2nd) > 0):
            if 'sdf_grad' not in out:
                raise ValueError(
                    "sdf is required for eikonal loss, no sdf is found in the output."
                )
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            
            if renderer == "1st":
                self.log(f"train/loss_eikonal_{step}", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
            else:
                self.log(f"train/loss_eikonal_2nd_{step}", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal_2nd)

        # normal consistency loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_normal_consistency) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_normal_consistency_2nd) > 0):
            if 'mesh' in out:
                if not isinstance(out["mesh"], list):
                    out["mesh"] = [out["mesh"]]
                loss_normal_consistency = 0.0
                for mesh in out["mesh"]:
                    assert isinstance(mesh, Mesh), "mesh should be an instance of Mesh"
                    loss_normal_consistency += mesh.normal_consistency()
            else:
                raise ValueError(
                    "mesh is required for normal consistency loss, no mesh is found in the output."
                )

            if renderer == "1st":
                self.log(f"train/loss_normal_consistency_{step}", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency)
            else:
                self.log(f"train/loss_normal_consistency_2nd_{step}", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency_2nd)
        
        # laplacian loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_laplacian_smoothness_2nd) > 0):
            if 'mesh' in out:
                if not isinstance(out["mesh"], list):
                    out["mesh"] = [out["mesh"]]
                loss_laplacian = 0.0
                for mesh in out["mesh"]:
                    assert isinstance(mesh, Mesh), "mesh should be an instance of Mesh"
                    loss_laplacian += mesh.laplacian()

            else:
                raise ValueError(
                    "mesh is required for laplacian loss, no mesh is found in the output."
                )
            
            if renderer == "1st":
                self.log(f"train/loss_laplacian_smoothness_{step}", loss_laplacian)
                loss += loss_laplacian * self.C(self.cfg.loss.lambda_laplacian_smoothness)
            else:
                self.log(f"train/loss_laplacian_smoothness_2nd_{step}", loss_laplacian)
                loss += loss_laplacian * self.C(self.cfg.loss.lambda_laplacian_smoothness_2nd)
            
        # lambda_normal_smoothness_2d
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_normal_smoothness_2d_2nd) > 0):
            normal = out["comp_normal"]
            loss_normal_smoothness_2d = (
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean() +
                (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
            )
            if renderer == "1st":
                self.log(f"train/loss_normal_smoothness_2d_{step}", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(self.cfg.loss.lambda_normal_smoothness_2d)
            else:
                self.log(f"train/loss_normal_smoothness_2d_2nd_{step}", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(self.cfg.loss.lambda_normal_smoothness_2d_2nd)

        if "inv_std" in out:
            self.log("train/inv_std", out["inv_std"], prog_bar=True)

        return {"loss": loss}


    def _save_image_grid(
        self, 
        batch,
        batch_idx,
        out,
        phase="val",
        render="1st",
    ):
        
        assert phase in ["val", "test"]

        # save the image with the same name as the prompt
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')
        # specify the image name
        image_name  = f"it{self.true_global_step}-{phase}-{render}/{name}/{str(batch['index'][batch_idx].item())}.png"
        # specify the verbose name
        verbose_name = f"{phase}_{render}_step"

        # normalize the depth
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())

        self.save_image_grid(
            image_name,
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][batch_idx] if not self.cfg.rgb_as_latents else out["decoded_rgb"][batch_idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][batch_idx],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][batch_idx, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": normalize(out["depth"][batch_idx, :, :, 0]),
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "depth" in out
                else []
            ),
            name=verbose_name,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-val-{render}"
            for render in ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                for prompt in tqdm(
                    os.listdir(os.path.join(self.get_save_dir(), filestem)),
                    desc="Generating validation videos",
                ):
                    try:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            multithreaded=True,
                        )
                    except:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            # multithreaded=True,
                        )

    def on_test_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-test-{render}"
            for render in ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                for prompt in tqdm(
                    os.listdir(os.path.join(self.get_save_dir(), filestem)),
                    desc="Generating validation videos",
                ):
                    try:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=30,
                            name="test",
                            step=self.true_global_step,
                            multithreaded=True,
                        )
                    except:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            # multithreaded=True,
                        )