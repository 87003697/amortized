import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from .base_callable import MultiRefProcessor, hash_prompt, hash_image
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.utils.misc import get_rank
from tqdm import tqdm

import torchvision.transforms.functional as TF
from diffusers import StableUnCLIPImg2ImgPipeline
import cv2, numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os

from functools import partial

# create custom image loader
class ImageLoader:
    def __init__(self, image_paths, image_root_dir, transform):
        self.transform = transform
        self.image_root_dir = image_root_dir
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root_dir, self.image_paths[idx])
        image = self.transform(image_path)
        return self.image_paths[idx], image
    
    def collate_fn(self, batch):
        paths = [x[0] for x in batch]
        images = [x[1] for x in batch]
        return paths, images



@threestudio.register("sd-unclip-multi-reference-processor-callable")
class StableUnclipCallableProcessor(MultiRefProcessor):
    @dataclass
    class Config(MultiRefProcessor.Config):        
        image_root_dir: str = ""

        use_latent: bool = False
        use_embed_global: bool = True
        use_embed_local: bool = False

        default_prompt: str = ""
        image_size: Tuple[int, int] = (512, 512)

    cfg: Config

    def load_model_text(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder",
        ).to(self.device)

        for p in text_encoder.parameters():
            p.requires_grad_(False)

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }

    def load_model_image(
        self,
    ):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
        ).to(self.device)
        del pipe.unet # we don't need the unet here
        return {
            "pipeline": pipe
        }
            
    @staticmethod
    def load_image(
        image_path,
        image_size: Tuple[int, int] = (512, 512),
    ):
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )

        rgba = (
            cv2.resize(rgba, image_size, interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        # convert to PIL image
        image = Image.fromarray((rgb * 255).astype(np.uint8))
        return image

    def func_image(
        self,
        pretrained_model_name_or_path, image_paths, cache_dir, 
        pipeline: Optional[StableUnCLIPImg2ImgPipeline] = None,
    ):

        if pipeline is None:
            modules = self.load_model_image()
            pipeline = modules.pop("pipeline")

        image_encoder = pipeline.image_encoder
        feature_extractor = pipeline.feature_extractor
        vae = pipeline.vae

        if type(image_paths) == str:
            image_paths = [image_paths]

        for image_path in image_paths:
            image = self.load_image(
                os.path.join(
                    self.cfg.image_root_dir,
                    image_path
                ),
                image_size=self.cfg.image_size
            )

            ## encode the images ############################
            global_embeddings: Float[Tensor, "B C"]
            local_embeddings: Float[Tensor, "B N C"]
            latents: Float[Tensor, "B C H W"]

            with torch.no_grad():
                # CLIP Encoder
                
                feat = feature_extractor(
                    images=[image],
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                feat = image_encoder(feat)

                global_embeddings = feat.image_embeds
                global_embeddings = pipeline.noise_image_embeddings(
                    image_embeds = global_embeddings,
                    noise_level = 0,
                )

                local_embeddings = feat.last_hidden_state

                # VAE Encoder
                image_pt = torch.stack(
                    [
                        TF.to_tensor(image)
                    ],
                    dim=0
                ).to(dtype=vae.dtype, device=self.device)
                image_pt = image_pt * 2 - 1
                latents = vae.encode(image_pt).latent_dist.mode() * vae.config.scaling_factor

            ##################################################

            for global_embedding, local_embedding, latent in zip(global_embeddings, local_embeddings, latents):
                # save the local text embeddings
                if self.cfg.use_embed_local:
                    torch.save(
                        local_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size,'local')}.pt",
                        ),
                    )
                
                # save the global text embeddings
                if self.cfg.use_embed_global:
                    torch.save(
                        global_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size, 'global')}.pt",
                        ),
                    )

                # save the latent embeddings
                if self.cfg.use_latent:
                    torch.save(
                        latent.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size, 'latent')}.pt",
                        ),
                    )

        del pipeline
        cleanup()

    def spawn_func_image(
        self,
        args
    ):
        pretrained_model_name_or_path, image_paths, cache_dir = args

        modules = self.load_model_image()
        pipeline = modules.pop("pipeline")

        image_encoder = pipeline.image_encoder
        feature_extractor = pipeline.feature_extractor
        vae = pipeline.vae

        if type(image_paths) == str:
            image_paths = [image_paths]

        dataset = ImageLoader(
            image_paths,
            self.cfg.image_root_dir,
            partial(
                self.load_image, 
                image_size=self.cfg.image_size
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, # hard coded batch size
            shuffle=False,
            num_workers=2,
            pin_memory=False, #True,
            collate_fn=dataset.collate_fn
        )

        rank = get_rank()
        for image_paths, images in tqdm(
            dataloader,
            desc="Saving image encodings in GPU {}".format(rank),
        ):
            
            ## encode the images ############################
            global_embeddings: Float[Tensor, "B C"]
            local_embeddings: Float[Tensor, "B N C"]
            latents: Float[Tensor, "B C H W"]

            with torch.no_grad():
                # CLIP Encoder
                feat = feature_extractor(
                    images=images,
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                feat = image_encoder(feat)

                global_embeddings = feat.image_embeds
                global_embeddings = pipeline.noise_image_embeddings(
                    image_embeds = global_embeddings,
                    noise_level = 0,
                )

                local_embeddings = feat.last_hidden_state

                # VAE Encoder
                image_pt = torch.stack(
                    [
                        TF.to_tensor(image) for image in images
                    ],
                    dim=0
                ).to(dtype=vae.dtype, device=self.device)
                image_pt = image_pt * 2 - 1
                latents = vae.encode(image_pt).latent_dist.mode() * vae.config.scaling_factor

            ##################################################

            for image_path, global_embedding, local_embedding, latent in zip(image_paths, global_embeddings, local_embeddings, latents):

                if self.cfg.use_embed_local:
                    torch.save(
                        local_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size, 'local')}.pt",
                        ),
                    )

                if self.cfg.use_embed_global:
                    torch.save(
                        global_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size, 'global')}.pt",
                        ),
                    )

                if self.cfg.use_latent:
                    torch.save(
                        latent.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_image(pretrained_model_name_or_path, image_path, self.cfg.image_size, 'latent')}.pt",
                        ),
                    )

        del pipeline
        cleanup()

    # def image_to_encoding(
    #     images: List[Image.Image],
    #     feature_extractor_dict: Dict[str, nn.Module],
    #     image_encoder: CLIPVisionModelWithProjection,
    #     vae: AutoencoderKL,
    #     device: str = "cuda"
    # ):
    #     if type(images) is not list:
    #         images = [images]

    #     # CLIP Encoder
    #     feat = feature_extractor(
    #         images=images,
    #         return_tensors="pt"
    #     ).pixel_values.to(device)
    #     feat = image_encoder(feat)

    #     # VAE Encoder
    #     image_pt = torch.stack(
    #             [
    #                 TF.to_tensor(image) for image in images
    #             ],
    #             dim=0
    #         ).to(dtype=vae.dtype, device=device)
    #     image_pt = image_pt * 2 - 1
    #     image_latents = vae.encode(image_pt).latent_dist.mode() * vae.config.scaling_factor
        
    #     return feat.image_embeds, feat.last_hidden_state, image_latents

    def spawn_func_text(
        self, 
        args,
    ):
        pretrained_model_name_or_path, prompt_list, cache_dir = args
        modules = self.load_model_text()
        tokenizer = modules.pop("tokenizer")
        text_encoder = modules.pop("text_encoder")

        batch_size = 32 # hard coded batch size
        rank = get_rank()
        for i in tqdm(
            range(0, len(prompt_list), batch_size),
            desc="Saving text embeddings in GPU {}".format(rank),
        ):

            prompts = prompt_list[i:i+batch_size]
            with torch.no_grad():
                input_ids = []
                for prompt in prompts:
                    tokens = tokenizer(
                        [prompt],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    # avoid exceeding max_length
                    tokens.input_ids = tokens.input_ids[:, :tokenizer.model_max_length]
                    input_ids.append(tokens.input_ids)
                
                outputs = text_encoder(torch.cat(input_ids, dim=0).to(text_encoder.device))
                # we need both the local and global text embeddings
                locals_text_embeddings, globla_text_embeddings= outputs[0], outputs[1]

            for prompt, globla_text_embedding, locals_text_embedding in zip(prompts, globla_text_embeddings, locals_text_embeddings):
                # save the global text embeddings
                torch.save(
                    globla_text_embedding.cpu(), # [0] is to remove the batch dimension
                    os.path.join(
                        cache_dir,
                        f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
                    ),
                )

                # save the local text embeddings
                torch.save(
                    locals_text_embedding.cpu(), # [0] is to remove the batch dimension
                    os.path.join(
                        cache_dir,
                        f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
                    ),
                )
                
        del text_encoder
        del tokenizer
        cleanup()

    def func_text(
        self,
        pretrained_model_name_or_path: str,
        prompts,
        cache_dir: str,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[CLIPTextModel] = None,
    ) -> Any:
        
        if tokenizer is None or text_encoder is None:
            modules = self.load_model_text()
            tokenizer = modules.pop("tokenizer")
            text_encoder = modules.pop("text_encoder")

        
        if type(prompts) == str:
            prompts = [prompts]

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            # avoid exceeding max_length
            tokens.input_ids = tokens.input_ids[:, :tokenizer.model_max_length]
            
            outputs = text_encoder(tokens.input_ids.to(text_encoder.device))
            # we need both the local and global text embeddings
            locals_text_embeddings, globla_text_embeddings= outputs[0], outputs[1]

        for prompt, globla_text_embedding, locals_text_embedding in zip(prompts, globla_text_embeddings, locals_text_embeddings):
            # save the global text embeddings
            torch.save(
                globla_text_embedding.cpu(), # [0] is to remove the batch dimension
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
                ),
            )

            # save the local text embeddings
            torch.save(
                locals_text_embedding.cpu(), # [0] is to remove the batch dimension
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
                ),
            )