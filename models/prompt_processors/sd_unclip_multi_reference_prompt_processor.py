import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKL

import threestudio
from .base_callable import MultiRefProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.utils.misc import get_rank
from tqdm import tqdm

import torchvision.transforms.functional as TF

import cv2, numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os

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
        return image_path, image
    
    def collate_fn(self, batch):
        paths = [x[0] for x in batch]
        images = [x[1] for x in batch]
        return paths, images

def image_to_encoding(
    images: List[Image.Image],
    feature_extractor: CLIPImageProcessor,
    image_encoder: CLIPVisionModelWithProjection,
    vae: AutoencoderKL,
    device: str = "cuda"
):
    if type(images) is not list:
        images = [images]

    # CLIP Encoder
    feat = feature_extractor(
        images=images,
        return_tensors="pt"
    ).pixel_values.to(device)
    feat = image_encoder(feat)

    # VAE Encoder
    image_pt = torch.stack(
            [
                TF.to_tensor(image) for image in images
            ],
            dim=0
        ).to(dtype=vae.dtype, device=device)
    image_pt = image_pt * 2 - 1
    image_latents = vae.encode(image_pt).latent_dist.mode() * vae.config.scaling_factor
    
    return feat.image_embeds, feat.last_hidden_state, image_latents

@threestudio.register("sd-unclip-multi-reference-processor-callable")
class StableZero123MultirefCallableProcessor(MultiRefProcessor):
    @dataclass
    class Config(MultiRefProcessor.Config):
        use_text_condition: bool = False
        
        image_root_dir: str = ""

        use_latent: bool = False
        use_embed_global: bool = True
        use_embed_local: bool = False

    cfg: Config


    def load_model_image(
        self,
    ):
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="feature_extractor",
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="image_encoder",
        ).to(self.device)

        vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="vae",
        ).to(self.device)

        # freeze the image encoder and vae encoder
        for param in image_encoder.parameters():
            param.requires_grad = False
        for param in vae.parameters():
            param.requires_grad = False
            
        encoder_func = lambda x: image_to_encoding(
            x, feature_extractor, image_encoder, vae, self.device
        )
        return {
            "image_encoder": image_encoder,
            "vae": vae,
            "encoder_func": encoder_func
        }
            
    @staticmethod
    def load_image(
        image_path
    ):
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )

        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
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
        image_encoder = None, vae = None, encoder_func = None, 
    ):

        if encoder_func is None or image_encoder is None or vae is None:
            modules = self.load_model_image()
            encoder_func = modules.pop("encoder_func")
            image_encoder = modules.pop("image_encoder")
            vae = modules.pop("vae")

        if type(image_paths) == str:
            image_paths = [image_paths]

        for image_path in image_paths:
            image = self.load_image(
                os.path.join(
                    self.cfg.image_root_dir,
                    image_path
                ) 
            )

            with torch.no_grad():
                global_embeddings, local_embeddings, latents = encoder_func(image)

            for global_embedding, local_embedding, latent in zip(global_embeddings, local_embeddings, latents):
                # save the local text embeddings
                if self.cfg.use_embed_local:
                    torch.save(
                        local_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'local')}.pt",
                        ),
                    )
                
                # save the global text embeddings
                if self.cfg.use_embed_global:
                    torch.save(
                        global_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'global')}.pt",
                        ),
                    )

                # save the latent embeddings
                if self.cfg.use_latent:
                    torch.save(
                        latent.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'latent')}.pt",
                        ),
                    )

        del image_encoder
        del encoder_func
        del vae
        cleanup()

    def spawn_func_image(
        self,
        args
    ):
        pretrained_model_name_or_path, image_paths, cache_dir = args

        modules = self.load_model_image()
        encoder_func = modules.pop("encoder_func")
        image_encoder = modules.pop("image_encoder")
        vae = modules.pop("vae")

        if type(image_paths) == str:
            image_paths = [image_paths]

        dataset = ImageLoader(
            image_paths,
            self.cfg.image_root_dir,
            self.load_image
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
            global_embeddings, local_embeddings, latents = encoder_func(images)

            for image_path, global_embedding, local_embedding, latent in zip(image_paths, global_embeddings, local_embeddings, latents):

                if self.cfg.use_embed_local:
                    torch.save(
                        local_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'local')}.pt",
                        ),
                    )

                if self.cfg.use_embed_global:
                    torch.save(
                        global_embedding.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'global')}.pt",
                        ),
                    )

                if self.cfg.use_latent:
                    torch.save(
                        latent.cpu(), # [0] is to remove the batch dimension
                        os.path.join(
                            cache_dir,
                            f"{hash_prompt(pretrained_model_name_or_path, image_path, 'latent')}.pt",
                        ),
                    )

        del image_encoder
        del encoder_func
        del vae
        cleanup()