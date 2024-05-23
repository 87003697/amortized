import torch
import torch.nn as nn
import numpy as np

from diffusers.models.attention_processor import Attention, AttnProcessor, LoRAAttnProcessor, LoRALinearLayer
from threestudio.utils.typing import *
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from diffusers.loaders import AttnProcsLayers
from threestudio.utils.base import BaseModule
from dataclasses import dataclass


class TriplaneSelfAttentionLoRAAttnProcessor(nn.Module):
    """
    Perform for implementing the Triplane Self-Attention LoRA Attention Processor.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        num_planes: int = 3,
    ):
        super().__init__()

        assert num_planes == 3, "The number of planes should be 3."

        self.hidden_size = hidden_size
        self.rank = rank

        # lora for 1st plane
        self.to_q_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # lora for 2nd plane
        self.to_q_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # lora for 3nd plane
        self.to_q_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        assert encoder_hidden_states is None, "The encoder_hidden_states should be None."
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        
        query = attn.to_q(hidden_states)
        _query_new = torch.zeros_like(query)
        # lora for 1st plane
        _query_new[0::3] = query[0::3] + scale * self.to_q_xy_lora(hidden_states[0::3])
        # lora for 2nd plane
        _query_new[1::3] = query[1::3] + scale * self.to_q_xz_lora(hidden_states[1::3])
        # lora for 3rd plane
        _query_new[2::3] = query[2::3] + scale * self.to_q_yz_lora(hidden_states[2::3])
        query = _query_new

        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        _key_new = torch.zeros_like(key)
        # lora for 1st plane
        _key_new[0::3] = key[0::3] + scale * self.to_k_xy_lora(encoder_hidden_states[0::3])
        # lora for 2nd plane
        _key_new[1::3] = key[1::3] + scale * self.to_k_xz_lora(encoder_hidden_states[1::3])
        # lora for 3rd plane
        _key_new[2::3] = key[2::3] + scale * self.to_k_yz_lora(encoder_hidden_states[2::3])
        key = _key_new

        value = attn.to_v(encoder_hidden_states)
        _value_new = torch.zeros_like(value)
        # lora for 1st plane
        _value_new[0::3] = value[0::3] + scale * self.to_v_xy_lora(encoder_hidden_states[0::3])
        # lora for 2nd plane
        _value_new[1::3] = value[1::3] + scale * self.to_v_xz_lora(encoder_hidden_states[1::3])
        # lora for 3rd plane
        _value_new[2::3] = value[2::3] + scale * self.to_v_yz_lora(encoder_hidden_states[2::3])
        value = _value_new

        # in self-attention, query of each plane should be used to calculate the attention scores of all planes
        key = attn.head_to_batch_dim(
            torch.cat(
                [
                    torch.cat([key[0::3], key[1::3], key[2::3]], dim=1), # sequence_length x 3 * hidden_size
                    torch.cat([key[1::3], key[2::3], key[0::3]], dim=1),
                    torch.cat([key[2::3], key[0::3], key[1::3]], dim=1)
                ], dim=0
            )
        )
        value = attn.head_to_batch_dim(
            torch.cat(
                [
                    torch.cat([value[0::3], value[1::3], value[2::3]], dim=1), # sequence_length x 3 * hidden_size
                    torch.cat([value[1::3], value[2::3], value[0::3]], dim=1),
                    torch.cat([value[2::3], value[0::3], value[1::3]], dim=1)
                ], dim=0
            )
        )

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        _hidden_states_new = torch.zeros_like(hidden_states)
        # lora for 1st plane
        _hidden_states_new[0::3] = hidden_states[0::3] + scale * self.to_out_xy_lora(hidden_states[0::3])
        # lora for 2nd plane
        _hidden_states_new[1::3] = hidden_states[1::3] + scale * self.to_out_xz_lora(hidden_states[1::3])
        # lora for 3rd plane
        _hidden_states_new[2::3] = hidden_states[2::3] + scale * self.to_out_yz_lora(hidden_states[2::3])
        hidden_states = _hidden_states_new

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class TriplaneCrossAttentionLoRAAttnProcessor(nn.Module):
    """
    Perform for implementing the Triplane Cross-Attention LoRA Attention Processor.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        num_planes: int = 3,
    ):
        super().__init__()

        assert num_planes == 3, "The number of planes should be 3."

        self.hidden_size = hidden_size
        self.rank = rank

        # lora for 1st plane
        self.to_q_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_xy_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_v_xy_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_out_xy_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # lora for 2nd plane
        self.to_q_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_xz_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_v_xz_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_out_xz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # lora for 3nd plane
        self.to_q_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_yz_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_v_yz_lora = LoRALinearLayer(cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_out_yz_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        
        assert encoder_hidden_states is not None, "The encoder_hidden_states should not be None."

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        _query_new = torch.zeros_like(query)        
        # lora for 1st plane
        _query_new[0::3] = query[0::3] + scale * self.to_q_xy_lora(hidden_states[0::3])
        # lora for 2nd plane
        _query_new[1::3] = query[1::3] + scale * self.to_q_xz_lora(hidden_states[1::3])
        # lora for 3rd plane
        _query_new[2::3] = query[2::3] + scale * self.to_q_yz_lora(hidden_states[2::3])
        query = _query_new

        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        _key_new = torch.zeros_like(key)
        # lora for 1st plane
        _key_new[0::3] = key[0::3] + scale * self.to_k_xy_lora(encoder_hidden_states[0::3])
        # lora for 2nd plane
        _key_new[1::3] = key[1::3] + scale * self.to_k_xz_lora(encoder_hidden_states[1::3])
        # lora for 3rd plane
        _key_new[2::3] = key[2::3] + scale * self.to_k_yz_lora(encoder_hidden_states[2::3])
        key = _key_new

        value = attn.to_v(encoder_hidden_states)
        _value_new = torch.zeros_like(value)
        # lora for 1st plane
        _value_new[0::3] = value[0::3] + scale * self.to_v_xy_lora(encoder_hidden_states[0::3])
        # lora for 2nd plane
        _value_new[1::3] = value[1::3] + scale * self.to_v_xz_lora(encoder_hidden_states[1::3])
        # lora for 3rd plane
        _value_new[2::3] = value[2::3] + scale * self.to_v_yz_lora(encoder_hidden_states[2::3])
        value = _value_new

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        _hidden_states_new = torch.zeros_like(hidden_states)
        # lora for 1st plane
        _hidden_states_new[0::3] = hidden_states[0::3] + scale * self.to_out_xy_lora(hidden_states[0::3])
        # lora for 2nd plane
        _hidden_states_new[1::3] = hidden_states[1::3] + scale * self.to_out_xz_lora(hidden_states[1::3])
        # lora for 3rd plane
        _hidden_states_new[2::3] = hidden_states[2::3] + scale * self.to_out_yz_lora(hidden_states[2::3])
        hidden_states = _hidden_states_new

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class OneStepTriplaneStableDiffusion(BaseModule):
    """
    One-step Triplane Stable Diffusion module.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        training_type: str = "lora_rank_4",
        timestep: int = 999,
        num_planes: int = 3,
        output_dim: int = 32,

    cfg: Config

    def configure(self) -> None:

        self.num_planes = self.cfg.num_planes
        self.output_dim = self.cfg.output_dim

        # we only use the unet and vae model here
        model_path = self.cfg.pretrained_model_name_or_path
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        alphas_cumprod = self.scheduler.alphas_cumprod
        self.alphas: Float[Tensor, "T"] = alphas_cumprod**0.5
        self.sigmas: Float[Tensor, "T"] = (1 - alphas_cumprod) ** 0.5

        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        # the encoder is not needed
        self.vae.encoder = None

        # transform the attn_processor to customized one
        self.timestep = self.cfg.timestep
        self.num_planes = self.cfg.num_planes

        # set the training type
        training_type = self.cfg.training_type
        assert training_type in [
            "lora_rank_4", "lora_rank_8", "lora_rank_16", "lora_rank_32", 
            "lora_rank_64", "lora_rank_128", "lora_rank_256", 
            "full", "lora"], "The training type is not supported."
        if "lora" in training_type:

            # parse the rank
            self.rank = int(training_type.split("_")[-1])

            # free all the parameters
            for param in self.unet.parameters():
                param.requires_grad_(False)
            for param in self.vae.parameters():
                param.requires_grad_(False)                

            # specify the attn_processor for unet
            lora_attn_procs = self._set_attn_processor(self.unet, self_attn_name="attn1.processor")
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_attn_unet = AttnProcsLayers(self.unet.attn_processors).to(self.unet.device)
            self.lora_attn_unet._load_state_dict_pre_hooks.clear()
            self.lora_attn_unet._state_dict_hooks.clear()

            # specify the attn_processor for vae
            lora_attn_procs = self._set_attn_processor(self.vae, self_attn_name="processor")
            self.vae.set_attn_processor(lora_attn_procs)
            self.lora_attn_vae = AttnProcsLayers(self.vae.attn_processors).to(self.vae.device)
            self.lora_attn_vae._load_state_dict_pre_hooks.clear()
            self.lora_attn_vae._state_dict_hooks.clear()

        else:
            raise NotImplementedError("The training type is not supported.")

        # overwrite the outconv
        conv_out_orig = self.vae.decoder.conv_out
        conv_out_new = nn.Conv2d(
            in_channels=128, out_channels=self.cfg.output_dim, kernel_size=3, padding=1
        )
        # zero init the new conv_out
        nn.init.zeros_(conv_out_new.weight)
        nn.init.zeros_(conv_out_new.bias)
        # overwrite this module its weight and bias, copy from the original outconv
        conv_out_new.weight.data[:3] = conv_out_orig.weight.data
        conv_out_new.bias.data[:3] = conv_out_orig.bias.data
        self.vae.decoder.conv_out = conv_out_new

    def _set_attn_processor(
            self, 
            module,
            self_attn_name: str = "attn1.processor",
            self_attn_procs = TriplaneSelfAttentionLoRAAttnProcessor,
            cross_attn_procs = TriplaneCrossAttentionLoRAAttnProcessor
        ):
        lora_attn_procs = {}
        for name in module.attn_processors.keys():

            if name.startswith("mid_block"):
                hidden_size = module.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(module.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = module.config.block_out_channels[block_id]
            elif name.startswith("decoder"):
                # special case for decoder in SD
                hidden_size = 512

            if name.endswith(self_attn_name):
                # it is self-attention
                cross_attention_dim = None
                lora_attn_procs[name] = self_attn_procs(
                    hidden_size, self.rank, num_planes = self.num_planes
                )
            else:
                # it is cross-attention
                cross_attention_dim = module.config.cross_attention_dim
                lora_attn_procs[name] = cross_attn_procs(
                    hidden_size, cross_attention_dim, self.rank, num_planes = self.num_planes
                )
        return lora_attn_procs

    def forward(
        self,
        text_embed,
        styles,
    ):
        batch_size = text_embed.size(0)
        noise_shape = styles.size(-2)

        # set timestep
        t = torch.ones(
            batch_size * self.num_planes,
            ).to(text_embed.device) * self.timestep
        t = t.long()

        # repeat the text_embed
        text_embed = text_embed.repeat_interleave(self.num_planes, dim=0)

        # reshape the styles
        styles = styles.view(-1, 4, noise_shape, noise_shape)
        noise_pred = self.unet(
            styles,
            t,
            encoder_hidden_states=text_embed
        ).sample

        # transform the noise_pred to the original shape
        alphas = self.alphas.to(text_embed.device)[t]
        sigmas = self.sigmas.to(text_embed.device)[t]
        latents = (
            1
            / alphas.view(-1, 1, 1, 1)
            * (styles - sigmas.view(-1, 1, 1, 1) * noise_pred)
        )

        # decode the latents to triplane
        latents = 1 / self.vae.config.scaling_factor * latents
        triplane = self.vae.decode(latents).sample
        
        # triplane = (triplane * 0.5 + 0.5).clamp(0, 1) # no need for  
        triplane = triplane.view(batch_size, self.num_planes, -1, *triplane.shape[-2:])

        return triplane
        
