import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

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

from diffusers.models.lora import LoRACompatibleConv


class LoRALinearLayerwBias(nn.Module):
    r"""
    A linear layer that is used with LoRA, can be used with bias.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        with_bias: bool = False
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        if with_bias:
            self.bias = nn.Parameter(torch.zeros([1, 1, out_features], device=device, dtype=dtype))
        self.with_bias = with_bias

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        if self.with_bias:
            up_hidden_states = up_hidden_states + self.bias

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    
class TriplaneLoRAConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
        with_bias: bool = False
    ):
        super().__init__()

        self.down_xy = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.down_xz = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.down_yz = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up_xy = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.up_xz = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.up_yz = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        if with_bias:
            self.bias_xy = nn.Parameter(torch.zeros([1, out_features, 1, 1]))
            self.bias_xz = nn.Parameter(torch.zeros([1, out_features, 1, 1]))
            self.bias_yz = nn.Parameter(torch.zeros([1, out_features, 1, 1]))
        self.with_bias = with_bias

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        # initialize the weights
        nn.init.normal_(self.down_xy.weight, std=1 / rank)
        nn.init.normal_(self.down_xz.weight, std=1 / rank)
        nn.init.normal_(self.down_yz.weight, std=1 / rank)
        nn.init.zeros_(self.up_xy.weight)
        nn.init.zeros_(self.up_xz.weight)
        nn.init.zeros_(self.up_yz.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down_xy.weight.dtype

        # xy plane
        down_hidden_states = self.down_xy(hidden_states[0::3].to(dtype))
        up_hidden_states_xy = self.up_xy(down_hidden_states)
        if self.with_bias:
            up_hidden_states_xy = up_hidden_states_xy + self.bias_xy

        # xz plane
        down_hidden_states = self.down_xz(hidden_states[1::3].to(dtype))
        up_hidden_states_xz = self.up_xz(down_hidden_states)
        if self.with_bias:
            up_hidden_states_xz = up_hidden_states_xz + self.bias_xz

        # yz plane
        down_hidden_states = self.down_yz(hidden_states[2::3].to(dtype))
        up_hidden_states_yz = self.up_yz(down_hidden_states)
        if self.with_bias:
            up_hidden_states_yz = up_hidden_states_yz + self.bias_yz

        # combine the hidden states
        up_hidden_states = torch.concat(
            [torch.zeros_like(up_hidden_states_yz)] * 3,
            dim=0
        )
        up_hidden_states[0::3] = up_hidden_states_xy
        up_hidden_states[1::3] = up_hidden_states_xz
        up_hidden_states[2::3] = up_hidden_states_yz    

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

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
        with_bias: bool = False
    ):
        super().__init__()

        assert num_planes == 3, "The number of planes should be 3."

        self.hidden_size = hidden_size
        self.rank = rank

        # lora for 1st plane
        self.to_q_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

        # lora for 2nd plane
        self.to_q_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

        # lora for 3nd plane
        self.to_q_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

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
        with_bias: bool = False
    ):
        super().__init__()

        assert num_planes == 3, "The number of planes should be 3."

        self.hidden_size = hidden_size
        self.rank = rank

        # lora for 1st plane
        self.to_q_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_xy_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_xy_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_xy_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

        # lora for 2nd plane
        self.to_q_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_xz_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_xz_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_xz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

        # lora for 3nd plane
        self.to_q_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_k_yz_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_v_yz_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
        self.to_out_yz_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

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
        gradient_checkpoint: bool = False

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

        # save trainable parameters
        self.trainable_params = torch.nn.ParameterDict()

        assert "lora" in training_type or "locon" in training_type or "full" in training_type, "The training type is not supported."
 


        if "lora" in training_type:
            # parse the rank from the training type, with the template "lora_rank_{}"
            rank = re.search(r"lora_rank_(\d+)", training_type).group(1)
            self.lora_rank = int(rank)

            # if the finetuning is with bias
            self.w_lora_bias = False
            if "with_bias" in training_type:
                self.w_lora_bias = True

            # specify the attn_processor for unet
            lora_attn_procs = self._set_attn_processor(self.unet, self_attn_name="attn1.processor")
            self.unet.set_attn_processor(lora_attn_procs)
            # update the trainable parameters
            self.trainable_params.update(self.unet.attn_processors)

            # specify the attn_processor for vae
            lora_attn_procs = self._set_attn_processor(self.vae, self_attn_name="processor")
            self.vae.set_attn_processor(lora_attn_procs)
            # update the trainable parameters
            self.trainable_params.update(self.vae.attn_processors)

        if "locon" in training_type:
            # parse the rank from the training type, with the template "locon_rank_{}"
            rank = re.search(r"locon_rank_(\d+)", training_type).group(1)
            self.locon_rank = int(rank)

            # if the finetuning is with bias
            self.w_locon_bias = False
            if "with_bias" in training_type:
                self.w_locon_bias = True

            # specify the conv_processor for unet
            locon_procs = self._set_conv_processor(self.unet)
            # update the trainable parameters
            self.trainable_params.update(locon_procs)

            # specify the conv_processor for vae
            locon_procs = self._set_conv_processor(self.vae)
            # update the trainable parameters
            self.trainable_params.update(locon_procs)
   
            
        if "full" in training_type:
            raise NotImplementedError("The full training type is not supported.")

        # free all the parameters
        for param in self.unet.parameters():
            param.requires_grad_(False)
        for param in self.vae.parameters():
            param.requires_grad_(False)

        # unfreeze the trainable parameters
        for param in self.trainable_params.parameters():
            param.requires_grad_(True)

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

        if self.cfg.gradient_checkpoint:
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()

    def _set_conv_processor(
        self,
        module,
        conv_name: str = "LoRACompatibleConv",
    ):
        locon_procs = {}
        for _name, _module in module.named_modules():
            if _module.__class__.__name__ == conv_name:
                # append the locon processor to the module
                locon_proc = TriplaneLoRAConv2dLayer(
                    in_features=_module.in_channels,
                    out_features=_module.out_channels,
                    rank=self.locon_rank,
                    kernel_size=_module.kernel_size,
                    stride=_module.stride,
                    padding=_module.padding,
                    with_bias = self.w_locon_bias
                )
                # add the locon processor to the module
                _module.lora_layer = locon_proc
                # update the trainable parameters
                key_name = f"{_name}.lora_layer"
                locon_procs[key_name] = locon_proc

        return locon_procs


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
                    hidden_size, self.lora_rank, num_planes = self.num_planes, with_bias = self.w_lora_bias
                )
            else:
                # it is cross-attention
                cross_attention_dim = module.config.cross_attention_dim
                lora_attn_procs[name] = cross_attn_procs(
                    hidden_size, cross_attention_dim, self.lora_rank, num_planes = self.num_planes, with_bias = self.w_lora_bias
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
        
