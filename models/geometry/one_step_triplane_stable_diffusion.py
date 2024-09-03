import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *

from ..geometry.utils import contract_to_unisphere_custom, sample_from_planes
from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp


@threestudio.register("one-step-triplane-stable-diffusion")
class StableDiffusionTriplaneAttention(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
                "training_type": "lora",
                "output_dim": 32,
                "gradient_checkpoint": False,
                "self_lora_type": "quadra_v1",
                "cross_lora_type": "quadra_v1",
                "locon_type": "quadra_v1",
            }
        )

        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 2, 
            }
        )

        backbone: str = "one_step_triplane_stable_diffusion"
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

        # rotate planes to fit the conventional direction of image generated by SD
        # in right-handed coordinate system 
        # xy plane should looks that a img from top-down / bottom-up view
        # xz plane should looks that a img from right-left / left-right view
        # yz plane should looks that a img from front-back / back-front view
        rotate_planes: Optional[str] = None # "v1"
        interpolate_feat: Optional[str] = None # "v1", "v2"


    def configure(self) -> None:
        super().configure()

        # set up the space generator
        if self.cfg.backbone == "one_step_triplane_stable_diffusion":
            from ...extern.one_step_triplane_sd_modules import OneStepTriplaneStableDiffusion as Generator
            self.space_generator = Generator(self.cfg.space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")

        # set up the mlp
        if self.cfg.interpolate_feat in ["v1"]:
            input_dim = self.space_generator.output_dim * 1 # feat_xy + feat_xz + feat_yz
        elif self.cfg.interpolate_feat in ["v2"]:
            input_dim = self.space_generator.output_dim * 3 # feat_xy concat feat_xz concat feat_yz
        elif self.cfg.interpolate_feat in ["v3"]:
            input_dim = self.space_generator.output_dim * 1 - 1 # alpha_xy * feat_xy + alpha_xz * feat_xz + alpha_yz * feat_yz 
        else:
            raise ValueError(f"Unknown interpolate_feat {self.cfg.interpolate_feat}")
        
        self.sdf_network = get_mlp(
            input_dim,
            1,
            self.cfg.mlp_network_config,
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                input_dim,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                input_dim,
                3,
                self.cfg.mlp_network_config,
            )

        self.noise_dim = None # not used
        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        # not used
        pass

    # this function is similar to the one in threestudio/models/geometry/impcit_sdf.py
    def get_shifted_sdf(
        self, 
        points: Float[Tensor, "*N Di"], 
        sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def generate_space_cache(
        self,
        styles: Float[Tensor, "B Z"],
        text_embed: Float[Tensor, "B C"],
    ) -> Any:
        output = self.space_generator(
            text_embed = text_embed,
            styles = styles,
        )
        return output
    
    def denoise(
        self,
        noisy_input: Any,
        text_embed: Float[Tensor, "B C"],
        timestep
    ) -> Any:
        output = self.space_generator.forward_denoise(
            text_embed = text_embed,
            noisy_input = noisy_input,
            t = timestep
        )
        return output
    
    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(
            latents = latents
        )
        return triplane

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ):
        batch_size, n_points, n_dims = points.shape
        # the following code is similar to EG3D / OpenLRM
        
        assert self.cfg.rotate_planes in [None, "v1", "v2"]

        if self.cfg.rotate_planes == None:
            return sample_from_planes(
                plane_features = space_cache,
                coordinates = points,
            ).view(*points.shape[:-1],-1)

        space_cache_rotated = torch.zeros_like(space_cache)
        if self.cfg.rotate_planes == "v1":
            # xy plane, diagonal-wise
            space_cache_rotated[:, 0::3] = torch.transpose(
                space_cache[:, 0::3], 3, 4
            )
            # xz plane, rotate 180° counterclockwise
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            # zy plane, rotate 90° clockwise
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )
        elif self.cfg.rotate_planes == "v2":
            # all are the same as v1, except for the xy plane
            # xy plane, row-wise flip
            space_cache_rotated[:, 0::3] = torch.flip(
                space_cache[:, 0::3], dims=(4,)
            )
            # xz plane, rotate 180° counterclockwise
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            # zy plane, rotate 90° clockwise
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )

        return sample_from_planes(
            plane_features = space_cache_rotated,
            coordinates = points,
            interpolate_feat = self.cfg.interpolate_feat
        ).view(*points.shape[:-1],-1)


    def rescale_points(
        self,
        points: Float[Tensor, "*N Di"],
    ):
        # transform points from original space to [-1, 1]^3
        points = contract_to_unisphere_custom(
            points, 
            self.bbox, 
            self.unbounded
        )
        return points

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Any,
        output_normal: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:

        batch_size, n_points, n_dims = points.shape


        grad_enabled = torch.is_grad_enabled()
        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points
        points = self.rescale_points(points)

        enc = self.interpolate_encodings(points, space_cache)
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {
                "sdf": sdf.view(batch_size * n_points, 1) # reshape to [B*N, 1]
            }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims)
            output.update(
                    {
                        "features": features.view(batch_size * n_points, self.cfg.n_feature_dims) # reshape to [B*N, n_feature_dims]
                    }
                )

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                    [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                ).to(points_unscaled)
                points_offset: Float[Tensor, "... 3 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
                sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                    points_offset, space_cache
                )
                sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "analytic":
                # QUESTION: the sdf is >0 for points outside the shape in the original space
                # so its normal should point outwards, but the normal is pointing inwards if we put a negative sign
                # so we need to flip the normal by multiplying it by -1
                sdf_grad = torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=grad_enabled, # not implemented in the test
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    torch.set_grad_enabled(False)
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise NotImplementedError(
                    f"normal_type == {self.cfg.normal_type} is not implemented yet."
                )
            output.update(
                {
                    "normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "shading_normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "sdf_grad": sdf_grad.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                }
            )
        return output

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Float[Tensor, "*N 1"]:
        batch_size = points.shape[0]
        assert points.shape[0] == batch_size, "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = self.rescale_points(points)

        # sample from planes
        enc = self.interpolate_encodings(
            points.reshape(batch_size, -1, 3),
            space_cache
        ).reshape(*points.shape[:-1], -1)
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)

        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_field(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        # TODO: is this function correct?
        batch_size = points.shape[0]
        assert points.shape[0] == batch_size, "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = self.rescale_points(points)

        # sample from planes
        enc = self.interpolate_encodings(points, space_cache)      
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # TODO: is this function correct?
        return field - threshold

    def export(
        self, 
        points: Float[Tensor, "*N Di"], 
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    **kwargs) -> Dict[str, Any]:
        # TODO: is this function correct?
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = self.rescale_points(points)

        # sample from planes
        enc = self.interpolate_encodings(points, space_cache)
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
        # else:
        #     raise NotImplementedError(
        #         f"normal_type == {self.cfg.normal_type} is not implemented yet."
        #     )

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()
