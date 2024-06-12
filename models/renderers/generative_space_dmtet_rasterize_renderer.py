from dataclasses import dataclass
from functools import partial
from tqdm import tqdm

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer

from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.models.mesh import Mesh

from threestudio.utils.ops import scale_tensor as scale_tensor


@threestudio.register("generative-space-dmtet-rasterize-renderer")
class GenerativeSpaceDmtetRasterizeRenderer(NVDiffRasterizer):
    @dataclass
    class Config(NVDiffRasterizer.Config):
        # the following are from NeuS #########
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True

        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        context_type: str = "cuda"
        isosurface_method: str = "mt" # "mt" or "mc-cpu"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        # overwrite the geometry
        self.geometry.isosurface = self.isosurface

        assert self.cfg.isosurface_method in ["mt", "mc-cpu"]
        if self.cfg.isosurface_method == "mt":
            from threestudio.models.isosurface import MarchingTetrahedraHelper
            self.isosurface_helper = MarchingTetrahedraHelper(
                self.cfg.isosurface_resolution,
                f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
            )
        elif self.cfg.isosurface_method == "mc-cpu":
            from threestudio.models.isosurface import  MarchingCubeCPUHelper
            self.isosurface_helper = MarchingCubeCPUHelper(
                self.cfg.isosurface_resolution,
            )


    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Float[Tensor, "B ..."]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        batch_size = mvp_mtx.shape[0]
        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        num_views_per_batch = batch_size // batch_size_space_cache

        if space_cache is None:
            space_cache = self.geometry.generate_space_cache(
                styles = noise,
                text_embed = text_embed,
            )
        # the isosurface is dependent on the space cache
        mesh_list = self.isosurface(space_cache)

        out_list = []
        # if render a space cache in multiple views,
        for batch_idx, mesh in enumerate(mesh_list):
            _mvp_mtx: Float[Tensor, "B 4 4"]  = mvp_mtx[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
            v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                mesh.v_pos, _mvp_mtx
            )

            # do rasterization
            if self.training: # requires only 4 views, memory efficient:
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                gb_feat, _ = self.ctx.interpolate(v_pos_clip, rast, mesh.t_pos_idx)
                depth = gb_feat[..., -2:-1]
            else: # requires about 40 views, GPU OOM, need a for-loop to rasterize
                rast_list = []
                depth_list = []
                n_views_per_rasterize = 4
                for i in range(0, v_pos_clip.shape[0], n_views_per_rasterize):
                    rast, _ = self.ctx.rasterize(v_pos_clip[i:i+n_views_per_rasterize], mesh.t_pos_idx, (height, width))
                    rast_list.append(rast)
                    gb_feat, _ = self.ctx.interpolate(v_pos_clip[i:i+n_views_per_rasterize], rast, mesh.t_pos_idx)
                    depth_list.append(gb_feat[..., -2:-1])
                rast = torch.cat(rast_list, dim=0)
                depth = torch.cat(depth_list, dim=0)

            mask = rast[..., 3:] > 0

            # special case when no points are visible
            if mask.sum() == 0: # no visible points
                # set the mask to be the first point
                mask[:1] = True

            mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
            out = {"opacity": mask_aa, "mesh": mesh, "depth": depth}

            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_aa = self.ctx.antialias(
                gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

            if render_rgb:

                # slice the space cache
                if torch.is_tensor(space_cache): #space cache
                    space_cache_slice = space_cache[batch_idx: batch_idx+1]
                elif isinstance(space_cache, Dict): #hyper net
                    # Dict[str, List[Float[Tensor, "B ..."]]]
                    space_cache_slice = {}
                    for key in space_cache.keys():
                        space_cache_slice[key] = [
                            weight[batch_idx: batch_idx+1] for weight in space_cache[key]
                        ]

                selector = mask[..., 0]

                gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
                gb_viewdirs = F.normalize(
                    gb_pos - camera_positions[
                        batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch,
                        None, None, :
                    ], dim=-1
                )
                gb_light_positions = light_positions[
                    batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch,
                    None, None, :
                ].expand(
                    -1, height, width, -1
                )

                positions = gb_pos[selector]

                # # special case when no points are selected
                # if positions.shape[0] == 0:                
                #     out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})
                #     continue

                geo_out = self.geometry(
                    positions[None, ...],
                    space_cache_slice,
                    output_normal= False, #self.training, # only output normal and related info during training
                )

                extra_geo_info = {}
                if self.material.requires_normal:
                    extra_geo_info["shading_normal"] = gb_normal[selector]
                
                if self.material.requires_tangent:
                    gb_tangent, _ = self.ctx.interpolate_one(
                        mesh.v_tng, rast, mesh.t_pos_idx
                    )
                    gb_tangent = F.normalize(gb_tangent, dim=-1)
                    extra_geo_info["tangent"] = gb_tangent[selector]

                # remove the following keys from geo_out
                geo_out.pop("shading_normal", None)

                # add sdf values for computing loss
                if "sdf_grad" in geo_out:
                    out.update({"sdf_grad": geo_out["sdf_grad"]})

                rgb_fg = self.material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    **extra_geo_info,
                    **geo_out
                )
                gb_rgb_fg = torch.zeros(num_views_per_batch, height, width, 3).to(rgb_fg)
                gb_rgb_fg[selector] = rgb_fg

                # background
                if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
                    gb_rgb_bg = self.background(
                        dirs=gb_viewdirs, 
                        text_embed=text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"]
                    )
                else:
                    gb_rgb_bg = self.background(dirs=gb_viewdirs)

                gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
                gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

                out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

            out_list.append(out)

        # stack the outputs
        out = {}
        for key in out_list[0].keys():
            if key not in ["mesh"]: # hard coded for mesh
                out[key] = torch.concat([o[key] for o in out_list], dim=0)
            else:
                out[key] = [o[key] for o in out_list]

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        pass

    def isosurface(self, space_cache: Float[Tensor, "B ..."]) -> List[Mesh]:

        # get the batchsize
        if torch.is_tensor(space_cache): #space cache
            batch_size = space_cache.shape[0]
        elif isinstance(space_cache, Dict): #hyper net
            # Dict[str, List[Float[Tensor, "B ..."]]]
            for key in space_cache.keys():
                batch_size = space_cache[key][0].shape[0]
                break

        # scale the points to [-1, 1]
        points = scale_tensor(
            self.isosurface_helper.grid_vertices.to(self.device),
            self.isosurface_helper.points_range,
            [-1, 1], # hard coded isosurface_bbox
        )
        # get the sdf values    
        sdf_batch, deformation_batch = self.geometry.forward_field(
            points[None, ...].expand(batch_size, -1, -1),
            space_cache
        )
        # get the isosurface
        mesh_list = []

        # for sdf, deformation in zip(sdf_batch, deformation_batch):
        for index in range(sdf_batch.shape[0]):
            sdf = sdf_batch[index]

            # the deformation may be None
            if deformation_batch is None:
                deformation = None
            else:
                deformation = deformation_batch[index]

            # special case when all sdf values are positive or negative, thus no isosurface
            if torch.all(sdf > 0) or torch.all(sdf < 0):
                threestudio.info("All sdf values are positive or negative, no isosurface")


                # attempt 1
                # # if no sdf with 0, manually add 5% to be 0
                # sdf_copy = sdf.clone()
                # with torch.no_grad():
                #     # select the 1% of the points that are closest to 0
                #     sdf_abs = torch.abs(sdf_copy)
                #     # get the threshold
                #     threshold = torch.topk(sdf_abs.flatten(), int(0.02 * sdf_abs.numel()), largest=False).values[-1]
                #     # find the points that are closest to 0
                #     idx = torch.where(sdf_abs < thres hold)
                # sdf[idx] = 0.0 * sdf[idx]

                # # attempt 2
                # # subtract the mean
                # # sdf_mean = torch.mean(sdf)
                # sdf_mean = torch.mean(sdf).detach()
                # sdf = sdf - sdf_mean

                # attempt 3
                # set the sdf values to be the norm of the points
                ratio_factor = 1.0
                sdf_manually = self.geometry.get_shifted_sdf(points, torch.zeros_like(sdf))
                # sdf_manually = torch.norm(points, dim=-1) - 0.1 # the sdf will be forced to be a very small ball
                sdf = sdf * (1 - ratio_factor) + sdf_manually * ratio_factor # allow limited effect of original sdf
                if deformation is not None:
                    deformation = deformation * (1 - ratio_factor) + torch.zeros_like(deformation) * ratio_factor # allow limited effect of original deformation

            mesh = self.isosurface_helper(sdf, deformation)
            mesh.v_pos = scale_tensor(
                mesh.v_pos,
                self.isosurface_helper.points_range,
                [-1, 1], # hard coded isosurface_bbox
            )
            if self.cfg.isosurface_remove_outliers:
                mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
            mesh_list.append(mesh)
            
        return mesh_list


    def train(self, mode=True):
        if hasattr(self.geometry, "train"):
            self.geometry.train(mode)
        return super().train(mode=mode)

    def eval(self):
        if hasattr(self.geometry, "eval"):
            self.geometry.eval()
        return super().eval()