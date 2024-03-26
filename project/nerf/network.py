"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 22日 星期四 03:54:39 CST
# ***
# ************************************************************************************/
#
import os
import numpy as np

import torch
from torch import nn
import tinycudann as tcnn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from nerfacc import (
    accumulate_along_rays,
    render_weight_from_density,
)

from typing import Callable, List, Union

import todos
import pdb

# Come from nerfacc/examples/radiance_fields/ngp.py

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

class RadianceField(nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ):
        super().__init__()
        # aabb = tensor([-1.500000, -1.500000, -1.500000,  1.500000,  1.500000,  1.500000],
        #        device='cuda:0')
        if not isinstance(aabb, torch.Tensor):
            pdb.set_trace()
            aabb = torch.tensor(aabb, dtype=torch.float32)

        # Turns out rectangle aabb will leads to uneven collision so bad performance.
        # We enforce a cube aabb here.
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)).tolist() # 1.4472

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=num_dim, # 3
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                ],
            },
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=(self.direction_encoding.n_output_dims + self.geo_feat_dim),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def query_density(self, x, return_feat: bool = False):
        # todos.debug.output_var("x", x)
        # tensor [x] size: [2097152, 3], min: -1.5, max: 1.5, mean: -2e-06
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min + 1e-5)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        # tensor [x] size: [2097152, 16], min: -0.00016, max: 0.000146, mean: 1e-06
        density_before_activation, base_mlp_out = torch.split(x, [1, self.geo_feat_dim], dim=-1)
        density = self.density_activation(density_before_activation) * selector[..., None]
        # todos.debug.output_var("density", density)
        # tensor [density] size: [2097152, 1], min: 0.36785, max: 0.367932, mean: 0.367887

        if return_feat: # True | False
            return density, base_mlp_out
        else:
            return density

    def query_rgb(self, dir, embedding):
        # tensor [dir] size: [280255, 3], min: -0.999418, max: 0.99991, mean: -0.152141
        # tensor [embedding] size: [280255, 15], min: -0.000129, max: 0.000128, mean: -1e-06

        # tcnn requires directions in the range [0, 1]
        dir = (dir + 1.0) / 2.0
        d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        rgb = torch.sigmoid(rgb)

        # tensor [rgb] size: [280255, 3], min: 0.483968, max: 0.635222, mean: 0.562799
        return rgb

    def forward(self, positions, directions):
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self.query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore

    def occ_eval_fn(self, x):
        render_step_size = 5e-3
        density = self.query_density(x)
        return density * render_step_size

    def sigma_fn(self, t_starts, t_ends, rays_o, rays_d, ray_indices):
        """ Define how to query density for the estimator."""
        if t_starts.shape[0] == 0:
            sigmas = torch.empty((0, 1), device=t_starts.device)
        else:
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.query_density(positions) 
        return sigmas.squeeze(-1)  # (n_samples,)


    def rgb_sigma_fn(self, t_starts, t_ends, rays_o, rays_d, ray_indices):
        """ Query rgb and density values from a user-defined radiance field. """
        if t_starts.shape[0] == 0:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0, 1), device=t_starts.device)
        else:
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.forward(positions, t_dirs)  
        return rgbs, sigmas.squeeze(-1)  # (n_samples, 3), (n_samples,)


    # Come fom nerfacc/volrend/rendering
    def rendering(self,
        t_starts, t_ends,
        rays_o, rays_d, ray_indices,
        render_bkgd=None,
        expected_depths=True,
    ):
        if ray_indices is not None:
            assert (t_starts.shape == t_ends.shape == ray_indices.shape), \
                "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

        rgbs, sigmas = self.rgb_sigma_fn(t_starts, t_ends, rays_o, rays_d, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
        assert (sigmas.shape == t_starts.shape), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)

        n_rays = rays_o.shape[0]
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(t_starts, t_ends, sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays)
        opacities = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        depths = accumulate_along_rays(weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if expected_depths:
            depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

        # Background composition.
        if render_bkgd is not None:
            colors = colors + render_bkgd * (1.0 - opacities)

        return colors, opacities, depths, extras
