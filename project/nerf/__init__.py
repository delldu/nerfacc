"""Nerf Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024, All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 26 Mar 2024 07:32:28 PM CST
# ***
# ************************************************************************************/
#

import os
import random
import torch
import pdb

__version__ = "1.0.0"

import collections
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

from .dataset import SubjectLoader
from .network import RadianceField
from .occgrid import GridEstimator

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def save_model(estimator, network, psnr, model_path):
    print(f"Saving model to {model_path} ...")
    save_dict = {
        'estimator': estimator.state_dict(),
        'network': network.state_dict(),
        'psnr': psnr,
    }
    torch.save(save_dict, model_path)


def load_model(estimator, network, model_path):
    if os.path.exists(model_path):
        sd = torch.load(model_path)
        estimator.load_state_dict(sd['estimator'])
        network.load_state_dict(sd['network'])
        print(f"Loading model from {model_path}, PSNR={sd['psnr']:.2f}.")

def batch_forward(estimator, network, rays, batch_size=8*1024, render_bkgd=None):
    """batch rays forward."""

    rays_shape = rays.origins.shape # [1024, 3]
    num_rays, _ = rays_shape

    results = []
    for i in range(0, num_rays, batch_size):
        chunk_rays = namedtuple_map(lambda r: r[i : i + batch_size], rays)

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.viewdirs

        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o, rays_d, 
            sigma_fn=network.sigma_fn, 
            stratified=network.training,
        )

        color, opacity, depth, extras = network.rendering(
            t_starts, t_ends, 
            rays_o, rays_d, ray_indices,
            render_bkgd=render_bkgd,
        )

        chunk_results = [color, opacity, depth, len(t_starts)]
        results.append(chunk_results)
        del ray_indices, rays_o, rays_d
        torch.cuda.empty_cache()

    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples), # 280255
    )


__all__ = [
    "__version__",
    "SubjectLoader",
    "RadianceField",
    "GridEstimator",
    "save_model",
    "load_model",
    "batch_forward",
]


