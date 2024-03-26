import torch

from nerfacc import (
    traverse_grids,
    render_visibility_from_density,
    OccGridEstimator,
)

import todos
import pdb

class GridEstimator(OccGridEstimator):
    def __init__(self,
        roi_aabb,
        resolution=128,
        levels=1,
        **kwargs,
    ):
        super().__init__(roi_aabb, resolution, levels)

    @torch.no_grad()
    def sampling(self,
        rays_o,  # [n_rays, 3]
        rays_d,  # [n_rays, 3]
        sigma_fn,
        near_plane=0.0,
        far_plane=1e10,

        # rendering options
        render_step_size=5e-3,
        early_stop_eps=1e-4,
        alpha_thre=0.0, # !!!!
        stratified=True,
        cone_angle=0.0,
    ):
        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane) # size() -- [1024]
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane) # size() -- [1024]

        if stratified: # True or False
            near_planes += torch.rand_like(near_planes) * render_step_size

        intervals, samples, _ = traverse_grids(rays_o, rays_d, self.binaries, self.aabbs,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices

        # skip invisible space
        if (early_stop_eps > 0.0) and (sigma_fn is not None):
            alpha_thre = min(alpha_thre, self.occs.mean().item())

            # Compute visibility of the samples, and filter out invisible samples
            if t_starts.shape[0] != 0:
                sigmas = sigma_fn(t_starts, t_ends, rays_o, rays_d, ray_indices)
            else:
                sigmas = torch.empty((0,), device=t_starts.device)
            assert (sigmas.shape == t_starts.shape), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)

            packed_info = samples.packed_info
            masks = render_visibility_from_density(
                t_starts=t_starts,
                t_ends=t_ends,
                sigmas=sigmas,
                packed_info=packed_info,
                early_stop_eps=early_stop_eps,
                alpha_thre=alpha_thre,
            )
            ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends

