"""Model trainning."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024, All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 20 Mar 2024 11:50:36 AM CST
# ***
# ************************************************************************************/
#

import os
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

from nerf.dataset import SubjectLoader
from nerf.network import RadianceField
import nerfacc 
from nerfacc.estimators.occ_grid import OccGridEstimator

from tqdm import tqdm

import todos
import pdb  # For debug

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Counter(object):
    """Class Counter."""
    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(step, train_dataset, estimator, network, optimizer, device):
    """Trainning model ..."""
    grad_scaler = torch.cuda.amp.GradScaler(2**10)


    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = nn.SmoothL1Loss(reduction='mean')

    render_step_size = 5e-3
    def occ_eval_fn(x):
        density = network.query_density(x)
        return density * render_step_size

    def sigma_fn(t_starts, t_ends, ray_indices):
        """ Define how to query density for the estimator."""
        if t_starts.shape[0] == 0:
            sigmas = torch.empty((0, 1), device=t_starts.device)
        else:
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = network.query_density(positions) 
        return sigmas.squeeze(-1)  # (n_samples,)


    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        """ Query rgb and density values from a user-defined radiance field. """
        if t_starts.shape[0] == 0:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0, 1), device=t_starts.device)
        else:
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = network(positions, t_dirs)  
        return rgbs, sigmas.squeeze(-1)  # (n_samples, 3), (n_samples,)


    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"] # rays.origins.size() -- [1024, 3], rays.viewdirs.size() -- [1024, 3]
    color_gt = data["pixels"]
    # tensor [color_gt] size: [1024, 3], min: 0.0, max: 1.0, mean: 0.837546

    estimator.update_every_n_steps(step=step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2)

    rays_o = rays.origins
    rays_d = rays.viewdirs
    ray_indices, t_starts, t_ends = estimator.sampling(rays_o, rays_d, 
        # sigma_fn=sigma_fn, near_plane=0.2, far_plane=1.0, early_stop_eps=1e-4, alpha_thre=1e-2,
        sigma_fn=sigma_fn, near_plane=0.0, far_plane=1e10, 
        render_step_size=5e-3, stratified = network.training,
        cone_angle = 0.0, 
        early_stop_eps=1e-4, alpha_thre=0.0,
    )
    color, opacity, depth, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, 
        n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn,
        render_bkgd=render_bkgd,
    )

    if len(t_starts) > 0:
        loss = F.smooth_l1_loss(color, color_gt)
        last_loss = loss.item()
        last_psnr = -10.0 * math.log10(loss.item() + 1e-5)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        lr_scheduler.step()
    else:
        last_loss = 1e5
        last_psnr = 0.0


        # for batch in loader:
        #     # "rgbs", "grid", "tseq", "idxs"
        #     # Transform data to device
        #     rgbs = batch["rgbs"].to(device) # [2, 3, 720, 1280]
        #     grid = batch["grid"].to(device) # [2, 921600, 2]
        #     tseq = batch["tseq"].to(device) # [2, 1]

        #     count = len(rgbs) 
        #     # assert count == 1, "Current only support 1 batch"

        #     outputs = model(grid, tseq) # size() -- outputs.size() -- [2, 3, 921600]

        #     # Statics
        #     loss = loss_function(rearrange(rgbs, 'b c h w -> b c (h w)'), outputs)
        #     pred_rgbs = outputs.reshape(rgbs.size())
        #     total_loss.update(loss.item(), 1) # 1 for loss_function reduce mean

        #     # psnr = -10.0 * math.log10(loss.item() + 1e-5)
        #     psnr = -10.0 * math.log10(total_loss.avg + 1e-5)
        #     t.set_postfix(loss="{:.6f}, psnr={:.3f}".format(total_loss.avg, psnr))
        #     t.update(count)

        #     # Optimizer
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

    return (last_loss, last_psnr)

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_root", type=str, default="../data/nerf_synthetic/", help="Data root directory")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="Checkpoint file")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=512, help="Training epochs")
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    set_random_seed(42)
    device = todos.model.get_device()

    # training parameters
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    # near_plane = 0.0
    # far_plane = 1.0e10
    # # dataset parameters

    # # render parameters
    # alpha_thre = 0.0
    # cone_angle = 0.0

    # Step 1: get data loader
    init_batch_size = 16*1024
    train_dataset = SubjectLoader(
        subject_id="lego",
        root_fp=args.data_root,
        split="train",
        num_rays=init_batch_size,
        device=device,
    )

    # test_dataset = SubjectLoader(
    #     subject_id="lego",
    #     root_fp=args.data_root,
    #     split="test",
    #     num_rays=None,
    #     device=device,
    # )


    # Step 2: get net
    estimator = OccGridEstimator(roi_aabb=aabb, resolution=128, levels=1).to(device)
    estimator.train()
    network = RadianceField(aabb=estimator.aabbs[-1]).to(device)
    network.train()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Construct Optimizer and Learning Rate Scheduler
    # ***
    # ************************************************************************************/
    #
    optimizer = optim.Adam(network.parameters(), 
        lr=args.lr, 
        eps=1e-15, 
        weight_decay=1e-6,
    )
    lr_decay_step=[args.epochs*1//2, args.epochs*3//4, args.epochs*9//10]
    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=lr_decay_step,
                gamma=0.33,
            ),
        ]
    )

    total_loss = Counter()
    pbar = tqdm(total=args.epochs)

    for epoch in range(args.epochs):
        pbar.update(1)

        last_lr = lr_scheduler.get_last_lr()[0]

        last_loss, last_psnr = train_epoch(epoch, train_dataset, estimator, network, optimizer, device)

        if last_psnr > 0.01:
            total_loss.update(last_loss, 1) # 1 for loss_function reduce mean

        pbar.set_postfix(loss="{:.6f}, psnr={:.3f}, lr={:.6f}".format(total_loss.avg, last_psnr, last_lr))
        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Save Model Strategy
        # ***
        # ************************************************************************************/
        #
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            print(f"Saving model to {args.checkpoint} ...")

    todos.model.reset_device()

