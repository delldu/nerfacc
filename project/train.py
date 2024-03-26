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

import torch
import torch.nn.functional as F
import torch.optim as optim

import nerf
from tqdm import tqdm

import todos
import pdb  # For debug

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def train_step(step, train_dataset, estimator, network, optimizer):
    """Trainning model ..."""
    grad_scaler = torch.cuda.amp.GradScaler(2**10)

    estimator.update_every_n_steps(step=step, occ_eval_fn=network.occ_eval_fn, occ_thre=1e-2)

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]
    render_bkgd = data["color_bkgd"]
    rays = data["rays"] # rays.origins.size() -- [1024, 3], rays.viewdirs.size() -- [1024, 3]
    color_gt = data["pixels"]
    # tensor [color_gt] size: [1024, 3], min: 0.0, max: 1.0, mean: 0.837546

    color, opacity, depth, n_sample = nerf.batch_forward(estimator, network, rays, 
        batch_size=16*1024, render_bkgd=render_bkgd)

    if n_sample > 0:
        loss = F.smooth_l1_loss(color, color_gt)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        lr_scheduler.step()
    else:
        loss = 1.0 # log(1.0) == 0
    return loss


if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_root", type=str, default="../data/nerf_synthetic/", help="Data root directory")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="Checkpoint file")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--bs", type=int, default=8*1024, help="Batch size for rays")
    parser.add_argument("--max_steps", type=int, default=2000, help="Training max_steps")
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    set_random_seed(42)
    device = todos.model.get_device()
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)

    # Step 1: get data loader
    train_dataset = nerf.SubjectLoader(
        subject_id="lego",
        root_fp=args.data_root,
        split="train",
        num_rays=args.bs,
        device=device,
    )

    # test_dataset = nerf.SubjectLoader(
    #     subject_id="lego",
    #     root_fp=args.data_root,
    #     split="test",
    #     num_rays=None,
    #     device=device,
    # )


    # Step 2: get net
    estimator = nerf.GridEstimator(roi_aabb=aabb, resolution=128, levels=1).to(device)
    network = nerf.RadianceField(aabb=estimator.aabbs[-1]).to(device)
    nerf.load_model(estimator, network, args.checkpoint)

    estimator.train()
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
    lr_decay_step=[args.max_steps*1//2, args.max_steps*3//4, args.max_steps*9//10]
    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=lr_decay_step,
                gamma=0.33,
            ),
        ]
    )

    pbar = tqdm(total=args.max_steps)
    pbar.set_description("trainning")
    for step in range(args.max_steps):
        pbar.update(1)

        last_lr = lr_scheduler.get_last_lr()[0]
        last_loss = train_step(step, train_dataset, estimator, network, optimizer)

        last_psnr = -10.0*math.log10(last_loss + 1e-5)
        pbar.set_postfix(loss="{:.6f}, psnr={:.3f}".format(last_loss, last_psnr))

        lr_scheduler.step()

        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Save Model Strategy
        # ***
        # ************************************************************************************/
        #
        # if step == (args.max_steps // 2) or (step == args.max_steps - 1):
        if (step == args.max_steps - 1):
            nerf.save_model(estimator, network, last_psnr, args.checkpoint)

    todos.model.reset_device()

