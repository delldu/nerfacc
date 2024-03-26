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


def test_step(step, train_dataset, estimator, network):
    """Test model ..."""
    data = train_dataset[step]
    render_bkgd = data["color_bkgd"]
    # render_bkgd -- tensor([1., 1., 1.], device='cuda:0')
    rays = data["rays"]
    # rays is tuple: len = 2
    #     tensor [item] size: [800, 800, 3], min: 0.0, max: 2.959292, mean: 1.898851
    #     tensor [item] size: [800, 800, 3], min: -0.92056, max: 0.338345, mean: -0.452351
    color_gt = data["pixels"]
    # tensor [color_gt] size: [800, 800, 3], min: 0.0, max: 1.0, mean: 0.812961

    H, W, C = color_gt.size()
    rays_o = rays.origins.reshape(-1, C)
    rays_d = rays.viewdirs.reshape(-1, C)
    rays = nerf.Rays(origins=rays_o, viewdirs=rays_d)

    color, opacity, depth, n_sample = nerf.batch_forward(estimator, network, rays, 
        batch_size=16*1024, render_bkgd=render_bkgd)

    if n_sample > 0:
        color = color.reshape(H, W, C)
        loss = F.smooth_l1_loss(color, color_gt)
    else:
        loss = 1.0 # log(1.0) == 0
        color = None

    return loss, color


if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_root", type=str, default="../data/nerf_synthetic/", help="Data root directory")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="Checkpoint file")
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    set_random_seed(42)
    device = todos.model.get_device()
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)

    # Step 1: get data loader
    test_dataset = nerf.SubjectLoader(
        subject_id="lego",
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
    )

    # Step 2: get net
    estimator = nerf.GridEstimator(roi_aabb=aabb, resolution=128, levels=1).to(device)
    network = nerf.RadianceField(aabb=estimator.aabbs[-1]).to(device)
    nerf.load_model(estimator, network, args.checkpoint)

    # set grad to false for reduce memory !!!
    for p in network.parameters():
        p.requires_grad = False
    for p in estimator.parameters():
        p.requires_grad = False
    estimator.eval()
    network.eval()

    pbar = tqdm(total=len(test_dataset))
    pbar.set_description("testing")
    for step in range(len(test_dataset)):
        pbar.update(1)

        loss, color = test_step(step, test_dataset, estimator, network)
        psnr = -10.0*math.log10(loss + 1e-5)
        pbar.set_postfix(loss="{:.6f}, psnr={:.3f}".format(loss, psnr))

        if color is not None:
            output_filename = f"{args.output}/{step:06d}.png"
            color = color.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
            todos.data.save_tensor(color, output_filename)

    todos.model.reset_device()

