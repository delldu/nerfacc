"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays
import todos
import pdb

def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    # root_fp = 'data/nerf_synthetic'
    # subject_id = 'lego'
    # split = 'train'/'test'

    if not root_fp.startswith("/"): # True
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    # data_dir -- '/media/dell/8t/Workspace/3D/nerfacc/examples/datasets/../../data/nerf_synthetic/lego'
    with open(os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    # array [images] shape: (100, 800, 800, 4), min: 0, max: 255, mean: 45.75339
    # array [camtoworlds] shape: (100, 4, 4), min: -3.939868, max: 4.030528, mean: 0.255044

    # (Pdb) camtoworlds[0]
    # array([[-9.999022e-01,  4.192245e-03, -1.334572e-02, -5.379832e-02],
    #        [-1.398868e-02, -2.996591e-01,  9.539437e-01,  3.845470e+00],
    #        [-4.656613e-10,  9.540372e-01,  2.996883e-01,  1.208082e+00],
    #        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
    # focal -- 1111.1110311937682

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        # tensor [self.K] size: [3, 3], min: 0.0, max: 1111.111084, mean: 335.913574
        # tensor([[ 1111.111084,     0.000000,   400.000000],
        #         [    0.000000,  1111.111084,   400.000000],
        #         [    0.000000,     0.000000,     1.000000]], device='cuda:0')

        self.images = self.images.to(device)
        # tensor [self.images] size: [100, 800, 800, 4], min: 0.0, max: 255.0, mean: 45.753387

        self.camtoworlds = self.camtoworlds.to(device)
        # tensor [self.camtoworlds] size: [100, 4, 4], min: -3.939868, max: 4.030528, mean: 0.255044

        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)


    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
        # ==> alpha.size() -- [1024, 1]

        # tensor [rgba] size: [1024, 4], min: 0.0, max: 1.0, mean: 0.179809
        # rays is tuple: len = 2
        #     tensor [item] size: [1024, 3], min: -3.939868, max: 4.030528, mean: 0.643094
        #     tensor [item] size: [1024, 3], min: -0.999418, max: 0.99991, mean: -0.15511

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device, generator=self.g)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha) #  color_bkgd --- tensor([1., 1., 1.], device='cuda:0')

        # tensor [pixels] size: [1024, 3], min: 0.0, max: 1.0, mean: 0.837546
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays # 1024

        if self.training:
            if self.batch_over_images: # True
                image_id = torch.randint(
                    0,
                    len(self.images), # 100
                    size=(num_rays,),
                    device=self.images.device,
                    generator=self.g,
                ) # [1024]
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0,
                self.WIDTH,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
            y = torch.randint(
                0,
                self.HEIGHT,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0), # self.OPENGL_CAMERA ---- True
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1) # [1024, 3]--> [1024, 1, 3] --> [1024, 3]
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape) # size() -- [1024, 3]
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True) # size() -- [1024, 3]

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        # tensor [origins] size: [1024, 3], min: -3.939868, max: 4.030528, mean: 0.643094
        # tensor [viewdirs] size: [1024, 3], min: -0.999418, max: 0.99991, mean: -0.15511
        rays = Rays(origins=origins, viewdirs=viewdirs)
        # rays is tuple: len = 2
        #     tensor [item] size: [1024, 3], min: -3.939868, max: 4.030528, mean: 0.643094
        #     tensor [item] size: [1024, 3], min: -0.999418, max: 0.99991, mean: -0.15511

        # tensor [rgba] size: [1024, 4], min: 0.0, max: 1.0, mean: 0.179809

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
