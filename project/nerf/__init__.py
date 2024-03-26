"""Nerf Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#
import os
import torch

__version__ = "1.0.0"

from .dataset import SubjectLoader
from .network import RadianceField
from .occgrid import GridEstimator

def save_model(estimator, network, optimizer, model_path):
	print(f"Saving model to {model_path} ...")
	save_dict = {
		'estimator': estimator.state_dict(),
		'network': network.state_dict(),
		'optimizer': optimizer.state_dict(),
	}
	torch.save(save_dict, model_path)


def load_model(estimator, network, optimizer, model_path):
	if os.path.exists(model_path):
		print(f"Loading model from {model_path} ...")
		sd = torch.load(model_path)
		estimator.load_state_dict(sd['estimator'])
		network.load_state_dict(sd['network'])
		optimizer.load_state_dict(sd['optimizer'])


__all__ = [
	"__version__",
	"SubjectLoader",
	"RadianceField",
	"GridEstimator",
	"save_model",
	"load_model",
]


