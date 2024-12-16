
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro

from gymnasium import spaces
from torch.utils.data.dataset import Dataset
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
import pickle
from mani_skill.utils import gym_utils

@dataclass
class Args:
    env_id: str = "StackCube-v1"
    """the id of the environment"""
    demo_path: str = (
        "/home/zqy/code/RL2/dataset/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    )
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = 150
    """number of trajectories to load from the demo dataset"""
    depth: bool = True
    """use depth to train"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

def save_data_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class DemoDataset_DP(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, num_traj):
        if data_path.endswith(".pkl"):
            raise NotImplementedError("Loading from .pkl is not implemented.")
        else:
            from diffusion_policy.utils import load_demo_dataset

            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        print("Raw trajectory loaded, start to pre-process the observations...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if args.depth:
                depth = _obs_traj_dict["depth"].astype(np.float32) / 1024
                depth = np.clip(depth, 0, 1)  # Clip depth values > 1 to 1
                _obs_traj_dict["depth"] = torch.tensor(depth, dtype=torch.float32)
            _obs_traj_dict["rgb"] = torch.from_numpy(
                _obs_traj_dict["rgb"]
            ).float() / 255  # 转换为浮点数并归一化到 [0, 1]
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).float()
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list

        # 计算均值和标准差
        self.compute_mean_std(trajectories)
        self.trajectories = trajectories

        # 存储obs参数
        self.act_dim = trajectories["actions"][0].shape[1]
        self.state_dim = trajectories["observations"][0]["state"].shape[1]
        self.cam_num = int(trajectories["observations"][0]["rgb"].shape[1] / 3)

    def compute_mean_std(self, trajectories):
        """
        Compute the mean and standard deviation for state, rgb, and depth
        """
        state_list = []
        rgb_list = []
        depth_list = []

        # Collect all data for state, rgb, depth
        for traj in trajectories["observations"]:
            state_list.append(traj["state"])
            rgb_list.append(traj["rgb"])
            if args.depth and "depth" in traj:
                depth_list.append(traj["depth"])

        # Concatenate data along the batch dimension
        states = torch.cat(state_list, dim=0)  # (total_samples, state_dim)
        rgbs = torch.cat(rgb_list, dim=0)      # (total_samples, C, H, W)
        if depth_list:
            depths = torch.cat(depth_list, dim=0)  # (total_samples, C, H, W)

        # Compute mean and standard deviation for each modality
        self.state_mean = states.mean(dim=0, keepdim=True)
        self.state_std = states.std(dim=0, keepdim=True) + 1e-6

        # Normalize across the channel dimension (C)
        self.rgb_mean = rgbs.mean(dim=(0, 2, 3), keepdim=True)  # (C, 1, 1)
        self.rgb_std = rgbs.std(dim=(0, 2, 3), keepdim=True) + 1e-6

        if depth_list:
            self.depth_mean = depths.mean(dim=(0, 2, 3), keepdim=True)  # (C, 1, 1)
            self.depth_std = depths.std(dim=(0, 2, 3), keepdim=True) + 1e-6

        print("Mean and standard deviation computed for state, rgb, and depth.")
        print(f"State Mean: {self.state_mean.size()}")
        print(f"State Std: {self.state_std.size()}")
        print(f"RGB Mean: {self.rgb_mean.size()}")
        print(f"RGB Std: {self.rgb_std.size()}")
        if depth_list:
            print(f"Depth Mean: {self.depth_mean.size()}")
            print(f"Depth Std: {self.depth_std.size()}")

    def save_normalization_params(self, output_json):
        """
        Save the computed mean and std to a JSON file
        """
        params = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "rgb_mean": self.rgb_mean.tolist(),
            "rgb_std": self.rgb_std.tolist(),
            "depth_mean": self.depth_mean.tolist() if hasattr(self, 'depth_mean') else None,
            "depth_std": self.depth_std.tolist() if hasattr(self, 'depth_std') else None,
            "act_dim": self.act_dim,
            "state_dim": self.state_dim,
            "cam_num": self.cam_num,

        }

        with open(output_json, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Normalization parameters saved to {output_json}")



if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    obs_process_fn = partial(
        convert_obs,
        use_depth=args.depth,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(
            np.transpose, axes=(0, 3, 1, 2)
        ),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
    )
    if args.depth == True:
        obs_mode_ = "rgbd"
    else:
        obs_mode_ = "rgb"
    tmp_env = gym.make(args.env_id, obs_mode=obs_mode_)
    original_obs_space = tmp_env.observation_space
    print(original_obs_space)
    dataset = DemoDataset_DP(
        args.demo_path, obs_process_fn, original_obs_space, args.num_demos
    )
    norm_path = f'./norm/{args.env_id}_{args.control_mode}_norm_params.json'
    dataset.save_normalization_params(norm_path)
    # Example usage
    obs_space_path = f'./norm/{args.env_id}_{args.control_mode}_obs_space.pkl'
    save_data_pickle(original_obs_space, obs_space_path)
