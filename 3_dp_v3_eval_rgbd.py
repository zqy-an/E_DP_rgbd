## Adapted from https://github.com/arnavg115/ManiSkill/tree/main
import json

from tqdm import tqdm

ALGO_NAME = "BC_Diffusion_rgbd_UNet"

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pickle

from gymnasium import spaces
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from mani_utils.evaluate_cd import evaluate
from mani_utils.make_env import make_eval_envs
from mani_utils.utils import (IterationBasedBatchSampler,
                              build_state_obs_extractor, convert_obs,
                              worker_init_fn, FlattenRGBDObservationWrapper_zqy)
from mani_utils.condition_RepVit import RepVitObsCondition
from mani_utils.condition_Conv import CNNObsCondition
from mani_utils.condition_Resnet import ResnetObsCondition
from mani_utils.theia import Theia_DINOv2

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d,ChiTransformer,ChiUNet1d
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.diffusion.edm import EDM
from torch.optim.lr_scheduler import CosineAnnealingLR

from argparse import ArgumentParser, Namespace

def load_args_from_json(file_path):
    """Load arguments from a JSON file."""
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
    args = Namespace()
    args.__dict__.update(args_dict)
    return args

def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

def load_data_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_diffusion_model(args):
    """
    Constructs a diffusion model based on provided configuration parameters.

    Args:
        args: An object containing the following configuration attributes:
            - nn_diffusion (str): Type of neural network for diffusion ('DiT1d', 'chi_unet', or 'chi_transformer').
            - nn_condition (str): Type of condition model ('RepVitObsCondition', 'CNNObsCondition', or 'ResnetObsCondition').
            - mani_utils (str): Type of diffusion policy ('DDPM' or 'EDM').
            - act_dim (int): The dimension of the action space.
            - emb_dim (int): The dimension of the embedding space.
            - obs_horizon (int): Number of observation steps.
            - horizon (int): Time horizon for the agent.
            - device (str): Device to deploy the model ('cuda' or 'cpu').
            - lr (float): Learning rate.
            - sample_steps (int): Number of diffusion steps.
            - pre_model (str, optional): Path to a pre-trained model.
            - Other necessary fields specific to the condition models.

    Returns:
        agent: A configured diffusion policy model (either DDPM or EDM) with the specified neural network.
    """

    # Diffusion Model
    if args.nn_diffusion == "DiT1d":
        nn_diffusion = DiT1d(
            args.act_dim, emb_dim=args.emb_dim * args.obs_horizon, d_model=320, n_heads=10, depth=2,
            timestep_emb_type="fourier").to(args.device)
    elif args.nn_diffusion == "chi_unet":
        nn_diffusion = ChiUNet1d(
            args.act_dim, args.emb_dim, args.obs_horizon, model_dim=256, emb_dim=256, dim_mult=args.unet_dims,
            obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)
    elif args.nn == "chi_transformer":
        nn_diffusion = ChiTransformer(
            args.act_dim, args.emb_dim, args.horizon, args.obs_horizon, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)
    else:
        raise NotImplementedError(f"Diffusion model {args.nn_diffusion} is not implemented.")

    # Condition Model
    if args.nn_condition == "RepVitObsCondition":
        nn_condition = RepVitObsCondition(
            state_dim=args.state_dim,cam_num=args.cam_num,use_depth=args.depth,use_dino=args.dino,emb_dim=args.emb_dim, ).to(args.device)
    elif args.nn_condition == "CNNObsCondition":
        nn_condition = CNNObsCondition(
            state_dim=args.state_dim,cam_num=args.cam_num,use_depth=args.depth,use_dino=args.dino,emb_dim=args.emb_dim, ).to(args.device)
    elif args.nn_condition == "ResnetObsCondition":
        nn_condition = ResnetObsCondition(
            state_dim=args.state_dim,cam_num=args.cam_num,use_depth=args.depth,use_dino=args.dino,emb_dim=args.emb_dim, ).to(args.device)
    else:
        raise NotImplementedError(f"Condition model {args.nn_condition} is not implemented.")

    # Diffusion Policy
    if args.diffusion_policy == "DDPM":
        x_max = torch.ones((1, args.pred_horizon, args.act_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.pred_horizon, args.act_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    elif args.diffusion_policy == "EDM":
        agent = EDM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            optim_params={"lr": args.lr})
    else:
        raise NotImplementedError(f"Diffusion policy {args.diffusion_policy} is not implemented.")

    # Load Pre-trained Model if available
    if args.pre_model is not None:
        print(f"Loading pre-trained model from {args.pre_model}")
        savepath = args.pre_model
        agent.load(savepath)

    return agent

def save_args_to_json(args, filename):
    # 将参数转换为字典
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    args_dict = vars(args)
    # 将字典保存为JSON文件
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

def load_normalization_params(json_path):

    with open(json_path, "r") as f:
        params = json.load(f)

    # Convert parameters to torch tensors
    return {
        "state_mean": torch.tensor(params["state_mean"], dtype=torch.float32),
        "state_std": torch.tensor(params["state_std"], dtype=torch.float32),
        "rgb_mean": torch.tensor(params["rgb_mean"], dtype=torch.float32),
        "rgb_std": torch.tensor(params["rgb_std"], dtype=torch.float32),
        "depth_mean": torch.tensor(params["depth_mean"], dtype=torch.float32),
        "depth_std": torch.tensor(params["depth_std"], dtype=torch.float32),
    }

if __name__ == "__main__":
    run_name = 'StackCube-v1__3_dp_v2_train_rgbdd__1__1734428340'
    eval_path = f'./runs/{run_name}'
    eval_model_weight = '200000.pt'
    args = load_args_from_json(os.path.join(eval_path, 'params.json'))
    args.device = "cuda:0"
    args.capture_video = True
    args.sim_backend = "cpu"
    args.num_eval_episodes = 50
    args.num_eval_envs = 10
    args.max_episode_steps = 50

    # 归一化
    norm_path = f'./norm/{args.env_id}_{args.control_mode}_norm_params.json'
    # Load normalization parameters
    norm_params = load_normalization_params(norm_path)

    obs_space_path = f'./norm/{args.env_id}_{args.control_mode}_obs_space.pkl'
    original_obs_space = load_data_pickle(obs_space_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
        # 执行pipeline函数之前打印所有参数
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


    if args.depth == True:
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode="rgbd",
            render_mode="rgb_array",
        )
    else:
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode="rgb",
            render_mode="rgb_array",
        )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    if args.depth == True:
        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[partial(FlattenRGBDObservationWrapper_zqy,
                              # depth=True
                              )],
        )
    else:
        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[partial(FlattenRGBDObservationWrapper,
                              depth=False,
                              # sep_depth=True
                              )],
        )

    # env 参数
    args.act_dim = envs.single_action_space.shape[0]
    args.state_dim = envs.single_observation_space["state"].shape[1]
    args.cam_num = int(envs.single_observation_space["rgb"].shape[-1] / 3)
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    model_dinov2 = Theia_DINOv2( args.theia_model_path,args.target_theia_model,args.device)
    agent = create_diffusion_model(args)
    # Cosine LR schedule with linear warmup
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.total_iters)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    import time

    last_tick = time.time()
    agent.model.eval()
    agent.model_ema.eval()

    best_eval_metrics = defaultdict(float)
    best_loss = np.inf
    timings = defaultdict(float)
    if args.diffusion_policy == "DDPM":
        solver = None
    elif args.diffusion_policy == "DDIM":
        solver = "ddim"
    elif args.diffusion_policy == "DPM":
        solver = "ode_dpmpp_2"
    elif args.diffusion_policy == "EDM":
        solver = "euler"

    eval_metrics = evaluate(
        args.num_eval_episodes, agent, envs, args, model_dinov2, args.sim_backend, norm_params
    )
    timings["eval"] += time.time() - last_tick

    print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        #writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
        print(f"{k}: {eval_metrics[k]:.4f}")

    save_on_best_metrics = ["success_once", "success_at_end"]
    envs.close()
