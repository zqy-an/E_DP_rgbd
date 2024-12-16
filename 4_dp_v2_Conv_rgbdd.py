## Adapted from https://github.com/arnavg115/ManiSkill/tree/main

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

from gymnasium import spaces
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter


from diffusion_policy.evaluate_cd import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn,FlattenRGBDObservationWrapper_zqy)
from diffusion_policy.condition_RepVit import RepVitObsCondition
from diffusion_policy.condition_conv import CNNObsCondition
from diffusion_policy.theia import Theia_DINOv2

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.nn_diffusion import ChiTransformer
from cleandiffuser.nn_diffusion import ChiUNet1d
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.diffusion.edm import EDM
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = 'cuda:0'
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "StackCube-v1"
    """the id of the environment"""
    demo_path: str = (
        "/home/zqy/code/RL2/dataset/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5"
    )
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = 200
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 500_000
    """total timesteps of the experiment"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""

    # Diffusion Model Args
    nn_condition: str = 'CNNObsCondition'
    nn_diffusion: str = 'chi_unet'
    diffusion_policy: str = 'DDPM'
    sample_steps: int = 100
    pre_model: str = None
    theia_model_path: str = "/home/zqy/code/RL2/hugface/theia-base-patch16-224-cddsv"
    target_theia_model: str = "facebook/dinov2-large"

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    emb_dim: int = 256  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [1, 2, 4]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each
        # group have at least 8 channels; it seems 4 and 8 are simila
    )
    depth: bool = True
    dino: bool = True
    """use depth to train"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = 150
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 2000
    """the frequency of logging the training metrics"""
    eval_freq: int = 20_000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = 100_000
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 50
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 5
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, num_traj,normalization_params,theia_model):
        if data_path[-4:] == ".pkl":
            raise NotImplementedError()
        else:
            from diffusion_policy.utils import load_demo_dataset

            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        print("Raw trajectory loaded, start to pre-process the observations...")

        # 归一化
        self.state_mean = normalization_params["state_mean"]
        self.state_std = normalization_params["state_std"]
        self.rgb_mean = normalization_params["rgb_mean"]
        self.rgb_std = normalization_params["rgb_std"]
        self.depth_mean = normalization_params["depth_mean"]
        self.depth_std = normalization_params["depth_std"]

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)

            # Normalize depth if depth is included
            if args.depth and "depth" in _obs_traj_dict:
                depth = _obs_traj_dict["depth"].astype(np.float32) / 1024
                depth = np.clip(depth, 0, 1)  # Clip depth values > 1 to 1
                depth = torch.tensor(depth, dtype=torch.float32)
                _obs_traj_dict["depth"] = (depth - self.depth_mean) / self.depth_std

            # Normalize dino_v2
            rgb = torch.tensor(_obs_traj_dict["rgb"], dtype=torch.float32)
            rgb1 = rgb[:, :3, :, :].permute(0, 2, 3, 1)
            rgb1_dino = theia_model(rgb1)
            if rgb.shape[1]>3:
                rgb2 = rgb[:, 3:, :, :].permute(0, 2, 3, 1)
                rgb2_dino = theia_model(rgb2)
                rgb_dino = np.concatenate((rgb1_dino, rgb2_dino), axis=-1)
                _obs_traj_dict["rgb_dino"] = torch.tensor(rgb_dino, dtype=torch.float32).permute(0, 3, 1,2)
            else:
                _obs_traj_dict["rgb_dino"] = torch.tensor(rgb1_dino, dtype=torch.float32).permute(0, 3, 1,2)

            # Normalize RGB
            _obs_traj_dict["rgb"] = (rgb / 255.0 - self.rgb_mean) / self.rgb_std

            # Normalize state
            state = torch.tensor(_obs_traj_dict["state"], dtype=torch.float32)
            _obs_traj_dict["state"] = (state - self.state_mean) / self.state_std

            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list

        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
                "delta_pos" in args.control_mode
                or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                         max(0, start): start + self.obs_horizon
                         ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
                obs_seq["state"].shape[0] == self.obs_horizon
                and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


def create_diffusion_model(args):
    """
    Creates a diffusion model based on the specified arguments.

    Args:
    - args: An object containing the necessary configuration parameters such as:
      - nn_diffusion: type of the neural network for diffusion ('DiT1d' or 'chi_unet')
      - nn_condition: condition model ('RepVitObsCondition')
      - diffusion_policy: type of diffusion policy ('DDPM' or 'EDM')
      - act_dim: the action dimension
      - emb_dim: the embedding dimension
      - obs_horizon: the number of observation steps
      - horizon: the time horizon for the agent
      - device: the device to place the model on ('cuda' or 'cpu')
      - lr: learning rate
      - sample_steps: number of diffusion steps
      - pre_model: pre-trained model path (optional)
      - other necessary fields for condition models

    Returns:
    - agent: The diffusion policy model (DDPM or EDM) configured with the appropriate neural network.
    """

    # Diffusion Model
    if args.nn_diffusion == "DiT1d":
        nn_diffusion = DiT1d(
            args.act_dim, emb_dim=args.emb_dim * args.obs_horizon, d_model=320, n_heads=10, depth=2,
            timestep_emb_type="fourier").to(args.device)
    elif args.nn_diffusion == "chi_unet":
        nn_diffusion = ChiUNet1d(
            args.act_dim, args.emb_dim, args.obs_horizon, model_dim=256, emb_dim=256, dim_mult=args.unet_dims,
            obs_as_global_cond=True, timestep_emb_type="positional").to(device)
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
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

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
            assert (
                    control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # 归一化
    norm_path = f'./norm/{args.env_id}_{args.control_mode}_norm_params.json'
    # Load normalization parameters
    norm_params = load_normalization_params(norm_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_args_to_json(args, os.path.join(f"runs/{run_name}", 'params.json'))
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
    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=gym_utils.find_max_episode_steps_value(envs),
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,

            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    orignal_obs_space = tmp_env.observation_space
    tmp_env.close()
    model_dino = Theia_DINOv2( args.theia_model_path,args.target_theia_model,args.device)
    dataset = SmallDemoDataset_DiffusionPolicy(
        args.demo_path, obs_process_fn, orignal_obs_space, args.num_demos,norm_params,model_dino
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        pin_memory=True,
        persistent_workers=(args.num_dataload_workers > 0),
    )

    args.act_dim = envs.single_action_space.shape[0]
    args.state_dim = envs.single_observation_space["state"].shape[1]
    args.cam_num = int(envs.single_observation_space["rgb"].shape[-1] / 3)
    agent = create_diffusion_model(args)
    # Cosine LR schedule with linear warmup
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.total_iters)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    # ema = EMAModel(parameters=agent.parameters(), power=0.75)
    # ema_agent = Agent(envs, args).to(device)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    import time

    start_time = time.time()
    print("Training started at: ", time.ctime(start_time))

    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for iteration, data_batch in enumerate(train_dataloader):
        # # copy data from cpu to gpu
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward and compute loss
        obs_batch_dict = data_batch["observations"]
        obs_batch_dict = {
            k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()
        }
        act_batch = data_batch["actions"].cuda(non_blocking=True)

        # forward and compute loss
        # update diffusion
        diffusion_loss = agent.update(act_batch, obs_batch_dict)['loss']
        lr_scheduler.step()
        # diffusion_loss_list.append(diffusion_loss)
        last_tick = time.time()
        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {diffusion_loss}")
            writer.add_scalar("losses/total_loss", diffusion_loss, iteration)

        # Evaluation
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            agent.model.eval()
            agent.model_ema.eval()


            if args.diffusion_policy == "DDPM":
                solver = None
            elif args.diffusion_policy == "DDIM":
                solver = "ddim"
            elif args.diffusion_policy == "DPM":
                solver = "ode_dpmpp_2"
            elif args.diffusion_policy == "EDM":
                solver = "euler"

            eval_metrics = evaluate(
                args.num_eval_episodes, agent, envs,args,model_dino, args.sim_backend,norm_params
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
                    agent.save(f'runs/{run_name}/checkpoints/{k}.pt')
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

            agent.model.train()
            agent.model_ema.train()

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
            agent.save(f'runs/{run_name}/checkpoints/{iteration}.pt')


    end_time = time.time()
    print("Training finished at: ", time.ctime(end_time))

    training_time = end_time - start_time
    print("Total training time:{:.2f} minutes".format(training_time / 60))
    print("Total training time:{:.2f} hours".format(training_time / 3600))

    envs.close()
    writer.close()
