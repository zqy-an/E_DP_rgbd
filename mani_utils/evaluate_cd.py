from collections import defaultdict
import gymnasium
import numpy as np
import torch

from mani_skill.utils import common

def collect_episode_info(infos, result):
    if "final_info" in infos: # infos is a dict

        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            result['return'].append(ep['r'][0])
            result['episode_len'].append(ep["l"][0])
            if "success" in info:
                result['success'].append(info['success'])
            if "fail" in info:
                result['fail'].append(info['fail'])
    return result

def evaluate(n_ep: int, agent, eval_envs, args, theia_model,sim_backend: str, norm_params = None):

    if norm_params is not None:
        norm_params = {key: value.unsqueeze(0).to(args.device) if value is not None else None for key, value in norm_params.items()}
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        n, seq, h, w ,cn = obs["rgb"].shape
        while eps_count < n_ep:
            obs = common.to_tensor(obs, args.device)
            # theia
            rgb_ = obs["rgb"].view(seq * n, *obs["rgb"].shape[2:])
            rgb1 = rgb_[:, :, :, :3]
            rgb1_dino = theia_model(rgb1)
            if rgb_.shape[-1]>3:
                rgb2 = rgb_[:, :, :, 3:]
                rgb2_dino = theia_model(rgb2)
                rgb_dino = np.concatenate((rgb1_dino, rgb2_dino), axis=-1)
                rgb_dino = torch.tensor(rgb_dino, dtype=torch.float32).permute(0, 3, 1,2).to(args.device)
            else:
                rgb_dino = torch.tensor(rgb1_dino, dtype=torch.float32).permute(0, 3, 1,2).to(args.device)
            obs["rgb_dino"] = rgb_dino.view(n, seq, *rgb_dino.shape[1:])
            if norm_params is not None:
                # Apply normalization to 'rgb', 'depth', and 'state' in the observation
                if "rgb_mean" in norm_params and "rgb" in obs:
                    rgb = obs["rgb"] / 255.0
                    rgb = rgb.permute(0, 1, 4, 2, 3)
                    obs["rgb"] = (rgb - norm_params["rgb_mean"]) / norm_params["rgb_std"]

                else:
                    obs["rgb"] = obs["rgb"].permute(0, 1, 4, 2, 3)


                if "depth_mean" in norm_params and "depth" in obs:
                    depth = obs["depth"]
                    depth = torch.clamp(depth / 1024.0, 0.0, 1.0)  # Normalize depth between [0, 1]
                    depth = depth.permute(0,1, 4, 2, 3)  # Assuming depth is (B, H, W, C)
                    obs["depth"] = (depth - norm_params["depth_mean"]) / norm_params["depth_std"]


                if "state_mean" in norm_params:
                    state = obs["state"]
                    obs["state"] = (state - norm_params["state_mean"]) / norm_params["state_std"]
                if args.state_fix:
                    indices = torch.cat((torch.arange(0, 9), torch.arange(18, obs["state"].shape[-1]))).to(args.device)
                    obs["state"] = obs["state"].index_select(-1, indices)

            # action_seq = agent.get_action(obs)
            prior = torch.zeros((n, args.pred_horizon, args.act_dim), device=args.device)
            action_seq, _ = agent.sample(prior=prior, n_samples=n, sample_steps=args.sample_steps,
                                      solver=None, condition_cfg=obs, w_cfg=1.0, use_ema=True)

            if sim_backend == "cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics