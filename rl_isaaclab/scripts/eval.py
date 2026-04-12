# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Quantitative evaluation script for trained policies.
# Loads a checkpoint, runs N episodes, reports mean/std reward, success rate, episode length.

import argparse
import sys
import shutil
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained RL policy.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs for eval.")
parser.add_argument("--task", type=str, default=None, help="Gym task ID.")
parser.add_argument("--seed", type=int, default=42, help="Env seed.")
parser.add_argument("--cache", type=str, default=None, help="Grasp cache path.")
parser.add_argument("--load_path", type=str, required=True, help="Checkpoint to evaluate.")
parser.add_argument("--algorithm", type=str, default="PPO", help="PPO or ProprioAdapt.")
parser.add_argument("--num_episodes", type=int, default=256, help="Minimum episodes to evaluate.")
parser.add_argument("--output_json", type=str, default=None, help="Write metrics to JSON file.")
parser.add_argument("--eval_name", type=str, default="eval", help="Label for this evaluation run.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
from datetime import datetime

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if os.path.isdir("outputs"):
        shutil.rmtree("outputs/")

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm
    agent_cfg["load_path"] = args_cli.load_path
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    agent_cfg["algorithm"]["minibatch_size"] = min(args_cli.num_envs * 8, 32768)

    # Disable training-time randomization for cleaner eval
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True  # keep friction rand (play.py also keeps this)
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache

    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    log_dir = os.path.join("logs", "eval", datetime.now().strftime(f"{args_cli.eval_name}_%Y-%m-%d_%H-%M-%S"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)

    print(f"[EVAL] Loading checkpoint: {args_cli.load_path}")
    agent.restore_test(args_cli.load_path)
    agent.set_eval()

    # Run evaluation loop
    num_envs = args_cli.num_envs
    device = env.device
    obs_dict = env.reset()

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    completed_rewards = []
    completed_lengths = []

    max_ep_len = env.unwrapped.max_episode_length
    total_steps = 0
    # Run for enough steps to collect target number of episodes.
    # Each env produces one episode every max_ep_len steps (plus early terminations).
    max_total_steps = (args_cli.num_episodes // num_envs + 2) * max_ep_len + 200

    print(f"[EVAL] max_episode_length={max_ep_len}, running up to {max_total_steps} steps")
    print(f"[EVAL] Target: {args_cli.num_episodes} episodes across {num_envs} envs")

    is_proprio_adapt = args_cli.algorithm == "ProprioAdapt"

    with torch.no_grad():
        while len(completed_rewards) < args_cli.num_episodes and total_steps < max_total_steps:
            if is_proprio_adapt:
                input_dict = {
                    'obs': agent.running_mean_std(obs_dict['obs']),
                    'proprio_hist': agent.sa_mean_std(obs_dict['proprio_hist'].detach()),
                }
            else:
                input_dict = {
                    'obs': agent.running_mean_std(obs_dict['obs']),
                    'priv_info': obs_dict['priv_info'],
                }
            mu = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, rewards, dones, infos = env.step(mu)

            episode_rewards += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            episode_lengths += 1
            total_steps += 1

            done_mask = dones.bool()
            if done_mask.any():
                for idx in done_mask.nonzero(as_tuple=True)[0].tolist():
                    completed_rewards.append(episode_rewards[idx].item())
                    completed_lengths.append(episode_lengths[idx].item())
                episode_rewards[done_mask] = 0
                episode_lengths[done_mask] = 0

            if total_steps % 50 == 0:
                print(f"[EVAL] step {total_steps}, episodes collected: {len(completed_rewards)}")

    # Compute metrics
    rewards_t = torch.tensor(completed_rewards[:args_cli.num_episodes])
    lengths_t = torch.tensor(completed_lengths[:args_cli.num_episodes], dtype=torch.float32)

    metrics = {
        "checkpoint": args_cli.load_path,
        "eval_name": args_cli.eval_name,
        "num_episodes": int(rewards_t.numel()),
        "num_envs": num_envs,
        "mean_reward": float(rewards_t.mean().item()),
        "std_reward": float(rewards_t.std().item()),
        "min_reward": float(rewards_t.min().item()),
        "max_reward": float(rewards_t.max().item()),
        "median_reward": float(rewards_t.median().item()),
        "mean_length": float(lengths_t.mean().item()),
        "max_length": float(lengths_t.max().item()),
        "full_episode_rate": float((lengths_t >= max_ep_len - 1).float().mean().item()),
    }

    print()
    print("=" * 60)
    print(f"EVAL RESULTS: {args_cli.eval_name}")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 60)

    if args_cli.output_json is not None:
        os.makedirs(os.path.dirname(args_cli.output_json), exist_ok=True)
        with open(args_cli.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[EVAL] Metrics written to {args_cli.output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
