# Copyright (c) 2022-2025. SPDX-License-Identifier: BSD-3-Clause
#
# Does the xhand grasp cache actually HOLD the object under full gravity?
#
# The M3A "+54" run trained under a gravity curriculum that starts at g=0, so a
# positive reward there does NOT prove the grasp holds weight. This script loads
# the real grasp cache, places the hand joints + object pose EXACTLY as the env
# does on reset (cache row = [D hand dofs | 3 obj pos | 4 obj quat]), then turns
# on full gravity and measures how many of the cached configs keep the object up.
#
# Faithful to the env (same asset, same PD gains, same friction, same placement)
# and free of any guesswork about where the object "should" sit — the cache says
# where. This is the trustworthy baseline the geometric probe lacked.
#
# Usage:
#   pixi run python -u rl_isaaclab/scripts/cache_hold_check.py --headless \
#       --cache cache/xhand_sphere_grasp_linspace_1.0-1.0-1.npy --n 256 --steps 200

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check whether the xhand grasp cache holds under gravity.")
parser.add_argument("--cache", type=str, default="cache/xhand_sphere_grasp_linspace_1.0-1.0-1.npy")
parser.add_argument("--n", type=int, default=256, help="Number of cache rows (envs) to test.")
parser.add_argument("--steps", type=int, default=200, help="Sim steps under gravity.")
parser.add_argument("--radius", type=float, default=0.05, help="Sphere radius (matches grasp cfg).")
parser.add_argument("--friction", type=float, default=5.0)
parser.add_argument("--stiffness", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

REPO_ROOT = Path(__file__).resolve().parents[2]
XHAND_USD = str(REPO_ROOT / "assets/xhand/xhand_right.usd")
# Joint order matches xhand_env_cfg.actuated_joint_names (= cache column order).
ACTUATED = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2", "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]
PALM_FLAT = (0.7071068, 0.0, -0.7071068, 0.0)   # cache was generated at the flat palm
HAND_POS = (0.0, 0.0, 0.5)


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    hand = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=XHAND_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=HAND_POS, rot=PALM_FLAT),
        actuators={"joints": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=20.0, damping=1.0)},
        soft_joint_pos_limit_factor=1.0,
    )
    obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=5.0, dynamic_friction=5.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.075, 0.0, 0.62)),
    )


def main():
    cache = np.load(args_cli.cache)
    n = min(args_cli.n, cache.shape[0])
    D = len(ACTUATED)
    print(f"[cache_hold] cache={args_cli.cache} shape={cache.shape}; testing {n} configs, "
          f"{args_cli.steps} steps @ g=-9.81", flush=True)

    sim = SimulationContext(SimulationCfg(dt=1/240, gravity=(0.0, 0.0, -9.81), device=args_cli.device))
    scene = InteractiveScene(ProbeSceneCfg(num_envs=n, env_spacing=2.0))
    sim.reset()

    hand, obj = scene["hand"], scene["obj"]
    # reorder cache joint columns -> sim joint order
    sim_names = hand.joint_names
    col_for_sim = [ACTUATED.index(nm) if nm in ACTUATED else None for nm in sim_names]
    rows = torch.tensor(cache[:n], dtype=torch.float32, device=sim.device)

    jpos = hand.data.default_joint_pos.clone()
    for j, src in enumerate(col_for_sim):
        if src is not None:
            jpos[:, j] = rows[:, src]
    jvel = torch.zeros_like(hand.data.default_joint_vel)
    hand.write_joint_state_to_sim(jpos, jvel)
    hand.set_joint_position_target(jpos)

    obj_pos = rows[:, D:D+3] + scene.env_origins
    obj_quat = rows[:, D+3:D+7]
    obj_state = torch.cat([obj_pos, obj_quat, torch.zeros((n, 6), device=sim.device)], dim=1)
    obj.write_root_pose_to_sim(obj_state[:, :7])
    obj.write_root_velocity_to_sim(obj_state[:, 7:])

    z0 = obj_pos[:, 2].clone()
    for _ in range(args_cli.steps):
        hand.set_joint_position_target(jpos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    z1 = obj.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
    drop = z0 - obj.data.root_pos_w[:, 2]
    held = (drop < 0.05)                              # fell less than 5 cm
    print("=" * 60, flush=True)
    print(f"held (drop<5cm):  {held.float().mean().item()*100:5.1f}%  ({int(held.sum())}/{n})", flush=True)
    print(f"mean z drop:      {drop.mean().item()*1000:6.1f} mm", flush=True)
    print(f"median z drop:    {drop.median().item()*1000:6.1f} mm", flush=True)
    print(f"final obj z mean: {z1.mean().item():.3f}  (start {z0.mean().item():.3f})", flush=True)
    print("=" * 60, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
