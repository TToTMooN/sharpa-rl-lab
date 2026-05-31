# Copyright (c) 2022-2025. SPDX-License-Identifier: BSD-3-Clause
#
# Camera-free geometric probe for the xhand palm angle.
#
# Rendering (Camera/TiledCamera) segfaults on this RTX 5090 laptop, so the VLM
# visual_check loop is unavailable. This probe answers the same question —
# "does the object actually sit cradled in the hand at palm angle X?" — using
# only physics, no rendering. For each candidate (roll, pitch, yaw) palm
# orientation it:
#   1. sets the hand to that root rotation + the training finger pose,
#   2. drops a sphere from just above the fingertip centroid,
#   3. simulates under gravity,
#   4. reports whether the sphere stayed up, how far it drifted, and how many
#      fingertips ended up within "touching" distance.
#
# A good (angled) cradle keeps the sphere up (z drop small), low horizontal
# drift, and >=3 fingertips touching. A flat palm lets it roll off (large drift
# / big z drop / few fingertips).
#
# Usage:
#   pixi run python rl_isaaclab/scripts/palm_probe.py --headless
#   pixi run python rl_isaaclab/scripts/palm_probe.py --headless \
#       --eulers "0,-90,0; 20,-90,0; 0,-65,0; 20,-70,0; 30,-70,0"

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Camera-free xhand palm-angle probe.")
parser.add_argument("--eulers", type=str,
                    default="0,-90,0; 30,-90,0; 30,-70,0",
                    help="Semicolon-separated roll,pitch,yaw triples (degrees).")
parser.add_argument("--j2_values", type=str, default="0.0,0.6,1.2",
                    help="Comma-separated finger tip-curl (joint2) values to sweep — the caging DOF.")
parser.add_argument("--radius", type=float, default=0.035, help="Sphere radius (m).")
parser.add_argument("--steps", type=int, default=200, help="Sim steps per trial.")
parser.add_argument("--touch_margin", type=float, default=0.015,
                    help="Fingertip counted as touching if within radius+margin of sphere center.")
parser.add_argument("--drop", type=float, default=-0.01, help="Start sphere this far above fingertip centroid (m); negative = nestled into the cluster.")
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

REPO_ROOT = Path(__file__).resolve().parents[2]
XHAND_USD = str(REPO_ROOT / "assets/xhand/xhand_right.usd")

# Training pre-grasp finger pose (matches xhand_env_cfg joint_pos).
XHAND_JOINTS = {
    "right_hand_thumb_bend_joint": 1.5,
    "right_hand_thumb_rota_joint1": 0.0,
    "right_hand_thumb_rota_joint2": 0.5,
    "right_hand_index_bend_joint": 0.0,
    "right_hand_index_joint1": 1.9,
    "right_hand_index_joint2": 0.0,
    "right_hand_mid_joint1": 1.9,
    "right_hand_mid_joint2": 0.0,
    "right_hand_ring_joint1": 1.9,
    "right_hand_ring_joint2": 0.0,
    "right_hand_pinky_joint1": 1.9,
    "right_hand_pinky_joint2": 0.0,
}
FINGERTIPS = [
    "right_hand_thumb_rota_tip", "right_hand_index_rota_tip",
    "right_hand_mid_tip", "right_hand_ring_tip", "right_hand_pinky_tip",
]
HAND_POS = (0.0, 0.0, 0.5)


def palm_quat(roll_deg, pitch_deg, yaw_deg):
    r, p, y = (math.radians(a) * 0.5 for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cy*cr*cp + sy*sr*sp, cy*sr*cp - sy*cr*sp, cy*cr*sp + sy*sr*cp, sy*cr*cp - cy*sr*sp)


def main():
    sim = SimulationContext(SimulationCfg(dt=1/240, gravity=(0.0, 0.0, -9.81)))
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    hand = Articulation(ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=XHAND_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=HAND_POS, rot=palm_quat(0, -90, 0), joint_pos=XHAND_JOINTS),
        actuators={"joints": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=20.0, damping=1.0)},
        soft_joint_pos_limit_factor=1.0,
    ))
    sphere = RigidObject(RigidObjectCfg(
        prim_path="/World/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=args_cli.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=5.0, dynamic_friction=5.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    ))
    sim.reset()

    tip_ids, _ = hand.find_bodies(FINGERTIPS)
    j2_ids, _ = hand.find_joints([f"right_hand_{f}_joint2" for f in ("index", "mid", "ring", "pinky")])
    jvel = torch.zeros_like(hand.data.default_joint_vel)

    eulers = []
    for chunk in args_cli.eulers.split(";"):
        chunk = chunk.strip()
        if chunk:
            eulers.append(tuple(float(x) for x in chunk.split(",")))
    j2_values = [float(x) for x in args_cli.j2_values.split(",") if x.strip()]

    print("\n" + "=" * 104)
    print(f"{'roll,pitch,yaw':>14}  {'j2':>4} | {'z_drop':>7} | {'xy_drift':>8} | {'final_z':>7} | {'n_touch':>7} | tip dists (cm)")
    print("=" * 104)

    results = []
    for (roll, pitch, yaw) in eulers:
      for j2 in j2_values:
        jpos = hand.data.default_joint_pos.clone()
        jpos[:, j2_ids] = j2                       # finger tip-curl = caging DOF
        quat = torch.tensor([palm_quat(roll, pitch, yaw)], device=sim.device, dtype=torch.float32)
        # place hand
        root = hand.data.default_root_state.clone()
        root[:, :3] = torch.tensor([HAND_POS], device=sim.device)
        root[:, 3:7] = quat
        root[:, 7:] = 0.0
        hand.write_root_pose_to_sim(root[:, :7])
        hand.write_root_velocity_to_sim(root[:, 7:])
        hand.write_joint_state_to_sim(jpos, jvel)
        hand.set_joint_position_target(jpos)
        # park sphere high & let hand settle into pose
        sphere.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        sphere.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        for _ in range(20):
            hand.set_joint_position_target(jpos)
            hand.write_data_to_sim(); sphere.write_data_to_sim()
            sim.step(render=False); hand.update(sim.get_physics_dt()); sphere.update(sim.get_physics_dt())
        # fingertip centroid → drop point
        tips = hand.data.body_pos_w[0, tip_ids]            # (5,3)
        centroid = tips.mean(0)
        start = centroid.clone(); start[2] += args_cli.radius + args_cli.drop
        sphere.write_root_pose_to_sim(torch.cat([start, torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)]).unsqueeze(0))
        sphere.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        # simulate
        for _ in range(args_cli.steps):
            hand.set_joint_position_target(jpos)
            hand.write_data_to_sim(); sphere.write_data_to_sim()
            sim.step(render=False); hand.update(sim.get_physics_dt()); sphere.update(sim.get_physics_dt())
        obj = sphere.data.root_pos_w[0]
        tips = hand.data.body_pos_w[0, tip_ids]
        dists = torch.norm(tips - obj.unsqueeze(0), dim=-1)          # (5,)
        n_touch = int((dists < (args_cli.radius + args_cli.touch_margin)).sum().item())
        z_drop = float((start[2] - obj[2]).item())
        xy_drift = float(torch.norm(obj[:2] - centroid[:2]).item())
        final_z = float(obj[2].item())
        dist_cm = " ".join(f"{d*100:4.1f}" for d in dists.tolist())
        held = final_z > 0.45 and z_drop < 0.12
        flag = "HELD" if held else "drop"
        obj_xyz = tuple(round(float(v), 4) for v in obj.tolist())
        print(f"{roll:4.0f},{pitch:4.0f},{yaw:3.0f}  {j2:4.1f} | {z_drop:7.3f} | {xy_drift:8.3f} | {final_z:7.3f} | {n_touch:5d}   | {dist_cm}   [{flag}] rest_xyz={obj_xyz}")
        results.append((roll, pitch, yaw, j2, n_touch, z_drop, xy_drift, final_z, held, obj_xyz))

    print("=" * 104)
    held = [r for r in results if r[8]]
    if held:
        best = max(held, key=lambda r: (r[4], -r[5]))   # most fingertips, then least drop
        print(f"BEST held cradle: roll={best[0]:.0f} pitch={best[1]:.0f} yaw={best[2]:.0f} j2={best[3]:.1f} "
              f"n_touch={best[4]} z_drop={best[5]:.3f} xy_drift={best[6]:.3f} rest_xyz={best[9]}")
    else:
        print("No (angle, j2) held the sphere — widen the sweep, shrink object, or revisit finger pose.")
    print("=" * 104 + "\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
