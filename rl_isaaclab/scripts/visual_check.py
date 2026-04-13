# Copyright (c) 2022-2025. SPDX-License-Identifier: BSD-3-Clause
#
# Visual sanity check: load a hand USD + object USD with initial joint pose,
# run a few physics steps, screenshot via Camera sensor, optionally ask Gemini.
#
# This bypasses the gym task entirely — much simpler than hooking into the
# DirectRLEnv. Used to verify that the hand+object are geometrically sensible
# BEFORE committing to a 20min grasp-cache run.
#
# Usage:
#   pixi run python rl_isaaclab/scripts/visual_check.py --robot xhand --gemini
#   pixi run python rl_isaaclab/scripts/visual_check.py --robot sharpa --gemini

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual sanity check for hand+object.")
parser.add_argument("--robot", choices=["sharpa", "xhand"], default="xhand")
parser.add_argument("--steps", type=int, default=10, help="Sim steps before screenshot.")
parser.add_argument("--out", type=str, default="experiments/screenshots")
parser.add_argument("--gemini", action="store_true")
parser.add_argument("--gemini_model", type=str, default=None,
                    help="Override Gemini model (e.g. gemini-2.5-flash).")
parser.add_argument("--camera_pos", type=float, nargs=3, default=[0.35, 0.35, 0.85])
parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.55])
parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 1024])
parser.add_argument("--obj_scale", type=float, default=0.5, help="Cylinder scale.")
parser.add_argument("--prompt", type=str, default=None,
                    help="Override the Gemini prompt.")
AppLauncher.add_app_launcher_args(parser)

# Must inject --enable_cameras before parse so AppLauncher picks it up
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
# Force headless rendering (offscreen)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import torch
import numpy as np
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext


REPO_ROOT = Path(__file__).resolve().parents[2]

# --- robot specs ---
SHARPA_USD = str(REPO_ROOT / "assets/SharpaWave/right_sharpa_wave.usda")
XHAND_USD = str(REPO_ROOT / "assets/xhand/xhand_right.usd")
CYLINDER_USD = str(REPO_ROOT / "assets/cylinder/cylinder.usd")

SHARPA_INIT_JOINTS = {
    "right_thumb_CMC_FE": math.pi/180 * 95.12771,
    "right_thumb_CMC_AA": math.pi/180 * -3.11244,
    "right_thumb_MCP_FE": math.pi/180 * 14.81626,
    "right_thumb_MCP_AA": math.pi/180 * -1.03493,
    "right_thumb_IP": math.pi/180 * 12.23986,
    "right_index_MCP_FE": math.pi/180 * 65.21091,
    "right_index_MCP_AA": math.pi/180 * 6.1133,
    "right_index_PIP": math.pi/180 * 15.58495,
    "right_index_DIP": math.pi/180 * 5.90325,
    "right_middle_MCP_FE": math.pi/180 * 31.74149,
    "right_middle_MCP_AA": math.pi/180 * -0.95812,
    "right_middle_PIP": math.pi/180 * 41.88173,
    "right_middle_DIP": math.pi/180 * 12.844,
    "right_ring_MCP_FE": math.pi/180 * 31.72383,
    "right_ring_MCP_AA": math.pi/180 * 9.84458,
    "right_ring_PIP": math.pi/180 * 35.22366,
    "right_ring_DIP": math.pi/180 * 18.02839,
    "right_pinky_CMC": math.pi/180 * 10.9712,
    "right_pinky_MCP_FE": math.pi/180 * 68.30895,
    "right_pinky_MCP_AA": math.pi/180 * 7.99151,
    "right_pinky_PIP": math.pi/180 * 5.89626,
    "right_pinky_DIP": math.pi/180 * 5.89875,
}

XHAND_INIT_JOINTS = {
    "right_hand_thumb_bend_joint": 1.0,
    "right_hand_thumb_rota_joint1": 0.5,
    "right_hand_thumb_rota_joint2": 0.8,
    "right_hand_index_bend_joint": 0.0,
    "right_hand_index_joint1": 0.96,
    "right_hand_index_joint2": 0.96,
    "right_hand_mid_joint1": 0.96,
    "right_hand_mid_joint2": 0.96,
    "right_hand_ring_joint1": 0.96,
    "right_hand_ring_joint2": 0.96,
    "right_hand_pinky_joint1": 0.96,
    "right_hand_pinky_joint2": 0.96,
}

HAND_INIT_POS = (0.0, 0.0, 0.5)
HAND_INIT_ROT = (0.819152, 0.0, -0.5735764, 0.0)
CYL_INIT_POS = (-0.09559, -0.00517, 0.61906)


def main():
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- sim context ---
    sim_cfg = SimulationCfg(dt=1/240, gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args_cli.camera_pos), target=list(args_cli.camera_target))

    # Ground plane + lights
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)))

    # Hand
    if args_cli.robot == "sharpa":
        usd_path = SHARPA_USD
        init_joints = SHARPA_INIT_JOINTS
        robot_name = "SharpaWave"
    else:
        usd_path = XHAND_USD
        init_joints = XHAND_INIT_JOINTS
        robot_name = "xhand"

    print(f"[VIZ] Robot: {robot_name}")
    print(f"[VIZ] USD:   {usd_path}")

    hand_cfg = ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_INIT_POS,
            rot=HAND_INIT_ROT,
            joint_pos=init_joints,
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    hand = Articulation(hand_cfg)

    # Cylinder
    obj_cfg = RigidObjectCfg(
        prim_path="/World/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CYLINDER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            scale=(args_cli.obj_scale,) * 3,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CYL_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    obj = RigidObject(obj_cfg)

    # Camera
    cam_cfg = CameraCfg(
        prim_path="/World/VizCamera",
        update_period=0.0,
        height=args_cli.resolution[1],
        width=args_cli.resolution[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(args_cli.camera_pos),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    cam = Camera(cam_cfg)

    # Start
    sim.reset()
    cam.set_world_poses_from_view(
        eyes=torch.tensor([args_cli.camera_pos], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([args_cli.camera_target], dtype=torch.float32, device=sim.device),
    )

    # Step so physics settles
    for _ in range(max(1, args_cli.steps)):
        sim.step()
        hand.update(sim.cfg.dt)
        obj.update(sim.cfg.dt)
        cam.update(dt=sim.cfg.dt)

    # Grab screenshot
    rgb = cam.data.output["rgb"][0]  # [H, W, 3] or [H, W, 4]
    rgb_np = rgb.detach().cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = out_dir / f"{robot_name}_{ts}.png"
    from PIL import Image
    Image.fromarray(rgb_np).save(png_path)
    print(f"[VIZ] Saved: {png_path}")

    # Gemini analysis
    if args_cli.gemini:
        from rl_isaaclab.diagnostics.vlm import check_initial_pose, MODEL_ROBOTICS
        model = args_cli.gemini_model or MODEL_ROBOTICS
        print(f"[VIZ] Gemini analyzing with {model}...")
        if args_cli.prompt:
            from rl_isaaclab.diagnostics.vlm import analyze_image
            report = analyze_image(png_path, args_cli.prompt, model=model)
        else:
            report = check_initial_pose(png_path, robot_name=robot_name, model=model)

        print("=" * 70)
        print(f"GEMINI REPORT ({model})")
        print("=" * 70)
        print(report)
        print("=" * 70)

        md_path = str(png_path).replace(".png", "_report.md")
        with open(md_path, "w") as f:
            f.write(f"# Visual check — {robot_name}\n\n")
            f.write(f"- PNG: {png_path.name}\n")
            f.write(f"- Model: {model}\n")
            f.write(f"- Obj scale: {args_cli.obj_scale}\n")
            f.write(f"- Steps: {args_cli.steps}\n\n")
            f.write("## Report\n\n")
            f.write(report)
        print(f"[VIZ] Report: {md_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
