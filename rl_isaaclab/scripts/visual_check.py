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
parser.add_argument("--hand_pos", type=float, nargs=3, default=None,
                    help="Override hand root position (default: (0, 0, 0.5)).")
parser.add_argument("--hand_rot", type=float, nargs=4, default=None,
                    help="Override hand root quaternion (w, x, y, z). Default: 70° Y.")
parser.add_argument("--cyl_pos", type=float, nargs=3, default=None,
                    help="Override cylinder initial position.")
parser.add_argument("--thumb_bend", type=float, default=None,
                    help="xhand: override right_hand_thumb_bend_joint initial value.")
parser.add_argument("--thumb_rota1", type=float, default=None,
                    help="xhand: override right_hand_thumb_rota_joint1 initial value.")
parser.add_argument("--thumb_rota2", type=float, default=None,
                    help="xhand: override right_hand_thumb_rota_joint2 initial value.")
parser.add_argument("--finger_flex", type=float, default=None,
                    help="xhand: override all finger joint1/joint2 initial values (default 0.96).")
parser.add_argument("--gravity", action="store_true",
                    help="Enable gravity (default off for pose inspection).")
parser.add_argument("--cache", type=str, default=None,
                    help="Load joint positions + cylinder pos from a grasp cache .npy file (row 0 used). "
                    "Shape: (N, num_hand_dofs+7). Overrides raw cfg init pose.")
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


def _apply_overrides():
    """Apply CLI overrides to the module-level pose constants and joint dicts."""
    global HAND_INIT_POS, HAND_INIT_ROT, CYL_INIT_POS
    if args_cli.hand_pos is not None:
        HAND_INIT_POS = tuple(args_cli.hand_pos)
    if args_cli.hand_rot is not None:
        HAND_INIT_ROT = tuple(args_cli.hand_rot)
    if args_cli.cyl_pos is not None:
        CYL_INIT_POS = tuple(args_cli.cyl_pos)
    if args_cli.thumb_bend is not None:
        XHAND_INIT_JOINTS["right_hand_thumb_bend_joint"] = args_cli.thumb_bend
    if args_cli.thumb_rota1 is not None:
        XHAND_INIT_JOINTS["right_hand_thumb_rota_joint1"] = args_cli.thumb_rota1
    if args_cli.thumb_rota2 is not None:
        XHAND_INIT_JOINTS["right_hand_thumb_rota_joint2"] = args_cli.thumb_rota2
    if args_cli.finger_flex is not None:
        v = args_cli.finger_flex
        for f in ("index", "mid", "ring", "pinky"):
            XHAND_INIT_JOINTS[f"right_hand_{f}_joint1"] = v
            XHAND_INIT_JOINTS[f"right_hand_{f}_joint2"] = v


def _load_cache_row(cache_path: str, joint_names: list[str]) -> tuple[dict, tuple]:
    """Load a row from a grasp cache npy and return (joint_pos_dict, cyl_pos).

    Cache row layout: [D hand DOFs | 3 object_pos | 4 object_quat]
    The joint order in the cache matches the hand's joint order in the Articulation,
    which we pass in as joint_names (from the cfg's actuated_joint_names list).
    """
    data = np.load(cache_path)
    print(f"[VIZ] Cache shape: {data.shape}, using row 0")
    row = data[0]
    D = len(joint_names)
    joint_dict = {name: float(row[i]) for i, name in enumerate(joint_names)}
    cyl_pos = tuple(row[D:D + 3].tolist())
    return joint_dict, cyl_pos


def main():
    _apply_overrides()
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- sim context ---
    gravity = (0.0, 0.0, -9.81) if args_cli.gravity else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(dt=1/240, gravity=gravity)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args_cli.camera_pos), target=list(args_cli.camera_target))

    # Ground plane + lights
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)))

    # Hand
    if args_cli.robot == "sharpa":
        usd_path = SHARPA_USD
        init_joints = dict(SHARPA_INIT_JOINTS)
        robot_name = "SharpaWave"
    else:
        usd_path = XHAND_USD
        init_joints = dict(XHAND_INIT_JOINTS)
        robot_name = "xhand"

    # Optionally load cyl_pos from a grasp cache (joint positions applied post-reset)
    global CYL_INIT_POS
    cache_row = None
    if args_cli.cache:
        cache_path = args_cli.cache
        if not os.path.exists(cache_path):
            print(f"[VIZ] ERROR: cache file not found: {cache_path}")
            raise FileNotFoundError(cache_path)
        cache_data = np.load(cache_path)
        cache_row = cache_data[0]
        D = len(init_joints)  # num hand DOFs
        CYL_INIT_POS = tuple(cache_row[D:D+3].tolist())
        print(f"[VIZ] Cache: {cache_data.shape} row 0, cyl_pos={CYL_INIT_POS}")
        # init_joints stays as default — we'll overwrite via write_joint_state_to_sim
        # AFTER the articulation is constructed (so we know its real joint_names order).

    print(f"[VIZ] Robot: {robot_name}")
    print(f"[VIZ] USD:   {usd_path}")
    print(f"[VIZ] Hand pose: pos={HAND_INIT_POS}, rot={HAND_INIT_ROT}")
    print(f"[VIZ] Cyl pos: {CYL_INIT_POS}")

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

    # --- Push default_joint_pos to sim.
    # Isaac Lab's init_state.joint_pos populates _data.default_joint_pos but does NOT
    # write to the physics sim automatically. The env wrapper does this during reset.
    # For our standalone script we must do it explicitly.
    default_pos = hand.data.default_joint_pos.clone()
    default_vel = hand.data.default_joint_vel.clone()
    hand.write_joint_state_to_sim(default_pos, default_vel)
    hand.set_joint_position_target(default_pos)

    # Also push default root pose to sim (so hand_init_pose / rot take effect)
    root_state = hand.data.default_root_state.clone()
    hand.write_root_state_to_sim(root_state)
    # Object default root state
    obj_default = obj.data.default_root_state.clone()
    obj.write_root_pose_to_sim(obj_default[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(obj_default[:, 7:]))

    # --- Debug: show actual joint names, limits, current positions ---
    print(f"[VIZ DEBUG] Articulation joint_names (order is what the cache row uses):", flush=True)
    names = hand.data.joint_names
    pos = hand.data.joint_pos[0]
    lim_lo = hand.data.joint_pos_limits[0, :, 0]
    lim_hi = hand.data.joint_pos_limits[0, :, 1]
    for i, n in enumerate(names):
        p = pos[i].item() if i < pos.numel() else float("nan")
        lo = lim_lo[i].item() if i < lim_lo.numel() else float("nan")
        hi = lim_hi[i].item() if i < lim_hi.numel() else float("nan")
        print(f"  [{i:2d}] {n}: pos={p:+.4f} default={default_pos[0, i].item():+.4f} limits=[{lo:+.4f}, {hi:+.4f}]", flush=True)
    print(f"[VIZ DEBUG] total joints={len(names)}", flush=True)

    # Apply grasp cache joint positions in the articulation's real DOF order
    # (cfg.init_state dict order doesn't necessarily match USD tree traversal order)
    if cache_row is not None:
        real_joint_names = hand.data.joint_names
        print(f"[VIZ] Articulation joint_names: {real_joint_names}")
        D = len(real_joint_names)
        cache_joint_values = torch.tensor(cache_row[:D], dtype=torch.float32, device=sim.device).unsqueeze(0)
        zero_vel = torch.zeros_like(cache_joint_values)
        hand.write_joint_state_to_sim(cache_joint_values, zero_vel)
        hand.set_joint_position_target(cache_joint_values)
        # also write object pose from cache (if shape supports it)
        if cache_row.shape[0] >= D + 7:
            obj_pos = torch.tensor([cache_row[D:D+3]], dtype=torch.float32, device=sim.device)
            obj_quat = torch.tensor([cache_row[D+3:D+7]], dtype=torch.float32, device=sim.device)
            obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
            obj.write_root_pose_to_sim(obj_pose)
        print(f"[VIZ] Applied cache joints + object pose")

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
        from rl_isaaclab.diagnostics.vlm import check_initial_pose, MODEL_PRO
        model = args_cli.gemini_model or MODEL_PRO
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
