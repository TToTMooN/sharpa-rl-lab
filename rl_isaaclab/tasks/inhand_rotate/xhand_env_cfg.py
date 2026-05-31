# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Environment config for RoboEra xhand.
#
# xhand vs SharpaWave summary:
# - 12 active DOF (vs 22)
# - 5 fingers: thumb (3 DOF), index (3 DOF including bend), mid/ring/pinky (2 DOF each)
# - No separate elastomer/DP contact bodies — contact sensors are on *_tip fingertip links
# - Observation: (12 joint pos + 12 targets + 5 contacts + 15 contact pos) × 3 frames = 132
# - Action space: 12

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .sharpa_wave_env_cfg import SharpaWaveEnvCfg, EventCfg
from .palm_pose import palm_quat


@configclass
class XhandEnvCfg(SharpaWaveEnvCfg):
    """Config for in-hand rotation on the RoboEra xhand.

    Inherits from SharpaWaveEnvCfg and overrides hand-specific fields:
    URDF, action/observation dims, fingertip/contact bodies, joint init pose,
    material randomization settings.
    """

    # --- env dims (hand-specific) ---
    action_space = 12
    observation_space = 132  # (12 + 12 + 5 + 15) * 3

    # --- hand-specific contact/material ---
    # xhand has no distinct elastomer bodies; use fingertip *_tip links as contact sensors.
    contact_sensor_body_names: list[str] = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]
    # Uniform friction across xhand materials — no elastomer distinction needed.
    # Setting elastomer_material_ids to empty skips the elastomer friction scaling.
    num_hand_materials: int = 30  # xhand has 30 links (approx materials)
    elastomer_material_ids: list[int] = []

    # --- hand pose (in world frame when spawned) ---
    # Palm orientation expressed as sweepable XYZ-Euler degrees (roll, pitch,
    # yaw). (0, -90, 0) reproduces the legacy FLAT palm-up pose exactly: palm
    # faces +z, fingers point +x and curl up into a ring.
    #
    # The flat palm is a weak cradle — the object only rests on a ring of
    # fingertips and rolls out through the thumb-opposite gap under gravity.
    # A non-flat (angled) palm tilts the hand so the object settles into the
    # corner between palm and curled fingers and gravity presses it INTO that
    # wall. Tune via the VLM loop (visual_check.py); see
    # roadmap/M3B_angled_palm.md. MUST match xhand_grasp_env_cfg.palm_euler_deg.
    palm_euler_deg = (0.0, -90.0, 0.0)
    hand_init_pose = ((0.0, 0.0, 0.5), palm_quat(*palm_euler_deg))


    # --- robot articulation ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../assets/xhand/xhand_right.usd",
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # xhand linkages are close; disable to avoid spurious collisions
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_init_pose[0],
            rot=hand_init_pose[1],
            # Side-pinch root-curl-tip-flat pose. Real grasp cache generated
            # by EXP-028 (cache/xhand_grasp_linspace_1.0-1.0-1.npy, 50k entries).
            joint_pos={
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
            },
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                # URDF-converted USD has drive stiffness=0; need explicit PD to hold pose.
                stiffness=20.0,
                damping=1.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # --- contact sensors (5 fingertips, no elastomer/DP split) ---
    contact_sensor = [
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_thumb_rota_tip",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_index_rota_tip",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_mid_tip",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_ring_tip",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_pinky_tip",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
    ]

    # --- actuated joints (12 DOF) ---
    actuated_joint_names = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    # --- fingertip body names (same as contact sensors for xhand) ---
    fingertip_body_names = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]

    # SPHERE object (radius 50mm) — switched from cylinder because xhand's
    # flat finger arrangement can't form a stable wrap grasp on a cylinder.
    # Sphere is symmetric in all directions and settles into the cup.
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.075, 0.0, 0.620), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Widened from (0.6, 0.64) to give episodes room to run. Per-env bounds
    # are set as (obj_default_z - window/2, obj_default_z + window/2) in
    # _reset_idx, so what matters is the delta: 0.4 gives a ±0.2m window.
    # Alive bonus tells PPO when the sphere is "safe" via the z threshold
    # separately — we don't need height_reset to be tight too.
    reset_height_lower = 0.2
    reset_height_upper = 0.6

    # High friction matching grasp gen.
    metal_base_friction = 5.0
    elastomer_base_friction = 5.0
    object_base_friction = 5.0

    # --- Reward shaping: "hold first, rotate later" ---
    # Stage 1 of the M3A curriculum is implicitly encoded in the reward
    # weights + gravity schedule below, without splitting training into
    # separate runs:
    #   - object_pos_reward_scale dominates (alive/proximity signal)
    #   - rotate_reward_scale small but nonzero (so the policy starts
    #     picking up rotation as a bonus once hold is stable)
    #   - penalties small to avoid drowning out the positive signal
    # As the agent learns to hold in the first few million steps, the
    # gravity schedule below will ramp up the difficulty and the policy
    # can transfer what it learned to rotate under real gravity.
    # EXP-M3A-7 diagnostic showed work_penalty raw ≈ 6000 per step, so even
    # -0.02 weight produces -120/step, DOMINATING the total. Optimal policy
    # becomes "minimize torque" which means "don't hold the sphere". Zero
    # them during curriculum; add back in final stage only.
    object_linvel_penalty_scale = -0.02
    pos_diff_penalty_scale = -0.02
    torque_penalty_scale = 0.0
    work_penalty_scale = 0.0
    object_pos_reward_scale = 0.1  # proximity-to-cache reward — small
    rotate_reward_scale = 1.0  # half of SharpaWave default (2.5)
    # Contact reward: +2 per fingertip touching the sphere (force > 0.1N).
    contact_reward_scale = 2.0
    # Alive bonus: +5 per step while sphere z > 0.55 (above the palm).
    # Sphere cache positions are z ~ 0.60-0.65 and palm is at z ~ 0.5, so
    # 0.55 catches "sphere has dropped off the hand" without false-firing
    # from small positional drift.
    alive_bonus_scale = 5.0
    alive_bonus_z_threshold = 0.55

    # Print reward components every 200 env steps for diagnosis.
    reward_debug_every = 200

    # --- Step-scheduled gravity curriculum ---
    # Interpolate world gravity z linearly between keypoints on the env's
    # common_step_counter (per-env-step count, shared across num_envs). The
    # env reads this in sharpa_wave_env._get_dones and calls set_gravity().
    #
    # For 4096 envs and PPO horizon=8, common_step_counter ≈ agent_steps / 4096.
    # EXP-M3A-2b showed the ramp from step 2000 to 10000 was WAY too fast —
    # reward degraded from -380 at step 2000 (g=0) to -609 at step 4800 (g=-3.4)
    # because PPO couldn't adapt quickly enough. Slowed 5x.
    # Schedule:
    #   0–3000 steps:     g=0 (pure hold phase, ~12M agent steps to converge)
    #   3000–50000 steps: g ramps linearly to -9.81 over ~190M agent steps
    #   50000+ steps:     full gravity, standard task
    gravity_curriculum = False  # disable the reset-rate-gated SharpaWave curriculum
    gravity_schedule = [
        (0,      0.0),
        (3_000,  0.0),
        (50_000, -9.81),
    ]

    # --- override grasp cache path for xhand ---
    # cache/xhand_sphere_grasp_linspace_*.npy generated by EXP-030b incremental
    # save mode with sphere object.
    grasp_cache_path = "cache/xhand_sphere_grasp_linspace"

    scale_range = [1.0, 1.0, 1]

    # Rebind events with xhand's scale_range (SharpaWaveEnvCfg binds at class body time)
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)

    # Start with PD gain randomization disabled to match xhand firmware tuning (TODO: sweep)
    randomize_pd_gains = False
