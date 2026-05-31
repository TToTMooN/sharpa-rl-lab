# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Grasp cache generation env config for xhand.

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg as SharpaWaveGraspEnvCfg, EventCfg
from .palm_pose import palm_quat


@configclass
class XhandGraspEnvCfg(SharpaWaveGraspEnvCfg):
    """Grasp cache generation config for xhand. Inherits from SharpaWave grasp cfg,
    overrides hand-specific fields."""

    # --- env dims ---
    action_space = 12
    observation_space = 132  # (12 + 12 + 5 + 15) * 3

    # --- hand-specific contact/material ---
    contact_sensor_body_names: list[str] = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]
    num_hand_materials: int = 30
    elastomer_material_ids: list[int] = []

    # --- hand init pose ---
    # Palm orientation as sweepable XYZ-Euler degrees. (0, -90, 0) reproduces
    # the legacy flat palm-up pose (palm faces +z, fingers point +x, curl up).
    # The grasp cache is generated AT this orientation, so it MUST match
    # xhand_env_cfg.palm_euler_deg or the cached holds won't transfer to
    # training. See palm_pose.py / roadmap/M3B_angled_palm.md.
    palm_euler_deg = (0.0, -90.0, 0.0)
    hand_init_pose = ((0.0, 0.0, 0.5), palm_quat(*palm_euler_deg))

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
                enabled_self_collisions=False,
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
            # Root-curl-tip-flat pose: 4 fingers curl at base (j1=1.9 max) but
            # tip stays straight (j2=0.0) — fingertips end up high (z≈0.60) and
            # form a wall. Thumb side-pinch (bend=1.5, rota1=0.0) puts tip at
            # (-0.027, 0.058, 0.634). Cylinder lives in the cup between them.
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
                # Explicit PD gains — URDF-converted USD leaves drive stiffness
                # at 0, so joints couldn't resist cylinder contact forces and
                # the hand deformed on every run. 20 Nm/rad is enough to hold
                # the pose while the cylinder presses in.
                stiffness=20.0,
                damping=1.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

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

    fingertip_body_names = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]

    # Grasp cache save prefix (training env will look at cache/xhand_grasp_linspace_*.npy).
    # NOTE: grasp_cache_path is None here (inherited) because the grasp env is CREATING
    # the cache, not loading it. grasp_cache_save_prefix controls where it's written.
    grasp_cache_save_prefix = "cache/xhand_grasp_linspace"

    # SPHERE object instead of cylinder. Cylinder rotation is too constrained
    # for xhand's flat finger arrangement (intermittent contacts won't sustain
    # the rotation axis). A sphere is symmetric so any rotation reward applies,
    # and a sphere settles naturally into the finger cup under gravity.
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
        # Sphere (radius 0.035) at the cup center, just above 4-finger cluster.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.075, 0.0, 0.620),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    reset_height_lower = 0.585
    reset_height_upper = 0.625

    # Very high friction to make the partial pinch grasp survive gravity cycling.
    metal_base_friction = 5.0
    elastomer_base_friction = 5.0
    object_base_friction = 5.0

    # randomize_friction is False in the SharpaWave grasp cfg, which means friction
    # is NEVER explicitly applied (uses USD defaults). For xhand the URDF-converted
    # USD has low default friction. Enable randomize_friction with a tight range so
    # base_friction gets multiplied by ~1.0 and applied deterministically.
    randomize_friction = True
    randomize_friction_scale_lower = 1.0
    randomize_friction_scale_upper = 1.0

    # Sphere — single radius, scale_range still required by code.
    scale_range = [1.0, 1.0, 1]
    # Output cache to a different path so we don't clobber the cylinder cache.
    grasp_cache_save_prefix = "cache/xhand_sphere_grasp_linspace"

    # Relaxed contact requirements for xhand (flat cup can't reach 3 simultaneous
    # side contacts; accept 2 fingers + weaker 0.2N threshold).
    grasp_min_contacts = 2
    grasp_force_thresh = 0.2
    # Incremental save mode: xhand's intermittent grasps can't survive 400-step
    # gravity cycling, so save any momentary cond-true state instead of waiting
    # for full episode end. Streak threshold filters for "held for N consecutive
    # steps" — higher = more stable grasps in cache (and smaller cache).
    grasp_save_incremental = True
    grasp_save_streak_threshold = 3
    grasp_target_size = 20000
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)
