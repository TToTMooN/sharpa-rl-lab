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
    hand_init_pose = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))

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
            # Cupped fingertip cage pose (verified TRAINABLE by Gemini 3.1 Pro,
            # iter3 in xhand_20260413_000438_report.md).
            joint_pos={
                "right_hand_thumb_bend_joint": 1.5,
                "right_hand_thumb_rota_joint1": 0.4,
                "right_hand_thumb_rota_joint2": 0.7,
                "right_hand_index_bend_joint": 0.0,
                "right_hand_index_joint1": 0.85,
                "right_hand_index_joint2": 0.85,
                "right_hand_mid_joint1": 0.85,
                "right_hand_mid_joint2": 0.85,
                "right_hand_ring_joint1": 0.85,
                "right_hand_ring_joint2": 0.85,
                "right_hand_pinky_joint1": 0.85,
                "right_hand_pinky_joint2": 0.85,
            },
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

    # Override cylinder spawn position to match the verified TRAINABLE pose.
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../assets/cylinder/cylinder.usd",
            ),
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
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.04, 0.0, 0.585), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    reset_height_lower = 0.565
    reset_height_upper = 0.605

    # xhand has no separate elastomer/metal materials — bump friction so cylinder
    # doesn't slip out of grasp during the gravity-cycling search.
    metal_base_friction = 0.8
    elastomer_base_friction = 0.8
    object_base_friction = 0.8

    # randomize_friction is False in the SharpaWave grasp cfg, which means friction
    # is NEVER explicitly applied (uses USD defaults). For xhand the URDF-converted
    # USD has low default friction. Enable randomize_friction with a tight range so
    # base_friction gets multiplied by ~1.0 and applied deterministically.
    randomize_friction = True
    randomize_friction_scale_lower = 1.0
    randomize_friction_scale_upper = 1.0

    # Start single-scale for M3 first pass
    scale_range = [0.5, 0.5, 1]
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)
