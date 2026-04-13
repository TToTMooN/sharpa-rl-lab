# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Grasp cache generation env config for xhand.

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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
        "right_hand_thumb_rota_link2",
        "right_hand_index_rota_link2",
        "right_hand_mid_link2",
        "right_hand_ring_link2",
        "right_hand_pinky_link2",
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
            joint_pos={
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
            prim_path="/World/envs/env_.*/Robot/right_hand_thumb_rota_link2",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_index_rota_link2",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_mid_link2",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_ring_link2",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_pinky_link2",
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
        "right_hand_thumb_rota_link2",
        "right_hand_index_rota_link2",
        "right_hand_mid_link2",
        "right_hand_ring_link2",
        "right_hand_pinky_link2",
    ]

    # Start single-scale for M3 first pass
    scale_range = [0.5, 0.5, 1]
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)
