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
    # Approximate palm-up orientation; may need tuning based on visualization.
    hand_init_pose = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))

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
            # Wrap pose (matches synth cache in cache/xhand_grasp_linspace_1.0-1.0-1.npy).
            # Gravity-cycling grasp gen couldn't find stable grasps for xhand's flat
            # cup pose — the 4-finger wall cannot wrap a cylinder radially. Instead we
            # seed training with a synthetic perturbed-pose cache.
            joint_pos={
                "right_hand_thumb_bend_joint": 1.8,
                "right_hand_thumb_rota_joint1": 1.5,
                "right_hand_thumb_rota_joint2": 1.0,
                "right_hand_index_bend_joint": 0.0,
                "right_hand_index_joint1": 0.95,
                "right_hand_index_joint2": 1.1,
                "right_hand_mid_joint1": 0.95,
                "right_hand_mid_joint2": 1.1,
                "right_hand_ring_joint1": 0.95,
                "right_hand_ring_joint2": 1.1,
                "right_hand_pinky_joint1": 0.95,
                "right_hand_pinky_joint2": 1.1,
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

    # --- override cylinder spawn (palm location is different from SharpaWave) ---
    # Verified TRAINABLE in iter3 visual_check.
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
        # Cylinder spawn matches synth cache entries (x=-0.06 ± 5mm).
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.06, 0.0, 0.585), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Reset height bounds for cylinder centered at z=0.585 (40mm window).
    reset_height_lower = 0.565
    reset_height_upper = 0.605

    # xhand has no separate elastomer/metal materials (elastomer_material_ids=[]),
    # so all hand materials get metal_base_friction. The SharpaWave default of 0.1
    # is essentially frictionless — cylinder slips out of grasp immediately.
    # Treat xhand's whole hand as if it were elastomer pads.
    metal_base_friction = 0.8
    elastomer_base_friction = 0.8  # unused since elastomer_material_ids=[], but consistent
    object_base_friction = 0.8

    # --- override grasp cache path for xhand ---
    # This loads cache/xhand_grasp_linspace_1.0-1.0-1.npy which is SYNTHETIC
    # (see rl_isaaclab/scripts/synth_xhand_cache.py). Real grasp-gen failed for
    # xhand because the flat cup pose can't wrap a cylinder; we seed training
    # with perturbed-init-pose entries and let PPO learn to hold.
    grasp_cache_path = "cache/xhand_grasp_linspace"

    scale_range = [1.0, 1.0, 1]

    # Rebind events with xhand's scale_range (SharpaWaveEnvCfg binds at class body time)
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)

    # Start with PD gain randomization disabled to match xhand firmware tuning (TODO: sweep)
    randomize_pd_gains = False
