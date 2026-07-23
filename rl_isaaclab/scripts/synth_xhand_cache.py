"""Synthesize a grasp cache for xhand without running gravity-cycling grasp gen.

Background: xhand's 12-DOF cupped pose can't form a stable wrap around a
cylinder (finger cluster geometry doesn't match cylindrical surface for all
4 main fingers simultaneously), so the gravity-cycling `gen_grasp.py` can't
find stable grasps. Rather than keep iterating on grasp generation, we
bypass it: write a synthetic cache of (hand_dof + obj_pose) entries using
the env's init pose with small noise. The training PPO will then learn
from this distribution without requiring true pre-grasps.

Cache row layout matches sharpa_wave_env.py loader: [D hand DOFs | 3 obj_pos | 4 obj_quat].

Usage:
  pixi run python rl_isaaclab/scripts/synth_xhand_cache.py

Output: cache/xhand_sphere_grasp_linspace_1.0-1.0-1.npy
"""

from __future__ import annotations
import os
import numpy as np

# Joint order must match the articulation's internal joint_names order (level-sequential),
# not the cfg's finger-sequential dict order. Verified from visual_check.py debug output:
#  [ 0] right_hand_index_bend_joint
#  [ 1] right_hand_mid_joint1
#  [ 2] right_hand_pinky_joint1
#  [ 3] right_hand_ring_joint1
#  [ 4] right_hand_thumb_bend_joint
#  [ 5] right_hand_index_joint1
#  [ 6] right_hand_mid_joint2
#  [ 7] right_hand_pinky_joint2
#  [ 8] right_hand_ring_joint2
#  [ 9] right_hand_thumb_rota_joint1
#  [10] right_hand_index_joint2
#  [11] right_hand_thumb_rota_joint2
# Q4 nest pose (palm_euler_deg=(5,-90,0)): moderate root curl + strong tip
# curl forms a palm bowl, thumb wraps over. Statically holds the r=0.04
# sphere under full gravity for 900+ steps.
JOINT_VALUES = [
    0.0,   # index_bend
    0.5,   # mid_joint1
    0.5,   # pinky_joint1
    0.5,   # ring_joint1
    1.2,   # thumb_bend
    0.5,   # index_joint1
    0.9,   # mid_joint2
    0.9,   # pinky_joint2
    0.9,   # ring_joint2
    1.4,   # thumb_rota1
    0.9,   # index_joint2
    0.8,   # thumb_rota2
]

# Joint limits (lower, upper) from URDF
JOINT_LIMITS = [
    (-0.1740, 0.1740),   # index_bend
    (0.0, 1.9190),       # mid_joint1
    (0.0, 1.9190),       # pinky_joint1
    (0.0, 1.9190),       # ring_joint1
    (0.0, 1.8320),       # thumb_bend
    (0.0, 1.9190),       # index_joint1
    (0.0, 1.9190),       # mid_joint2
    (0.0, 1.9190),       # pinky_joint2
    (0.0, 1.9190),       # ring_joint2
    (-0.6980, 1.5700),   # thumb_rota1
    (0.0, 1.9190),       # index_joint2
    (0.0, 1.5700),       # thumb_rota2
]

# Settled Q4 nest position (world frame, hand root at (0, 0, 0.5)).
OBJ_POS = [-0.128, -0.014, 0.558]
OBJ_QUAT = [1.0, 0.0, 0.0, 0.0]

NUM_ENTRIES = 50_000
JOINT_NOISE = 0.05
POS_NOISE = 0.005

# Must match the training env's grasp_cache_path ("cache/xhand_sphere_grasp_linspace",
# xhand_env_cfg.py) + the "_1.0-1.0-1.npy" suffix appended by sharpa_wave_env.py.
OUTPUT_PATH = "cache/xhand_sphere_grasp_linspace_1.0-1.0-1.npy"


def main():
    rng = np.random.default_rng(42)
    D = len(JOINT_VALUES)
    rows = np.zeros((NUM_ENTRIES, D + 7), dtype=np.float32)

    base_j = np.array(JOINT_VALUES, dtype=np.float32)
    lo = np.array([lh[0] for lh in JOINT_LIMITS], dtype=np.float32)
    hi = np.array([lh[1] for lh in JOINT_LIMITS], dtype=np.float32)
    base_p = np.array(OBJ_POS, dtype=np.float32)
    base_q = np.array(OBJ_QUAT, dtype=np.float32)

    for i in range(NUM_ENTRIES):
        j_noise = rng.uniform(-JOINT_NOISE, JOINT_NOISE, size=D).astype(np.float32)
        j = np.clip(base_j + j_noise, lo, hi)
        p_noise = rng.uniform(-POS_NOISE, POS_NOISE, size=3).astype(np.float32)
        p = base_p + p_noise
        rows[i, :D] = j
        rows[i, D:D + 3] = p
        rows[i, D + 3:D + 7] = base_q

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, rows)
    print(f"Wrote {rows.shape} → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
