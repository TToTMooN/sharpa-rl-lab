# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Helper for parameterizing the hand-root ("palm") orientation by Euler angles.
#
# Why this exists
# ---------------
# Early xhand experiments fixed the palm "flat" — facing straight up (+z), a
# pure -90 deg rotation about the world y-axis. A flat upward palm is a poor
# cradle: the object only rests on a ring of fingertips and rolls off through
# the thumb-opposite gap as soon as gravity turns on. Humans never hold a ball
# with a flat palm; they TILT the hand so the object settles into the corner
# between the palm and the curled fingers, and gravity presses it INTO that
# wall instead of letting it roll out.
#
# `palm_quat` lets a config express the palm orientation as (roll, pitch, yaw)
# degrees so the tilt is a single sweepable knob (used by the VLM pose-iteration
# loop in visual_check.py). The convention matches Isaac Lab's
# `isaaclab.utils.math.quat_from_euler_xyz` exactly, so values transfer 1:1.
#
#   palm_quat(0, -90, 0) == (0.7071068, 0.0, -0.7071068, 0.0)   # legacy flat palm-up
#
# Returns (w, x, y, z) to match ArticulationCfg.InitialStateCfg.rot.

import math


def palm_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) from XYZ Euler angles in degrees.

    Matches isaaclab.utils.math.quat_from_euler_xyz so the same numbers can be
    fed to either path. roll = rotation about world x, pitch about y, yaw about z.
    """
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return (qw, qx, qy, qz)
