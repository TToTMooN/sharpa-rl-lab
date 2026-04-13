# Copyright (c) 2022-2025. SPDX-License-Identifier: BSD-3-Clause
#
# Convert the xhand (RoboEra) URDF to a single USD file for Isaac Lab.
# Usage:
#   pixi run python rl_isaaclab/scripts/convert_xhand_urdf.py
#
# Produces: assets/xhand/xhand_right.usd

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert xhand URDF to USD.")
parser.add_argument(
    "--urdf",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../assets/xhand/XHAND1_URDF_v1.3/xhand1_right/urdf/xhand_right.urdf",
    ),
    help="Path to xhand URDF",
)
parser.add_argument(
    "--out_dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/xhand"),
    help="Output directory for USD",
)
parser.add_argument("--out_name", default="xhand_right.usd", help="Output USD filename")
parser.add_argument("--merge_fixed", action="store_true", default=True,
                    help="Merge fixed joints (recommended for the xhand passive linkages)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Force headless
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    urdf_path = os.path.abspath(args_cli.urdf)
    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Converting: {urdf_path}")
    print(f"[INFO] Output:     {out_dir}/{args_cli.out_name}")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=out_dir,
        usd_file_name=args_cli.out_name,
        fix_base=True,
        merge_fixed_joints=args_cli.merge_fixed,
        convert_mimic_joints_to_normal_joints=False,
        force_usd_conversion=True,
        make_instanceable=False,  # single hand, no need to instance
        collider_type="convex_decomposition",  # finger links have complex shapes
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,  # placeholder; overridden at runtime by actuator cfg
                damping=1.0,
            ),
        ),
    )

    converter = UrdfConverter(cfg)
    print(f"[INFO] USD written to: {converter.usd_path}")
    print(f"[INFO] Num joints converted: {getattr(converter, 'num_joints', 'N/A')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
