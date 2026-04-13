# xhand (RoboEra) assets

## Contents

```
XHAND1_URDF_v1.3/
  xhand1_right/
    urdf/
      xhand_right.urdf      # URDF with 30 links, 12 revolute + 17 fixed joints  (tracked)
      xhand_right.csv       # SW inertial + joint specs per link                  (tracked)
    meshes/                  # 60 STL collision/visual meshes                     (GITIGNORED)
    config/, launch/         # ROS config (not used by Isaac Lab)
  xhand1_left/
    ...                      # mirrored structure
xhand_right.usd              # converted USD for Isaac Lab (after EXP-017, tracked)
```

Source: `~/Downloads/XHAND1_URDF_ver 1.3.zip` (right), `~/Downloads/URDF_LH1.1.zip` (left — not yet unpacked)

## STL files not in git

`*.STL` is gitignored globally. To get the meshes back on a fresh clone:
```bash
cd assets/xhand
unzip "~/Downloads/XHAND1_URDF_ver 1.3.zip"
mv "XHAND1_URDF_ver 1.3" XHAND1_URDF_v1.3  # remove space in path
rm -rf __MACOSX
```

Once the USD conversion (EXP-017) is done, the `.usd` file has the meshes baked in and you don't need the STL files anymore for Isaac Lab runtime.

## Kinematics (right hand)

**12 active DOFs** (revolute joints):

| Finger | Joints | Joint limits (rad) |
|--------|--------|-------------------|
| Thumb | `thumb_bend_joint` | [0, 1.832] |
| | `thumb_rota_joint1` | [-0.698, 1.57] |
| | `thumb_rota_joint2` | [0, 1.57] |
| Index | `index_bend_joint` | [-0.174, 0.174] (abduction) |
| | `index_joint1` | [0, 1.919] |
| | `index_joint2` | [0, 1.919] |
| Middle | `mid_joint1` | [0, 1.919] |
| | `mid_joint2` | [0, 1.919] |
| Ring | `ring_joint1` | [0, 1.919] |
| | `ring_joint2` | [0, 1.919] |
| Pinky | `pinky_joint1` | [0, 1.919] |
| | `pinky_joint2` | [0, 1.919] |

**17 fixed (passive) joints** couple the underactuated linkages. These are `*_rotaback_*` and `*_joint3` joints — they attach intermediate linkage bodies to the driven joints.

## Fingertip link names

- `right_hand_thumb_rota_tip`
- `right_hand_index_rota_tip`
- `right_hand_mid_tip`
- `right_hand_ring_tip`
- `right_hand_pinky_tip`

## Differences vs SharpaWave

| Aspect | SharpaWave | xhand |
|--------|-----------|-------|
| Active DOF | 22 | **12** |
| Fingers | 5 | 5 |
| Thumb DOF | 3 | 3 |
| Index DOF | 4 (inc. splay?) | 3 (bend + 2) |
| Mid/ring/pinky DOF | 4 each | 2 each |
| Contact sensor links | `*_elastomer` + `*_DP` (10 distinct) | Only fingertips (5) — tactile is on-surface |
| Joint limits | ~[0, π/2] per joint | Similar range |
| Mesh count | ~100+ | ~60 STL files |

## URDF mesh paths

The URDF uses `package://xhand_right/meshes/...` paths. Isaac Lab's `urdf_to_usd` converter usually resolves these relative to the URDF file location, so we point it at `xhand_right/urdf/xhand_right.urdf` and it finds `../meshes/*.STL`.

## No elastomer bodies

The xhand URDF does NOT have separate elastomer/DP bodies like SharpaWave. Tactile sensors are integrated on the fingertips themselves (via the SDK; not represented in URDF). For our RL sim we'll use only the 5 fingertip links as contact sensors — drops from SharpaWave's 10 sensors to 5.

## Observation dim changes

SharpaWave obs_buf_lag_history dims (per frame):
- 22 joint pos + 22 targets + 5 contacts + 15 contact pos = **64 dims/frame × 3 = 192**

xhand:
- 12 joint pos + 12 targets + 5 contacts + 15 contact pos = **44 dims/frame × 3 = 132**

Action space: 22 → **12**

## SDK references (not under version control)

SDK archives in `~/Downloads/`:
- `xhand_control_sdk_py_x86_64_v117.zip` — Python SDK (for M4 deploy.py port)
- `xhand_control_sdk_x86_64_v141.tar.gz` — C++ SDK
- `xhand_control_ros_x86_64_v123.tar.gz`, `xhand_control_ros2_x86_64_v132.tar.gz` — ROS bridges
- `xhand_v1.1.15.0_release_20250625.deb` — driver
