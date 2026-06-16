# LEAP Hand Joint Map

The LEAP hand has 16 DOFs. The `canonical_pose` vector in
[LeapHandRot.yaml](../src/LEAP_Hand_Sim/leapsim/cfg/task/LeapHandRot.yaml)
indexes them in URDF joint-name order — **not** kinematic-chain order within
each finger. This document is the source of truth for what each index controls.

## Finger layout

The hand models a **right hand without a pinky**, mounted upside-down at init
(`Quat.from_axis_angle(x_axis, π)` in `env_setup.init_object_pose`). Three
fingers extend straight out in a row (index, thumb, middle in our labelling) and
one opposable finger (ring) is mounted at the base of the palm with a different
joint orientation.

This labelling is **empirical** — confirmed by zero-ing the canonical pose and
rotating one joint at a time to observe which physical link moves in rerun. The
URDF `<link>` names ("thumb_pip", "thumb_dip" in joints 12–15) are misleading;
those names correspond to what we call the ring finger, not the thumb.

## DOF table

| `canonical_pose` idx | URDF joint name | Parent → child link | Limits (rad) | Semantic |
|---|---|---|---|---|
| 0  | `0`  | `mcp_joint → pip`        | −1.047 / +1.047 | **Index** MCP side-side (abduction) |
| 1  | `1`  | `palm → mcp_joint`       | −0.314 / +2.230 | **Index** MCP flex (knuckle root) |
| 2  | `2`  | `pip → dip`              | −0.506 / +1.885 | **Index** PIP flex |
| 3  | `3`  | `dip → fingertip`        | −0.366 / +2.042 | **Index** DIP flex |
| 4  | `4`  | `mcp_joint_2 → pip_2`    | −1.047 / +1.047 | **Thumb** MCP side-side |
| 5  | `5`  | `palm → mcp_joint_2`     | −0.314 / +2.230 | **Thumb** MCP flex |
| 6  | `6`  | `pip_2 → dip_2`          | −0.506 / +1.885 | **Thumb** PIP flex |
| 7  | `7`  | `dip_2 → fingertip_2`    | −0.366 / +2.042 | **Thumb** DIP flex |
| 8  | `8`  | `mcp_joint_3 → pip_3`    | −1.047 / +1.047 | **Middle** MCP side-side |
| 9  | `9`  | `palm → mcp_joint_3`     | −0.314 / +2.230 | **Middle** MCP flex |
| 10 | `10` | `pip_3 → dip_3`          | −0.506 / +1.885 | **Middle** PIP flex |
| 11 | `11` | `dip_3 → fingertip_3`    | −0.366 / +2.042 | **Middle** DIP flex |
| 12 | `12` | `palm → pip_4`           | −0.349 / +2.094 | **Ring** base rotation (opposable) |
| 13 | `13` | `pip_4 → thumb_pip`      | −0.470 / +2.443 | **Ring** MCP flex |
| 14 | `14` | `thumb_pip → thumb_dip`  | −1.200 / +1.900 | **Ring** PIP flex |
| 15 | `15` | `thumb_dip → thumb_fingertip` | −1.340 / +1.880 | **Ring** DIP flex |

## Why this order looks strange

For the three "straight" fingers (index, thumb, middle), the four joints per
finger are stored as `[side-side, MCP-flex, PIP, DIP]` — but the side-side joint
is **not** the kinematic root. The MCP-flex joint (URDF joints 1/5/9) is the
palm-mounted root. The side-side joint (0/4/8) connects mcp_joint to pip. This
ordering matches the URDF declaration order, not the natural "root → tip" walk.

The ring finger's first joint (12, `palm → pip_4`) skips the mcp_joint link
entirely — it's mounted directly on the palm with a different `<origin rpy>`
(`(0, π/2, 0)` instead of `(π/2, π/2, 0)`) — making its base rotation
asymmetric (−0.349 to +2.094) and giving it a wider range of motion suited to
opposing the other three fingers.

## Sign-convention notes

All joint axes are `0 0 -1` in their local frame, so positive joint values
follow the right-hand rule with thumb pointing along **−local-z**. However, the
local frames differ per finger (especially the ring), so the **world-frame
direction of "curl toward palm" is finger-dependent**.

Working hypothesis from the cube canonical pose (index MCP at −1.047, thumb MCP
at +1.6139, middle MCP at +0.029):

- For joints 1, 5, 9 (MCP flex of index/thumb/middle): identical URDF rpy means
  identical world axes, and the larger positive limit (+2.23) suggests positive
  is "curl toward palm." The cube pose then has index hyper-extended and thumb
  fully curled — a tripod-style pinch, not a uniform grip.
- For joint 12 (ring base rotation): larger positive limit (+2.094) suggests
  positive is "rotate across the palm to oppose the others."

These signs should be verified empirically before relying on them for new
poses. The reliable probe is:

```bash
'task.env.canonical_pose=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'
```

then set the joint of interest to its max positive value and observe direction
of motion in rerun.
