# Outline: "How Robots Are Represented in Software"

**Audience:** Future contributors to this repo. Assumes Python/PyTorch fluency, no robotics background. Goal: confidently modify the LEAP URDF, swap objects, change action/obs space, and debug spawn / limit / contact issues.

**Format conventions:**

- Mermaid blocks for tree/graph structure and data flow
- 2D ASCII side-views for frame diagrams (X right, Z up, Y into page)
- LEAP code excerpts everywhere
- Each section: concept → code → pitfall callout

---

## Part 0 — Background Appendix

A 1-page reset on prerequisites.

- What a rigid body is
- What a frame is: origin point + 3 axes attached to a body
- Translation, rotation, transform; homogeneous-matrix idea
- "Frame X expressed in frame Y" notation; transform composition
- Pointers (no derivations): quaternion math, SO(3), Euler conventions

---

## Part I — Foundations: Frames, Origins, and Rotations

### 1.1 The keystone: "origin" is two different things

- **"Origin of frame X"** (math sense) = the point (0,0,0) of X's own coords. Always zero in X's own coords.
- **The XML tag `<origin xyz=... rpy=.../>`** = where [the containing thing]'s frame sits, measured in coords of the *containing* frame.
- Callout: **Every `<origin>` tag answers 'relative to what frame?'. The answer is always the containing frame in the XML nesting.**

### 1.2 Three orientation representations and their three consumers

| Representation | Numbers | Used by | Why |
|---|---|---|---|
| rpy (roll-pitch-yaw) | 3 (radians) | URDF | Human-readable |
| Quaternion (x,y,z,w) | 4 (unit-norm) | Simulator state tensors | Numerical stability, no gimbal lock |
| Rotation matrix | 9 (orthogonal) | Internal libraries | Composable via matmul |

Subtopics:
- DoFs vs storage numbers: pose is 6-DoF but 7 numbers as pos3+quat4
- Quaternion order footgun: IsaacGym `(x,y,z,w)`; scipy `(w,x,y,z)`
- Gimbal lock in one paragraph + external link

### 1.3 Coordinate conventions

- World frame: Z-up, right-handed
- Units: m, rad, kg, s, N, N·m
- Right-handed in one sentence

---

## Part II — The Kinematic Tree

### 2.1 Links are nodes, joints are edges

- Tree with N links has N−1 joints (if connected)
- Root link has no parent joint
- Mermaid diagram of LEAP hand tree (palm → 4 fingers, 4–5 links each)

### 2.2 Toy A — 2-link pendulum

A complete worked URDF: base-link + revolute joint + swing-arm pendulum. Annotated, with ASCII side-view showing world frame, arm frame, mesh draw position.

Used to nail down:
- Joint `<origin>` = parent-frame → child-frame offset
- Visual `<origin>` = child-frame → mesh offset
- Link frame has no XML tag; implicit at link's origin
- `<axis>` is in child's local frame, after joint origin's rpy is applied

### 2.3 The three (actually four) offsets

1. **Joint `<origin>`** — parent → child link frame (structural)
2. **Visual `<origin>`** — link frame → mesh draw (cosmetic)
3. **Collision/inertial `<origin>`** — link frame → collision shape / COM
4. **(Hidden) Mesh file's own internal origin** — baked into STL; not in URDF

### 2.4 Joint types

| Type | DoFs | Use case | In LEAP? |
|---|---|---|---|
| revolute | 1 | Hinges with limits | ✓ (×16) |
| prismatic | 1 | Sliders | ✗ |
| continuous | 1 | Wheels (no limits) | ✗ |
| fixed | 0 | Welds | ✓ |
| floating | 6 | *Theoretical; most sims reject* | ✗ |
| planar | 3 | Rare | ✗ |

Sidebar: **"How does the cube fly around with no joints?"** Floating-base story: simulators add a free 6-DoF pose to every root link, separate from URDF joints. `fix_base_link` (Python flag, not URDF tag) controls this per-actor.

### 2.5 `<axis>` is axis of rotation, NOT direction of motion

- Revolute joint: child rotates *around* axis
- `<axis>` is in the **child's local frame**, after `<origin rpy>` has rotated it
- Why Onshape URDFs look numerically chaotic: link frames rotated to align local axes with mechanical features

### 2.6 DoF accounting

- `total_dofs = sum(joint.dof_count)`
- LEAP: 16 revolute × 1 + N fixed × 0 = 16
- Configuration vector `q` is 16-dim, radians
- Control mode ≠ DoF count

### 2.7 Toy B — 3-link planar arm and Forward Kinematics

- Compose `T_world_elbow = T_world_shoulder ⊗ T_shoulder_elbow ⊗ R(axis_2, θ_2)`
- Concrete walkthrough θ₁=90°, θ₂=0° → elbow at (0, 1, 0)
- Point: every joint angle along the chain moves every downstream frame

### 2.8 FK in practice: you don't compute it, the sim does

- `rigid_body_state` IS forward kinematics, pre-computed
- Show `tasks/leap_hand_rot.py:87-93`: acquire and reshape
- When you compute FK yourself: real robot, debugging, retargeting
- Brief Inverse Kinematics mention (not used in this codebase)

---

## Part III — Geometry and Inertia

### 3.1 Three geometries per link

```
<link>
  <visual>     ← rendering (high-poly OK)
  <collision>  ← contact detection (low-poly preferred)
  <inertial>   ← dynamics integration
```

### 3.2 Visual ≠ collision: tradeoffs

- Visual: hi-poly fine; cosmetics
- Collision: high poly = expensive + unstable contacts. Norm = primitives, convex hulls, V-HACD.
- LEAP reuses STL for both — accurate but expensive
- Failure modes: "cube falls through palm" (too coarse); "contact jitter" (too sharp + stiff PD)

### 3.3 Inertial properties

- Mass: scalar (kg). Zero invalid.
- COM: `<inertial><origin>` in link frame
- Inertia tensor: 3×3 symmetric (6 unique)
  - Diagonal = moment of inertia per axis
  - Off-diagonal = products of inertia (nonzero iff body's principal axes ≠ link's coord axes)
  - Equal diagonals + zero off-diag = symmetric body
- Sanity formula: uniform solid cube `I_diag = m·s²/6`
- Worked: LEAP cube says `1e-4`, analytical says `4.7e-5` → 2× too high
- Why URDFs ship approximate inertials

### 3.4 Collision filtering

- Sims auto-disable collision between linked bodies
- Per-actor bitmask
- Failure: over-aggressive = self-intersection; under = ghost contacts

---

## Part IV — The Simulator Side (IsaacGym)

### 4.1 URDF → actor: loading

- `gym.load_asset(sim, asset_root, urdf, asset_options)`
- `asset_options.fix_base_link` (Python, not URDF). LEAP `env_setup.py:27`
- Per-DoF properties (`stiffness`, `damping`, `effort`, `velocity`) set after load

### 4.2 Actors, envs, replication

- Env = isolated simulation instance (8192 of them)
- Each env contains N actors (LEAP: 2 — hand + cube)
- Tensor-level state sharing across envs

### 4.3 The three-tensor view of state

| Tensor | Shape | One row = | LEAP example |
|---|---|---|---|
| `actor_root_state` | `(num_envs × num_actors_per_env, 13)` | Actor's root pose+vel | `(16384, 13)` |
| `dof_state` | `(num_envs × num_dofs_per_env, 2)` | DoF's (q, q̇) | `(131072, 2)` |
| `rigid_body_state` | `(num_envs, num_bodies_per_env, 13)` | Link's world pose+vel | `(8192, ~21, 13)` |

- Worked indexing: "get the cube's position in env 42" → `actor_root_state[42*2 + 1, 0:3]`
- Critical callout: **floating object motion lives in `actor_root_state`, NOT `dof_state`**

### 4.4 IsaacGym tensor API

- `acquire_*_state_tensor` (once at init)
- `gymtorch.wrap_tensor` (handle → PyTorch tensor)
- `refresh_*_state_tensor` (pull latest into your view)
- `set_*_state_tensor` / `set_*_state_tensor_indexed`

### 4.5 Reset patterns

- Episode reset writes new root + dof state for the env_ids being reset
- `_indexed` for subset writes (not Python loops)
- Walkthrough using LEAP reset code

---

## Part V — Actuation and Control

### 5.1 The pipeline

```
policy net → action [B, 16] in ~[-1, 1]
              ↓ targets = prev_targets + (1/24) * action
              ↓ clamp to joint limits
        cur_targets [B, 16] in radians
              ↓ set_dof_position_target_tensor
        IsaacGym per-DoF PD: τ = pgain·(q_tgt − q) + dgain·(0 − q̇)
              ↓ torques
        PhysX integrates rigid-body dynamics
              ↓ next q, q̇
        refresh state tensors → next observation
```

### 5.2 Control modes

| Mode | What you supply | When |
|---|---|---|
| `DOF_MODE_POS` | Target position | Most manipulation (LEAP) |
| `DOF_MODE_VEL` | Target velocity | Some legged locomotion |
| `DOF_MODE_EFFORT` | Torque | Advanced control |
| `DOF_MODE_NONE` | (passive) | Free-spinning joints / objects |

### 5.3 PD controllers in 4 lines

- pgain (P/stiffness): pushes hard when far from target
- dgain (D/damping): resists fast motion
- High pgain → fast tracking but contact instability
- Low pgain → drift, can't grip
- No derivation; link to controls reference

### 5.4 The delta-action pattern

- LEAP: `targets = prev_targets + (1/24) × action` (`leap_hand_rot.py:942`)
- Action is dimensionless, not radians; 1/24 is empirical
- Sim 120 Hz, control 20 Hz (6 substeps per control)
- Tradeoff: deltas = smoother / easier exploration; absolute targets = more authority

### 5.5 Tuning intuition (debugging guide)

- Jitter / buzzing: pgain too high
- Joint drifts under gravity: pgain too low
- Slow tracking: action_scale too small
- Wild swings / explosion: action_scale too high
- Joints exceed limits: clamp missing

---

## Part VI — Observations and the Policy Interface

### 6.1 Observations are constructed by hand

- Curated, normalized, possibly-noised, possibly-history-stacked subset of state
- LEAP components (`tasks/leap_hand_rot.py:657-751`):
  - 16 joint positions, normalized to `[-1, 1]` via `unscale`
  - 16 current targets
  - 3 obj_pos + 3 obj_rpy
  - Optional: scales, pd gains, friction, phase
- Default `numObservations: 102` = (16q + 16tgt) × 3 history + 6 obj_pose
- **Only q and targets are history-stacked**; object pose appended once

### 6.2 Design choices

- Normalize to [-1,1]: networks train best on standardized input
- Noise on positions: simulates encoder noise (sim-to-real prep)
- History stacking instead of velocity: encoder velocity noisy at deploy; network picks features; robust to stale state
- Object pose in rpy: smaller dim, more interpretable, acceptable gimbal-lock risk
- Targets in obs: gives policy tracking-error info for free

### 6.3 Privileged information / asymmetric actor-critic

- Measurable with sensors (object pose via mocap/vision): used at deploy
- Truly privileged (friction, scale, PD gains): sim-only
- Asymmetric pattern (`vec_task.py:549`): critic sees privileged, actor doesn't
- Teacher-student alternative

### 6.4 Framing line

**Observation design IS policy design.** What the obs contains determines the function class the policy can express.

---

## Part VII — Conventions and Common Pitfalls

### 7.1 Conventions cheat sheet

- World: Z-up, right-handed
- Units: m, rad, kg, s, N, N·m
- Quaternion order: IsaacGym `(x,y,z,w)`; scipy `(w,x,y,z)`
- rpy in URDF: fixed-axis (X then Y then Z)
- Indexing: env-major in IsaacGym

### 7.2 Top 12 pitfalls

1. Confusing joint `<origin>` with visual `<origin>` — different frames
2. Confusing axis-of-rotation with direction-of-motion
3. Floating object state lives in `actor_root_state`, not `dof_state`
4. Off-by-one indexing of actors per env
5. Setting fix_base_link in URDF (impossible — Python flag)
6. Treating action as a target instead of a delta (LEAP does delta)
7. Quaternion order mismatch
8. Mesh file's hidden internal origin (the 4th offset)
9. Inertia tensor not sanity-checked
10. Collision shape too coarse (cube falls through) or too sharp (jitter)
11. PD gains tuned without considering action scale (coupled)
12. Adding non-observable info to obs without asymmetric / teacher-student wrapper

---

## Part VIII — Appendices

### A. URDF tag quick reference

### B. IsaacGym API cheat sheet

### C. The numbers for LEAP

| Quantity | Value |
|---|---|
| Hand DoFs | 16 |
| Hand bodies | ~21 |
| Cube DoFs | 0 (6-DoF floating root) |
| Actors per env | 2 |
| Sim freq | 120 Hz |
| Control freq | 20 Hz (every 6 substeps) |
| Action scale | 1/24 |
| Default obs dim | 102 |

### D. Further reading

- Quaternions: 3Blue1Brown video; eater.net/quaternions
- Classical control / PD: any controls textbook ch. 4-5
- V-HACD convex decomposition: original paper
- Sim-to-real manipulation: OpenAI dexterous hand paper
- IsaacGym preview docs (`./isaacgym/docs/`)

---

## Writing-order recommendations

1. **Start with Part II + Toy A** (concrete URDF reading), not Part I — bounce back to foundations when needed.
2. **Each section's pitfall callout should reference a real bug** you've hit or could plausibly hit.
3. **Four most load-bearing concepts** (in order):
   1. Frames + keystone origin distinction (I.1)
   2. Three offsets (II.3)
   3. Three-tensor view (IV.3)
   4. Delta-action pipeline (V.4)
