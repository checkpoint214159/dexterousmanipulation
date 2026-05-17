# Plan: Rerun.io integration

**Status:** Ready to implement  
**ADR:** [0001-rerun-as-visualization](../adr/0001-rerun-as-visualization.md)

---

## Prerequisites

- [ ] Rerun Viewer installed on Windows: `winget install rerun-io.rerun` (or download from rerun.io). One-time. Not managed by this project.
- [ ] Docker image rebuilt after `pyproject.toml` change (step 1 below).

---

## Step 1 — Add `rerun-sdk` to dependencies

**File:** `pyproject.toml`

Add to `[project.dependencies]`:
```toml
"rerun-sdk>=0.22,<0.23"
```

Pin the minor version. Rerun's Python API has breaking changes across minor releases. Check https://github.com/rerun-io/rerun/releases for the latest stable when implementing.

Then rebuild the Docker image:
```bash
docker build -t dexmanip:latest .
```

---

## Step 2 — Add Rerun config block to task YAML

**File:** `src/LEAP_Hand_Sim/leapsim/cfg/task/LeapHandRot.yaml`

Add under `env:`:
```yaml
env:
  rerun:
    enabled: False                # off by default; enable with CLI flag
    window_length_steps: 400      # how many control steps to record per window (400 = 20s @ 20Hz)
    record_every_n_steps: 4000    # open a new window every N steps (~200s sim time)
    env_idx: 0                    # which environment to log
    output_dir: "."               # placeholder; train.py overwrites with experiment_dir/rerun
```

`output_dir: "."` is a struct placeholder — it exists so OmegaConf struct mode allows `train.py` to overwrite it at runtime. It is never used as-is.

---

## Step 3 — Wire `output_dir` in `train.py`

**File:** `src/LEAP_Hand_Sim/leapsim/train.py`

After the `experiment_dir` computation (existing lines):
```python
runs_dir = Path(to_absolute_path(cfg.runs_dir))
experiment_dir = runs_dir / run_name
```

Add:
```python
if cfg.task.env.rerun.enabled:
    cfg.task.env.rerun.output_dir = str(experiment_dir / "rerun")
```

This is the only Rerun-aware line in `train.py`. All other logic lives in the env class.

---

## Step 4 — Implement Rerun in `LeapHandRot`

**File:** `src/LEAP_Hand_Sim/leapsim/tasks/leap_hand_rot.py`

### 4a — Module-level import

At the top, alongside other imports. Import is unconditional (hard dependency):
```python
import rerun as rr
```

### 4b — Call `_rerun_setup()` in `__init__`

After `super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)` and after the state tensors are initialised (i.e., after the `gymtorch.wrap_tensor` block, ~line 90):
```python
self._rerun_setup()
```

### 4c — `_rerun_setup` method

```python
def _rerun_setup(self):
    rr_cfg = self.cfg["env"].get("rerun", {})
    self._rr_enabled = rr_cfg.get("enabled", False)
    if not self._rr_enabled:
        return

    self._rr_window_length = rr_cfg["window_length_steps"]
    self._rr_period        = rr_cfg["record_every_n_steps"]
    self._rr_env_idx       = rr_cfg["env_idx"]
    self._rr_output_dir    = Path(rr_cfg["output_dir"])
    self._rr_output_dir.mkdir(parents=True, exist_ok=True)

    self._rr_global_step  = 0
    self._rr_window_step  = 0
    self._rr_window_count = 0
    self._rr_in_window    = False

    # Fetch link names in rigid_body_states order — used as entity path suffixes.
    hand_handle = self.gym.find_actor_handle(self.envs[0], 'hand')
    self._rr_link_names = self.gym.get_actor_rigid_body_names(self.envs[0], hand_handle)
    # self._rr_link_names[i]  ↔  self.rigid_body_states[env, i, :]
    # Ordering: hand bodies first (indices 0..num_leap_hand_bodies-1), then object body.
```

### 4d — `_rerun_open_window` method

Opens a new `.rrd` file and logs all static geometry (once per window).

```python
def _rerun_open_window(self):
    path = (
        self._rr_output_dir
        / f"window_{self._rr_window_count:04d}_step_{self._rr_global_step:08d}.rrd"
    )
    rr.init("leap_hand", recording_id=path.stem)
    rr.save(str(path))
    print(f"[Rerun] recording window {self._rr_window_count} → {path}")

    # Static geometry: small box marker at each hand link (logged once, no time component).
    for name in self._rr_link_names:
        rr.log(
            f"world/hand/{name}",
            rr.Boxes3D(half_sizes=[[0.008, 0.008, 0.008]]),
            static=True,
        )

    # Object: cube. baseObjScale * 0.05m nominal half-size ≈ 0.04m at scale=0.8.
    # Using a fixed value here; actual scale varies per-env due to randomisation.
    rr.log("world/object", rr.Boxes3D(half_sizes=[[0.04, 0.04, 0.04]]), static=True)
```

### 4e — `_log_rerun_frame` method

Called once per control step during a recording window.

```python
def _log_rerun_frame(self):
    rr.set_time_sequence("step", self._rr_window_step)

    # --- Hand link transforms (world-frame) ---
    # rigid_body_states: (num_envs, num_bodies, 13) — [pos_xyz | quat_xyzw | linvel | angvel]
    # Quaternion convention: IsaacGym xyzw == Rerun xyzw. No reshuffle.
    rb = self.rigid_body_states[self._rr_env_idx, :self.num_leap_hand_bodies].cpu()
    for i, name in enumerate(self._rr_link_names):
        pos  = rb[i, 0:3].numpy()
        quat = rb[i, 3:7].numpy()
        rr.log(
            f"world/hand/{name}",
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat),
            ),
        )

    # --- Object transform (world-frame) ---
    obj_pos = self.object_pos[self._rr_env_idx].cpu().numpy()
    obj_rot = self.object_rot[self._rr_env_idx].cpu().numpy()
    rr.log(
        "world/object",
        rr.Transform3D(
            translation=obj_pos,
            rotation=rr.Quaternion(xyzw=obj_rot),
        ),
    )

    # --- A: Commanded vs achieved joint angles ---
    targets = self.cur_targets[self._rr_env_idx, :self.num_leap_hand_dofs].cpu().numpy()
    actual  = self.leap_hand_dof_pos[self._rr_env_idx].cpu().numpy()
    for i in range(self.num_leap_hand_dofs):
        rr.log(f"control/target/joint_{i:02d}", rr.Scalar(float(targets[i])))
        rr.log(f"control/actual/joint_{i:02d}", rr.Scalar(float(actual[i])))

    # --- C: Object velocity magnitudes ---
    rr.log("object/linvel_mag",  rr.Scalar(float(self.object_linvel[self._rr_env_idx].norm())))
    rr.log("object/angvel_mag",  rr.Scalar(float(self.object_angvel[self._rr_env_idx].norm())))
```

### 4f — `_rerun_tick` method

Drives the window state machine. Called at the end of `post_physics_step`.

```python
def _rerun_tick(self):
    if not self._rr_enabled:
        return

    self._rr_global_step += 1

    # Open a new window on the period boundary.
    if not self._rr_in_window and self._rr_global_step % self._rr_period == 0:
        self._rerun_open_window()
        self._rr_in_window   = True
        self._rr_window_step = 0

    # Log a frame if inside a window.
    if self._rr_in_window:
        # E: Reset event (log before frame so it lands at the right step).
        if self.reset_buf[self._rr_env_idx]:
            rr.log(
                "events",
                rr.TextLog(
                    f"env {self._rr_env_idx} reset — global step {self._rr_global_step}"
                ),
            )
        self._log_rerun_frame()
        self._rr_window_step += 1
        if self._rr_window_step >= self._rr_window_length:
            self._rr_in_window    = False
            self._rr_window_count += 1
```

### 4g — Hook into `post_physics_step`

At the very end of `post_physics_step` in `LeapHandRot`, after all `refresh_*_tensor` calls and after `self.object_pos`, `self.object_rot`, `self.object_linvel`, `self.object_angvel` are populated (they are assigned at ~line 944–949):

```python
self._rerun_tick()
```

---

## Step 5 — CLI usage

Enable Rerun from the command line without touching any YAML:
```bash
# Training run — periodic snapshots
uv run python train.py task=LeapHandRot task.env.rerun.enabled=true

# Eval/replay — record every step (set period = 1, window = large)
uv run python train.py task=LeapHandRot test=true checkpoint=... \
    task.env.rerun.enabled=true \
    task.env.rerun.record_every_n_steps=1 \
    task.env.rerun.window_length_steps=99999
```

Output files land at: `runs/<run_name>/rerun/window_0000_step_00004000.rrd`

On Windows, open with: `rerun <path>` in a terminal, or drag-and-drop onto the Viewer.  
WSL2 path prefix on Windows: `\\wsl$\Ubuntu\root\odyssey\dexterousmanipulation\runs\...`

---

## Entity-path reference

| Entity path | Type | Frequency | Source tensor |
|---|---|---|---|
| `world/hand/<link_name>` | `Transform3D` | per frame | `rigid_body_states[env_idx, i, 0:7]` |
| `world/hand/<link_name>` | `Boxes3D` | static | hardcoded 1.6cm half-size |
| `world/object` | `Transform3D` | per frame | `object_pos`, `object_rot` |
| `world/object` | `Boxes3D` | static | hardcoded 4cm half-size |
| `control/target/joint_<i>` | `Scalar` | per frame | `cur_targets[env_idx, i]` |
| `control/actual/joint_<i>` | `Scalar` | per frame | `leap_hand_dof_pos[env_idx, i]` |
| `object/linvel_mag` | `Scalar` | per frame | `object_linvel[env_idx].norm()` |
| `object/angvel_mag` | `Scalar` | per frame | `object_angvel[env_idx].norm()` |
| `events` | `TextLog` | on reset | `reset_buf[env_idx]` |

Link names (17 total, URDF order):
`palm_lower`, `mcp_joint`, `pip`, `dip`, `fingertip`, `mcp_joint_2`, `pip_2`, `dip_2`, `fingertip_2`, `mcp_joint_3`, `pip_3`, `dip_3`, `fingertip_3`, `pip_4`, `thumb_pip`, `thumb_dip`, `thumb_fingertip`

---

## Known limitations / v2 work items

- **Mesh geometry (Option C):** Replace per-link `Boxes3D` markers with actual link meshes from `assets/leap_hand/`. Requires parsing URDF for mesh file paths and logging `rr.Mesh3D` once per link at init. The `Transform3D` plumbing is unchanged.
- **True object scale:** The object `Boxes3D` uses a fixed half-size. Actual scale is randomised per-env (`randomizeScaleList: [0.95, 0.9, 1.0, 1.05, 1.1]`). To fix: read `self.object_scale[env_idx]` and pass scaled half-sizes at window open.
- **TensorBoard → Rerun migration (v2):** Subclass `CommonAgent.write_stats()` to route `writer.add_scalar(...)` calls to `rr.log(..., rr.Scalar(...))`. This would eliminate the TensorBoard dependency entirely.
- **Multi-env side-by-side:** Log `N` envs under separate roots (`world/env0/hand/...`, `world/env1/hand/...`). 5-line change; deferred until single-env view is proven useful.
