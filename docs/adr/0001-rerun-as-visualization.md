# ADR-0001: Rerun.io as the visualization backend

**Status:** Accepted  
**Date:** 2026-05-17

---

## Context

Training runs for dexterous manipulation (IsaacGym + PPO) are long and headless. The only way to understand whether a policy is improving spatially — whether the hand is grasping, how the cube moves, whether joints are tracking targets — is to watch the sim render. On this development setup (WSL2 on Windows), that is not possible:

- IsaacGym's built-in camera sensors require Vulkan. WSL2 has no NVIDIA Vulkan ICD (`libGLX_nvidia.so.0` does not exist on the Linux side; Windows WDDM owns the GPU for display). All Vulkan paths available in the container (LLVMpipe, Mesa) crash under IsaacGym's renderer (SIGSEGV in `libvulkan_lvp.so`).
- Enabling `record_video: True` in the task config triggers `enableCameraSensors=True`, which forces `graphics_device_id=0`, which initialises Vulkan — and crashes, even in headless training.

Alternatives considered:

| Option | Why rejected |
|---|---|
| Fix Vulkan on WSL2 | `libGLX_nvidia.so.0` is part of the Linux NVIDIA display driver, absent on WSL2 by design. No install path exists. |
| D3D12 Mesa (`dzn`) Vulkan ICD | `/dev/dxg` and `libd3d12.so` exist on this system, but no ICD JSON is installed. Dead end on this kernel+driver combination. |
| Run training in a Linux VM (VirtualBox) | VirtualBox cannot pass through consumer NVIDIA GPUs. VMSVGA is software only. GPU training not possible. |
| TensorBoard only | Gives scalars; gives no spatial/geometric view. Cannot answer "is the cube being grasped?" |
| Matplotlib/OpenCV periodic frames | Requires either a display or offscreen Mesa (which crashes). Same root problem. |
| Rerun.io | SDK runs in-process (Python, in Docker/WSL2), writes `.rrd` files to the mounted workspace filesystem. Viewer runs natively on Windows with the host GPU. Bypasses WSL2 Vulkan entirely. |

## Decision

Use **Rerun.io** as the periodic visualization backend for training runs.

**What Rerun is:** a structured logger for spatial-temporal scenes. The SDK logs typed primitives (`Transform3D`, `Boxes3D`, `Scalar`, `TextLog`) tagged with a timeline sequence value and serialises them to a `.rrd` file. The viewer reconstructs and renders the scene. The SDK and viewer are fully decoupled: SDK runs in the training process, viewer runs anywhere with filesystem access to the output file.

**What Rerun is not:** it does not render in-process, does not require Vulkan, and does not receive a pixel buffer from IsaacGym. It reads data we explicitly log from tensors that IsaacGym already maintains.

**Scope of this decision (v1):**

- Spatial scene: per-link world-frame transforms for all 17 rigid bodies of the hand (flat entity layout, URDF link names verbatim), plus the cube object pose.
- Control overlays: commanded vs. achieved joint angles per DOF (16+16 scalars); object linear and angular velocity magnitudes.
- Episode events: `TextLog` entries when env 0 resets, to make the timeline scrubbable.
- Scalars from the training algorithm (rewards, value loss, KL, etc.) are explicitly **not** in scope for v1. They remain in TensorBoard via rl_games' existing `SummaryWriter` path. A v2 that routes all logging through Rerun (replacing TensorBoard) is feasible — Rerun's `Scalar` primitive supports it — but requires subclassing `CommonAgent.write_stats()`.

**Recording regime:** periodic windowed snapshots, not continuous. Every `record_every_n_steps` control steps, env 0 is recorded for `window_length_steps` steps, producing one `.rrd` file per window under `<experiment_dir>/rerun/window_<N>_step_<M>.rrd`. This bounds file size and supports cross-checkpoint comparison (open two windows side-by-side in the viewer).

**Quaternion convention note:** IsaacGym `rigid_body_states[..., 3:7]` is `(x, y, z, w)`. Rerun `rr.Quaternion(xyzw=...)` is also `(x, y, z, w)`. No reshuffle needed. This must not be changed without verifying both sides.

## Consequences

**Positive:**
- Unblocks any spatial debugging on WSL2+Docker.
- Zero changes to IsaacGym, rl_games, or the PhysX pipeline.
- `rerun-sdk` is a pure Python package; no system-level dependencies.
- `.rrd` files are append-only and viewer-tolerant of partial/truncated files.

**Negative:**
- Requires installing the Rerun Viewer on Windows separately (not managed by `uv`/`pyproject.toml`).
- The hand appears as small box markers per link, not mesh geometry. True mesh rendering (Option C) requires loading per-link `.obj`/`.stl` files from the URDF and is deferred.
- Two logging backends (Rerun + TensorBoard) until v2 unifies them.
