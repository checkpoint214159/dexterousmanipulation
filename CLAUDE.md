# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment constraints

- **Python 3.8 only** — IsaacGym Preview 4 ships pre-compiled `.so` files only for Python 3.6–3.8. There is no source to recompile. The `.python-version` file pins 3.8.
- **numpy 1.21–1.23** — lower bound: `npt.NDArray` requires ≥1.21; upper bound: `np.float` removed in 1.24 and rl-games 1.5.2 still uses it.
- **WSL2 headless only** — `headless=false` is impossible. WSL2 has no `/dev/nvidia0`, so the NVIDIA Vulkan ICD fails. LLVMpipe has no CUDA-Vulkan interop extensions. Always use `headless=true`.
- **LD_LIBRARY_PATH** — must have `/usr/lib/wsl/lib` first. IsaacGym's `gymdeps.py` calls `ctypes.CDLL("libcuda.so", RTLD_GLOBAL)` at import time; if the distro's bare `libcuda.so` is loaded instead of the WSL2 stub, it poisons CUDA driver symbols process-wide and PyTorch sees zero GPUs. This is set in `.bashrc` and `.env`.

## Setup (after cloning)

```bash
git clone --recurse-submodules https://github.com/checkpoint214159/dexterousmanipulation.git
# Download IsaacGym Preview 4 from developer.nvidia.com/isaac-gym, extract to ./isaacgym/
uv sync
uv pip install -e ./isaacgym/python
```

## Running training

Training must be launched from `src/LEAP_Hand_Sim/leapsim/` (Hydra resolves config paths relative to cwd):

```bash
cd src/LEAP_Hand_Sim/leapsim
uv run python train.py headless=true
```

Key CLI overrides (passed as `key=value` after `train.py`):
- `task=LeapHandRot` or `task=LeapHandGrasp` — selects task + matching PPO config
- `num_envs=N` — override environment count
- `test=true checkpoint=runs/<name>/nn/<name>.pth` — run inference on a saved policy
- `wandb_activate=false` — disable W&B (default config has it on)
- `capture_video=false` — disable video capture

Checkpoints and per-run configs are saved to `runs/<run_name>/` relative to cwd (i.e., inside `leapsim/`).

## Package layout

This is a `uv` workspace with two packages:

| Package | Path | Purpose |
|---|---|---|
| `dexmanip` | `src/dexmanip/` | Our research code; `sim_backend.py` is the planned seam for isolating IsaacGym calls |
| `leapsim` | `src/LEAP_Hand_Sim/` | Git submodule (fork of upstream LEAP Hand Sim); contains training loop, tasks, and rl-games integration |

`isaacgym` is not committed (NVIDIA license). It lives at `./isaacgym/` and is installed as an editable path source via `[tool.uv.sources]` in the root `pyproject.toml`.

## Import order constraint

`import isaacgym` must precede any `import torch` in every entry point. `gymdeps.py` raises `ImportError` if `torch` is already in `sys.modules` when isaacgym is first imported. `train.py` already does this correctly; preserve this ordering in any new scripts.

## leapsim architecture

- **`train.py`** — Hydra entry point. Registers rl-games env (`RLGPU`) and algorithm builders (`amp_continuous`), then calls `runner.run()`.
- **`leapsim/__init__.py`** — `make()` factory used by rl-games to construct the vectorized environment.
- **`tasks/leap_hand_rot.py`** / **`tasks/leap_hand_grasp.py`** — task implementations; inherit from `tasks/base/vec_task.py` which wraps IsaacGym's gym API.
- **`learning/`** — AMP (Adversarial Motion Priors) agent/player/model overrides on top of rl-games.
- **`cfg/config.yaml`** — top-level Hydra config; defaults to `task: LeapHandRot`, `headless: true`, `sim_device: cuda:0`.
- **`cfg/task/`** — per-task environment configs (reward weights, physics params, object assets).
- **`cfg/train/`** — per-task PPO hyperparameters.
