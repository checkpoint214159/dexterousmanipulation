"""
Simulation backend interface.

All IsaacGym calls are funnelled through this module so the backend can later be
swapped for a subprocess/socket implementation when moving to Python 3.11+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_gym():
    """Return the isaacgym gymapi module, raising a clear error if not installed."""
    try:
        from isaacgym import gymapi  # type: ignore[import]
        return gymapi
    except ImportError as e:
        raise RuntimeError(
            "IsaacGym is not installed. Download Preview 4 from "
            "https://developer.nvidia.com/isaac-gym and run: "
            "pip install -e ./isaacgym/python"
        ) from e


def get_gymtorch():
    """Return the isaacgym gymtorch module."""
    try:
        from isaacgym import gymtorch  # type: ignore[import]
        return gymtorch
    except ImportError as e:
        raise RuntimeError(
            "IsaacGym is not installed. See get_gym() for instructions."
        ) from e
