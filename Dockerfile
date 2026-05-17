# ── Base ──────────────────────────────────────────────────────────────────────
# cuda 12.1 matches the PyTorch wheels in uv.lock (torch==2.4.1+cu121)
# ubuntu 20.04 ships Python 3.8 as the default python3
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Expose ALL NVIDIA capabilities (compute + graphics) so the container runtime
# mounts the real NVIDIA Vulkan ICD — required for IsaacGym camera sensors.
# Without "graphics" here, vkCreateInstance() would fall back to LLVMpipe and segfault.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        ffmpeg \
        libgl1 \
        libglu1-mesa \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-utils \
        libegl1 \
        libxi-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        gdb \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Driver 591.86 on WSL2 has a known bug where it does not inject the NVIDIA Vulkan
# ICD into containers even with NVIDIA_DRIVER_CAPABILITIES=all. The ICD JSON only
# points the Vulkan loader at libGLX_nvidia.so.0, which the container runtime DOES
# mount via LD_LIBRARY_PATH. Baking the file in sidesteps the injection bug.
# Source: isaacgym/docker/ (IsaacGym ships these for exactly this reason).
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
COPY isaacgym/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY isaacgym/docker/10_nvidia.json  /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Remove Mesa EGL so the loader never accidentally picks it up over the NVIDIA one.
# libegl1 (runtime) is still present; only the Mesa backend .so is removed.
RUN rm -f \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 \
    /usr/share/glvnd/egl_vendor.d/50_mesa.json

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Python environment ────────────────────────────────────────────────────────
# Venv lives at /opt/venv — OUTSIDE the /workspace mount point.
# Mounting /workspace at runtime does not touch /opt/venv.
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Dependency installation ───────────────────────────────────────────────────
# WORKDIR must be /workspace so that editable-install .pth files record paths
# like /workspace/src/LEAP_Hand_Sim — matching the runtime mount point exactly.
# When you mount your local checkout at /workspace the .pth files resolve to the
# live source without any reinstall.
WORKDIR /workspace

# Copy only what uv needs to resolve and install all dependencies.
# At runtime these files are hidden by the bind mount, but /opt/venv persists.
COPY pyproject.toml uv.lock .python-version ./
COPY isaacgym/python ./isaacgym/python
COPY src ./src
COPY README.md ./README.md

# Install everything: PyTorch, rl-games, hydra, leapsim, dexmanip, isaacgym.
# Editable packages (leapsim, dexmanip, isaacgym) get .pth files pointing to
# /workspace/{src/..., isaacgym/python} — resolved from the mount at runtime.
RUN uv sync

# ── Runtime defaults ──────────────────────────────────────────────────────────
CMD ["/bin/bash"]
