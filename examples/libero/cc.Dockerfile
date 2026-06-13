# Change base image to Ubuntu 24.04 to fix GLIBC_2.38 issue
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1 \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Fix PYTHONPATH
ENV PYTHONPATH=/app/src:/app:/app/packages/openpi-client/src:/app/third_party/libero

# Fix Numba cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Fix MuJoCo/EGL
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_EGL_DEVICE_ID=0

COPY ./examples/libero/requirements.txt /tmp/requirements.txt
COPY ./third_party/libero/requirements.txt /tmp/requirements-libero.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py $CLIENT_ARGS"]