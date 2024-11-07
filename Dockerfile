FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as cuda-base
FROM ghcr.io/astral-sh/uv:latest as uv-base
FROM python:3.11-slim-bullseye

# Copy CUDA runtime libraries and dependencies from cuda-base
COPY --from=cuda-base /usr/local/cuda/lib64 /usr/local/cuda/lib64
COPY --from=cuda-base /usr/local/cuda/include /usr/local/cuda/include
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install uv
COPY --from=uv-base /uv /uvx /bin/

CMD ["tail", "-f", "/dev/null"]
