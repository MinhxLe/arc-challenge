FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && apt install -y git

ENV VIRTUAL_ENV=$HOME/.venv
RUN uv python install 3.11
RUN uv venv --python 3.11 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# we install flash-attn and torch bc it takes forever
RUN uv pip install torch==2.4.1 
RUN uv pip install packaging setuptools ninja 
RUN uv pip install flash-attn --no-build-isolation
RUN uv pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

CMD ["tail", "-f", "/dev/null"]
