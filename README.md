# ARC Challenge

## Local Developer Setup. 
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install [just](https://github.com/casey/just).
3. Run the following commands within the project directory to create a virtual env:
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
pre-commmit install
```
4. Get your `.env` from Minh.

# Setting up remote instance.
## Building docker image. (need to be collaborator on dockerhub)
```
docker build -t minhxle/arc-challenge .
docker tag minhxle/arc-challenge minhxle:latest
# to test
docker run --gpus all --runtime=nvidia minhxle/arc-challenge:latest 
docker push minhxle/arc-challenge:latest

```

## Setting up docker container on vastai.ai.
1. Create vastai account.
2. Create a new template with minhxle/arc-challenge.
3. Launch an instance with 3090RTX with the template.

## Enabling ssh.
1. add SSH public key to instance

## Setting up instance
```
# ssh into instance 
mkdir arc && cd arc
uv venv --python 3.11
```

# Working with remote instance
## Syncing code into remote server
1. install [lsyncd](https://github.com/lsyncd/lsyncd).
2. `lsyncd lsyncd.conf.lua`
## (WIP) Running code
## (WIP) Running  notebook


# TODO
- use flash attention 2 (https://huggingface.co/docs/transformers/perf_infer_gpu_one)
