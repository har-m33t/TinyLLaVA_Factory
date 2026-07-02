# INSTALL-ENV-CLAUDE.md

End-to-end environment setup for TinyLLaVA_Factory on a SLURM/HPC login node,
using [uv](https://github.com/astral-sh/uv) instead of conda. Captured by
Claude Code during the first-time setup.

Host at time of setup:
- OS: Rocky/RHEL 9 (Linux 5.14, OpenHPC modules present)
- GPU driver: NVIDIA 580.x (driver-reported CUDA 13.0 compat)
- Shell: bash
- No prior `~/.ssh/`, no prior Python venv

---

## 1. Install `uv`

`uv` was not on `PATH`. Installed to `~/.local/bin` via the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc for future sessions
uv --version                            # confirm
```

## 2. Create the project virtualenv

From inside the repo, create a Python 3.10 venv. `uv` will download a
managed CPython 3.10 if the system doesn't already have one — no conda,
no system Python needed.

```bash
cd /home/u16/harmeets/TinyLLaVA_Factory
uv venv --python 3.10 .venv
source .venv/bin/activate
```

The venv lives at `.venv/` inside the repo and is ignored by `uv`'s auto-generated
`.venv/.gitignore`, so it will not be committed.

## 3. Install TinyLLaVA + dependencies

Editable install straight from `pyproject.toml`:

```bash
uv pip install -e .
```

This resolves and installs the pins from `pyproject.toml`, including:

- `torch==2.0.1`, `torchvision==0.15.2` (CUDA 11.7 wheels)
- `transformers==4.40.1`, `tokenizers==0.19.0`, `accelerate==0.27.2`
- `bitsandbytes==0.41.0`, `peft==0.10.0`, `deepspeed==0.14.0`
- `timm==0.6.13`, `einops`, `sentencepiece==0.1.99`
- `gradio==3.35.2`, `fastapi`, `uvicorn`, `wandb`

### SLURM notes

- The pinned `torch==2.0.1` ships with CUDA 11.7 runtime libs bundled in the
  wheel, so it runs against the 580.x driver even though the driver advertises
  CUDA 13.0. No `module load cuda` is required just to import torch.
- `bitsandbytes==0.41.0` and `deepspeed==0.14.0` may need a matching
  `nvcc` at first use / JIT-build time. If they fail on a compute node, do
  `module load cuda/11.8` (or the closest 11.x available) before running.
- For heavy installs or CUDA-linked builds prefer an interactive GPU node:
  ```bash
  srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
  ```
- If `$HOME` has a quota, redirect the pip cache before installing:
  ```bash
  export PIP_CACHE_DIR=$SCRATCH/pip-cache
  export UV_CACHE_DIR=$SCRATCH/uv-cache
  ```

## 4. GitHub SSH key

No existing `~/.ssh/`. Generated an ed25519 key with no passphrase for
non-interactive use on the cluster:

```bash
ssh-keygen -t ed25519 -C "harmeets130922@gmail.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub          # paste into GitHub → Settings → SSH keys
ssh -T git@github.com              # expect: "Hi har-m33t! You've successfully authenticated"
```

## 5. Point `origin` at the fork and push

The repo was cloned from the upstream `TinyLLaVA/TinyLLaVA_Factory` (HTTPS).
Repointed `origin` to the personal fork over SSH so pushes use the ed25519
key above:

```bash
git remote set-url origin git@github.com:har-m33t/TinyLLaVA_Factory.git
git remote -v
git push origin main
```

---

## Reproducing from scratch on a new node

```bash
# 1. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. clone via SSH (key must already be on GitHub)
git clone git@github.com:har-m33t/TinyLLaVA_Factory.git
cd TinyLLaVA_Factory

# 3. venv + install
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .

# 4. sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Activating the env in a SLURM job script

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

cd /home/u16/harmeets/TinyLLaVA_Factory
source .venv/bin/activate
# module load cuda/11.8   # uncomment if bitsandbytes/deepspeed need nvcc
python -m tinyllava.<your_entrypoint> ...
```
