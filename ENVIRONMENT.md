# Environment

The reported runs used NVIDIA H100 GPUs and CUDA 12.4. The Scale-RAE,
RAE-SigLIP2-B proxy, and PixArt-alpha environment used Python 3.10 and PyTorch
2.5.1. FLUX.2 used a separate overlay with PyTorch 2.6.0+cu124, torchvision
0.21.0+cu124, Transformers 5.12.1, Accelerate 1.14.0, bitsandbytes 0.49.2,
and Diffusers 0.39.0.dev0 at commit
`bd2c91958881b777260eedb1c3d61d01c03e800f`. Scale-RAE checkpoints and their
diffusion heads remain FP32; the RAE proxy and PixArt-alpha use BF16 autocast;
FLUX.2 uses BF16 compute with 4-bit weights and model CPU offload. TF32 is
enabled for CUDA/cuDNN matrix multiplication, with high float32 matmul
precision.

Create the small, backend-neutral environment first:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,media]'
```

PyTorch wheels are platform-specific. If the preceding command does not select
the CUDA 12.4 wheel on your system, install the official CUDA 12.4 build first:
PyTorch 2.5.1/torchvision 0.20.1 for Scale-RAE, the RAE proxy, and PixArt-alpha,
or PyTorch 2.6.0/torchvision 0.21.0 for FLUX.2. Then run `pip install -e .`.

Backend dependencies are intentionally separated:

- Scale-RAE: install the CUDA 12.4 PyTorch 2.5.1 wheels, then run
  `python -m pip install -r requirements/scale_rae.txt`. Install the public
  Scale-RAE repository code and this package into that environment.
- RAEv2 proxy: check out the public `diffusion-bench/diffusion-bench`
  repository at revision `a4a0f5c22e70f7f253f47a5250b478ae5084acd7`, run its
  frozen `uv` environment installation, and install this package into it. The
  same revision and exact checkpoint-relative path are recorded in
  `configs/raev2.json`.
- PixArt-alpha: `python -m pip install -r requirements/pixart.txt`.
- FLUX.2: first install the CUDA 12.4 PyTorch 2.6.0 wheels, then run
  `python -m pip install -r requirements/flux2.txt`. The requirements file pins
  the source-installed Diffusers revision recorded by the reported run.

The archive contains no checkpoints. Pass a Hugging Face repository identifier
or a local directory through your backend loader. Keep credentials in
environment variables such as `HF_TOKEN`; never write them into a config or
log committed with results.

Quick CPU-only verification (does not load a generator):

```bash
python scripts/validate_release.py
python -m unittest discover tests
python scripts/resolve_config.py --help
python scripts/prepare_benchmark.py --help
python scripts/export_benchmark.py --help
python scripts/aggregate_metrics.py --help
```
