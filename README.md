<p align="center">
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (HuggingFace)</a></h3>
</p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.

---


https://github.com/user-attachments/assets/7548c2db-fe03-4ba8-8dd3-52d2c6160739


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### GUI demo
```
python demo.py
```
Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

### CLI Inference
Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder and our [huggingface repo](https://huggingface.co/gdhe17/Self-Forcing/tree/main/checkpoints).

## Training
### Download text prompts and ODE initialized checkpoint
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
The DMD configuration in this reproduction uses `prompts/vidprom_filtered_extended.txt`, which is tracked with Git LFS in this private repository. If you clone the repository without Git LFS, install Git LFS and run `git lfs pull`; alternatively, download the prompt from Hugging Face with the command above. A smaller prompt file such as `prompts/MovieGenVideoBench.txt` can be used for a smoke test. Our training algorithm (except for the GAN version) is data-free (**no video data is needed**). For now, we directly provide the ODE initialization checkpoint. ODE initialization follows the process described in the [CausVid](https://github.com/tianweiy/CausVid) repo; `scripts/generate_ode_pairs.py` contains the local helper used to generate ODE pairs.

### Self Forcing Training with DMD
The reference 64-GPU command is:
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```

This reproduction adds explicit controls for shorter and resumable runs:
```
python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --max_steps 1 \
  --grad_accum_steps 2 \
  --disable-wandb
```
Use `--max_steps 600` for the full training schedule, increase `--grad_accum_steps` to fit a larger effective batch on fewer GPUs, and pass `--resume <checkpoint_directory>` to continue from a saved `model.pt`. The model-loading changes support local indexed safetensors shards and separate load dtypes for the 1.3B generator and 14B score model; when FlashAttention is unavailable, attention falls back to PyTorch scaled dot-product attention.

The reference training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. The four-GPU scripts under `scale_up_outputs/nm5_self_forcing_dmd/slurm_scripts/` document the NM5 dryrun/fullrun setup, but contain site-specific paths and scheduler settings and must be adapted to the target cluster before use. Keep credentials, model weights, checkpoints, logs, and machine-local configuration outside Git.

## Reproduction Layout
- `train.py`: distributed DMD/SiD/CausVid training entry point.
- `inference.py`: text-to-video and image-to-video batch inference; outputs MP4 files at 16 FPS.
- `demo.py`: optional GUI demo.
- `configs/`: default, DMD, and SiD experiment configurations.
- `model/` and `pipeline/`: model construction and causal rollout/KV-cache logic.
- `trainer/`: training loops, checkpointing, gradient accumulation, and resume support.
- `scripts/`: ODE initialization and LMDB preparation helpers.
- `scale_up_outputs/`: local NM5 launcher templates and runtime notes; do not copy private absolute paths unchanged to another machine.

## Reproduction Notes
1. Confirm the CUDA/PyTorch/FlashAttention combination on the target machine before launching distributed training.
2. Download model directories and checkpoints into the paths referenced by the selected config; model weights are not tracked in Git.
3. Run a one-step multi-GPU smoke test with the same `train.py` code path before starting the 600-step run.
4. Record the exact config, checkpoint step, GPU count, seed, and output directory for each run. W&B can be disabled for offline testing with `--disable-wandb`.

## VBench Evaluation (step 600)

当前仓库附带的 `vbench_eval_step600_results.json` 是对第 600 步 checkpoint 生成视频的 VBench 结果汇总；完整的逐视频结果保存在 `vbench_eval_step600_full_info.json`。本次结果使用 1 个样本/提示词，视频为 81 帧、16 FPS。不同指标由 VBench 的不同 prompt 子集计算，因此每个指标覆盖的视频数量不同，不能把所有指标简单视为同一个样本集上的平均分。

| Dimension | Score |
| --- | ---: |
| Subject consistency | 0.9251 |
| Background consistency | 0.9246 |
| Temporal flickering | 0.9850 |
| Motion smoothness | 0.9854 |
| Dynamic degree | 0.2917 |
| Aesthetic quality | 0.6399 |
| Human action | 0.7900 |
| Scene | 0.2892 |
| Temporal style | 0.2322 |
| Appearance style | 0.2141 |
| Overall consistency | 0.2459 |

### Why are the scores uneven?

这种分布是当前评测设置和模型目标共同造成的，并不表示所有维度都同样好：

1. **训练目标偏向时序稳定性。** Self-Forcing/DMD 主要解决自回归 rollout 的 train-test gap。因而主体一致性、背景一致性、帧间闪烁和运动平滑度较高是符合预期的。
2. **Dynamic degree 不是质量分数。** 它衡量视频中动作/变化的程度。许多 prompt 本身是静态场景或低运动场景，较低分不能直接解释为生成失败；它也与 temporal flickering、motion smoothness 的定义不同。
3. **Style 与 scene 分数依赖评测器和 prompt 分布。** `temporal_style`、`appearance_style`、`scene` 和 `overall_consistency` 使用专门的视觉/语义评测模型，容易受到 prompt 类型、文本措辞、画面构图以及模型域差异影响，不能由前四项时序指标推断。
4. **这是单 checkpoint、单样本结果。** 结果来自 step 600，且每个 prompt 只有一个视频，没有多 seed 方差或置信区间。因此它适合作为当前复现状态的记录，不应被当作完整的统计结论。
5. **评测覆盖范围按维度变化。** VBench 会为每个 dimension 选择对应的 prompt 集合；例如本次逐视频文件共有 616 条记录，但单项指标覆盖约 72--100 个视频。比较其他运行时必须保持同一 prompt 文件、视频规格、采样设置和 VBench 版本。

因此，当前结果更准确的结论是：**step 600 模型的时序稳定性和主体/背景保持能力较好，但动态幅度、风格和场景语义维度仍偏弱，且这些分数需要在统一设置下用多次采样进一步验证。** 生成和评测流程见本节说明以及 `scripts/prepare_vbench_videos.py`。

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
