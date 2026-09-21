# agent.md — looped-flow-matching / Temporal Loop 代码定位

记录时间：2026-09-13 04:17 UTC（最后更新：2026-09-21，加入对齐后的 formal 训练状态）

## 当前接管交接（优先于下方历史记录）

新对话开始时先读本节，再读取父仓库 memory、NM5 skills 和
`scale_up_outputs/nm5_looped_self_forcing_search_20260912/config/runtime.yaml`。
本节是本轮运行的最新状态；下方涉及 40 秒、162 latent frames 或旧实验的内容均为
历史资料，不能覆盖本节的硬约束。

**最新状态（2026-09-21）：R1–R4 四轮共 16 个候选全部跑完并完成收尾；
第五轮（block-wise LoRA DMD 训练接入）的对齐 dryrun 已通过，formal 作业已提交并排队。**
搜索结论见下方「十六配置合并结论」——最不伤动作的深度区间是
`layer_start=8, layer_end=15`（K=2），且该结论在 per-block（R3）与 stack（R4）
两种 loop 算子语义下都成立。该区间已被上游分支 `168d4de` 采纳为训练预设的固定配置。

**第五轮要点见下方「第五轮：block-wise LoRA DMD 训练接入」**：对齐后的 dryrun 已通过
（NM5 作业 `46239326`，30 steps，零 OOM、零 CheckpointError）；formal 作业
`formal_align_r1_20260921`（NM5 作业 `46240819`）已提交，目前等待 Slurm 调度，
尚无完成证据。三个非显然的根因（`is_init` 重放缺陷、
FSDP 切分阈值卡在 5e7、NM5 的 H100 只有 63.29 GiB）都记录在该节，**改动全部在分支外**，
`Self-Forcing-blockwise` 的 HEAD 仍是 `168d4de`、代码一字未改。

### 提交到 GitHub（强制要求，优先于其他收尾动作）

**本轮操作员覆盖（2026-09-21）**：本次对齐修复与 dryrun 证据已通过受控 rsync
同步到 NM5；用户最新指令已明确允许在 formal 完成前先 commit/push 当前版本到
GitHub。formal 作业仍需独立监控，不能把提交视为训练成功证据。

**推送目标（已配置好，直接 `git push` 即可）**

- 远端 `mine` = `https://github.com/duwuuw/self-forcing-reproduction`，
  本地 `main` 映射到远端分支 `looped-flow-matching`；
  `branch.main.pushRemote=mine` + `remote.mine.push=refs/heads/main:refs/heads/looped-flow-matching`
  已写入`.git/config`（**本地配置，不随仓库共享，换机器要重设**）。
- 远端 `origin` = `https://github.com/a1443356159/looped-flow-matching` 当前账号
  （`duwuuw`）**只有 READ 权限，推送会 403**，保留它仅用于 fetch。
- `mine` 上的 `github-upload` 分支是一份**独立的 Self-Forcing 复现源码树**
  （含 `vbench_assets/`），与本工作区历史无关，**绝对不要 force 覆盖它**。

- **每当出现新的可提交修改（代码、脚本、配置、launcher、文档），就立即 commit + push。
  不要攒批，不要等用户提醒。** 跑完一轮实验、改完一个 launcher、更新完本文件，都算一次。
  用户明确要求后续对话积极提交。
- **权重与大产物永不入仓**：`checkpoints/`、`wan_models/`、`*.pt`、`variants/`、`logs/`、
  `artifacts/`、`run_state/`、`scale_up_outputs/envs/` 一律不提交。
- **私有值永不入仓**：NM5 的 Slurm 账号、用户名、项目存储路径、SSH 路由、W&B key
  都不得出现在提交内容里 —— 只写键名不写值，具体值从私有 sandbox 配置现场解析。
  注意 `artifacts/experiment_results.csv` 记录了 NM5 的项目存储路径，**已在排除列表内，
  不要把它移出**。
- 排除规则写在 `.git/info/exclude`，**该文件是本地的、不随仓库共享**；换机器或重新克隆后
  必须重新配置，否则要么漏提交 bundle 源码，要么把运行产物和私有值推上去。
- 提交前自查：`git status --short` 里不应出现 `*.pt`、`checkpoints/`、`variants/` 或私有路径。
- 代码树以 **submodule** 形式记录：`Self-Forcing-layerwise` →
  `https://github.com/vibe-hust/self-forcing-reproduction` @ `e1df619`
  （branch `looped-self-forcing`，即 layer-wise selected-layer stack loop）。
  本地 `Self-Forcing/` 是另一棵上游树（`guandeh17/Self-Forcing`）且带 89 处未提交改动，
  **不提交**；`agent.me`、`HANDOFF_NM5.md`、`1.md` 等内部交接笔记同样不提交。

### 当前目标与硬约束

- 目标：在 NM5 上对 looped Self-Forcing 做 rollout → VBench-Long eval，完成第一轮候选
  配置后继续执行备选配置。
- 正式 checkpoint：`Self-Forcing/checkpoints/self_forcing_dmd.pt`，必须使用
  `--use_ema`；不要把 checkpoint 重新判为缺失。
- 唯一可配置帧数：`num_output_frames: 123`，严禁改为任何其他值。
- 489（raw decode）和 480（VBench processing）是由 123 派生的内部处理数，不是可调
  配置；16 fps 下最终只保留 30 秒视频。
- 允许变化的只有 YAML 中的 `k_min` / `k_max`（用户口头称 kmin/kmax）和 loop layer
  范围；不要改 prompt manifest、帧数、fps 或输出时长协议。
- 当前 active bundle：`scale_up_outputs/nm5_looped_self_forcing_search_20260912/`。
  其中有 26 个候选配置和 128 条 prompt；第一轮推荐
  `k1_full_19_26`、`k4_full_19_26`、`k1_k4_full_19_26`、`k1_k6_late_23_26`，
  第一轮完成后备选为 `k3_full_19_26`、`k5_full_19_26`、`k6_full_19_26`、
  `k2_k6_full_19_26`。

### 环境与传输状态（最后核验时间：2026-09-13 05:35 UTC — 已解除阻塞）

- transfer 已完成：`TRANSFER_EXIT_CODE=0`，`TRANSFER_END_UTC=2026-09-13T04:31:45Z`。
- 环境已解包并落盘：`scale_up_outputs/envs/miniconda3/envs/looped-self-forcing`
  （`conda-unpack` 已执行）。旧 `scale-rae` 环境确认不存在（`loop_diffusion_envs/` 为空目录），
  没有再复用。
- **环境门已关闭**：NM5 单 GPU generation smoke 作业 `silly-nm5-env-smoke-r4`（job
  45808311）`COMPLETED`、`ExitCode 0:0`，证据
  `scale_up_outputs/nm5_looped_self_forcing_search_20260912/readiness/nm5_env_smoke_result.json`
  → `status=success`、`raw_frames=489`、`num_output_frames=123`、`use_ema=true`。
  同一 smoke 产出的 `0-0_ema.mp4` 经 ffprobe 校验为 832x480 / 16 fps / 489 帧。
- `readiness/nm5_env_manifest_verified_20260913.json` = verified；
  `readiness/nm5_env_import_smoke_20260913.json` = success。
- 5 个传输 tarball 已在完整性核验（env import、10 个 vbench_cache 模型目录、overlay310、
  静态 ffmpeg、checkpoint 大小、wan 权重、dinov2/dreamsim 资产、无断链）后删除，
  `scale_up_outputs/envs/` 由 51G 降至 17G；`*_manifest*.json` / `*_readiness*.json` 等
  证据文件全部保留。

### 本轮修复的 launcher 缺陷（durable，已落在 active bundle）

前一轮的 launcher 从未真正执行过，接手时发现并修复：

1. `gen_packed.sbatch` 未 `cd "$SF"`，而 `inference.py:59` 以相对路径加载
   `configs/default_config.yaml` → 每次生成都会 `FileNotFoundError`。改为在 subshell 中
   `cd "$SF"` 并把 `--checkpoint_path` 改为绝对路径。
2. `_env.sh` 未定义 `sue_tools()` / `sue_preflight()`，但两个 packed launcher 都会调用
   → `set -e` 下 exit 127。已从旧 bundle 移植并改写，`sue_preflight` 现在还会断言
   `flash_attn` 可导入。
3. `eval_packed.sbatch` 有 3 个资产路径多写了一层 `hub/`。已加入自愈式 symlink 桥接
   （`$TORCH_HOME/<repo>` → `$TORCH_HOME/hub/<repo>`），保留原有 fail-loud 断言。
4. NM5 侧 `slurm_scripts/` 落后于本地版本（6 个文件中 5 个不同），已重新同步。
5. smoke 脚本缺 `module load ffmpeg/7.0.1-gcc` + `sue_tools`，`ffprobe` 不在 PATH 上
   （imageio-ffmpeg 不带 ffprobe）→ 生成成功后仍在校验步骤失败。已修复，并把 shebang
   改为 `#!/bin/bash -l`（module 系统需要 login shell）。
6. smoke 脚本的 `#SBATCH --job-name` 缺少 `<username>-` 前缀，已改为带前缀的默认值，
   并注明 canonical submitter 会用 `user.yaml` 解析值覆盖。
7. `submit_search.sh` 的 tmux 加载需要先 `module load ncurses/6.4`（tmux/3.3a 声明了
   `prereq_any("ncurses/6.4")`），否则 monitor 会话起不来；且现在 tmux 不可用时只告警
   不致命（Slurm 的 `--output/--error` 才是权威日志）。

**环境侧新增依赖**：`flash_attn==2.7.4.post1`。`wan/modules/attention.py:flash_attention`
在 T2V 路径上硬断言 `FLASH_ATTN_2_AVAILABLE`，但 `Self-Forcing/requirements.txt` 里没有它，
NM5 也无外网无法编译。已用预编译 wheel
`flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
（torch 2.5.1+cu124 是 `_GLIBCXX_USE_CXX11_ABI=0`，必须 abiFALSE）装机。`nm5_env_install_20260913.sh`
增加 `STEP=install_flash_attn`，`requirements-nm5-core.txt` 里以注释形式记录。

### 第一轮结果（2026-09-13，128 prompts，30 s @ 16 fps，123 帧协议）

`fullrun_r1_20260913` 四个配置，VBench-Long 六维：

| variant | subject | background | motion | dynamic | aesthetic | imaging |
| --- | --- | --- | --- | --- | --- | --- |
| `k1_full_19_26`（K=1，等价关循环） | **0.9722** | **0.9625** | 0.9859 | **0.3755** | **0.5412** | **0.7027** |
| `k4_full_19_26` | 0.9563 | 0.9492 | 0.9851 | 0.2969 | 0.4675 | 0.5391 |
| `k1_k4_full_19_26` | 0.9688 | 0.9569 | 0.9870 | 0.3552 | 0.5177 | 0.6708 |
| `k1_k6_late_23_26` | 0.9669 | 0.9537 | **0.9882** | 0.3661 | 0.5198 | 0.6557 |

结论（务必如实写进后续分析，不要淡化）：

- `k1_full_19_26` 在 `strength=1.0` 下 `eta = strength/K = 1`，递推退化为
  `x = B(x)`，**它就是关循环 baseline**。第一轮没有任何 looped 配置在任何一维上超过它。
- 固定 K 的代价随 K 单调变差：`k4` 的 dynamic −0.0786、imaging −0.1636 是全轮最差。
- **层范围比 K 更关键**：只在后段层（23–26）做 K 1→6 的 ramp，dynamic 只掉 −0.0094，
  motion_smoothness 反而 +0.0024；同样 ramp 铺满 19–26 代价大一个数量级。
- 这重现并加强了 `dyn30` 的旧结论（K=2 相对 K=1 掉 −0.0867 dynamic），样本从 n=30 扩到 128。
- dryrun 阶段 `dynamic_degree=0.0` 确认是单条 prompt 选样问题，不是回归。

durable 产物：`artifacts/experiment_results.csv`（4 行 fullrun + 4 行 dryrun）、
`artifacts/eval_results_history.md`、`readiness/readiness_nm5.json`、
`fullrun_r1_20260913/run_state/eval_complete.json`、各 variant 的 `eval/metrics.json`。

### 第二轮结果（2026-09-13，同协议）

`fullrun_r2_20260913`：generation job `45825388`（COMPLETED 3:36:59，四个 variant 各
128/128），packed eval job `45825389`（COMPLETED 3:37:17）。相对 round-1 的
`k1_full_19_26`（关循环 baseline）的差值：

| variant | subject | background | motion | dynamic | aesthetic | imaging |
| --- | --- | --- | --- | --- | --- | --- |
| `k3_full_19_26` | −0.0112 | −0.0125 | −0.0009 | −0.0750 | −0.0591 | −0.1163 |
| `k5_full_19_26` | −0.0181 | −0.0128 | −0.0008 | −0.0693 | −0.0788 | −0.1935 |
| `k6_full_19_26` | −0.0189 | −0.0120 | −0.0007 | −0.0760 | −0.0843 | −0.2187 |
| `k2_k6_full_19_26` | −0.0106 | −0.0116 | −0.0010 | −0.0547 | −0.0547 | −0.1060 |

### 第三轮结果（2026-09-14，同协议）

`fullrun_r3_20260914`：固定 `K=[2,2]`，把 8 层宽的 loop 窗口沿深度平移，回答"loop 放在
哪一段最不伤动作"。generation 与 eval 分离（`eval_variant_1gpu.sbatch`，一 GPU 一个
variant），4 个 eval 作业全部 `COMPLETED`、`ExitCode 0:0`：

| variant | loop 层（零基闭区间） | eval Slurm job | 状态 |
| --- | --- | --- | --- |
| `k2_l00_07` | 0..7 | 45861588 | COMPLETED 0:0（1:15:48） |
| `k2_l08_15` | 8..15 | 45861589 | COMPLETED 0:0（1:16:53） |
| `k2_l16_23` | 16..23 | 45861590 | COMPLETED 0:0（1:16:48） |
| `k2_l22_29` | 22..29 | 45861591 | COMPLETED 0:0（1:17:08） |

**2026-09-20 只读复核（本收尾任务的取证）**：`sacct` 重查这四个作业仍为 `COMPLETED`、
`ExitCode 0:0`（四个 `.batch`/`.extern` 子步骤同），结束时刻 09-14 19:02:03 / 19:03:08 /
19:03:03 / 19:03:23；NM5 上 `fullrun_r3_20260914/variants/<v>/eval/metrics.json` **4 个齐全**
（2259–2271 B，mtime 与作业结束时刻一致）；`run_state/` 内 `eval_complete.json` 与 4 个
`eval_<v>_complete.json` 均在；workspace 级 `artifacts/eval_results_history.md`（188 行 /
16 条）与 bundle 级 history **都已含 R3 的 4 个 `<!-- eval:fullrun_r3_20260914:<v> -->` 标记**
⇒ 本轮收尾产物完备。`squeue` 中无本工作区作业（在排队的属于同一共享账号下其他项目）。

VBench-Long 六维绝对值，括号内为相对关循环 baseline `k1_full_19_26`
（`fullrun_r1_20260913`）的差值：

| variant | subject | background | motion | dynamic | aesthetic | imaging |
| --- | --- | --- | --- | --- | --- | --- |
| `k2_l00_07` | 0.9795 (+0.0073) | 0.9756 (+0.0131) | 0.9918 (+0.0060) | **0.0333 (−0.3422)** | 0.4377 (−0.1035) | 0.5986 (−0.1041) |
| `k2_l08_15` | 0.9770 (+0.0048) | 0.9638 (+0.0013) | 0.9874 (+0.0015) | **0.3385 (−0.0370)** | 0.5464 (+0.0052) | 0.7033 (+0.0006) |
| `k2_l16_23` | 0.9675 (−0.0046) | 0.9584 (−0.0041) | 0.9826 (−0.0033) | **0.4286 (+0.0531)** | 0.5073 (−0.0339) | 0.6981 (−0.0046) |
| `k2_l22_29` | 0.9315 (−0.0407) | 0.9413 (−0.0212) | 0.9708 (−0.0151) | **0.5516 (+0.1760)** | 0.4487 (−0.0925) | 0.5334 (−0.1693) |

baseline 绝对值：subject 0.9722 / background 0.9625 / motion 0.9859 / **dynamic 0.3755** /
aesthetic 0.5412 / imaging 0.7027。

关键读数：

- **K 完全固定、层宽完全相同，只平移 8 层窗口，`dynamic_degree` 就从 0.0333 变到
  0.5516（约 16.5 倍）。** 这是"层范围比 K 更重要"最干净的证据：K=2 的代价几乎完全
  由 loop 落在深度哪一段决定，而不是由 K 决定。
- **深度对 dynamic 单调递增**：越靠前越抑制运动，越靠后越放大运动。0..7 把视频几乎
  冻住（0.0333，只有 baseline 的 9%）。
- **质量与 dynamic 方向相反。** `k2_l00_07` 的 subject/background/motion 高于 baseline
  是"几乎不动 → 一致性指标自然偏高"的假象，其 aesthetic/imaging 反而各掉
  −0.1035 / −0.1041。
- **`k2_l08_15` 是唯一在其他五维（subject/background/motion/aesthetic/imaging）上都不
  低于 baseline 的 looped 配置**，且 dynamic 代价仅 −0.0370，是全轮 looped 配置里
  最小的动态损失。**8..15 是最不伤动作的深度区间。**
- `k2_l16_23` 用 −0.0339 aesthetic 换 +0.0531 dynamic，若要有意增强运动可用；
  `k2_l22_29` 的 +0.1760 dynamic 代价过大（imaging −0.1693），不建议。

### 十六配置合并结论

`dynamic_degree` 相对关循环 baseline（`k1_full_19_26` = 0），从小到大：

```text
k2_l00_07 (R3 per-block, 0-7)      -0.3422   ← 全轮最差：把运动冻住
k2_s00_07 (R4 stack,     0-7)      -0.1984   ← 同窗口换 stack 语义，损失减半
k4_full_19_26                      -0.0786
k6_full_19_26                      -0.0760
k3_full_19_26                      -0.0750
k5_full_19_26                      -0.0693
k2_k6_full_19_26                   -0.0547
k2_l08_15 (R3 per-block, 8-15)     -0.0370
k2_s08_15 (R4 stack,     8-15)     -0.0240   ← 两种语义都最好；唯一五维不劣于 baseline
k1_k4_full_19_26                   -0.0203
k1_k6_late_23_26                   -0.0094   ← 全轮 dynamic 损失最小：4 层窄 ramp，K 多为 1
k2_s16_23 (R4 stack,     16-23)    -0.0068
k1_full_19_26 (K=1, 关循环)         0        ← 关循环 baseline
--- 以下 dynamic 高于 baseline ---
k2_l16_23 (R3 per-block, 16-23)    +0.0531
k2_s22_29 (R4 stack,     22-29)    +0.0656   ← 画质崩坏后的伪运动（imaging -0.4585）
k2_l22_29 (R3 per-block, 22-29)    +0.1760
```

四条可写进论文的结论（第 1、3 条已在 R4 上复现并加固）：

1. **"looped 配置在任何一维都不如关循环 baseline"只对 R1/R2 的 8 个配置成立，R3 与 R4
   都推翻了它。** `k2_l08_15`、`k2_s08_15` 在 subject/background/motion/aesthetic/imaging
   五维上都不低于 baseline。正确的表述是：**loop 的代价不是必然的，而是由 loop 所在的
   深度区间决定的。**
2. **层范围（深度）比 K 重要；两种算子语义下都成立。** 固定 K=2、固定 8 层宽，仅平移窗口
   就让 dynamic 从 0.1771 变到 0.4411（R4）/ 0.0333 变到 0.5516（R3）；跨 R1–R4，深度
   带来的动态差异（−0.3422 ~ +0.1760）远大于 K 从 2 到 6 带来的差异（−0.037 ~ −0.079）。
3. **深度对运动单调、方向与质量相反；R4 进一步证明"深层高 dynamic"是伪信号。**
   前段层抑制运动，后段层放大运动；但后段放大运动的同时 aesthetic/imaging 塌陷，
   且 **stack 语义下 22–29 的 imaging 掉到 −0.4585**。**中间段 8–15 是唯一两头都不亏
   的区间，这一点在 per-block 与 stack 两种语义下都成立。**
4. **K 的代价在 K≈3 之后饱和。** 固定 K 的 k3/k4/k5/k6 都聚在 −0.069 ~ −0.079，
   再增大 K 不会显著更差；ramp 型（`k1_k4`、`k2_k6`）一致优于同量级的固定 K。
5. **算子语义（per-block vs stack）只改两端，不改中段结论。** 8–15 在两语义下都最好；
   stack 在前段更温和（0–7 损失减半）、在后段更凶（22–29 imaging 恶化 2.7 倍），
   16–23 分歧最大（imaging −0.0046 → −0.1460）。

工程结论：默认推荐 `K=2` + `layer_start=8, layer_end=15`；若要更强运动，把窗口扩到
`16..23`，而不是继续加大 K，也不是继续向深层推。**若只看 dynamic 数值会误选 22–29
（R4 甚至为正），必须同时看 imaging/aesthetic。**

**结论溯源与收尾状态（2026-09-20 复核）**：16 个配置的六维数值全部可溯源到各轮的
`variants/<v>/eval/metrics.json`。R3 四条已与本地
`artifacts/fullrun_r3_20260914/variants/*/eval/metrics.json` 对照，小数点后 4 位与上表
逐条一致；R1–R4 的 16 条也都在 `artifacts/eval_results_history.md`（188 行）里。
**本轮复核不改变任何结论**：R3 的四条读数、深度单调性、以及"8..15 是唯一两头都不亏的
区间"均维持原样；搜索阶段依旧封盘，不提交任何作业。

### 第四轮：layer-wise 代码迁移与 dryrun（2026-09-15）

上游 `vibe-hust/self-forcing-reproduction@looped-self-forcing`（提交 `e1df619`
"Implement selected layer stack loop"）是 layer-wise 版本：**选中的连续层区间作为一个
loop body 整体重复 K 次**（`causal_model.py` 的 `run_block` + stack 循环），而不是旧的
逐 block 各自循环 K 次。旧树 `Self-Forcing/` 与新树 `Self-Forcing-layerwise/` 与上游相比
只有 11 个文件不同。

- 新树以 submodule 形式挂在本仓库（见上方「提交到 GitHub」）。
- NM5 上跑新树的开关是 `SUE_SF_DIR`（`_env.sh` 覆盖，`submit_search.sh` 显式透传）。
- 新树是 git clone，缺 `VBench/`、`checkpoints/`、`wan_models/` 三个非 git 资产，已用
  symlink 指向旧树复用（零拷贝）。**`_env.sh` 已加守卫**，缺资产时带修复命令 exit 2。
- 教训全文见 `SCALE_UP.md` 的 "Search round 4" 一节。

**dryrun 结论（run id `dryrun_r5_20260915`，已闭环）**：
- 生成 job `45886786` `COMPLETED 0:0`（4:46），4 个 variant 各 1 个视频；
  worker 日志确认走的是 `Self-Forcing-layerwise/`，`[TemporalLoop] K=2 layer_range=[...]` 正确。
- eval job `45886787` `COMPLETED 0:0`（3:43），**4 个 variant 的 `metrics.json` 全部产出**，
  层范围/K/prompt 数与各自配置一致。
- 1-prompt 下 `dynamic_degree` 多为 0，是单样本选样现象，不是回归（与 R3 dryrun 同）。
**R4 fullrun 结果（run id `fullrun_r4_20260915`，2026-09-17 05:47 UTC 收尾）**：
生成 job `45886006` `COMPLETED 0:0`（2:22:32），4 个 variant 各 **128/128**；
packed eval job `45886007` `COMPLETED 0:0`（3:49:34），4 个 `metrics.json` 全部产出
（写入 09-16 19:33–20:29）。`finalize_variant_evals.py` 与 `reduce_vbench_history.py`
均 `EXIT=0`。**eval 起初申请了 24 h wall time 而长时间 `PENDING/Priority`；就地
`scontrol update jobid=45886007 TimeLimit=05:00:00` 把 backfill 窗口从 24 h 缩到 5 h 后
才拿到节点**（实测只需 3:49，见下方"排队教训"）。

VBench-Long 六维绝对值，括号内为相对关循环 baseline `k1_full_19_26`
（`fullrun_r1_20260913`）的差值：

| variant | loop 层（零基闭区间） | subject | background | motion | dynamic | aesthetic | imaging |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `k2_s00_07` | 0..7 | 0.9445 (−0.0276) | 0.9504 (−0.0122) | 0.9895 (+0.0036) | **0.1771 (−0.1984)** | 0.4560 (−0.0851) | 0.5569 (−0.1458) |
| `k2_s08_15` | 8..15 | 0.9795 (+0.0073) | 0.9651 (+0.0026) | 0.9864 (+0.0006) | **0.3516 (−0.0240)** | 0.5397 (−0.0015) | 0.7055 (+0.0028) |
| `k2_s16_23` | 16..23 | 0.9639 (−0.0083) | 0.9548 (−0.0078) | 0.9907 (+0.0049) | **0.3688 (−0.0068)** | 0.5051 (−0.0360) | 0.5567 (−0.1460) |
| `k2_s22_29` | 22..29 | 0.9071 (−0.0650) | 0.9485 (−0.0141) | 0.9892 (+0.0033) | **0.4411 (+0.0656)** | 0.3357 (−0.2055) | 0.2442 (−0.4585) |

关键读数：

- **`k2_s08_15` 再次是唯一五维不低于 baseline 的窗口**（subject/background/motion/imaging
  为正，aesthetic 仅 −0.0015），dynamic 只掉 −0.0240。**R3 的结论在 stack 语义下复现。**
- dynamic 仍随深度单调递增（0.1771 → 0.3516 → 0.3688 → 0.4411），但 **stack 语义把前段的
  运动损失砍掉近一半**：0–7 从 R3 的 −0.3422 收窄到 −0.1984。
- **stack 语义显著放大深层质量塌陷**：22–29 的 imaging 从 R3 的 −0.1693 恶化到 −0.4585，
  aesthetic 从 −0.0925 到 −0.2055。该窗口 dynamic 高于 baseline（+0.0656）是画质崩坏后的
  伪运动，**不是收益，不能据此推荐深层**。
- **16–23 是两种语义分歧最大的窗口**：imaging 在 R3 几乎无损（−0.0046），在 R4 掉 −0.1460。

语义对比（Δ vs baseline，同 K=2、同四窗口，唯一变量是 loop 算子）：

```text
窗口      R3 per-block（逐 block 各循环 K 次）   R4 stack（整个层区间作 loop body）
0-7       dyn −0.3422   img −0.1041             dyn −0.1984   img −0.1458
8-15      dyn −0.0370   img +0.0006             dyn −0.0240   img +0.0028   ← 两种语义都最好
16-23     dyn +0.0531   img −0.0046             dyn −0.0068   img −0.1460
22-29     dyn +0.1760   img −0.1693             dyn +0.0656   img −0.4585
```

- **"最不伤动作的深度区间"这一结论对算子语义稳健**：两种语义下 8–15 都同时保住 dynamic
  与画质；差异只出现在两端（前段 stack 更温和，后段 stack 更凶）。
- 若只看 dynamic 数值会误选 22–29（R4 甚至为正）；**必须同时看 imaging/aesthetic**，
  否则会把画质崩坏读成运动增强。

`artifacts/eval_results_history.md` 现状：**R4 收尾时发现该文件在 NM5 与本地都不存在，
agent.md 早前"R3 已写出该文件"的记录与实际不符。** 已用各轮留存的 `metrics.json`
重建并补齐 **R1–R4 全部 16 条**（`eval:<run_id>:<variant>` 标记各 1 条，188 行），
并已 scp 回本地 bundle。若后续再出现"收尾脚本报 EXIT=0 但产物找不到"的情况，
先 `find` 确认路径再信任记录。

### 第五轮：block-wise LoRA DMD 训练接入（2026-09-18 → 09-20）

**目标**：把上游分支 `block-wise-temporal-loop`（tip `168d4de`，selected-block LoRA DMD 训练）
**第一次真正跑在 GPU 上**。该分支自己的 `AGENTS.md` 把 GPU 明确划在证据边界之外：
`GPU, Wan2.1-14B, real training ... remain unverified`。本轮就是在撞这条边界。

**代码位置**：`Self-Forcing-blockwise/`（git worktree，HEAD `168d4de`，105 tracked files）。
NM5 上经 `git bundle` → `fetch` → `worktree add` 部署，**HEAD 与本地一致**。
大资产（`VBench` / `checkpoints` / `wan_models`）从默认树软链 —— 沿用 §"NM5 asset reuse" 的约定。

**结果：训练冒烟通过。** 作业 `46169313`（`<username>-blockwise-lora-dmd-smoke_r8_20260919-smoke`，
`--account=<NM5_ACCOUNT> --partition=acc --qos=acc_debug --gres=gpu:4`；作业名前缀与账号均
按硬规则从私有 `deepresearch-sandbox/config_nm5.txt` 现场解析，不写入本文件）
**`COMPLETED 41:44`**，`rc=124`（step-unbounded loop 的预期收尾）、**`OOM=0` / `CheckpointError=0`**、
`TRAIN_EVIDENCE_OK checkpoints=1 offline_wandb_runs=1`。
写出 7 个 adapter 存档（step 10..70）；`generator_optimizer` state `step`=**14**（=70÷`dfake_gen_update_ratio` 5）、
`critic_optimizer` `step`=**70** ⇒ **优化器确实在走**。

**三个非显然的根因（按发现顺序，全部为实测）**

1. **`is_init` 重放缺陷（分支真 bug）**：`wan/modules/model.py:174-183` 的
   `WanT2VCrossAttention.forward` 在 `torch.utils.checkpoint(..., use_reentrant=False)` **体内部**
   翻转共享标志 `crossattn_cache["is_init"]`。backward 重放时该标志已是 `True`，于是走缓存分支、
   **整个跳过 K/V 投影**，两次 pass 保存的张量数不同 ⇒
   `CheckpointError: ... forward 82 vs recomputation 72`。四个 rank 计数完全相同 ⇒ 不是竞态。
   上游**真正跑过 600 iterations** 的 `self_forcing_dmd.yaml` 同样是 `gradient_checkpointing: true`
   但没有 `temporal_loop`/`lora` ⇒ 缺陷出在新增代码。
2. **FSDP 切分阈值刚好卡住（也是最反直觉的一条）**：`utils/distributed.py:80` 的 `fsdp_wrap`
   默认 `min_num_params=int(5e7)`，而本模型单个 transformer block ≈ **4.7e7** —— 差一点点，
   于是整个 1.42B student 落成**一个** FSDP 单元，每次 forward 一次性 all-gather **2.64 GiB**，
   这就是 OOM 的那个申请。降到 `4e7` 后按 block 切分（~94 MB）。
   **顺带解释了"LoRA 为什么没省显存"**：整个 student 在同一个 flat param 里，
   `use_orig_params=True` 下 `flat_param.requires_grad=True`，冻结 block 的激活照样进图。
3. **显存账**：NM5 的 H100 只有 **63.29 GiB 可用**（`nvidia-smi` 报 `65247 MiB`），
   **不是 80 GB** —— 上游参考是按 80 GB 调的。4 卡下 14B teacher 的 fp32 分片是 **~14 GiB/卡**
   （64 卡时仅 ~0.9 GiB）。最终靠 **teacher + T5 + critic 三者 CPU offload**（合计约 25 GB）装下。

**迭代轨迹**：step 20（r5）→ 30（r6，加 `expandable_segments`）→ 70（r7，加 FSDP 阈值）
→ 跑满 40 分钟（r8，加 critic offload）。中途一次 `ValueError: training_gradient_window_frames
cannot exceed num_training_frames`（`pipeline/self_forcing_training.py:93`）—— 分支自有校验器
禁止该方向，**改帧数必须同步改 window**。

**硬约束（本轮全程遵守）**：`temporal_loop` 与其 layer 范围 **8..15 不得改动**。
最终存档 metadata 自证未变：`{layer_start: 8, layer_end: 15, k_min: 2, k_max: 2, strength: 1.0}`。

**与上游 preset 的偏离（全部记录在配置注释里）**

| 项 | 上游 | 本次 | 原因 |
| --- | --- | --- | --- |
| `generator_ckpt` | `checkpoints/ode_init.pt` | `self_forcing_dmd_ema_as_generator.pt` | 用户批准：从**已 DMD** 的 Self-Forcing 权重起步。发布文件顶层是 `{"generator_ema": …}` 而加载器只认 `{"generator": …}`（`model/base.py:89-96` + `strict=True`），故做了**位级重包装**（825 tensor `torch.equal` 全真） |
| `num_training_frames` / `min_training_frames` | 21 | **21（当前）** | 早先为显存做过 15 帧诊断，但当前 dryrun 恢复官方 21 帧；`training_gradient_window_frames` 保持与帧数相等，满足分支 `window ≤ frames` 校验 |
| `log_iters`（冒烟） | 50 | 10 | 只用 submitter 白名单派生，fail-closed 断言"除白名单 key 外无差异" |

**全部修复在分支外**：`scripts/train_blockwise_lora_dmd.py`（含 offline W&B 强制、
`is_init` 重放修复、teacher/critic CPU offload、FSDP 切分阈值），每个补丁都**断言自己生效**
并打审计行（`WRAPPER: fsdp_wrap <模块> params=X.XXB cpu_offload=… min_num_params=…`）。
分支 `git status` 只有那个 preset YAML 是 M。详见 `HANDOFF_NEXT.md` §10。

**断言边界（不要过度声称）**：本轮**证明了**管线通、优化器在走、存档与 W&B 落盘；
**没有证明模型学到东西** —— `lora_B` 70 步后仅 2/16 非零、总 |sum|≈0.10，
这是 `lr=2e-6` × 14 次更新的必然结果，**不是 bug，但说明 70 步远不够**（上游参考 600 步）。
**正式 600-step run 已提交但尚无完成证据**：实测 **~29 s/step** ⇒ 600 步约 **4.8 小时**，其中 critic offload 约贡献 2.5× 的减速。
**协议**：当前训练是 **21 latent 帧**，**不可**与 123-latent 的 VBench-Long 评测分数直接比较；此前 15 帧只代表历史诊断跑。

### Pipeline 状态与下一步

- `sue-nm5-env-install` 与 `sue-dryrun` 均已按证据记录为 completed。
- **dryrun 已通过**：run id `dryrun_r7_20260913`，generation job `45810581`
  （COMPLETED 3:55）、evaluation job `45810582`（COMPLETED 2:15）。
  `readiness/readiness_nm5.json` → `status=ready`、`fullrun_ready=true`；
  `artifacts/experiment_results.csv` 已有 4 行 `status=completed`，W&B offline run id
  分别为 `08ujo6tt` / `6ibvkpbk` / `mp2whe4z` / `g840o4kd`。
- **四轮 fullrun 均已结束**（核验时间 2026-09-17 05:47 UTC；`squeue` 中该账号残留的
  PD 作业属于其他项目，与本 run 无关）：
  - R1 `fullrun_r1_20260913`：4 配置，128/128，生成与 packed eval 完成。
  - R2 `fullrun_r2_20260913`：generation job `45825388`、packed eval `45825389`，
    均 COMPLETED。
  - R3 `fullrun_r3_20260914`：4 个 1-GPU eval 作业 `45861588` / `45861589` /
    `45861590` / `45861591` 全部 `COMPLETED`、`ExitCode 0:0`；
    `variants/<v>/eval/metrics.json` 4 个齐全。
  - R4 `fullrun_r4_20260915`：generation job `45886006` `COMPLETED 0:0`（2:22:32）、
    packed eval `45886007` `COMPLETED 0:0`（3:49:34）；4 个 `metrics.json` 齐全。
- **R3 收尾已执行**（NM5 登录节点，`EXIT=0`）：
  `scripts/finalize_variant_evals.py --exp-dir <exp> --run-id fullrun_r3_20260914
  --mode fullrun --variants k2_l00_07 k2_l08_15 k2_l16_23 k2_l22_29
  --reference k1_full_19_26 --reference-run-id fullrun_r1_20260913`，写出
  `fullrun_r3_20260914/run_state/eval_complete.json`。
  ⚠️ **本文件此前记录的"R3 已写出 `artifacts/eval_results_history.md`"经 2026-09-17
  核查为不实**——该文件当时在 NM5 与本地均不存在。
- **R4 收尾已执行**（NM5 登录节点，`finalize` 与 `reduce` 均 `EXIT=0`）：
  `finalize_variant_evals.py --run-id fullrun_r4_20260915 --variants k2_s00_07
  k2_s08_15 k2_s16_23 k2_s22_29 --reference k1_full_19_26
  --reference-run-id fullrun_r1_20260913`，写出
  `fullrun_r4_20260915/run_state/eval_complete.json`；随后 `reduce_vbench_history.py`
  写入 history。**并已用各轮留存的 `metrics.json` 回填 R1–R3，使 history 恢复为
  R1–R4 全部 16 条**（188 行），文件已 scp 回本地 bundle。
- `artifacts/experiment_results.csv` 已含 R1–R4 全部 16 行（另加 4 行 dryrun）；2026-09-18
  重跑 finalize 后核验，与各轮 `metrics.json` 逐位一致（见「实验结果 CSV Ledger」）。
  **R1–R4 共 16 个候选已全部测完，不要再提交任何作业。**
- fullrun 完成判据：`<run>/run_state/generation_complete.json` 与 `eval_complete.json`、
  `variants/<v>/eval/metrics.json`、`artifacts/experiment_results.csv` 新增 4 行、
  `artifacts/eval_results_history.md` 更新。
- **提交入口**：`slurm_scripts/nm5_round_submit.sh <run_id> <dryrun|fullrun> <variant...>`，
  在 NM5 HPC 登录节点（`nm5`）执行；它会在执行面读取
  `$NM5_DEEPRESEARCH_ROOT/deepresearch-sandbox/config_nm5.txt`，凭据不离开 NM5。
  注意 `nm5-transfer` 是数据搬运集群（`projects/tapes/archive` 分区、无 GPU），
  `--gres` 在那里非法，不能用来提交作业。
- 可选 `SUE_GEN_QOS` / `SUE_EVAL_QOS` 覆盖单阶段 QoS。`acc_debug` 优先级权重是
  `acc_ehpc` 的 100 倍（10000 vs 100）但每用户只允许 1 个作业且 wall ≤ 02:00:00，
  只适合 dryrun；fullrun 必须留在 `acc_ehpc`。
- 第一轮完成后按用户确认**自动续跑**备选 `k3_full_19_26`、`k5_full_19_26`、
  `k6_full_19_26`、`k2_k6_full_19_26`。
- Slurm 作业名前缀由执行面 `<workspace>/user.yaml` 的 `username` 解析；无法解析时回落到
  默认 `silly-`。前缀取值、账号、SSH 路由都**不写入本文件**——需要时从私有
  `deepresearch-sandbox/config_nm5.txt` 和 `user.yaml` 现场解析。

### 接手时的止损规则

- 不要在 `nm5-env-transfer-20260913`（已结束）之外重复传输；payload 已删除，若需重建
  环境必须从 build 侧重新打包。
- 没有 `readiness/nm5_env_smoke_result.json` 的 `status=success` 时，不得把 environment
  标为 ready，不得提交 rollout/eval。
- 生成进程必须在 `cd "$SF"` 的上下文中运行 `inference.py`，否则 `default_config.yaml`
  找不到；这是本轮最常见的失败根因。

本文件根据同目录的 `agent.me` 和 `Self-Forcing` 当前代码整理。`agent.me`
中的上一轮 128-prompt rollout + VBench-Long eval 已完成；本轮 dryrun 已提交，
不要把历史完成记录误判为本轮已经提交。接管时以本文件上方的“当前接管交接”为准。

## 当前实验入口与配置

- 推理入口：`Self-Forcing/inference.py`
- 本轮 loop 配置：`Self-Forcing/configs/self_forcing_dmd_temporal_loop.yaml`
- checkpoint：`Self-Forcing/checkpoints/self_forcing_dmd.pt`，正式推理使用 `--use_ema`
- active bundle 中所有 YAML 的唯一帧字段均为 `num_output_frames: 123`；本轮不允许通过
  CLI 覆盖帧数。
- 本轮视频协议是 16 fps、最终 30 秒；raw 489 和 eval 480 是派生内部数。Loop 参数
  由每个候选 YAML 的 `k_min`、`k_max`、`layer_start`、`layer_end` 决定，不能把源代码
  中的历史 40 秒默认值当作本轮协议。
- 上一轮正式 run：`vbench128_k2_k1k6_full_20260910`；k2 和 k1_k6 的 rollout/eval
  作业均已正常结束，结果与 prompt 说明见 `agent.me`、`HANDOFF_NM5.md`。

## 1. 修改 k 的入口：YAML

在以下文件修改实验配置：

`Self-Forcing/configs/self_forcing_dmd_temporal_loop.yaml:58-73`

```yaml
temporal_loop:
  enabled: true
  mode: dense
  layer_start: 19
  layer_end: 26
  k_min: 1
  k_max: 6
  target_duration_sec: 40.0
  strength: 1.0
  schedule: temporal_uniform
```

字段名是带下划线的 `k_min` / `k_max`，不是 `kmin` / `kmax`。`k_min <= k_max`，
且 K 必须是非负整数；loop layer 的范围也必须合法。

## 2. 随时间调度 K 的文件

真正负责 temporal K schedule 的生产代码是：

`Self-Forcing/utils/temporal_loop.py`

关键位置：

- `TemporalLoopConfig`：约第 109-239 行，读取并校验 `k_min`、`k_max`、
  `target_duration_sec`、`schedule` 和 layer 范围。
- `temporal_k_schedule()`：第 251-303 行，按 rollout progress 将 K 映射到
  `[k_min, k_max]`。
- `temporal_loop_state()`：第 306-334 行，为一个 AR block 计算 progress、虚拟视频时间
  和 K。
- `effective_denoising_k()`：第 337-363 行，只对已经由 AR rollout 得到的 K 做可选的
  denoising gate；它不会从 diffusion timestep 重新计算 K。

当前 `schedule: temporal_uniform` 的生产公式（`N` 是真实生成 AR block 总数，
`block_idx` 是零基索引）是：

```text
p = 0                         if N == 1
p = block_idx / (N - 1)       otherwise
R = k_max - k_min + 1
D = N - 1
K = k_min + min(R - 1, floor(R * block_idx / D))
```

因此，K 由自回归 rollout 的 block 进度决定；prefill 不计入 `N`，diffusion/flow
denoising timestep 不参与 K 的计算。同一个 AR block 的所有 denoising steps 共享
该 block 的 K。旧的 `temporal_linear`（`ceil(k_min + (k_max-k_min)*p)`）仍在
`temporal_k_schedule()` 中保留兼容，但当前实验显式使用 `temporal_uniform`。

## 3. K 的调用链

`Self-Forcing/inference.py:58-79` 先加载并合并 YAML，然后调用
`TemporalLoopConfig.from_config(config)`，再按是否存在 `denoising_step_list` 选择：

- `Self-Forcing/pipeline/causal_inference.py`：few-step/DMD causal rollout，约第
  210-277 行。
- `Self-Forcing/pipeline/causal_diffusion_inference.py`：multi-step diffusion causal
  rollout，约第 228-304 行。

两条 pipeline 都在 AR block 循环中调用 `_temporal_block_state()`，再通过
`_denoising_k()` 把该 K 传给 `WanDiffusionWrapper`；clean-context cache update
通过 `_clean_cache_k()` 单独处理。`utils/wan_wrapper.py` 只负责在 loop enabled
时把 `temporal_loop_config` 和已解析的 `temporal_loop_k` 继续传给模型。

## 4. 决定哪些 layer 做 loop 的文件

这里没有一个独立的 `loop_layers` 列表文件。选择由 YAML 的
`temporal_loop.layer_start` 和 `temporal_loop.layer_end` 决定，实际执行位置是：

`Self-Forcing/wan/modules/causal_model.py`

关键位置：

- 第 473-479 行：`CausalWanModel` 用 `num_layers` 构造真实的
  `self.blocks` `ModuleList`。
- 第 997-1002 行：模型侧读取 `TemporalLoopConfig`，并用
  `len(self.blocks)` 校验 layer 范围。
- 第 1103-1138 行：遍历 `self.blocks`；只有满足
  `loop_config.layer_start <= block_index <= loop_config.layer_end` 的 block
  才调用 `apply_dense_token_loop()`，范围外只调用一次普通 `block()`。
- 第 914-955 行：对同一个 inclusive layer range 做 Dense-loop cache post-check。

该范围是**零基、两端 inclusive**。所以当前 `19..26` 表示代码中的 block index
19、20、21、22、23、24、25、26；按人类从第 1 层开始编号是第 20-27 层。
实际层数以运行时 `len(self.blocks)` 为准，不要硬编码模型深度。

Dense loop 的数学 helper 在：

`Self-Forcing/wan/modules/temporal_loop.py:228-255`

其中 `apply_dense_token_loop()` 执行 `K` 次固定 conditioning/cache 的残差更新；
它负责 loop 数学和配置归一化，不负责决定哪些 block 被选中。

## 5. 相关但不是权威 selector 的文件

- `Self-Forcing/docs/temporal_loop.md`：实现说明、公式、调用层次和限制。
- `Self-Forcing/tests/test_temporal_loop_config.py`：配置与 schedule 测试。
- `Self-Forcing/tests/test_temporal_loop_core.py`：inclusive layer range、K=1、loop
  数学测试。
- `Self-Forcing/tests/test_temporal_loop_pipeline.py`：AR block/pipeline 调用测试。
- `Self-Forcing/eval/vbench_long/prepare_assets.py:133-138`：把
  `k_min/k_max/layer_start/layer_end` 写入 eval manifest 的元数据；它不是正式
  inference 的 runtime layer selector，修改实验时应以 YAML 和模型代码为准。

## 6. 历史运行摘要（不覆盖当前接管交接）

- 方法：`looped-self-forcing` = Self-Forcing + training-free Dense Token Loop。
- backend：NM5；partition/account/QoS 等具体值属于私有配置，不要写入本文件，
  应从本地 sandbox 配置解析。
- 40 秒协议：162 latent frames、54 个 generated AR blocks，输出 645 raw frames，
  裁到 640 frames @ 16 fps 做 VBench-Long。
- 正式比较只有 `k2`（`k_min=k_max=2`）和 `k1_k6`（`k_min=1,k_max=6`）；没有单独
  提交 `k_min=k_max=1` 的正式任务。
- 正式结果根目录：
  `artifacts/vbench_long_selected128/vbench128_k2_k1k6_full_20260910/`
- 本地 CSV ledger：
  `scale_up_outputs/nm5_looped_self_forcing_selected128/artifacts/experiment_results.csv`

完整的 prompt 选择、作业 ID、VBench-Long aggregate 数值和运行交接细节仍以
`agent.me` 与 `HANDOFF_NM5.md` 为准。
