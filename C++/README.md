# Flashone — C++ 版 AlphaZero 五子棋引擎

规则、启发式先验、PUCT 搜索、回放池、自对弈、训练步、门控全部是纯 C++，
神经网络张量运算用 libtorch（直接复用 pip 安装的 torch，不需要单独下载 libtorch）。

> **历史说明**：这套引擎最初是 `PYTHON/` 原型的逐层重写，早期目标是「逐位对齐 Python 语义」。
> 现在它已经是主线（训练曲线 +200 Elo 跑在这套 C++ 上），Python 侧退化为看板与对弈页。
> 下面「已知差异」一节保留的是当初对齐阶段记下的与 Python 实现的语义差别，仍是有效参考。

---

## 1. 构建

前置：

- **MSVC**（Visual Studio 2022 BuildTools 或 Community，需勾选「使用 C++ 的桌面开发」）
- **CMake ≥ 3.20** 与 **Ninja**（放在 PATH 里即可；仓库内若存在 `tools\cmake-*` / `tools\ninja` 会优先用自带的）
- **Python 3.13 + `pip install torch`**（直接复用 pip 装的 torch，不需要单独下载 libtorch）

```bat
cd C++
build.bat
```

`build.bat` 会自动：

1. 用 `vswhere` 找到最新的 VS 并 `call vcvars64.bat`；
2. 问 `python -c "import torch; print(torch.__file__)"` 拿到 torch 根目录，
   填给 `CMAKE_PREFIX_PATH`（找不到 `TorchConfig.cmake` 会明确报错）；
3. 把 `TEMP/TMP` 指到仓库内 `_tmp`（否则 cl 在含空格的 TEMP 下会报 `D8050 无法执行 c1xx.dll`）；
4. 用 CMake + Ninja 生成到 `build\` 并编译。

可覆盖的环境变量（机器上有多套 Python / VS 时用）：

| 变量 | 作用 |
|---|---|
| `TORCHROOT` | torch 根目录，跳过 python 查询 |
| `CMAKE` / `NINJA` | 指定可执行文件路径 |

产出：

| 可执行 | 用途 |
| --- | --- |
| `build\gk.exe` | 主程序（`bench` / `selfplay` / `gate` / `loop`） |
| `build\test_rules.exe` | 规则与张量编码（25 项） |
| `build\test_mcts.exe` | 搜索树不变量 |
| `build\test_dataset.exe` | 8 变换与 numpy 逐格对比、回放池淘汰 |
| `build\test_net.exe` | 参数数、与 Python 权重的前向逐值对比、延迟 |
| `build\test_selfplay.exe` | 整盘自对弈的样本不变量（13917 项检查） |
| `build\test_train.exe` | Adam/裁剪/EMA 的可学习性回归 |
| `build\test_gate.exe` | Wilson 区间与门控判定 |
| `build\probe_eval.exe` | 评估路径延迟（含 CPU 负载干扰模式） |
| `build\probe_mcts.exe` | 单模拟步进诊断 |

运行任何 exe 都需要 `PATH` 里有 `<site-packages>\torch\lib`，否则缺 DLL
（`torch_cpu.dll` 等）。`test_net` / `test_dataset` 的相对路径基准文件在 `C++\_tmp\`
下，**要在 `C++\` 根目录运行**。

## 2. 用法

```bat
REM 单局搜索基准：n_sim=400 走 20 手，打印平均/中位 ms/手
gk bench --net ..\_tmp\v4_gen37.bin --moves 20 --sims 400

REM 并行自对弈：16 局、8 线程、跨局共享批量评估器
gk selfplay --net ..\_tmp\v4_gen37.bin --workers 8 --games 16 --out ..\_tmp\sp

REM 门控：候选 vs 基线，20 局，阈值 0.55
gk gate --cand ..\_tmp\a.bin --best ..\_tmp\b.bin --eval-games 20

REM 门控配对开局：同一盘面走两局、只换黑白，扣掉先手优势（实测黑 71% vs 白 36%）
gk gate --cand ..\_tmp\a.bin --best ..\_tmp\b.bin --eval-games 60 --opening-plies 2

REM 棋力曲线（KataGo 风格：锚点必须显式指定某一代 = 0 Elo，每点带 95% 置信区间）
REM 做法：相邻代两两对局（链）+ 每一代都对锚点打一次（回指），用最小二乘把链解成绝对 Elo
gk elo --dir model/run_v2 --gens 1,10,24,43 --anchor 10 --eval-games 24 --sims 400

REM VCF 战术基准：判定局面是否存在连续冲四必胜（穷举证明，不是启发式）
REM   输入每行一个局面 "x1 y1 x2 y2 ..."，逐行输出 JSON
gk vcfsuite --vcf-max-ply 8 --vcf-max-nodes 2000 < positions.txt

REM 判定某个着法是否为必胜着（比 VCF 本身更严：要求对方任意应手后仍能取胜）
gk vcfsuite --check-x 9 --check-y 9 --vcf-max-ply 8 --vcf-max-nodes 2000 < positions.txt

REM 跑一遍已发布的题库（41 题，需要权重）
python tools\vcf_bench.py --replay ..\docs\tactics\vcf_puzzles.tsv

REM 结果写 model\run_v2\elo.json，看板「C++ 棋力曲线」卡直接读它


REM 完整训练循环（V4 配方默认值，自对弈→训练→门控→落盘→日志）
gk loop --dir ..\models\V5 --gens 10

REM 多进程自对弈（每个子进程一个独立 CUDA 上下文；功能可用，但吞吐不如单进程）
gk loop --dir ..\models\V5 --gens 10 --procs 16

REM 与游戏/其它 GPU 程序共存：进程降到 BelowNormal
gk loop --dir ..\models\V5 --gens 10 --nice
```

`gk loop` 的默认参数是「保守配方」（6×128、320 局/代、`n_sim=600`、`--batch-mcts 16`、
`--min-steps 160`、`--batch 256`、`--lr 3e-4`、`--fresh-ratio 0.5`、`--eval-games 20`、
`--eval-every 2`、`--hw 0.5 --pw 1.0`、回放池 60 代/半衰期 20）。
这组值是在 V4 配方（160 局、`n_sim=400`、`--min-steps 320`、`--lr 1e-3`、fresh 0.7）实测
出「7 代后强度与初始持平」之后调的，理由见第 9 节。
`--cpu` 切到 CPU；`--blocks/--channels` 改架构（`value_hidden` 固定 64）。
训练默认用 **bf16 autocast**（`--no-amp` 回到 fp32）；门控默认 8 局并行
（`--eval-workers N`），候选权重默认是**刚训完的原始 net**（`--gate-ema` 恢复 EMA 行为）。

**配方档案与可测量性（2026-10-28 加）**

- 每次训练都带 `--recipe N`，这个编号写进 `train_log.jsonl` 每一行的 `recipe_id`；看板的损失
  曲线据此在配方切换处画竖虚线并标「配方N」。配方本体存 `C++\peifang\配方NN-*.json`，
  含参数、动机、结果、教训四件套（配方 1 = 历史基线，配方 3 = 可测分辨率，配方 4 = 现行）。
  已废止的两份（配方 5 = `min_steps 1600` 过拟合、配方 2 = 160/320 事故）2026-10-31 移出到
  `_archive\peifang\`，教训并入 `docs/DEVLOG.md` §16.5 / §16.6。
- `--lr-gen-decay 0.97 --lr-floor 1e-4`：从第 2 代起按代退火。原来的 `lr_decay` 只在门控
  胜率 < 0.30 时触发，43 代一次没触发过；门控拒绝**不**回退学习率（否决的梯度不是反向证据）。
- 门控 `--opening-plies 2`：正反两局用同一盘面前两手，只换黑白。19 路自由五子棋先手优势实测
  黑 71% / 白 36%，不配对开局时颜色噪声比可测信号还大。
- `gk elo` 是独立评测子命令，只读权重不改权重，产物 `elo.json` 由看板读。

两个循环行为上的细节：

- **启动时会先做一次预热**（打印 `[预热] 首步初始化 N s（一次性，已还原权重）`）。
  第一次反向传播要触发 cuDNN 的算法选择，本机约 15 s；不预热的话这笔时间会被算进
  第 1 代的训练耗时里（看起来像 300 ms/步）。
- **第 1 代没有历史样本可抽**，与 Python 的 `train_step` 一样退回「只用当代新数据」，
  不会白跳过一整代。

产物目录默认写到 `C++\model\<run>`，与 Python 的 `PYTHON\models\` 分开，两套版本不混用：

```
model/run/gen_001.bin           每代权重（GKAZ1 格式）
model/run/best.bin              门控基线
model/run/buffer/gen_001.bin    该代样本（GKD1 格式，只保留最近 3 代）
model/run/train_log.jsonl       每行一代的指标
```

`python tools\analyze_log.py model\run\train_log.jsonl` 把日志渲染成逐代表格。

## 3. 已验证的等价性

| 项 | 结果 |
| --- | --- |
| 网络参数数 | **1,819,554**，与 Python `GomokuNet(6,128)` 完全一致 |
| 前向（加载 V4 gen_37 权重，4 个局面） | 策略 `max|dp| = 4.32e-06`，价值 `max|dv| = 3.87e-07`（GPU TF32 口径） |
| 数据增广 | 8 种变换与 numpy `rot90(k)+fliplr` 逐格一致（用 Python 生成参照表） |
| 搜索树 | 根访问数、子访问数之和、强制点、必挡点、复用子树等不变量与 mcts.py 一致 |
| 门控 | Wilson 区间与 `ladder.py` 逐值一致；奇局数自动 +1、偶局候选执黑、无认输 |
| 权重往返 | `.pt → .bin → .pt` 后 96 个张量逐值相同（`convert_ckpt.py` 双向） |
| 断点续训 | 从 `gen_003.bin` + `buffer/` 续训与一次性连跑行为一致 |

## 4. 性能（本机 RTX 5070 Ti Laptop 12GB + Ultra 9 275HX）

评估路径单独测量（`probe_eval`，单线程、GPU 空闲时）：

| batch | 16 | 32 | 48 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| ms/call | 2.27 | 3.51 | 5.37 | 6.97 | 16.6 | 32.4 |
| states/s | 7.1k | 9.1k | 8.9k | 9.2k | 7.7k | 7.9k |

结论：**GPU 前向在 b32–b64 就接近饱和（≈9k 状态/s）**，再加大 batch 只是延迟变长。
所以 C++ 侧的瓶颈不在前向本身，而在「每个评估请求太小 + 线程等待」。

单进程搜索（`gk bench`，`n_sim=400`，各 20 手的中位 ms/手）：

| `--batch-mcts` | 16 | 32 | 64 |
| --- | --- | --- | --- |
| ms/手 | 54.9 | 44.9 | 42.5 |

对照 Python `bench.py`：均值 119 / 中位 105 ms/手。**单进程约 2×**。

并行自对弈（`gk probe`，条件：8 组 5 手开局、每线程 240 手、
`n_sim=400`、`hw=6.0/pw=0.3`、temp=1；与当年 Python 侧同名口径一致）：

| 线程数 | 1 | 2 | 4 | 8 | 12 | 16 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ 手/s | 10.3 | 16.9 | 27.9 | 35.2 | 42.6 | **50.3** | **50.8** |
| Python 手/s | 9.2 | 14.3 | 29.6 | 41.5 | — | 46.1 | — |

最佳组合是 **`--workers 16 --eval-batch 24 --eval-threads 3`**（≈50 手/s，比 Python 的 46.1 高约 10%）；
`--eval-threads` 是净增项：3 条独立 CUDA stream 才能填掉单 stream 的同步空转。
再加线程反而下降（16→20 只多 1%），因为已经被 GPU 前向的 9k 状态/s 上限挡住。

多进程自对弈（`--procs 8`，每个子进程独立 CUDA 上下文）实测**更慢**：32 局 18.3 手/s，
而同样 32 局用单进程 8 线程 + 共享评估器是 33.3 手/s。原因是每进程的 batch 只有 16，
GPU 被十几条小请求来回切换；子进程还有 ~10 s 的 torch/CUDA 初始化。
所以 `--procs` 保留作为隔离选项，默认仍走单进程共享评估器。

⚠️ 这些数字是在有游戏占用 GPU 时测的，会被压低；最终对比应在 GPU 空闲时重测。
`--nice` 只是把进程优先级降下来，不改变吞吐比值。

### 训练步

`probe_train`（batch=256、6×128、20 步不含预热）：

| 实现 | 总计 | 前向+反向 | 裁剪 | Adam | EMA | 数据+增广 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 72.5 ms | 70.2 | 0.40 | 0.61 | 0.08 | 1.07 |
| bf16 autocast（默认） | **38.8 ms** | 36.7 | 0.29 | 0.57 | 0.08 | 1.02 |

同一时刻用 `PYTHON/_tmp_time2.py` 在 Python 侧量对照：fp32 81.97 ms、bf16 44.93 ms —— 
**前向+反向两边一致**（都是 cuDNN），C++ 省下的是数据管线（1.1 ms vs Python 的约 25 ms）
和 optimizer 的 kernel 数（96 个张量 → `_foreach_*` 每步 ~8 个 kernel）。
所以「训练步快约 1.8×」这部分收益来自 **bf16**，不是来自语言。

`--no-amp` 可关掉 bf16 回到 fp32。

⚠️ **bf16 与 `_foreach_*` 优化器同用必须关掉 autocast 的权重转换缓存**
（`at::autocast::set_autocast_cache_enabled(false)`）：缓存按张量版本号失效，而
`_foreach_sub_` 这类就地更新不一定会碰版本号，于是前向一直用「上一步的 bf16 权重」，
训练会缓慢劣化。实测过一次：3 代训练后价值头饱和到 -1、策略熵从 1.6 涨到 3.8，
第 2 代自对弈 160 局只产出 320 个样本（每局 2 手就全部认输）。
关掉缓存后第 1 步 loss 与 fp32 一致（23.5 vs 23.4），200 步收敛到同样的 0.0007。

## 6. 两套配方的对照实验（这是本项目最重要的实测结论）

### 6.1 V4 配方：7 代没有进步

`160 局/代、n_sim=400、320 训练步、lr 1e-3、fresh 0.7`，从 V4 `gen_37` 起训 7 代
（`model/run_v4amp`）：自对弈 110–166 s/代、训练 13–17 s/代。逐代 `价值 mse` 从
0.318 涨到 0.411，两次门控都是 **7/20 = 0.35（拒绝，best 从第 2 代后再没更新）**。
40 局一对一：

| 对局 | 候选胜率 |
| --- | --- |
| gen1 vs gen3 | 0.425 |
| gen3 vs gen5 | 0.475 |
| gen5 vs gen7 | 0.625 |
| **gen1 vs gen7** | **0.550（Wilson[0.398, 0.693]，与 0.5 无法区分）** |

诊断：① EMA(0.999) 的时间常数约 1000 步 ≈ 3 代，门控拿滞后 3 代的重均去挑战冻结的
best，结构性看不到进度；② 320 步 × 256 = 8.2 万次更新 / 8.9k 新样本 = 每代把新数据
过 9 遍，且 lr 恒为 1e-3，每代都把上一代拽走；③ 自对弈黑棋胜率约 70%，白方防守样本质量低。

### 6.2 保守配方：4 代 +190 Elo

`320 局/代、n_sim=600、160 训练步、lr 3e-4、fresh 0.5、门控候选=原始 net`
（`model/run_v2`，从 `run_v4amp/gen_007` 续起，回放池沿用 7 代）：

| 代 | 样本 | 手/局 | 自对弈 | 训练 | 门控 |
| --- | --- | --- | --- | --- | --- |
| 1 | 17633 | 55.1 | 429.9 s | 8.3 s | — |
| 2 | 19199 | 60.0 | 473.5 s | 8.4 s | 建立基线 |
| 3 | 19693 | 61.5 | 484.2 s | 8.6 s | — |
| 4 | 20713 | 65.0 | 496.1 s | 8.0 s | **14/20 = 0.70 采纳** |

40 局验证（`--eval-workers 8`）：

| 对局 | 结果 | Wilson |
| --- | --- | --- |
| run_v2 gen4 vs **起点** run_v4amp gen7 | **30:9 → 0.750** | [0.598, 0.858] |
| run_v2 gen4 vs run_v2 gen2 | **28:12 → 0.700** | [0.546, 0.819] |

即 4 代约 **+190 Elo**（对祖先）、2 代约 **+147 Elo**（对内）。同时手/局从 55 涨到 65
（棋变长=防守变好），这与“变强”的方向一致。

代价：320 局 × 600 模拟是 430–500 s/代（旧配方约 2.3 分钟/代）。若想快，把
`--games 240 --sims 500` 大致能压到 4 分钟/代，学习速度按数据量近似线性。

两个可以复用的经验：

- **门控默认用原始 net**（`--gate-ema` 才用 EMA）。EMA 滞后 3 代时，门控测到的是历史平均。
- **loss / mse 不能当进步指标**：新配方 mse 反而更高（0.57 vs 0.33），因为每代只过 2.4 遍
  新数据、目标又随策略漂移；强度只能靠门控或对局测。

### 6.3 保守配方跑到 43 代：进步变慢 + 门控量不出来（2026-10-28）

继续用 6.2 的配方从 gen 4 跑到 **gen 43**（中间 gen 5–7 被"160 局/320 步"污染过一次，
10-27 已回滚，那 3 代留在 `run_v2\_bad160\`）。结论有两层：

**① 早期确实在涨，而且是靠 40 局对局测出来的**（回放池还在长的时候）：

| 阶段 | 数据 | 读数 |
| --- | --- | --- |
| gen 4 | 40 局 vs 起点 | 30:9 = 0.750 → **+190 Elo** |
| gen 25–43 | 20 局门控 ×19 次 | 198/400 = **0.495**，**19 连拒** |
| gen 43 | 200 局门控 vs best | 107:92:1 = **0.535**，Wilson[0.466, 0.603] → 拒绝（≈ +24 Elo） |
| gen 43 | `gk elo`（40 版本 / 78 边） | **+134 Elo** 相对 gen_010 锚点 |

**② 20 局门控在这个强度区间就是噪声，"连续拒绝"不能当"没进步"的证据。**

- 20 局的二项 sd ≈ 0.11，要过 0.55 阈值需要 ≥12 胜；一个真实的 **+25 Elo** 优势只有
  **41%** 概率被这个门控看见 ⇒ 零采纳是"进步很慢"下的**默认结果**，不是异常。
- 更硬的证据是自相矛盾：`gen_026` 对 best（gen_024）门控只赢 **6/20 = 0.30**，而同一天
  的 Elo 拟合给 gen_026 = **+286**、gen_024 = **+65**（差 220 Elo，期望胜率 0.78）。
  两个数不可能同时对 —— 20 局门控在这个量级上测的是运气。
- Elo 自己的区间也全是 ±100：gen 43 = 134 [17, 284]、gen 34 = 263 [152, 405]、
  gen 17 = 176 [61, 305]。**gen 25 之后没有任何一代能把自己的区间和 gen 17 分开。**

**③ 但也不能全赖测量**：`ce` 在 gen 25–43 钉在 1.245–1.276，`mse` 0.52–0.59，回放池
43.5 万 → 48.9 万封顶。真实进步很慢（大概 +0～+10 Elo/代）。零采纳 = **进步小 × 测量粗**。

所以下一步不是调 lr，而是把"每代有效更新"和"每次测量的分辨率"同时抬上去 ——
配方档案 `peifang\配方03-可测分辨率.json`（`eval-games 60`、`opening-plies 2`、
`lr-gen-decay 0.97`、`buffer-gens 30`、`min-steps 320`），**尚未跑过一代**。

### 6.4 代边界 OOM（已修）

43 代里训练进程死在代边界两次（`gen 11`/`gen 13`/`gen 44`），日志都是
`memory allocation failed with OOM on device 0 ... free: 0, total: 12,792,037,376`。
原因是 `GraphRunner` 每代权重版本一变就 `release_all()` 再重新捕获 CUDA Graph，
每张图一个私有显存池，而本机是 12 GB 卡跑到 12.0–12.8 GB 峰值。
**2026-10-28 已把 CUDA Graph 改成默认关闭**（`GK_GRAPH=1` 才开），代价在噪声内（−5.9% ~ ±5%）。

## 7. 已知差异（相对 PYTHON/）

> **⚠ 最重要的一条：C++ 没有 aux heads（辅助预测头）。**
> Python 侧 `network.py` / `train.py` 实现了一组辅助头（V4 的 `aux_heads` / `aux_loss` 日志字段
> 就是它），**C++ 侧完全没有对应实现**——`src/` 里 grep `aux` 零命中。
> 后果：**用 C++ 无法复现 V4 的训练配方**，只能训纯 policy + value 的网络。
> 载入 V4 权重本身没问题（辅助头的张量不参与前向，转换时会忽略），曲线也仍然可比，
> 但"V4 那 37 代到底靠什么涨的"这个问题，在 C++ 里复现不了。

1. **随机数与浮点顺序不同**：Zobrist 用 `splitmix64`（Python 用 numpy 序列），RNG 用
   `std::mt19937_64`（Python 用 PCG64）。同一 seed 不会产生同一批对局，所以
   **不能逐位复现 Python 的训练曲线**；强度数字必须重新基线。
2. **checkpoint 是自有格式**：`GKAZ1`（magic + arch + 命名张量）。两个方向都能转：
   `python tools\convert_ckpt.py --ckpt a.pt --out a.bin`（Python → C++）与
   `python tools\convert_ckpt.py --bin a.bin --pt a.pt`（C++ → Python，供 ladder/analysis 用；
   已用 V4 `gen_37` 验证往返后 96 个张量逐值相同）。不追求 `resume.pt` 位级续训（本来也不可能）。
3. **训练用的是原始 net**，门控候选是 EMA 影子权重 —— 与 train.py 一致。
4. 门控对局没有认输；自对弈有认输（连续 3 手 value < -0.95，认输步及之后样本丢弃）。
5. `n_sim` 必须远大于 `--batch-mcts`（一次 flush 至少 16 个叶），否则根的子节点会被
   少数模拟分摊，策略目标退化 —— Python 有同样行为，`gk loop` 会打印警告。
6. **门控颜色优势很大**（19 路自由五子棋，无禁手）：200 局实测候选执黑 71/100、执白 36/100
   ⇒ 单边偏差 ±17pp。2026-10-28 加了 `--opening-plies N`（默认 2）：同一对局用同一盘面
   只换黑白（配对开局），把颜色从比较里扣掉。旧日志（`opening_plies=0`）的 20 局读数
   有相当一部分是颜色抽样噪声。
7. **`--opening-plies` 只在门控里生效**，不影响自对弈数据生成；自对弈仍是空盘开局。

## 8. 尚未完成

与训练策略有关的开放问题（lr 退火 vs 容量、白方样本、回放池新鲜度）都在
`docs/ROADMAP.md` 里，每条都有命令和判据。这一节只留**实现层面**欠的账：

- **aux heads 未实现**（见 §7 顶部）：C++ 无法复现 V4 的训练配方。
- **多进程路径（`--procs`）**功能可用，但吞吐不如单进程共享评估器，一直是实验状态。
- **长时间稳定性**：已连续跑到 96 代；早期两次代边界 OOM 在 CUDA Graph 默认关闭后没再复现，
  但样本还不够下结论（每次都是训练几天才碰上一次）。
