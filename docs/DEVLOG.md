# 开发日志（DEVLOG）

> 这份文件是项目从 Python 原型迁到 C++、并把训练推到 +200 Elo 的完整工作记录，
> 保留了当时的**实测数据和踩过的坑**。它不是教程，读起来会有点乱，但每个结论后面
> 都有对应的实验数据。
>
> 路径已脱敏（`<repo>` 表示仓库根目录）。

---

# C++ 版五子棋 AlphaZero — 交接文档

> 写于 2026-10-26 11:35，2026-10-26 15:40 增补第 12/13 节，**2026-10-28 19:45 增补第 14 节（配方 3 / Elo 曲线 / 面板配方档案）并更新第 0/1/5/6/7/8/9/10/11/12 节**，**2026-10-30 增补 §6.2（配方 3 的 8 连拒、门控判据改含等号、lr 退火续跑修复）**。
> 这份文档是给**下一个会话**看的：现状、怎么跑、哪些数字已验证、哪些坑已经踩过、下一步该干什么。
>
> 相关文档的分工（别重复写，也别只看一份）：
>
> | 文档 | 管什么 |
> |---|---|
> | `docs/DEVLOG.md`（本文） | 现状 + 交接 + 坑 + 历次改动的技术记录（第 12/13/14 节）。**原 `HANDOFF.md` 已并入本文件，不再单独分发** |
> | `C++/README.md` | 给用户看的：构建、子命令、性能表、配方对照实验（§6） |
> | `C++/model/RUNS.md` | 哪个 run 该续训、每个目录里有什么 |
> | `docs/WEIGHTS.md` | Python V1–V4 到 C++ 主线的权重沿革与引用陷阱 |
> | `C++/peifang/*.json` | 配方档案（机器可读，看板「填入配方」直接吃） |
>
> 文中提到的「SPEED_HANDOFF §x」「ANALYZE_HANDOFF」是当时的另一份工作记录，
> 没有随仓库分发；相关结论已经在本文对应章节里复述过。
>
> 同理，文中引用的 `probe_*.py` / `verify_*.py` 是开发期用来量这些数字的一次性脚本，
> 也没有随仓库分发（它们绑本机路径与端口）。**结构性正确性**由 `C++/tests/` 那 8 个测试程序
> 保障（1.4 万项检查），性能数字则只保留结论与测量条件，供后来者用同样条件复测。

---

## 0. 一句话现状

C++ 引擎（`C++\build\gk.exe`）**功能完整、可训练、可对弈、已接进 Python 看板**。

**当前（2026-10-31 14:40）**：`model/run_v2` 已跑到 **gen_083**，`best.bin` 内容 = **gen_082**
（md5 核对；gen_082 以 33/60 = 0.5500 过门控，gen_083 是门控空档代）。配方 **4**
（`--min-steps 800` + `--gate-threshold 0.52` + `--eval-games 60` + `--eval-every 2`）。
目录与配方沿革明细见 §9。

**棋力**：`gk elo` 的拟合停在 gen 61、之后没重跑，而且那条曲线**不能用来比较相邻代**（原因见 §17.2）。
可信的进展数字只有定点对打：配方 4 前 10 代 **+109.5 Elo**（400 局，95%CI [+75.4, +143.5]）；
全期 gen_016 → gen_061 是 **+139 Elo / 45 代 = +3.09 Elo/代**。
⇒ **判断进展一律用 200 局以上的定点对打，不要看门控胜率。**

历史（保留原因）：gen 25–43 曾连续 19 代没过门控，根因是 20 局门控的分辨率（±100 Elo）
比后期真实速度（+1.3 Elo/代）粗两个数量级 —— 这是配方 3 把门控提到 60 局 + 配对开局的起因（§14）。
但即便如此，60 局也只够「防止 `best.bin` 被崩坏权重污染」，不够判断单代强弱（§17.3）。

2026-10-26 下午那一段：① CUDA Graph 化推理（当时生产形态 +10%，见第 12 节；**10-28 已改默认关闭**，见第 14.2 节）；
② 借鉴 Rapfi 实现 VCF 静态搜索与「强制着法不剪枝」（**两项实测均无收益，已默认关闭**，见第 13 节）；
③ 结论是**当前引擎的瓶颈不在搜索算法，而在并行形态**（第 13.4 节）。

---

## 1. 目录与文件

```
C++\
  build.bat           构建脚本（必须 CRLF！逻辑见第 2 节）
  CMakeLists.txt      C++20 + libtorch，静态库 gk_core + gk.exe + 测试 + probe
  src\
    board.*           19×19 棋盘、Zobrist、合法点、to_tensor(3×361)、成五/活四判定
    heuristics.*      与 heuristics.py 逐条对齐的先验（分数表 1e6/1e5/1e4/1e3/500/100/4）
    mcts.*            PUCT 搜索：虚拟损失、先验宽度裁剪、根噪声、树复用
    vcf.*             VCF 静态搜索：连续冲四必胜的廉价证明（默认关闭，见第 13 节）
    dataset.*         Sample 结构、8 变换增广、ReplayBuffer（按代淘汰/加权）、GKD1 落盘
    net.h / net_torch.cpp   TorchNet：与 Python state_dict 逐名对齐、手写 EMA、GKAZ1 权重格式
    eval_torch.h      TorchEvaluator：Evaluator 接口 → net.infer_batch
    selfplay.*        play_game：温度/认输/样本策略目标
    trainer.*         Trainer + FusedAdam：bf16 autocast、梯度裁剪、EMA
    gate.*            门控对局与 Wilson 判定（含配对开局 `--opening-plies`，见第 14 节）
    elo.*             【10-28 新增】跨代 Elo 拟合：链式相邻代对局 + 每代回指锚点，bootstrap 出 CI，写 elo.json
    batcher.*         NetService：跨对局共享批量评估器（多评估线程 + 独立 CUDA stream）
    pipeline.*        collect_games[_mp] / run_training_loop（含门控、落盘、run.json、配方编号与 lr 退火）
    config.h          TrainConfig = 当前配方默认值（改这里就改了面板默认值）
    sysutil.*         优先级、exe_path、run_parallel（多进程 worker）
    main.cpp          CLI：bench/selfplay/worker/gate/loop/play/probe/defaults/elo
  tests\              test_rules/mcts/vcf/dataset/net/selfplay/train/gate + probe_mcts/eval/multi/train
  tools\              cmake 4.1.2、ninja、mingw64（便携），convert_ckpt.py、analyze_log.py、
                      label_runs.py、check_page_js.py
  peifang\            【10-28 新增】配方档案 配方NN-*.json（参数/动机/结果/教训，看板「填入配方」直接读）
  model\              训练产物（run_v2 = 主线，见 RUNS.md；elo.json 是棋力曲线）
  build\              构建输出（gk.exe 等）
```

Python 侧（看板）相关文件：

```
PYTHON\
  serve_dashboard.py   看板服务（8791）+ C++ 状态/启动/停止 + SSE + 对弈页拉起 + 配方参数白名单
  watch_progress.ps1   生成 progress_live.html（模板在脚本里，**启动时解析**）；含配方档案下拉、棋力曲线、释义块
  web_play.py          对弈页（8770），支持 engine=python / cpp
  _restart_dash.ps1    一键重启看板服务 + watcher 并自检
  _restart_train.ps1   停掉 C++ 训练并按给定参数重开
  _tmp\supervise_loop.ps1  【10-28 新增】固定配方拉起 gk loop、崩了自动重启的监督器
```

---

## 2. 构建（只有一种方式）

```powershell
cd <repo>\C++
cmd /c build.bat
```

`build.bat` 里做了四件必须做的事，**不要绕过它自己敲 cmake**：

1. `TEMP=TMP=<C++>\_tmp` —— 用户目录的 Temp 拒绝写入，否则 `cl` 报 `D8050 无法执行 c1xx.dll`
2. `call vcvars64.bat`（`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`）
3. `CMAKE_PREFIX_PATH=<site-packages>\torch` —— 直接用 pip 装好的 libtorch，不需要单独下载
4. 便携的 cmake 4.1.2 + ninja 1.13.1（`tools\` 下）

已知构建坑：

- **C++ 标准必须是 20**：libtorch 的 `TensorImpl.h` 在 C++17 下报 C7582
- `build.bat` **必须 CRLF**，LF 会被 cmd 截断成 `'et'`/`'ld]'` 这种诡异报错
- `gk.exe` 正在运行时链接会 `LNK1104: 无法打开文件"gk.exe"` —— 先停进程（有时要等 2–3 秒句柄释放）
- `torch::nn::Module::register_parameter` 不接受带 `.` 的名字 ⇒ `net_torch.cpp` 内部把 `.` 编码成 `__`，对外仍输出 Python 点分名

## 3. 运行

**必须用 PowerShell 跑 exe**（bash 工具起 MSVC 产物会得到 rc=127、无输出），并且 PATH 要含 torch 的 DLL 目录：

```powershell
$env:PATH = '<TORCHROOT>\lib;' + $env:PATH
cd <repo>\C++
.\build\test_net.exe            # 等价性自测（必须在 C++ 根目录跑：读 _tmp\v4_gen37.bin）
.\build\gk.exe bench --moves 20 --sims 400
.\build\gk.exe probe --workers 16 --eval-batch 24 --eval-threads 3 --moves 240
.\build\gk.exe loop --dir model/run_v2 --gens 2 --games 8 --workers 8 --sims 64 `
    --batch-mcts 4 --min-steps 4 --batch 8 --eval-games 2 --eval-every 1
```

- `test_dataset.exe` 需要 `_tmp\aug_ref.txt`（numpy 参照），没有会 `[SKIP]`
- 训练产物：`gen_%03d.bin` / `best.bin` / `buffer\gen_*.bin` / `train_log.jsonl` / `run.json` / `loop_stdout.log`

## 4. 已验证的等价性（重写没改算法）

| 项 | 结果 |
|---|---|
| 参数量 | **1,819,554**，与 Python 完全一致 |
| 载入 Python `V4/gen_37.pt` 后前向 | 策略 `max|Δ|=4.32e-06`、价值 `3.87e-07` |
| CUDA vs CPU（TF32） | `3.719e-04` ⇒ 容差按 2e-3 写 |
| 8 种旋转变换 | 与 `np.rot90 + fliplr` 逐格一致（用 Python 生成参照） |
| Wilson 区间 | 15/20→[0.531295,0.888140] 等 5 组与 Python 逐值相同 |
| 自测总量 | test_rules 25 项 / test_selfplay 13917 项 / test_train 853 项 / test_gate 24 项，全 0 失败 |

**不可逐位复现**的差异（设计如此，不是 bug）：Zobrist 用 splitmix64（Python 是 numpy 序列）、RNG 用 `mt19937_64`（Python 是 PCG64）⇒ 同样的种子得不到同样的对局；权重格式是自有 `GKAZ1`，用 `tools\convert_ckpt.py` 双向转换。

## 5. 性能实测（本机，NVIDIA GeForce RTX 5070 Ti Laptop，专用 11.7 GB）

| 场景 | Python | C++ |
|---|---|---|
| 单进程搜索（400 模拟，中位 ms/手） | 105 | **52.7** |
| 吞吐 1/2/4/8/12/16/20 线程（手/s） | 9.2/14.3/29.6/41.5/—/46.1/— | 10.3/16.9/27.9/35.2/42.6/50.3/**50.8** |
| 训练步 batch=256 | 81.97 ms（fp32）/44.93（bf16） | **72.5（fp32）/38.8（bf16）** |
| 最优并行配置 | — | `workers 20 / eval-batch 24 / eval-threads 3` |

- 前向延迟（CUDA，含拷回）：b1 1.54 / b16 2.41 / b32 3.51 / b48 5.35 / b64 7.03 / b128 16.8 / b256 33.2 ms ⇒ **b32–b64 饱和 ≈9k 状态/s**
- **2026-10-28 复测（`gk probe --eval`，同一张 `best.bin`）把天花板钉正**：开启 CUDA Graph 时
  b16 1.36 ms（11,791 状态/s）/ b32 2.68（11,955）/ b48 4.51（10,644）/ b64 6.19（10,336）/
  b128 13.92 / b256 25.86 ⇒ **单流 GPU 峰值 ≈12k 状态/s**；关图后小 batch 略退化（b16 1.70、b32 3.02），
  大 batch 不变。生产形态（20 线程 / 600 sims / `best.bin`）= **8.2–8.8k 状态/s = 峰值的 68–73%**，
  且平均 batch 卡在 26–27（每请求 16 叶 + `kGraphMaxBatch=32` ⇒ 最多合两个请求）。
  用 `--dummy`（随机权重）跑同一负载是 **1592 手/s / 401k 状态/s**，即搜索只占 2.3%、
  **整条链路 43× 的差距全部在推理路径上**（读法见 `README.md` §4）。
- 所以**高并发时两边都撞 GPU 上限**（6.4–7k 状态/s），C++ 的优势只在低并发；`--procs`（多进程）实测 **18.3 手/s，比单进程 8 线程的 33.3 更慢**（每进程 batch 只有 16，GPU 被切碎），默认 0、只作隔离选项
- 一代 320 局 / 600 模拟 ≈ 自对弈 430–500 s + 训练 8 s；160 局 / 400 模拟 ≈ 200 s + 14 s

## 6. 配方与实验结论（重要，别重复踩）

**当前默认配方**（`src\config.h`，面板 C++ 模式的默认值）：`games=320, n_sim=600, min_steps=160, lr=3e-4, fresh_ratio=0.5, batch=256, eval_batch=32, eval_threads=3, buffer_gens=60, half_life=20, gate_use_ema=false, keep_gens=0（全留）, keep_buffer_gens=10, ema_decay=0.999, gate_threshold=0.55, eval_games=20, eval_every=2, seed=12345`；
**10-28 新增的四个旋钮默认值**：`recipe=0`（不标注配方）、`lr_gen_decay=1`（不退火）、`lr_floor=1e-4`、`opening_plies=2`（门控配对开局，可 `--opening-plies 0` 关）。
生产上实际用的是 `workers 20 / eval-batch 24 / eval-threads 3`（见 §5 第 128 行），配方档案见 §14.4。

- **旧 V4 配方（160 局 / 400 模拟 / 320 步 / lr 1e-3）7 代没进步**：gen1 vs gen7 = 0.550，Wilson[0.398,0.693]，两次门控 7/20 拒绝，mse 0.318→0.411
- 诊断：① EMA(0.999) 时间常数 ≈1000 步 ≈3 代，门控用 EMA 测的是"历史平均"；② 320 步 × 256 = 8.2 万更新 / 8.9k 新样本 = 每代过 9 遍且 lr 恒定；③ 自对弈黑胜率 ~70%
- **新保守配方有效**：从 `run_v4amp/gen_007.bin` 续训 4 代，gen4 门控 **14/20 采纳**，40 局对起点 **30:9（0.750）**、对 gen2 **28:12（0.700）** ⇒ 约 +190 Elo
- **loss/mse 不能当进步指标**：新配方 mse 0.57 反而高于旧配方 0.33，但棋力明显更强
- 门控 20 局的统计功效只有 ±0.11 ⇒ 一两代的胜率波动说明不了问题

**⚠️ 一个刚发生的混淆**：`Games` / `MinSteps` 是 Python 与 C++ **共有**的字段，两套配方数值不同（Python 160/320，C++ 320/160）。草稿或"从上次参数填充"会把 C++ 也留在 160/320 上——那样一代只产 1 万样本（320 局是 2 万）。已经在面板上加了检测：切到 C++ 时若与默认不同会显示「⚠ 与 C++ 默认配方不同：局数 160（默认 320）…」并给「用 C++ 默认配方」按钮。**看下一个会话要不要重新验证一遍这个提示**（需要浏览器刷新一次才能拿到新 JS）。

### 6.1 平台期诊断与配方 3（2026-10-28）

43 代只采纳 4 次（gen 4/18/22/24），gen 25–43 连续 19 代被拒，`best.bin` 冻在 gen_024。
200 局复核 gen_043 vs best = **0.535（Wilson [0.466, 0.603]）**，折合 +24 Elo、区间 −24~+43 ⇒ 确实没测出变强。

两个"总进步"的数字要分清口径（别混用，它们的参照物不同）：

| 口径 | 读数 | 含义 |
|---|---|---|
| 40 局对局（人工，gen 4 vs 起点） | 30:9 = 0.750 ⇒ **+190 Elo** | 前 4 代的真实进度 |
| `gk elo` 全量拟合（锚点 gen_010） | gen_043 = **+134 Elo**〔CI 17–284〕 | 43 代累计，但 CI 宽到 ±134 |
| 200 局门控（gen_043 vs best=gen_024） | 0.535 ⇒ **+24 Elo**〔−24~+43〕 | 最近 19 代的净变化：几乎为零 |

- **主因是测量，不是模型**：20 局的置信半宽约 ±100 Elo，而后期真实速度只有 +1.3 Elo/代（全期 +3.1/代）。
  连续拒绝是这个量级差的**统计学必然**（20 局门控对 +25 Elo 真差异的检出率只有 25%）。
  最硬的证据是自相矛盾：`gen_026` 对 best 门控 6/20 = 0.30，而 `elo.json` 给 `gen_026` = +286〔172, 461〕、
  `gen_024` = +65〔−40, 197〕（差 220 Elo ⇒ 期望 0.78）——两个数不可能同时对。
- 次要原因：lr 恒定 3e-4 一次没降（`lr_decay` 只在胜率 < 0.30 时触发）；`min_steps` 是硬下限、永远被抬到 160；
  回放池混 60 代旧策略；门控不配对开局，而先手优势实测黑 71% / 白 36%。
- **配方 3**（`C++\peifang\配方03-可测分辨率.json`，gen 44 起）：`--recipe 3 --min-steps 320 --eval-games 60
  --opening-plies 2 --buffer-gens 30 --lr-gen-decay 0.97 --lr-floor 1e-4`，其余同配方 1。
  代价约 +35 s/代（门控 23 s→70 s，但 eval_every=2），换来门控分辨率 ~100→~55 Elo、新旧数据遍数 2.1→4.2。
- **新开关**：`--recipe N`（写进 train_log 的 `recipe_id`，看板据此画配方断点）、`--lr-gen-decay / --lr-floor`
  （按代退火，门控拒绝不回退 lr）、`--opening-plies N`（门控配对开局，正反两局同盘面只换色）。
  ⚠️ 统计口径：配对后两局不独立，而 `Wilson` 仍按「局」算 ⇒ 区间偏乐观（约相当于 n 打八折）。
  要更严的结论就把 `--eval-games` 加大，或直接看 `gk elo` 的同代区间——那里每对也是配对开局。
- **新子命令 `gk elo`**（`src\elo.cpp`）：相邻代链式对局 + 每代对锚点回指，最小二乘解 Elo，带闭式置信区间；
  产物 `model\<run>\elo.json`，看板「C++ 棋力曲线」卡渲染（KataGo 读法：显式锚点 + 区间重叠 = 没测出差别）。
  **2026-10-29 加**：图上按 `train_log.jsonl` 的 `recipe_id` 画换配方竖虚线（老日志没有该字段时按每代
  samples 跨 15000 推断），隔段底色标出同一配方的连续区间——跨虚线那一步包含"换配方"本身，不能算训练变强；
  新配方若超出 `elo.json` 的代数范围，卡片上用黄字提示"重跑 `gk elo` 后才会出现"。
- 配方档案统一存 `C++\peifang\配方NN-*.json`（参数/动机/结果/教训），`/api/cpp-state` 带 `peifang` 与 `elo` 字段。
- **面板「填入配方」**（2026-10-28 加）：训练控制区 → C++ 引擎 → 「配方档案」下拉（服务端每 15 s 重读
  `C++\peifang\*.json`，把 `args` 烘进页面）+「填入配方」按钮，一套参数一次性回填到表单，仍走「开始训练」的
  干跑确认。新增配方只要丢一个 json 进目录，不用改代码——`args` 的键用 gk 的开关名（`games`/`min_steps`/
  `eval_games`/`opening_plies`/`recipe_id`/`gate_threshold`…），页面里的 `RECIPE_TO_FIELD` 负责翻译成表单 id。
  回填后状态条改用以该配方为基准比对（「参数与配方3 一致」/「⚠ 与配方3 不同：局数 160（配方3 为 320）」）。
  自检脚本：`_tmp\check_recipe.mjs`（映射静态校验）、`_tmp\check_recipe_fill.mjs`（DOM 桩里真跑一遍回填）。

### 6.2 配方 3 的 8 连拒与两个修（2026-10-30）

gen 46–60 连续 8 次门控全拒（采纳仍停在 gen 44）。逐次净胜局数（胜局数 − 30，黑白对称时应为 0）：
−5, −3, −3, −5, −1, +2, +3, +3 ⇒ **相对 gen 44 那个 best 一直在变强，只是没跨过 `wr > 0.55`（净胜 ≥ +4 局）**。
30 对配对局的噪声 1σ ≈ 5.5 局，所以 8 连拒是这门控的正常行为（29 次门控共采纳 5 次 = 17%），不是配方 3 失效；
CE 1.228–1.261 / MSE 0.512–0.561 全程无退化，权重每代都落盘，`best.bin` 冻结不动训练用的 `net`。

两个真问题，2026-10-30 已改（**用户明确「先不训练」，改完没跑**）：

1. **判据严格大于 ⇒ 33/60 = 0.5500 被卡两次**（gen 58、gen 60）。`gate.cpp` 的 `pass_score` 改成
   `win_rate >= threshold`；`gate.h` 注释与 `tests\test_gate.cpp` 的断言同步（25 checks 全绿）。
   `gate_ci_ll` 默认 0（关闭），所以这道判据现在是唯一的门槛。**注意**：含等号等于把门槛降半局，
   若之后嫌采纳太松，正确做法是提 `--eval-games`，不是把等号改回去。
2. **lr 按代退火被每次重启打断**：`pipeline.cpp` 原来是 `double lr = cfg.lr;`，不看第几代续跑，
   于是 train_log 里 lr 在 gen 49、59 两次复位到 0.000291（0.97 只连乘了 4–5 次就断），
   最低只到 2.2e-4 就被拉回 3e-4——lr 一直高，正是配方 3 想消除的随机游走。
   现在改成**读 train_log 最后一行的 `lr` 接着退火**（`read_last_log_lr()`），只有 `recipe_id` 变了
   才从 `--lr` 重设；要主动重设退火用新开关 `--lr-reset 1`。
   干跑验证（`--gens 0`，不训练）：`[学习率] 续跑接着配方 3 的上一代 lr 0.000274（不回到 --lr 0.000300）`。

### 6.4 gen_050 vs gen_061 定点对打（2026-10-30 20:00）

Elo 曲线上 gen_50 = +273.8、gen_61 = +219.8，看着像「后 11 代退步了」。曲线拟合的 CI 宽±150，
这个「下降」在噪声里 ⇒ 用户要求单独打 200 局以上来定论。实际打了两轮共 **400 局**：

```
gk.exe gate --cand model/run_v2/gen_061.bin --best model/run_v2/gen_050.bin \
    --eval-games 200 --sims 600 --blocks 6 --channels 128 --opening-plies 2 --seed 777001
```

| 轮次 | 种子 | 胜 | 负 | 平 | 执黑胜 | 执白胜 | 胜率 |
|---|---|---|---|---|---|---|---|
| 1 | 777001 | 112 | 88 | 0 | 86/100 | 26/100 | 0.560 |
| 2 | 913337 | 112 | 87 | 1 | 85/100 | 27/100 | 0.560 |

**合并 400 局：得分率 0.5613 → gen_061 比 gen_050 强 +42.8 Elo，95% CI [+8.5, +77.0]，单侧 p = 0.0071。**
两轮胜局数完全相同（112 / 112），先后手分布也一致 —— 可复现，不是抽签。
详细记录在 `model/run_v2/match_gen050_vs_gen061.json`。

**结论**：曲线上那段「下降」是噪声，gen 50→61 实际是**缓慢上升**（约 +3.9 Elo/代），不是退化。

**`gk.exe gate` 的两个坑（都踩了）**：
1. 局数参数是 **`--eval-games`**，不是 `--games`（`--games` 是自对弈每代局数，gate 读不到就会静默回落到 20 局）。
   第一次错写成 `--games 200`，30 秒就「跑完」了，还打印「采纳」—— 差点当成 200 局结果。
2. 先手优势极大：候选执黑胜率 85.5%、执白 26.5%。**必须开着 `--opening-plies 2` 配对开局**
   （同开局正反各一局），否则一半对局的信息量会被颜色差吃掉。历史上 gen 2–43 的曲线是 `opening-plies 0` 口径，
   与这次的配对口径不能直接相加。

复现用的完整命令见本节表格上方；`--sims 600` 与当前配方一致（`config.h` 的 `n_sim`）。

### 6.5 曲线高点复验：gen_016 / gen_025 / gen_050 三相循环（2026-10-30 20:32）

曲线前三高是 gen_060(+402.8)、gen_016(+307.1)、gen_025(+287.9)。用户问「这三个点是什么模型」，
查下来 gen_060 只有 **12 局**数据（对 gen_059 实际 1:3 落败、对 gen_010 和 gen_061 各 4:0），
CI 宽达 [+180, +816] ⇒ 小样本虚高，剔除。真实候选是 gen_016 / gen_025 / gen_050，
但曲线给它们的顺序相差仅 33 Elo，远小于 CI±150。于是两两各打 200 局（配对开局）：

| 对局 | 胜 | 负 | 平 | 得分率 | Elo 差 | 95% CI | 单侧 p |
|---|---|---|---|---|---|---|---|
| gen_025 vs gen_016 | 114 | 85 | 1 | 0.5725 | +50.7 | [+2.2, +99.2] | 0.020 |
| gen_050 vs gen_025 | 109 | 90 | 1 | 0.5475 | +33.1 | [−15.1, +81.3] | 0.090 |
| gen_050 vs gen_016 | 126 | 72 | 2 | 0.6350 | +96.2 | [+46.4, +146.0] | 0.0001 |

**三相自洽**：50.7 + 33.1 = 83.8，与直测的 96.2 差 12.4 Elo，在 ±34 噪声内。

**真实排名（以 gen_016 = 0）：gen_050 = +96.2 > gen_025 = +50.7 > gen_016 = 0。原曲线的顺序是反的。**

为什么会反：gen_016 / gen_025 的高点来自**对锚点 gen_010 的压倒性胜率**（10:2 和 11:1），
但它们对相邻代并不强（gen_016 负 gen_015 1:11，gen_025 负 gen_024 5:7）。
锚点口径下「赢锚点赢得狠」就等价于高 Elo，而 Elo 高不代表对任何对手都强。

数据在 `model/run_v2/match_top3_gens.json`。

**教训**：曲线用于**看趋势**可以，用于**排实力名次不可靠** —— 尤其当相邻点的 CI 区间大幅重叠时。
真要定名次就得像本节这样对打；`--eval-games` 低于 200 时 30–50 Elo 的差距测不出来。

### 6.6 曲线上自动标注前三高（2026-10-30 20:27）

用户要求「给 elo 曲线上自动标注前三高的模型」。实现在 `PYTHON\watch_progress.ps1` 的 `cppEloSvg()` 里，
按 `versions` 的 `elo` 降序取前三，画 ①②③ 圈标 + 代数文字，图下方再给一行文字排名。

**关键设计：标签必须带可信度**。直接标名次会误导 —— 实测 gen_060 曾用 **12 局**冲到第一（+402.8），
而它对 gen_059 的 4 局是 **1 胜 3 负**。所以：
- 标签文字带上「（N局）」；
- 局数没打满该点应有场次的用 `--warn` 警告色，文字版再加 `⚠仅N局`；
- 图例末句写明「⚠ 标黄 = 累计对局没打满该点应有的场次，名次可能只是小样本抽签」。

**阈值口径（踩过两次）**：主曲线是「链式相邻 2 条边 + 锚点 1 条边，每边 12 局」。
① 最初写 40 ⇒ 全图最高才 36，永远全标黄、信息量为 0；
② 改成 36 ⇒ 中间代对了，但**两端点（gen_002 / gen_061）只有 2 条边、满样本就是 24 局**，会被误标黄。
现在的判据是**按边数现算**：满样本 = `12 × 该点参与的边数`（从 `elo.edges` 现取，不硬编码 3）。

当前数据下 ①②③ 是 gen_056(372.2)、gen_046(291.5)、gen_057(281.6)，全部 36 局满样本、不标黄。

回归测试：`PYTHON\_tmp\check_elo_top3.mjs`（奖章数量、代数正确、文字行顺序、样本不足必须标黄、
样本够的不标黄、标签坐标不出画布）。

**踩到的坑（看板改了不生效，浪费了一圈排查）**：
`progress_live.html` 是 `watch_progress.ps1` 生成的**静态产物**，而 `Flash UI.cmd` 会拉起一个**常驻的
`pwsh watch_progress.ps1 -Html progress_live.html -Interval 5`**，每 5 秒整份重写一次。
只改脚本、不重启那个常驻进程 ⇒ 手动生成的新页面在 5 秒内被旧进程覆盖回旧版本，
现象是「源码里有新代码、页面里没有」，极易误判为自己改错。

**正确做法**：改完 `watch_progress.ps1` 必须
①`Stop-Process` 杀掉常驻 watch 进程 → ②重新拉起（或用 `Flash UI.cmd -Restart`）→ ③确认
`Get-CimInstance Win32_Process | ? CommandLine -match 'watch_progress\.ps1'` **只有一个**。
多个 watch 进程会互踩（实测同时出现 2 个，页面在两种版本间跳）。

### 6.7 Elo 是**相对锚点 gen_010** 的刻度，且 gk.exe 有短边偏差（2026-10-30 21:00）

**锚点是什么**：`elo.json` 的 `anchor_name = gen_010`，它的 `elo/lo/hi` 全是 `0.0`。
其它所有点的数字都是**相对 gen_010 的差值**，不是绝对棋力。gen_010 自己有 **708 局**
（几乎每一代都与它打过 12 局），所以大多数数字本质上就是「相对 gen_010 的直接胜率换算」。

**发现的偏差**：gen_060 补测到 36 局后（含对锚点 gen_010 的 12 局，5 胜 7 负），
用同一份边做标准 Bradley-Terry（MM 迭代）拟合，结果与 gk.exe 的输出对不上：

| 代 | 对 gen_010 战绩 | gk.exe 旧值 | 标准 BT 拟合 | 差 |
|---|---|---|---|---|
| gen_058 及以前 | 7/12 等 | 112.2 | 111.9 | −0.3（**差 <8，正常**）|
| gen_059 | 3/4 | 191.8 | 74.2 | **−117.6** |
| gen_060 | **1/6** | 402.8 | 107.6 | **−295.2** |
| gen_061 | 3/4 | 219.8 | 53.8 | **−166.0** |

**只有最后三代表现异常，共同点是它们与锚点的边只有 4 或 6 局**（正常是 12 局）。
gen_060 对锚点 **1 胜 6 负** 却被算成全图最高分 402.8 —— 这在 BT 模型下不可能成立，
说明 gk.exe 对**不足 12 局的短边有偏置**（疑似最小局数门槛或权重处理）。中段拟合完全正常（差 <8 Elo），
所以不是 gk.exe 整体有问题，是短边的问题。

**处置**：把 60 代全部按 12 局/边重跑 `gk.exe elo`（1422–1431 s），让所有边同口径。结果：

| 代 | 旧 | 新 |
|---|---|---|
| gen_059 | 191.8 | 117.2 |
| gen_060 | **402.8** | **123.5** |
| gen_061 | 219.8 | 128.2 |

gen_060 从第 1 名（402.8）落到 123.5，虚高消除，与它 17 胜 19 负的真实战绩一致。

**教训**：给某一代补测时，**必须同时保证它与锚点的边也打满 12 局**，否则短边会被算歪；
补完不要用手工合并 + 自行拟合（会与 gk.exe 口径不一致），直接全量重跑最稳。

**文件**：`refit_elo.py`（标准 BT + bootstrap 重拟合，可独立验证 gk.exe 的结果）、
`merge_gen060_edges.py`（把补测局数并进 edges，注意 `wins_a` 是 **a 方**胜场，必须翻视角）、
`elo_manual_merge_backup.json`（手工合并版，保留备查）、`elo_new.json`（全量重跑产物）。

## 7. 看板集成（Python 侧）

**踩到的坑（看板改了不生效，浪费了一圈排查）**：
`progress_live.html` 是 `watch_progress.ps1` 生成的**静态产物**，而 `Flash UI.cmd` 会拉起一个**常驻的
`pwsh watch_progress.ps1 -Html progress_live.html -Interval 5`**，每 5 秒整份重写一次。
只改脚本、不重启那个常驻进程 ⇒ 手动生成的新页面在 5 秒内被旧进程覆盖回旧版本，
现象是「源码里有新代码、页面里没有」，极易误判为自己改错。

**正确做法**：改完 `watch_progress.ps1` 必须
①`Stop-Process` 杀掉常驻 watch 进程 → ②重新拉起（或用 `Flash UI.cmd -Restart`）→ ③确认
`Get-CimInstance Win32_Process | ? CommandLine -match 'watch_progress\.ps1'` **只有一个**。
多个 watch 进程会互踩（实测同时出现 2 个，页面在两种版本间跳）。

## 7. 看板集成（Python 侧）

用户要求「在实时看板上增加 elo 对战功能，方便我看 elo 提升没有」⇒ 在「C++ 棋力曲线」卡下方
加了对战面板（选最近 N 代 + 每对局数 + 锚点 → 后台 `gk.exe elo` → 进度条 → 跑完报「比上次 +X Elo」）。
实现分三处：`PYTHON\serve_dashboard.py` 的 `elo_plan/elo_start/elo_status/elo_stop`（后台任务 + 状态落盘）、
`PYTHON\watch_progress.ps1` 的面板 HTML 与 `eloPaintJob/eloPaint`（渲染）、`checkpoints\history\elo_job.json`（状态）。

**踩到的坑（事故级，必须记住）**：`gk.exe elo --gens a,b,c` **只写被点名的这几代**。
第一版直接 `os.replace(elo.json.new, elo.json)` ⇒ 用户原有 40 代（gen 2–43）的完整曲线，
被一次「最近 5 代 + 锚点」的快速对比**碾成 3 个点**，且备份也是在第一次覆盖之后才拍的，当时救不回来。
（最后靠 60 个 `gen_*.bin` 全量重跑 1431 s 才恢复。）
修法：`_elo_merge(old, new)` —— 旧曲线里本次没测的版本原样保留、测过的用新值覆盖、`edges` 按 (a,b) 取并集；
**锚点变了就拒绝合并**（两边 0 Elo 不在同一点，硬拼会画出假跳变），提示改用命令行全量重跑。
替换前另存 `elo.json.bak`。回归测试 `PYTHON\_tmp\check_elo_merge.py`（18 项，覆盖"只测 5 代不能削短 42 代"）。

**第二个坑**：`elo_status()` 是白名单式构造返回字典，新加的 `merge_note / merged_versions`
一开始没透传 ⇒ 后端合并对了、状态文件也写了，页面却看不到，用户会以为整条曲线都重算了。
教训：给 `elo_status()` 的结果字典加字段时，**两边都要改**。

**恢复用命令**（全量重建，约 24 分钟）：

```
CD C++\build
set PATH=<python -c "import torch,os;print(os.path.dirname(torch.__file__)+'\lib')">;%PATH%
gk.exe elo --dir ..\model\run_v2 --gens 2,3,...,61 --anchor 10 --eval-games 12 --sims 400 --bootstrap 200 --out ..\model\run_v2\elo.json
```

`--eval-games 12`、锚点 gen_10 与原曲线一致（原曲线 40 版本 / 78 边是同一口径）。
重建后 60 个版本 / 118 条边，`gen_010` 的 games=732（每代都回指锚点，所以锚点的对局数远大于其它点）。

**还没做的**：`_tmp\check_elo_recipe.mjs` 原先硬编码「gen 44 超出 Elo 范围」，
已改成按 `elo.json` 实际覆盖范围分情况断言（`has44` 分支），以后补数据不用再改测试。

## 7. 看板集成（Python 侧）

- 看板：`http://127.0.0.1:8791`（`PYTHON\serve_dashboard.py`）
- 版本切换：`#verBar`（Python / C++），**C++ 面板现在 7 张卡**：概览 / 损失曲线（ce·mse·配方断点）/
  **棋力曲线（Elo）** / 逐代表 / 各 run 训练量 / champion / stdout 尾巴
- 训练引擎下拉 `#p_Engine`：`python` → `train.py`；`cpp` → `gk.exe loop`
  - C++ 参数来自 `gk.exe defaults`（即 `config.h`），启动前走 dry-run 拿服务端拼好的命令行再确认
  - **配方档案**下拉 + [填入配方]：读 `C++\peifang\*.json`，一键把整套参数回填（见 §6.1 与 §14.5）
  - C++ 运行目录是**下拉**（`#p_CppDirSel`，显示 `目录 · 角色 · 全局第几代`，来自各目录 `run.json`），默认选主线
  - **空目录保护**：目标目录没有 `gen_*.bin`/`best.bin` 时服务端**拒绝启动**（`error_kind: "scratch"`），避免拿随机权重从零开训
- 对弈页：`web_play.py`（8770），通过 iframe 嵌进看板；引擎可选 C++ / Python，C++ 走 `gk.exe play` 常驻进程（行协议 `GO/MOVE/INFO/QUIT`），树复用后单步 30 ms
- 实时同步：`GET /api/stream`（SSE，chunked 手写），页面 `EventSource` + 120 ms 防抖 + 原地 patch（`patchNode/patchInto`），不再整页刷新

**10-28 新增/修好的看板行为**（细节见 §14.5）：损失曲线 x 轴按**代数**（原来按行号，断代被抹平）+ 配方切换竖虚线；
「自对弈累计」用小时；逐代表 14/14 列对齐 + 表头 tooltip + 五列释义块；新增「C++ 棋力曲线」卡（读 `elo.json`）；
训练控制新增「配方档案」下拉 + [填入配方]。
另外 `Get-TrainRecipe` 现在能认 `gk.exe loop`——之前只认 `train.py`/`auto_train.ps1`，于是永远回落到
`checkpoints\last_train_params.json`（Python V4 的 160/320），这就是「看板一直显示 160 局」的根因。

**重启看板**（改过 `watch_progress.ps1` 模板后必须做，模板是启动时解析的）：

```powershell
pwsh -NoLogo -NoProfile -File <repo>\PYTHON\_restart_dash.ps1
```

脚本会：按**端口**找 8791 进程杀掉 → 杀掉命令行含 `watch_progress.ps1` 的 pwsh → 重启服务 → 用 `-Html progress_live.html -Interval 3 -MaxMinutes 1440` 重启 watcher（**不带 `-Html` 就只打印控制台、不写页面**）→ 自检 `/api/cpp-state` 与页面关键字。

## 8. 踩过的坑（按"再遇到会不会卡很久"排序）

1. **`setvbuf(stdout, nullptr, _IOLBF, 0)` 在 MSVC 上直接终止进程**（退出码 `0xC0000409`，一个字都不打印，看起来像 exe 坏了）。本会话为了让看板日志实时刷新踩了一次，已改成 `_IONBF`。
2. **`NetService` 拆回请求时策略与价值共用游标 `soff`** ⇒ 从第二个请求起越界读 value（不崩时在悄悄降质）。ASan 才抓到（`batcher.cpp:151`）。修法：策略/价值各一个游标（`soff`/`voff`）。
3. **`cv_.wait_until` 的 Windows 等待粒度 ≈15.6 ms** ⇒ 每批凭空多 12–15 ms，吞吐掉到 1/5。改自旋 `yield()`。
4. **`MCTS::seek` 只按深度回退**：目标节点在另一分支时会停在同一深度的**别的局面**上。改成按 parent 链求最长公共前缀再重放。连带 `set_root_move` 必须重置 `board_`/`placed_`，否则提升新根后整棵树从错局面展开（症状：策略目标全 0）。
5. **`n_sim` 必须远大于 `batch_mcts`**：`n_sim=8, batch-mcts=1` 时策略目标退化，CE 卡在 `log(361)=5.8889`。`cmd_loop` 已加警告。
6. **`Pos::place()` 返回"是否成五"而不是"是否成功"**；`Pos::p[]` 是 27×27 填充数组（`pad_index(x,y)=(y+4)*27+(x+4)`），必须用 `pos.at(x,y)`。
   搭配的坑：**`check_win_after()` 只对「空点」有效**（函数体第一句就是 `if (p[base] != 0) return false;`），
   不能用来判「盘上已有一颗 who 的棋子是否构成五连」。写成 `p[..]==who && check_win_after(..)` 会得到一个
   **恒 false 的死条件**，不报错、不告警，只是那段逻辑永远不执行。已在两处测试里踩到，见 §17.1。
7. **`torch::nn::utils::clip_grad_norm_` 直接返回 double**（裁剪前范数），别写 `.item<double>()`。
8. **bf16 autocast 必须配 `at::autocast::set_autocast_cache_enabled(false)`**：`_foreach_*` 就地更新不 bump 版本号，否则缓存返回旧 bf16 权重，训练缓慢劣化（历史上跑坏过 `broken_amp_do_not_use`）。
9. **pwsh 不支持 heredoc**（`python - <<'PY'` 会 ParserError）；改代码一律用 edit/write 工具，别用 shell 内联。
10. **bash 起不了 MSVC 的 exe**（rc=127 无输出）；`pwsh` 里 `cd` 不改变后续工具调用的相对路径基准。
11. **跑 `build\*.exe` 前 PATH 必须含 torch 的 `lib` 目录**，否则静默失败（退出码 `-1073741515` = DLL 找不到）。前缀：`$env:PATH = "<TORCHROOT>\lib;" + $env:PATH`（`<TORCHROOT>` = `python -c "import torch,os;print(os.path.dirname(torch.__file__))"` 的输出）
12. **CUDA Graph 必须在非默认流上捕获**，否则抛 `CUDA graphs must be captured on a non-default stream`。而 `NetService` 的多评估线程跑在默认流上 ⇒ 用 `c10::cuda::CUDAStreamGuard` 明确罩住 `capture_begin`/`capture_end` 整段。
13. **后台跑 PowerShell 脚本要自己 `Set-Location`**：`bash` 工具的 `run_in_background: true` 起的是新 shell，**不继承 workdir**，`& .\build\gk.exe` 会报「术语不会被识别为 cmdlet」。脚本里写死绝对路径。
14. **`web_fetch` 抓 github.com / raw.githubusercontent.com 一律 `TypeError: fetch failed`**；`api.github.com` 无 token 时 403 限流。取源码用 `Invoke-WebRequest -UseBasicParsing -Headers @{'User-Agent'='Mozilla/5.0'}` 抓 HTML，或直接下 `https://codeload.github.com/<owner>/<repo>/zip/refs/heads/master`。
15. **⚠ 最贵的一个：`watch_progress.ps1` 每轮往 `$env:PATH` 里追加 torch 的 `lib` 目录**（`$env:PATH = "$torchLib;$env:PATH"` 写在渲染函数里，每 3 秒执行一次）⇒ 进程环境块每次 +91 字节，十几分钟撑到 Windows 的 32 KB 上限 ⇒ 之后**每一次** `CreateProcess` 都失败，症状是满屏
    「python.exe - Application Error／应用程序无法正常启动(0xc0000017)」弹窗、`gk.exe` 起不来、看板按钮全部失效。
    修法：加 `$script:TorchPathAdded` 幂等标志 + 只在**真的不在 PATH 里**时才拼。**任何每轮渲染的代码都不要改环境变量/改全局状态。**
16. **`gk.exe` 往 stdout 写的是 UTF-8，而本机 `[Console]::OutputEncoding` 是 CP936** ⇒ 把输出 Tee 到日志文件后中文全是乱码
    （`杩涚▼`=进程）。在启动脚本里先 `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false);
    $OutputEncoding = same`，再 `Start-Process`（**不要**用 `-RedirectStandardOutput` 再自己转码）。
17. **CUDA Graph 是「每代 OOM」的头号嫌疑**：`GraphRunner` 每代权重版本一变就 `release_all()` 再重新捕获，
    每张图一个私有显存池（12 张图），而本机是 **12 GB 卡跑 12 GB 峰值**（self-play 期间 12.0–12.8 GB）⇒
    `gen 44` 一启动就 `memory allocation failed with OOM on device 0 ... free: 0, total: 12,792,037,376`。
    已改默认关闭（`GK_GRAPH=1` 才开）。若将来要重开，先做两件事：`release_all()` 前 `cudaDeviceSynchronize()` +
    `emptyCache()`，并且**不要**每代重建图。
18. **别用「命令行里含 watch_progress.ps1」去杀进程**：你自己的 pwsh 命令行里也含这个字符串，会把自己杀掉
    （症状：`exit 4294967295`、后续命令全跑不动）。用 `-and $_.ProcessId -ne $PID` 排除自己，或干脆按端口/进程名找。
19. **`build.bat` 在 `gk.exe` 正在运行时会失败**（`LNK1104 无法打开文件 gk.exe`）：先停训练/对弈页进程再编译。
20. **PowerShell 的 `OrderedDictionary` 没有 `ContainsKey()`**（那是 .NET 的 `Dictionary`）⇒ 调用它会抛
    「数组索引的计算结果为 null」，看起来像 JSON 解析坏了。用 `.Contains($k)`。
21. **`last_train_params.json` 是会骗人的历史文件**：它记录的是上一次**Python** run（V4 的 160/320 +
    一个不存在的 `--buffer-decay-half-life`）。看板"从上次参数填充"若回落到它，就会把 C++ 也拉回 160/320。
    现在 `Get-TrainRecipe` 优先解析 `gk.exe` 命令行，并且不认识的参数一律不显示（但**不会**替你纠正目录里的历史文件）。
22. **想"从某一代"续训，必须先把更新的 `gen_*.bin` 移出目录**：`pipeline.cpp:331-341` 只看 `find_last_gen()`
    取编号最大的那个覆盖权重，`--init-weights` 仅在目录里没有 `gen_*.bin` 时才生效。目录里还留着坏代的权重时，
    无论命令行怎么写都会载入最差的那一代，而日志只有一行「[续训] 载入 gen_XXX」，极容易看漏。详见 §16.4-1。
23. **`best.bin` 是哪一代只能哈希对，别信时间戳**：`md5sum gen_*.bin | awk '$1=="<best 的 md5>"{print $2}'`。
    本次靠它才发现 best 冻在 gen_074（配方 5 六代全被拒），而时间戳指向 gen_081。见 §16.4-2。
24. **沙箱里编 mingw 独立测试：`$env:TMP` 必须指到项目内可写目录**（否则 `Cannot create temporary file`），
    运行时 `tools\mingw64\bin` 必须在 `PATH` 里（否则 `0xC0000135` DLL not found、且不打印任何东西）。
    完整命令见 §16.2。另外 `gk.exe` 里的 MSVC 编译不受影响——两套工具链各自独立。
25. **⚠ `build\` 只认 `.cpp` 的修改时间，改头文件不会重编**——`ninja` 会回「no work to do」，
    而产物其实是旧的。实测（2026-10-31）：`config.h` 12:47 改动后跑 `build.bat`，`vcf.cpp.obj` 重编了、
    `gk.exe` 重新链接（12:50:24），但 `main.cpp.obj` 仍是 03:56:07、`pipeline.cpp.obj` 仍是 00:32:59，
    而**这两个文件都 `#include "config.h"`**。原因：`build.ninja`（10-29 生成）里 `deps =` 出现 **0 次**，
    也没有 `/showIncludes` / `dyndep` / `cmake_ninja_depends` ⇒ 头文件依赖根本没被记录。
    危害：改了 `config.h` / `mcts.h` / `gate.h` 里的**值**（不是注释）却只跑一遍 `build.bat`，
    会得到一个"看着编译成功、其实还是旧逻辑"的 `gk.exe`，而且没有任何报错。
    **做法：只要动过头文件，就 `build.bat clean` 之后再 `build.bat`；或者编完核对
    `build\CMakeFiles\gk_core.dir\src\*.obj` 的时间戳是否都新于被改的头文件。**
    2026-10-31 03:55 那次之所以是全量重编（所有 obj 都在 03:55:58–03:56:07），是因为之前做过 clean。
26. **⚠ Windows PowerShell 5.1 下，`$ErrorActionPreference = 'Stop'` + 外部命令 `*>&1` 会抛 `NativeCommandError`**：
    5.1 会把外部程序的 stderr 包成 `ErrorRecord`，配 `Stop` 直接终止脚本（PS7 才没这问题）。
    症状极隐蔽——`gk.exe` 只要往 stderr 写一个字，整个 A/B 就在中途死掉，而终端看起来像"跑完了"。
    **跑外部程序一律用 `Continue` + 显式检查 `$LASTEXITCODE`。** 另：本机交互式控制台是 **5.1 不是 pwsh 7**，
    §8-16 的编码修法要用 5.1 验证。见 §17.4-4/5。

## 9. 当前模型状态（`C++\model`）

```
run_v2\            主线，**gen_083**（2026-10-31 14:37 落盘，配方 4）。
                   `best.bin` 内容 = **gen_082**（md5 核对，不是 mtime 最新的那一代，见 §16.4-2）。
                   门控记录：gen 78 = 34/60 采纳、gen 82 = 33/60 采纳、gen 76/80 被拒（0.43 / 0.47）；
                   gen 83 是门控空档代（`--eval-every 2`）。
                   配方沿革：gen 1–4、8–43 = 配方 1；gen 5–7 = 事故的 160/320；gen 44–65 = 配方 3；
                   gen 66–83 = 配方 4（`--min-steps 800`）。
                   ⚠ 这里的 gen 76–81 是**配方 4 重训出来的**权重，和 2026-10-31 凌晨那批被否决的
                   配方 5 同名代**不是同一份东西**（那批已删，见下方 `_rejected_recipe5` 与 §16.3）。
                   gen_001…gen_083.bin、buffer\（26 代）、train_log.jsonl（83 行，gen 1–83 无重复编号）、
                   `train_log_rejected_recipe5.jsonl`（配方 5 那 6 行的旁路存档）、elo.json（停在 gen 61）、loop_stdout.log。
                   `_bad160\` 里躺着 10-27 回滚出来的 gen_011/012 + 完整日志备份。
~~model\_rejected_recipe5\~~  **2026-10-31 曾归档、同日已删除**：配方 5 的 gen_076–081 六个权重（41.7 MB，
                   被门控全拒、定点对打 −88.7 Elo）。它们唯一的用途是复现那个负结果，而 400 局原始日志
                   （`tsest\g81v75_s1.log` / `s2.log`）与 §16.3 的记录已足够，权重本身不含额外信息。
                   删前先把它们移出 `run_v2`，是为了让 `find_last_gen()` 回到 75（见 §16.4-1）。
run_v4amp\         历史（旧配方 7 代无进步），run_v2 的起点。**2026-10-27 已瘦身**：
                   gen_001–006.bin / best.bin / buffer\ 已删（~192 MB），只留 gen_007.bin —
                   它同时是「+190 Elo」里那个起点，删了就没法再复现 30:9。
run_fp32\          实验（fp32 对照）——**2026-10-27 已删**（不产出模型；速度/精度数字在本文件 §5）
broken_amp_do_not_use\  损坏（早期 bf16 缓存 bug）——**2026-10-27 已删**（结论在 §6 第 8 条）
```

**2026-10-27 清理记录**（用户：「没用的模型可以删」）：C++ 侧删 304 MB（上面三项）；
PYTHON\models 删 1,831 MB —— 只留各 run 的 `best.pt` / `resume.pt` / 有引用价值的单代权重：
`V3/gen_110.pt`（**champion**，SHA1 `4D799CB0…`，删前删后都核过）、`V3/gen_30.pt`（`web_play.py` 的 `DEFAULT_MODEL`）、
`V2/gen_150.pt`（`web_play` 回落）+ `V2/gen_175.pt`（`V3/INIT.json` 的 `weights_from`）、`V4/gen_37.pt`（§4 数值对齐的参照）。
`run3` 的 60 代权重、`exp\`、以及 V2/V3 的 `data\*.npz` 分片一并删掉。

每个目录都有 `run.json`（role/note/gen_offset/recipe），`gk.exe loop` 第一次进目录时自动写、已存在不覆盖。

## 10. 下一步该干什么（按价值排序）

### A. 用配方 3 续训 10–20 代，检验「平台期是测量问题」（第一优先）

前提：43 代的 +134 Elo（全期 ≈+3.1/代，后期 ≈+1.3/代）与 20 局门控 ±100 Elo 的分辨率对不上，所以真正要验的不是"模型还行不行"，
而是"配方 3 之后能不能看见进步"。看两个数：

1. 门控通过率是否从 20% 抬到 ~50%（`--eval-games 60 --opening-plies 2` 之后）；
2. 跑一次 `gk elo --anchor 10` 重新拟合，**每代增益是否还在 +1.3 ~ +3 Elo 量级**。

- 若两者都改善 ⇒ 继续跑，别再动旋钮。
- 若 elo 也平坦 ⇒ 才是真到顶，转去改容量/特征/开局多样性（`--blocks/--channels`、开局库），**不要**继续调 lr。
- 启动方式：看板训练控制 → 引擎 C++ → 「配方档案」选**配方3** → [填入配方] → [开始训练]（会先干跑给你确认命令行）。

### B. 查清「并行形态」为什么差 3 倍（第二优先，唯一还有明显收益的工程项）

**现象**（下面两个数不是同一套 harness，只算强线索，不算结论）：

| 形态 | 吞吐 |
|---|---|
| `gk probe`：20 线程共用 1 个 NetService（单进程多线程） | **63.77 手/s** |
| 生产：8 进程 × 1 worker（各自独立 CUDA context） | **24.57 手/s** |
| 单个 worker 进程单独跑 | 7.62 手/s |

8 进程合计 24.57，而 8 × 单进程 = 61 手/s ⇒ 多进程效率只有 40%。更关键的是 **8 进程形态下 GPU util 只有 10–13%**（单 worker 也是 10–13%），即大部分时间 GPU 在空转、等 CPU/调度；而 probe 16 线程时 util 高达 97%。**同一个 GPU，一个形态能打满、另一个打不满**——这比「谁快」更值得追。

这与第 5 节「`--procs` 更慢、每进程 batch 只有 16 被切碎」的旧结论方向一致，但差 3 倍的量级还没解释清。
（10-28 的推理路径实测把上限钉得更死了：单流 GPU 峰值 ≈12k 状态/s，20 线程生产形态 8.2k = 68%，
再加评估线程只会更差（e6 比 e3 低 5.9%）——所以这条路的收益上限约 1.4×，且瓶颈是"在飞局数只有 16"，
不是 CUDA 侧。见 `C++\README.md` §4 与第 14 节。）

要做的实验：写死总工作量（16 局 × 600 sims），分别用 `gk selfplay --workers 16/12/8`（单进程多线程）和 N 个 `gk worker --workers 1`（多进程）跑同一份量，比墙钟。脚本草稿在 `C++\_tmp\ab_shape.ps1`（**从未成功执行过**，跑之前先看它的 PATH 前缀）。

若确认单进程多线程快得多，就把 `config.h` 的 `mp_procs` 彻底废弃、并检查 `run_parallel` 这整条多进程路径。

### C. 配方实验（便宜，但需要 GPU 空闲时间）

- `n_sim 600 → 400`：一代省约 1/3 时间。V4 时代用 400，需要门控/对局验证是否掉强度。
- `workers 20 / eval-batch 24 / eval-threads 3`：实测 50.8 vs 46 手/s，白捡 8%（已进配方 3）。
- 两套配方的干净对照（160/320 vs 320/160，**只变 games/steps，其他全固定**）才能回答"样本 1 万 vs 2 万"到底差多少；现有证据被 lr 和 gate-ema 混淆过。

### D. 别再碰的方向

- CUDA Graph：**已做完（第 12 节），但 10-28 已改默认关闭**（−5.9%/噪声内，却是每代 OOM 的头号嫌疑，见 §14.1）。
  要重开先修 `release_all()` 前后的同步 + 不要每代重建图。
- VCF 静态搜索 / 强制着法不剪枝：**已实现并实测，两项都没有收益，已默认关闭，见第 13 节**。

### E. 收尾杂项

- 用浏览器刷新一次看板，确认「配方档案」下拉 + [填入配方] + 棋力曲线卡真的显示（我只在无头渲染 + DOM 桩里验过）
- `PYTHON\progress_live.html` 是 `watch_progress.ps1` 的产物，别手改
- 陈留文件：`model\run_v2_run.log`、`PYTHON\_restart_dash.ps1`/`_restart_train.ps1`、`PYTHON\_tmp\supervise_loop.ps1`（运维脚本，建议保留）
- `C++\_tmp\` 已于 2026-10-26 15:30 清理（393 MB → 11.5 MB）。**剩下的 8 项不要删**，它们被代码或文档引用：`v4_gen37.bin`、`ref\`（`test_net.exe` 的基准）、`aug_ref.txt`（`test_dataset.exe` 的 numpy 参照）、`ab_shape.ps1`（第 10 节 B 的实验脚本）、`ab_mp.ps1`、`ab_rush_400.txt` / `ab_vcf_400.txt`（第 13 节的原始数据）、`rapfi_src\`（Rapfi 源码，第 13 节引用）。
- `src\net_torch.cpp` 里 `atexit` 相关的调试残留可以清掉（配合第 12 节的析构告警）

## 11. 千万别做的事

- 别把 `--procs` 当加速手段（实测更慢，见第 5 节与第 10 节 A）
- 别用「单局叶批量 16→64」当加速手段（Python 时代已判死：+59% 吞吐但同权重对照弱 ~240 Elo）
- 别拿 loss/mse 判断棋力
- 别在 `model/` 下用空目录开训（会随机权重起步）；面板已拦，CLI 靠 `[提示]` 文本
- 别直接改 `progress_live.html`（会被 watcher 覆盖）
- **别再用 CUDA Graph 固定桶**（`{16,24,32}` 那种把 batch 补齐的写法）：实测在 20 线程下反而慢 32%
- **别指望 VCF / 强制着法不剪枝能提强度**：400 局实测都没收益，VCF 还显著变弱。想重开先读完第 13 节
- **别用 20 局门控回答"有没有进步"**：分辨率 ≈ ±100 Elo，+25 Elo 的真进步只有 41% 概率被看见。
  要判趋势就用 `gk elo`（多版本联合拟合 + CI）或把 `--eval-games` 抬到 60/200（见第 14.1 节）
- **别在 `--opening-plies 0` 下读门控胜率**：19 路无禁手，执黑 71% / 执白 36%，20 局里黑白怎么分
  就能造出 ±17pp 的假信号。门控要比棋力就开配对开局
- **别把"Elo 区间重叠"当"变强了"**：`elo.json` 的 CI 现在普遍 ±100，两代区间重叠意味着没测出差别，
  不是"略强"
- **别在训练跑着的时候 `build.bat`**：`gk.exe` 被占用会 `LNK1104`
- **别在每轮渲染的代码里改 `$env:PATH` / 全局状态**（见第 8 节第 15 条，代价是两台机器的整整一下午）

## 12. 本期工作记录：CUDA Graph 化推理

**结论：做完并验证，生产形态 +10%（22.31 → 24.57 手/s）。但 2026-10-28 已改成默认关闭**
（复测 −5.9%，落在噪声内；而它是"每代边界 OOM"的头号嫌疑 —— 每代 `release_all()` + 重新捕获、
每张图一个私有显存池，12 GB 卡上跑到 12.8 GB。现在要 `GK_GRAPH=1` 才开，见第 14.1 节）。

实现（`src\net_torch.cpp`）：

- `class GraphRunner` 持有 `GraphBucket buckets_[kMaxGraphs]`（`kMaxGraphs=12`），每桶一张 `at::cuda::CUDAGraph` + 静态输入/输出缓冲。
- **按实际 batch 尺寸惰性捕获**，不补齐到固定桶。这是关键——第一版用固定桶 `{16,24,32}` 把 17–18 补齐到 24，20 线程下慢了 32%（55→37 手/s），因为高并发时 GPU 是瓶颈，多算的 FLOPs 吃掉了省下的 launch 开销。
- `kGraphMinBatch=8`（小于此值图的开销大于收益），`batch > kGraphMaxBatch` 走 eager；`n_ >= kMaxGraphs` 也回退 eager。
- 捕获必须罩在 `c10::cuda::CUDAStreamGuard` 里（见第 8 节第 12 条），否则抛 `must be captured on a non-default stream`。
- 环境变量：**默认关闭**；`GK_GRAPH=1` 开启（`GK_NO_GRAPH=1` 无条件关，优先级更高），`GK_GRAPH_DEBUG=1` 打日志。

实测（`model\run_v2\best.bin`，GPU 空闲）：

| 负载 | eager | graph | 增益 |
|---|---|---|---|
| probe 20 线程共用一个 NetService | 61.85 手/s | 63.77 手/s | +3%（噪声级） |
| probe 16 线程 | 45.53 手/s | 45.02 手/s | −1%（噪声级） |
| worker 单进程 1 worker × 600 sims | 7.32 手/s | 7.62 手/s | +4% |
| **生产形态：8 进程 × 1 worker** | 22.31 手/s | **24.57 手/s** | **+10%** |

读法：GPU 已被打满时（probe 16 线程 97% util）省 launch 开销没用；生产形态 GPU 只有 10–13% util，省 CPU 侧开销就有用。**这正是第 10 节 B 说瓶颈在并行形态而不是 kernel 的旁证。**

> **2026-10-28 复测（现行形态：单进程 20 线程，`procs=0`）**：开图 47.73 手/s（11,790 状态/s）
> vs 关图 44.92 手/s（11,097 状态/s）→ −5.9%；**重建后** 默认(关) 46.97 vs `GK_GRAPH=1` 46.61，
> 差值落在噪声内（±5%）。考虑到它是每代边界 OOM 的头号嫌疑（第 14.1 节），已改成默认关闭。
> 上表"生产形态 8 进程"那一行是 10-26 的旧形态，现在不再使用 `--procs`。

残留问题：进程退出时 stderr 有一串 `device is busy or unavailable (function reset)`（`CUDAGraph.cpp:404`），来自析构期重置图。曾用 `atexit` 尝试提前释放，**无效**——告警来自 `reset()` 本身。纯噪声，不影响结果与产物。

## 13. 本期工作记录：借鉴 Rapfi 的两项改动（均实测无收益）

评估对象：`https://github.com/dotb1ue/rapfi`（开源五子棋/连珠引擎，C++17 + alpha-beta + NNUE，**GPL v3**，与本项目许可不兼容，只能读思路不能拷代码）。源码解压在 `C++\_tmp\rapfi_src\`。

### 13.1 VCF 静态搜索（新增 `src\vcf.*`）

**思路**（抄自 `Rapfi\search\mcts\search.cpp:36-233` 的 `namespace SimpleVCF`）：在叶子交给网络之前，用「只走冲四」的廉价穷举证明「当前行棋方有必杀」，命中就直接给 ±1，不花 GPU。Rapfi 把它接在 `evaluateNode<Root>()` 里、网络评估之前。

实现：

- `src\vcf.h`：`struct VCFOptions { int max_ply = 6; int max_nodes = 128; };`，`int vcf_win(const Pos&, const VCFOptions& = {})`。返回 1=必胜，0=**未证明**（不代表必败）。不修改入参。
- `src\vcf.cpp`（243 行）：5 格窗口统计 `ca==4 && |E|==1`（攻方成五点）/ `cd==4 && |E|==1`（守方成五点）/ `ca==3 && |E|==2`（冲四点）；落子后数新成五点，`nf>=2` 双威胁直接胜、`nf==1` 逼挡后递归。廉价预筛 `att_stones < 3 → return 0`。**只做正向 VCF**（与 Rapfi 一致）。
- 接入点 `src\mcts.cpp:385-395` 的 `terminal_value()`，位于 `immediate_win_loss` 两个判定之后，**且带 `node != root_` 条件**（根节点短路会让策略目标全零）。
- 命中计数 `MCTSOptions::vcf_hits` → `MCTSStats::vcf_hits` → `main.cpp:210,227` 打印。

正确性：`tests\test_vcf.cpp`（365 行）含与朴素 `brute_vcf` 的**双向**随机交叉验证（300 次随机生成 → 155 个有效局面，其中 brute 报胜 33 个，**0 不一致**），另覆盖空盘/子数不足/活三/守方活四在先/单端空四等定点局面，全部通过。

开销：约 `3.5–5.3 µs/call`（不同机器状态下波动；quiet 与 busy 局面几乎相同——成本在整盘 5 格窗口扫描，不在搜索深度）。

**实测（400 局同网络自战，400 sims，seed 4242）：**

| 对照 | 结果 | 胜率 | Wilson | 判定 |
|---|---|---|---|---|
| A 双方关 VCF | 190胜201负9平 | 0.475 | [0.427, 0.524] | 拒绝 |
| B 候选开 VCF | 156胜231负13平 | **0.390** | [0.343, 0.439] | **显著变弱** |

**为什么先证明测试有分辨力**：同一套对局跑「候选 1600 sims vs 对手 400 sims」，100 局得到 0.760（Wilson [0.668, 0.833]）。所以上面两项的负结果不是统计功效不足，是真没收益。

**变弱的根因没查清**。怀疑方向（**都未验证**）：① `v=1.0` 是「对手最优」的确定性结论，混进 MCTS 的平均 q 语义后扭曲访问分配；② `mcts.cpp:433` 把 `tv` 写进以 hash 为键的 `value_cache_`，若 `seek` 的 hash 与实际局面有偏差会污染。

### 13.2 强制着法不剪枝（`guarantee_rush`）

**先纠正一个我最初说错的判断**：`child_priors`（`mcts.cpp:198-274`）**本来就**会无条件保留「一步成五」和「堵五」的点，所以强度清单 #2 描述的缺口并不存在。真正的缺口是 tier-1：「我方能做四（`c_me==4`）/ 对方能成四（`c_op==4`）」只靠分数（1e5/1e4）排进前 6，没有保证。

实现：`PriorBuf` 加 `uint8_t tier`（3=成五 / 2=堵五 / 1=四的争夺 / 0=其余），`child_priors` 里 tier≥1 必进候选。

**实测（400 局，seed 4242）：**

| 对照 | 结果 | 胜率 | Wilson | 判定 |
|---|---|---|---|---|
| A 双方都不保护 | 194胜182负24平 | 0.485 | [0.436, 0.534] | 拒绝 |
| B 候选保护 | 180胜208负12平 | 0.450 | [0.402, 0.499] | 无收益 |

推断的机制（未单独验证）：把非根宽度 6 撑到 10，模拟预算不变 ⇒ 每个分支访问数被稀释 40%，等效降低搜索深度。

### 13.3 当前状态：两个开关默认关闭

`config.h:42,45`、`mcts.h:57,62`、`gate.h:35,38,41,42`、`selfplay.h:32,35` 全部改为 `false`，理由以注释形式写在原地。

CLI 现在是**双向**的（原来只有关闭选项，默认改 false 后就再也开不起来了，所以补了正向开关）：

```
默认                          → 关闭
--vcf / --guarantee-rush      → 开启
--no-vcf / --no-guarantee-rush → 关闭（显式覆盖，便于 A/B）
```

gate 侧另有 `--cand-vcf` / `--cand-no-vcf` / `--cand-guarantee-rush` / `--cand-no-guarantee-rush`，让同一个网络只换搜索配置做强度 A/B；`--cand-sims N` 用于校准测试的分辨力。

8 套测试（`test_rules/test_vcf/test_dataset/test_mcts/test_selfplay/test_gate/test_net/test_train`）全部通过。

### 13.4 Rapfi 清单里还没做的（按价值）

Rapfi 值得借鉴的东西远不止这两项。按对**我们当前瓶颈**的相关性排序：

1. **并行形态**（不是 Rapfi 的东西，但同一次调研暴露出来）：见第 10 节 A。
2. **「活四捷径 bug」已被我核实为不成立，别再追**。我在调研中一度报告 `expand()`（`mcts.cpp:296-341`）漏了「对手是否已有一手成五」的检查，这是误报：`terminal_value()`（`mcts.cpp:365-381`）在叶子进入 `expand` **之前**就先判 `node_win_moves()`（我方一步成五 → +1）与 `opponent_has_immediate_win()`（对手一步成五 → −1），命中即短路返回，`flush()` 根本不会把这种叶子送进 `expand`。所以 `expand` 只在双方都没有一步成五的安静局面上跑，此时活四确实是必胜。

   唯一前提是 `opt_.immediate_win_loss == true`（默认，`mcts.h:54`）。**若有人把它关掉，活四捷径就变得不可靠**——真要动这个开关，得同时给 `expand()` 补对手成五判定。

   顺带核实：`Node::value`（`mcts.h`）是**死字段**，全仓库只有写（`mcts.cpp:120,303,330,334`）没有读，活四捷径的实际效果完全来自 `is_terminal=1, expanded=1`（选择时不再下钻）。这个字段可以删，删了不改变行为。
3. **LCB 选点**（Rapfi 强度 #4）：按方差下界选最终着法而非最大访问数。我们的 400–600 sims 访问数很低，理论上收益明显。半天工作量。
4. **增量棋型数组**（Rapfi 速度 #1）：`move_scores` 现在每次对每个候选点现场扫 4 个方向；Rapfi 维护每格每方向 1 字节的 `Pattern2x` 增量更新。3–5 天，要先打点确认它真占大头。
5. **位棋盘 + Stencil 邻域扩张**（Rapfi 速度 #2，1 周+）、**融合查表 PCodeP4**（速度 #3）、**搜索图/置换表而非树**（强度 #4）、**棋型先验+NNUE 残差**（强度 #5）、**cpuct 随访问数变化**（强度 #6）。
6. **clang 对照编译**（Rapfi README 明确说 MSVC 慢不少，半天工作量，最便宜的一项）。

---

## 14. 本期工作记录：配方 3 + Elo 曲线 + 面板配方档案（2026-10-28）

日期：2026-10-28（会话时间 13:28–19:40）。触发问题：用户看板发现「连续拒绝了好多代」。

### 14.1 触发点：43 代的平台期（先把数字钉死）

| 事实 | 数字 |
|---|---|
| `run_v2` 已跑代数 | 43 代（含 10-27 回滚后从 gen_010 续起） |
| 门控采纳 | **只有 4 次**：gen 4（14/20）、gen 18（16/20）、gen 22（13/20）、gen 24（14/20） |
| `best.bin` | SHA1 前 12 位 `D06F8CF0F6B1` = **gen_024**，即 gen 25–43 连续 19 代没换过 |
| 最近 9 次门控 | 79/180 = 0.439 |
| 全部 20 次门控 | 198/400 = 0.495（≈ 掷硬币） |
| **200 局复核** gen_043 vs best | **107 胜 92 负 1 平 = 0.535**，Wilson [0.466, 0.603] ⇒ 约 +24 Elo，区间 −24…+43 ⇒ **拒绝** |
| 43 代累计（`gk elo`，锚点 gen_010） | **+134 Elo**〔17, 284〕（全期 ≈ +3.1 Elo/代，前期快） |
| 最近 19 代（200 局门控，gen_043 vs best） | **+24 Elo**〔−24, +43〕≈ +1.3 Elo/代 |
| 回放池 | 从 gen 29 起冻在 ~48.8 万样本（说明新数据与淘汰量收支平衡） |
| ce / mse | ce 1.286 → 1.245（缓降），mse 0.575 → 0.586（微升，与「新配方 mse 更高」的旧结论一致） |

诊断（四条，按贡献排序）：

1. **主因是测量分辨率，不是模型**：20 局门控的置信半宽约 ±100 Elo，而后期真实速度只有 +1.3 Elo/代。
   对「+25 Elo 的真差异」，20 局门控的检出率只有 **25%**（p=0.55 时 41%、p=0.60 时 60%、p=0.65 时 76%）。
   ⇒ 连续 19 代拒绝是这个量级差的**统计学必然**，不是"模型坏了"。
2. **lr 恒定 3e-4，一次没降**：旧的 `lr_decay` 只在门控胜率 < 0.30 时触发，43 代没触发过。
3. **回放池混 60 代旧策略**：每代只过 2.1 遍新数据，旧数据的策略目标早已过时。
4. **门控既没配对开局、又只打 20 局**：19 路自由五子棋先手优势实测**黑 71% / 白 36%**，颜色噪声比可测信号大；
   每代只更新 20 局 `eval.games`。

**最硬的一条证据（门控与 Elo 自相矛盾）**：`gen_026` 对 best（`gen_024`）的 20 局门控只赢
**6/20 = 0.30**，而同日 `gk elo` 给 `gen_026 = +286 [172, 461]`、`gen_024 = +65 [−40, 197]`
（差 220 Elo ⇒ 期望胜率 0.78）。两个数不可能同时对 ⇒ 20 局门控在这个强度区间测的是运气。
Elo 自己的 CI 也普遍 ±100（gen_043 = +134 [17, 284]、gen_034 = +263 [152, 405]、
gen_017 = +176 [61, 305]），**gen 25 之后没有任何一代能把自己的区间与 gen_017 分开**——
"19 连拒"与"其实在涨"都能被这份数据兼容，这正是要用配方 3 换分辨率的原因。

附带发现（同日修掉）：`gen 44` 启动即 **CUDA OOM**（`free: 0, total: 12,792,037,376`，即 11.9 GB 卡），
原因是每代边界 `GraphRunner::release_all()` + 重新捕获各留一个私有显存池；训练在 15:06:48 静默死掉、
监督器也没了。CUDA Graph 已在 14.2 里改成默认关闭。

### 14.2 引擎改动（`C++\src`）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `gate.h` | `GateOptions` 加 `int opening_plies = 0;` | 门控要能配对开局 |
| `gate.cpp` | 新增 `build_opening()`；`play_gate_game(..., const std::vector<int>* opening)`；偶数局候选持黑、`g/2` 决定开局种子 ⇒ 正反两局同一盘面只换黑白 | 扣掉先手优势（黑 71% / 白 36%） |
| `elo.h` / `elo.cpp` | **新文件**。`EloModel{name,path,gen,elo,lo,hi,games}`、`EloEdge{a,b,score,n}`、`solve_elo()`（最小二乘 + 岭正则 + Gauss-Jordan 消元）、`fit_elo()`（bootstrap 200 次出 95% CI）、`elo_json()` | 20 局门控测不出 +1.3 Elo/代的后期速度，需要一个能跨代读强度的量 |
| `config.h` | 加 `recipe_id = 0`、`lr_gen_decay = 1.0`、`lr_floor = 1e-4`、`gate_opening_plies = 2`；`defaults` 打印这四个字段 | 配方编号要进日志；退火与配对开局要可配 |
| `main.cpp` | 新增 `cmd_elo`；`apply_common` 接受 `--recipe/--lr-gen-decay/--lr-floor/--opening-plies/--gate-threshold`；`defaults` 输出新字段；`gate` 打印配对信息 | 把新能力暴露到 CLI 与看板 |
| `pipeline.cpp` | ① 每代把 `recipe_id` 写进 `train_log.jsonl` 每一行；② 按代退火：`gen ≥ 2` 时 `lr = max(lr * decay^(gen-1), floor)`，**门控拒绝不回退 lr**（否决的梯度不是反向证据）；③ 门控传 `opening_plies` | 配方 3 的三根支柱 |
| `net_torch.cpp` | **CUDA Graph 默认关闭**，`GK_GRAPH=1` 才开，`GK_NO_GRAPH=1` 仍可显式关 | 实测 −5.9%（重建后噪声内），但它是每代 OOM 的头号嫌疑 |
| `CMakeLists.txt` | `GK_SRC` 加 `src/elo.cpp` | 新文件必须显式登记（本项目不用 glob） |

### 14.3 新增 CLI（都在 `gk.exe` 上，已进看板白名单）

```
gk loop ... --recipe 3 --min-steps 320 --eval-games 60 --opening-plies 2 \
              --buffer-gens 30 --lr-gen-decay 0.97 --lr-floor 1e-4

gk gate --cand X.bin --best Y.bin --eval-games 60 --opening-plies 2 --gate-threshold 0.55

gk elo  --dir model/run_v2 --gens 1,4,10,24,34,43 --anchor 10 \
        --eval-games 24 --sims 400 [--bootstrap 200] [--no-ref-edge] [--out elo.json]
```

`gk elo` 的做法（KataGo 读法）：**相邻代链式对局**（`i` vs `i+1`）+ **每代回指锚点**（`i` vs `anchor`，
默认开、`--no-ref-edge` 关），把链解成绝对 Elo；锚点显式 = 0 Elo；每点带 bootstrap 95% CI。
产物 `model\<run>\elo.json`，看板「C++ 棋力曲线」卡直接读。

### 14.4 新增产物

- `C++\peifang\配方01-基线.json`：320 局 / 600 模拟 / 160 步 / lr 3e-4 / fresh 0.5 / 门控 20 局 ——
  gen 1–4、8–43 的实际配方（gen 5–7 是事故的 160/320）。
- `C++\peifang\配方02-事故160局.json`：160 局 / 320 步 —— 看板回填污染的那两代，留作反例。
  **⚠ 2026-10-31 已移出 `peifang\`（留档 `_archive\peifang\`）**：它的 `args` 仍是 160/320，
  留在下拉里等于给"一键复现事故"留了个按钮，理由见 §16.6。
- `C++\peifang\配方03-可测分辨率.json`：320 局 / **320 步** / **门控 60 局** / **配对开局 2 手** /
  **buffer 30 代** / **lr 0.97 退火至 1e-4** / 门控阈值 0.55 —— gen 44 起。
  代价约 +35 s/代（门控 23 s → 70 s，但 `eval_every=2` 摊一半），换来门控分辨率 ~100 → ~55 Elo、
  新旧数据遍数 2.1 → 4.2 遍。
- `C++\model\run_v2\elo.json`：40 个版本、78 条边（39 链 + 39 回指）× 12 局 × 400 模拟，总耗时 **995.6 s**。
- 每份配方 json 的字段固定为：`id / name / status / created / args（键 = gk 开关名，snake_case）/
  command（完整命令行）/ changed_vs_recipe_1（相对配方 1 的差异）/ ingredients / motivation / result / lesson`。

### 14.5 看板改动（`PYTHON\`）

| 位置 | 改动 |
|---|---|
| `watch_progress.ps1` | **① PATH 幂等修复**（见 §8 第 15 条，0xC0000017 弹窗风暴的根因）；② C++ 损失/ce/mse 曲线 **x 轴改成按代数**（原来按行号，断代看不出来）+ **配方切换竖虚线** + 端点代数标签；③ 「自对弈累计」秒 → **小时**；④ 逐代表 `th` 右对齐修好（数字列与表头差 0 列，14/14 对齐）+ 表头 tooltip + **五列释义块**（loss = ce + mse 直接相加、ce/mse/lr/回放池各是什么、为什么不能用它们判断强弱）；⑤ 新卡「**C++ 棋力曲线**」（`cppEloSvg`，读 `elo.json`，画 CI 竖条 + 折线 + 锚点标记）；⑥ 新控件「**配方档案**」下拉 + [填入配方] + [说明]（数据由服务端每 15 s 重读 `C++\peifang\*.json` 烘进页面）；⑦ `$cppSpec` 加 5 个字段：`Recipe / LrGenDecay / LrFloor / OpeningPlies / GateThreshold`；⑧ `Get-TrainRecipe` 改成能认 `gk.exe loop`（原来只认 `train.py`/`auto_train.ps1`，于是永远回落到 Python 的 `last_train_params.json` ⇒ 看板一直显示 160/320） |
| `serve_dashboard.py` | `OPT` 白名单加 `Recipe / LrGenDecay / LrFloor / OpeningPlies / GateThreshold`（少了这些，面板根本凑不出配方 3 的命令行）；`/api/cpp-state` 补 `peifang` 与 `elo` 字段；`OrderedDictionary` 没有 `ContainsKey()` 的崩溃修掉 |
| `_tmp\supervise_loop.ps1` | 监督器：固定配方拉起 `gk loop`，退出就自动重启（最多 40 次），日志 `_tmp\loop_supervisor.log` |

**「填入配方」的实现要点**（以后加配方不用改代码）：`args` 的键就是 gk 的开关名，页面里一张
`RECIPE_TO_FIELD` 表把键翻成表单 id（`games → p_Games`、`sims → q_Sims`、`recipe_id → q_Recipe`…）；
回填后状态条改以该配方为比对基准（「参数与配方3 一致」/「⚠ 与配方3 不同：局数 160（配方3 为 320）」）；
仍走「开始训练」的干跑确认，不绕过任何校验。新增一份 json 丢进 `peifang\` 即可。

### 14.6 验证（都可复现）

```powershell
# 1) 推理路径（CUDA Graph 默认关闭后）
cd C++\build; .\gk.exe probe --net model/run_v2/best.bin --workers 20 --sims 600 `
    --eval-batch 24 --eval-threads 3 --moves 80        # 默认(图关) 46.97 手/s；GK_GRAPH=1 → 46.61（噪声内）

# 2) 门控分辨率 + 配对开局
.\gk.exe gate --cand model/run_v2/gen_043.bin --best model/run_v2/best.bin --eval-games 200
    # → 107:92:1 = 0.535，Wilson [0.466, 0.603]，225 s

# 3) 棋力曲线
.\gk.exe elo --dir model/run_v2 --anchor 10 --eval-games 12 --sims 400

# 4) 面板自检（不跑浏览器）
node ..\_tmp\check_recipe.mjs         # 配方 → 表单 id 的静态映射（3 份全通过）
node ..\_tmp\check_recipe_fill.mjs    # DOM 桩里真跑 recipeFill + cppDefCheck（18/18 字段）
node ..\_tmp\check_spark.mjs          # 损失曲线的 x 轴/配方断点
node ..\_tmp\check_hist_table.mjs     # 逐代表 14/14 列对齐
```

面板干跑出来的完整命令（配方 3，可直接复制）：

```
gk.exe loop --dir model/run_v2 --gens 30 --games 320 --workers 20 --min-steps 320
  --eval-games 60 --eval-every 2 --sims 600 --batch-mcts 16 --batch 256 --lr 0.0003
  --lr-gen-decay 0.97 --lr-floor 0.0001 --opening-plies 2 --gate-threshold 0.55 --recipe 3
  --fresh-ratio 0.5 --eval-batch 24 --eval-threads 3 --half-life 20 --procs 0 --keep-gens 0
  --keep-buffer-gens 10 --buffer-gens 30 --hw 0.5 --pw 1 --seed 12345 --nice
```

### 14.7 状态与下一步

- 训练**当前是停的**（用户 19:2x 明确要求「不要训练现在」）；GPU 空闲；看板与 watcher 在跑
  （watcher 用 `-Html progress_live.html -Json checkpoints\dashboard_state.json -Interval 3`）。
- 下一步就是**用配方 3 从 gen 44 续训**，跑 10–20 代后看两件事：
  ① `elo.json` 重新拟一次，每代增益是否还在 +1.3 ~ +3 Elo 量级（若掉到 0，说明是棋力真到顶而不是测不准）；
  ② 门控通过率是否从 20% 抬到 50% 附近（说明分辨率够用了）。
- 若 elo 也显示平坦：再谈改模型容量/加特征（`--blocks/--channels`）、或引入开局库/自对弈多样性，而不是继续调 lr。

---

## 15. 本期工作记录：训练侧加量、门控解冻与推理提速否决（2026-10-31）

### 15.0 一句话结论

63 代的"平台期"**不是棋力到顶，也不是测量分辨率不够**，而是两件各自独立、都被忽略的事：

1. **`best.bin` 自 gen_44 起冻结了 24 代**——门控在自锁。
2. **训练侧严重欠量**——每代只训 12–17 s，占单代 3–4%，而自对弈 340–460 s 占 83%。

把 `--min-steps` 320→800、`--gate-threshold` 0.55→0.52（配方 4），10 代换来
**+109.5 Elo**（400 局定点对打，95%CI [+75.4, +143.5]）= **10.95 Elo/代**，
是配方 3 期间 3.9 Elo/代的 **2.8 倍**。同期实测否决了 fp16 推理提速（0.899×，反而慢 11%）。

### 15.1 触发点：把"损失平坦"误判成棋力停滞

gen 1–63 的 value 损失一直在 0.51–0.53、策略损失 1.23 附近不动，**代间波动（±0.03）比 63 代的总降幅（0.05）还大**。
据此一度判断"网络已饱和、加数据无用"——**这个判断是错的**，被定点对打直接推翻。

教训写死在这里：**损失平坦 ≠ 棋力停滞。** 那个 mse 0.52 曾被认为是贝叶斯误差限（自对弈双方同等强度 +
温度采样 + 无禁手先手优势），实际其中相当一部分只是**欠训练造成的假地板**——
把训练步数从 320 提到 800（仅占单代 8% 时间）就压到 0.45–0.48，并换来 2.8 倍的进步速率。

### 15.2 根因一：门控自锁，`best.bin` 冻了 24 代

- 门控比较的是 `gen_N` vs `best_net`（`pipeline.cpp:593`、`gate.cpp`），而 `best_net` 只在
  `gr.accepted` 时才更新（`pipeline.cpp:610-614`）。gen 45–63 连续拒绝 ⇒ best 停在 gen_44。
- **但 `best.bin` 每代都被无条件重写**（`pipeline.cpp:642`），所以它的**时间戳永远是最新的一代，
  内容却是 24 代前的**。看 mtime 判断不了，必须靠定点对打。
- 更麻烦的是传播：`web_play.py:157-158` 把 `best.bin` 硬编码排在权重列表第一、
  `serve_dashboard.py:114` 把它当"冠军"展示 ⇒ 对弈界面和看板默认用的都是那个旧模型。
- **为什么必然拒绝**：单代真实进步只有 3–4 Elo，而 60 局门控的标准误是 **±45 Elo**，
  信噪比 0.08。要把它测到 2σ 需要约 **39,400 局**。配方 3 把 20 局提到 60 局，方向对但差两个数量级。
- 因为 best 冻结，门控实际在比 `gen_N` vs `gen_44`，差距随代数累积，`gate_wr` 从 0.417 一路爬到 0.55；
  一旦采纳差距归零、又要重攒——这就是那个锯齿。

### 15.3 根因二：训练侧欠量

每代时间构成（配方 3 实测）：自对弈 ~340–460 s（83%）／训练 12–17 s（3–4%）／门控摊薄 ~40 s。
配方 3 把 `min_steps` 160→320 时速率就从 1.3 涨到 3.9 Elo/代 ⇒ 这个杠杆真实有效，但只推了一档。

### 15.4 本期的代码改动（两处，都已编译进 `gk.exe`）

| 文件 | 改动 | 用途 |
|---|---|---|
| `src/pipeline.cpp`（"6) 日志"段） | `train_log.jsonl` 行末新增 `pool_gens` / `hist_mean_age` / `hist_share20` / `hist_gens` | 回放池诊断。数据取自 `buf.last_stats()`（`dataset.h:70` 的 public getter，本来就算了只是没输出） |
| `src/gate.cpp`（`run_gate_threads` 汇总循环） | 新增逐局 CSV 日志 `GATE1,<局号>,<配对号>,<候选是否执黑>,<胜者颜色>,<候选视角>` | 事后做颜色拆分；`winners[]` 本来就在，只差输出 |

实测基线（gen 66）：`pool_gens=26`、`hist_mean_age=9.68`——**池中实际 26 代、一半梯度来自约 10 代前的旧策略**。
注意 `hist_mean_age` 的 age 0 是 `exclude_latest` 后的次新代，真实代龄要 +1。

### 15.5 配方 4 与验证（`peifang\配方04-训练侧加量.json`）

改动只有两项：`--min-steps 800`、`--gate-threshold 0.52`，其余与配方 3 完全一致（保证可比）。

**结果（gen 66–75，10 代，82 分钟）**

| 指标 | 配方 3（gen 56–65） | 配方 4（gen 66–75） |
|---|---|---|
| value 损失 | 0.5221 | **0.4713**（最低 0.4508） |
| 策略损失 | 1.2337 | 1.2297 |
| 门控采纳 | 0 / 17 场 | **2 / 5 场**（gen 68、74） |
| 每代耗时 | ~478 s | 491.5 s |

**棋力验证（`gen_075` vs `gen_065`，两个 seed 各 200 局合并 400 局）**

```
胜 260 / 负 138 / 平 2   得分率 0.6525
Elo +109.5   95%CI [+75.4, +143.5]   ⇒ 10.95 Elo/代
候选执黑 132/200 = 66%   候选执白 128/200 = 64%
```

两侧对称 ⇒ 是**全面的进步，不是只变强了进攻**。对照历史：gen_061（强方）vs gen_050 时执白只赢 **26%**，
现在 gen_075 执白赢 **64%**——防守能力有了质的变化，和 mse 被压下去是对得上的。

### 15.6 推理提速否决：fp16 实测 0.899×（慢 11%）

前提（gen 66 实测）：评估器前向 248,152 次 / 状态 4,449,051 个，按 SPEED_HANDOFF §1 的
每位置 ~90 μs 算 = 400 s，而该代自对弈实测 396.5 s ⇒ **自对弈 100% 是 GPU 前向瓶颈，
CPU 树操作不占时间**（推翻了早先"瓶颈在树操作"的推测，那是拿 Python 版直觉外推的）。

所以提速只能降推理精度。用 `GK_HALF=1` + `--eval-batch 64` 跑 gen_076：

| | gen_075（fp32, b24） | gen_076（fp16, b64） |
|---|---|---|
| 自对弈 | 480.3 s | **534.5 s（0.899×）** |
| 平均 batch | 17.9 | 51.3 |
| 前向次数 | 248,152 | 84,367（−66%） |

批确实凑大了、调用次数也降了，**但总耗时反而 +54 s**。fp16 的 half→float 转换开销在 20 worker
并发下吃掉了批量收益还倒贴。与 SPEED_HANDOFF §3.7「真实中盘 fp16@b16 = 0.923×」同源。

**关键教训：`gk analyze` 单线程口径的加速比不能外推到自对弈。** 自对弈本来就靠 20 个 worker
天然凑批，analyze 那种"批太小喂不饱张量核"的困境在并发下并不存在。

⇒ 推理侧提速在本机基本走完了：**BN 折叠 1.02×（噪声内）、fp16 负面、fp32 加大 batch 也负面（§3.7 的 0.915×）**。
剩下只有 CUDA Graph / 更大图桶，收益未知。**自对弈 ~460 s/代 当作固定成本。**

### 15.7 配方 5（已否决）：`--min-steps 1600`

> **本节是"开跑前"的预期，已被实测推翻。** 结论看 §16.3（−88.7 Elo），
> 全部教训与原配方档案内容见 **§16.5**（原 `peifang\配方05-训练再加档.json` 已移出 peifang，留档 `_archive\peifang\`）。

依据：160→320 时 1.3→3.9，320→800 时 3.9→10.95，**两次加档都没看到拐点**；800 步只占单代 8%，
翻倍也才 15%。gen_076 已跑：`mse 0.4330`、`ce 1.21751`（两个损失头**都**比配方 4 更低），训练 83.6 s。

代价：每代约 584 s（460 + 84 + 40），比配方 4 慢 19% ⇒ **要划算，速率得超过 13.0 Elo/代**。
当前正在跑 gen_077–081，跑完用 `gen_081 vs gen_075` 定点对打 400 局验收。
判据：5 代 ≥ +50 Elo 就更划算；+30~50 看取舍；< +30 退回 800。

### 15.8 本期新增的踩坑

1. **损失平坦不等于棋力停滞**（§15.1）。判断"饱和"之前先确认训练量给够了——这块只占单代 3–15% 的时间，是最便宜的排查项。
2. **`best.bin` 的时间戳永远是最新的，但内容可能冻在十几代前**（§15.2）。要确认它内容有没有变，只能定点对打，不能看 mtime；也不要按"最近修改"挑权重。
3. **`gk.exe gate` 不显式带 `--hw/--pw` 会走 `MCTSOptions` 默认值 6.0/0.3**（`mcts.h:51-52`），与训练时的 0.5/1.0 **相差 20 倍**。候选和基线用同一组值就公平，但要知道测出来的是哪个口径。
4. **换配方会把 lr 重设回 `--lr`**（`pipeline.cpp:439-447`），代码里没有"换配方但接着退火"的选项。日志里记到的 lr 又是退火**之后**的值，所以会比 `--lr` 小一档（×0.97），别误读。
5. **`keep_buffer_gens` 的清理循环在续跑时失效**（`pipeline.cpp:654`：`for (old = gen - keep_buf; old >= start_gen; --old)`，当 `gen - keep_buf < start_gen` 时一次都不执行）⇒ 多次短续跑会让样本文件无限堆积。本期实测 `buffer/` 已堆到 **65 个文件（约 1.3 GB）**。启动时会被全量读回、再被 `buffer_size=500000` 截断，所以不影响训练结果，只占磁盘。
6. **回放池的真实约束是 `buffer_size=500000`（池中 26 代），不是 `buffer_gens=30`**——后者比实际代数大，从未触发。配方 3 想把池降到 24 万，改 `buffer_gens` 没用，得改 `--buffer-size`。
7. **看板配方档案的字段是固定的 20 项**（`serve_dashboard.py` / `progress_live.html:696-705` 的 `RECIPE_TO_FIELD`，含 `hw`/`pw`，**不含 `buffer_size`**）。配方 json 的 `args` 写多了会被报"未填入"；另外 `sims` 那一项填入时会漏（映射表有 `sims→q_Sims` 但实测没覆盖成功），**填完必须人工核对卡片值**。
8. **跑 `gk.exe` 的环境**：Git Bash 直接跑报 `api-ms-win-crt-locale-l1-1-0.dll` 找不到（exit 127）；PowerShell 也要先把 `...\site-packages\torch\lib` 加进 `$env:PATH`，否则 `0xC0000135`（DLL not found）。看板自己的做法是 `watch_progress.ps1:1312`。
9. **`GK_HALF` 是环境变量不是命令行开关**（`net_torch.cpp:489` 用 `std::getenv` 读），只改命令行参数没用；反过来，设过之后在同一个 shell 里再跑也会继承，要显式清掉。

### 15.9 状态与下一步

- 训练已停：配方 5（gen_076–081）**实测否决**，见 §16.2。
- 待办（按价值）：
  1. ~~配方 5 验收 → 定下 `min_steps` 的甜点~~ **已定：800**（1600 过拟合，−88.7 Elo）。下一步从 gen_075 按配方 4 续跑。
  2. `--half-life 8 --buffer-size 250000` 的 A/B：基线已实测（`hist_mean_age ≈ 9.7`、池中 26 代），预期能把历史样本平均代龄压到 ~4。动完盯 `mse`。
  3. `--hw/--pw` 的 A/B：结构默认 6.0/0.3（启发式主导）与实际 0.5/1.0（网络主导）差 20 倍且从未认真对比过，它决定剪枝时选哪 6 个候选点。
  4. 清 `buffer/` 里堆积的旧样本文件（保留最近 12 个即可）——**删除不可逆，先列出来再决定**。
  5. 中期：无禁手先手优势仍是结构性问题（800 模拟下先手胜率 89%，黑 356 : 白 44）。`vcf_root` 已在 600 sims 重测为**中性**（§16.1）；`guarantee_rush` 的旧实测是 400 sims、早期模型做的，值得在 600 sims + 当前强度下重测一次。
- 别再碰：推理侧提速（§15.6 已否决）、只加 `fp32` batch（§3.7 实测负面）、`min_steps > 800`（§16.2）。

## 16. 本期工作记录：根节点 VCF + 配方 5 否决（2026-10-31）

### 16.0 改动清单

代码（`C++\src`）：

| 文件 | 改动 |
|---|---|
| `vcf.h` / `vcf.cpp` | 新增 `vcf_find_move(pos, out_cell, opt)`——原 `vcf_win()` 只返回 1/0，现在能返回取胜**第一手**；`vcf_win()` 变成它的薄包装。另修一个根层不写 `first_move` 的契约 bug（§16.2） |
| `mcts.h` / `mcts.cpp` | `MCTSOptions.vcf_root`（默认 false）；`get_best_action()` 开头先算一次 VCF，命中就 `return cell` 跳过 MCTS。**不进叶子、不改树** |
| `gate.h` / `gate.cpp` | `GateOptions.vcf_root` + `cand_vcf_root`；`make_opts()` 里 `mo.vcf_root = candidate ? cand_vcf_root : vcf_root`（候选侧单开，做同权重 A/B） |
| `config.h` / `main.cpp` | `vcf_max_ply=6` / `vcf_max_nodes=128` 配置项；CLI `--vcf-root` / `--cand-vcf-root`（后者带 `--cand-no-vcf-root` 反向覆盖） |

新增测试（**不在项目内**，放在 `D:\DHSGZQ\tsest\`）：`test_vcf.cpp` ~170 行，独立验证 VCF 可靠性，见 §16.2。

文档/记录：

| 文件 | 改动 |
|---|---|
| `peifang\配方05-训练再加档.json` | `status` 改「已否决」；新增 `result` 块（400 局原始数据、Elo、颜色拆分、遗留混淆、mse 是陷阱的警告）；`lesson` 追加 4 条。**⚠ 该文件已于 2026-10-31 移出 `peifang\`（留档 `_archive\peifang\`），全部教训与命令行已移入 §16.5** |
| `HANDOFF.md` | 新增 §16（本节）；§15.7 标题改「已否决」；§15.9 待办与「别再碰」同步；§9 模型状态从陈旧的 gen_043 更新到 gen_075 |
| `model\_rejected_recipe5\` | 新建用于归档 gen_076–081 六个权重，**同日已删**（41.7 MB，§9） |

### 16.1 根节点 VCF（`vcf_root`）：400 局实测中性

§13.1 的 VCF 接在**叶子**（`terminal_value()`），400 sims 下实测 0.390 显著变弱，根因未查清。
这次换一个接入点：**只在根节点**算一次，命中就直接落子、完全跳过 MCTS，叶子一律不碰。好处是不改变搜索行为与自训数据分布，每手只多一次 VCF 调用（3.5–5.3 µs）。

新增开关：`--vcf-root` / `--cand-vcf-root`（gate 侧单开，做同权重 A/B）。`MCTSOptions.vcf_root`、`MCTSStats::vcf_hits` 复用。

**同权重自打（gen_075 vs gen_075，600 sims，两个 seed 各 200 局）：**

| seed | 结果 | 得分率 |
|---|---|---|
| 777001 | 95胜105负 | 0.475 |
| 913337 | 103胜97负 | 0.515 |
| **合并** | **198胜202负** | **0.4950** |

Elo −3.5，95%CI [−37.5, +30.6]。两个 seed 跨在 0.5 两侧 ⇒ **确认为中性，既不变强也不变弱**。

结论：600 sims 下 MCTS + 网络自己就能找到这些强制胜，根节点 VCF 是**纯冗余**。它唯一可能有价值的场景是低 sims（快棋/网页对弈），因为那里 MCTS 看不透 3–4 步强制胜——这一项正在测（见 §16.4）。

### 16.2 顺带修掉一个契约 bug：`vcf_find_move` 会"返回 1 但没写回首手"

写了一个不依赖 torch 的独立测试 `tsest\test_vcf.cpp`（只链 `board.cpp` + `vcf.cpp`，mingw 编），做两件事：① 随机生成局面，用**独立实现**的成五点扫描复现整条链，逐步检查"这一手是不是真冲四""最后是不是真收在五连上"；② 手摆局面。

首轮 2000 个随机局面就抓到：**724 次 `vcf_find_move` 返回 1，但 `*out_cell` 一个字节都没写**。全部属于同一类——根节点我方已经有"一步成五"的点，`search()` 在 `if (c.s.n_att_five > 0) return 1;` 处直接返回，**没记 `first_move`**（那是 `ply > 0` 才赋值的）。

危害有限：`mcts.cpp` 里 `cell` 初值 −1 且带 `cell >= 0` 判断，所以不会落错子，只是这次命中被静默丢掉、白算一次搜索。但契约是坏的，任何新调用方都会踩。

修法（`vcf.cpp`，3 行）：根层命中时把那个成五点记成 `first_move`。

```cpp
if (c.s.n_att_five > 0) {
    if (ply == 0) c.first_move = pad_to_cell(c.s.att_five[0]);
    return 1;
}
```

**修后重跑 2000 局面：首手未写回 0；首手有效 1334 个，链验证 1334 通过、假必胜 0。**

> ⚠ **2026-10-31 补充**：这个「假必胜 0」是**偏乐观**的。当时的 `verify_line()` 有一条恒 false 的死代码
> （把 `check_win_after` 用在已落子的点上，它对已占点一律返回 false），会漏掉一整类反杀判定。
> 修好后的口径见 §17.1：998 个局面、236 个真必胜、`brute_vcf` 双向零分歧。
> 「首手未写回 0」这一条仍然有效（那是 `first_move` 契约问题，与本节无关）。

另三个定点局面：安静盘面（无三无四）报 0 ✓；活三 `_XXX_` 报 1 且链验证 O ✓；交叉双四报 1、找到 (2,9)、链验证 O ✓（不是最短杀，但同样是必胜，符合头文件"只保证是一条必胜路径的首手"）。

⇒ VCF 搜索本身**没有假必胜**，"变弱"不是它算错导致的。§13.1 那个 0.390 的根因仍不明，但与本 bug 无关。

**编译方式（这个沙箱里能跑通）**：

```powershell
$env:PATH = "<repo>\C++\tools\mingw64\bin;$env:PATH"
$env:TMP  = "<repo>\C++\build\tmp"; $env:TEMP = $env:TMP   # 必须指到可写目录，否则 g++ 报 Cannot create temporary file
Set-Location "<repo>\C++"
.\tools\mingw64\bin\g++.exe -std=c++17 -O2 -I src -o ".\build\tmp\test_vcf.exe" "D:\DHSGZQ\tsest\test_vcf.cpp" src/board.cpp src/vcf.cpp
.\build\tmp\test_vcf.exe 2000
```

### 16.3 配方 5 否决：`min_steps 1600` 过拟合

`gen_081 vs gen_075` 定点对打 400 局（两个 seed）：

| | 结果 |
|---|---|
| 合并 | 150胜250负，得分率 **0.375** |
| Elo | **−88.7**，95%CI [−122.8, −54.7] |
| 分 seed | 0.405 / 0.345（同向，不是波动） |
| 执黑 / 执白 | 100/200 = 50% / 50/200 = **25%** |

配方 4 时 gen_075 打 gen_065 执白能赢 **64%**，现在掉到 25% ⇒ 换来的防守能力全丢了。

**mse 一路降到 0.39 是陷阱**：损失下降来自对回放池旧数据的拟合，不是棋力。以后验收只认定点对打得分率。

保留的流程教训：只跑到 gen_081 就停下来做对打，没跑满 10 代 —— 若跑满就是白烧约 88 分钟去造一个更弱的模型。**每 5 代必做一次定点对打**这条纪律继续保留。

遗留混淆：gen_076 是 fp16 那批数据，gen_077–081 从它继续训，所以 −88.7 Elo 里可能混有 fp16 血统影响，与过拟合分不开。但 fp16 已独立否决、不会再碰，且"更弱就是更弱"，不做区分实验。

### 16.4 本期新增的踩坑

**1. 从"某一代"续训，不能只写 `--init-weights`——`loop` 根本不看它**

`pipeline.cpp:331-341`：启动时先 `find_last_gen(resume_dir)`，**只要目录里有 `gen_*.bin` 就用编号最大的那个覆盖权重**，`--init-weights` 只在这个分支不成立（`last_gen == 0`）时才生效。

后果：`run_v2` 里还躺着 gen_076–081 时，想"从 gen_075 续训"写什么都没用——它会老老实实载入 gen_081，也就是刚被否决的那个弱模型，而且日志只打一行「[续训] 载入 gen_081」，很容易看漏。

正确做法（本次采用）：把不要的代数**移出目录**（`model\_rejected_recipe5\`），让 `find_last_gen` 自然回到 75；回放池和 `best.bin` 一并保留。另一种是换新目录 + `--init-weights`，代价是回放池从空开始。

顺带：`--resume <dir>` 也不是"只借回放池"——`resume_dir` 同时用于**载权重**，所以 `--resume run_v2 --dir run_v3` 依然会载 run_v2 的最大代，起不到隔离作用。

**2. `best.bin` 是哪一代，只能用哈希对，不能看时间戳**

§15.8-2 已经记过"mtime 永远最新"，这次栽在**反过来用**：看到 `best.bin` 时间戳 = gen_081 的时间，就以为 best 已经是 gen_081。实测：

```
md5sum gen_*.bin | awk '$1=="<best.bin 的 md5>"{print $2}'   # → gen_074.bin
```

`best.bin` 内容 = **gen_074**，配方 5 的六代**全被门控拒了**（门控这次判断是对的）。若按时间戳误判，会得出"best 已经是最好的、直接拿它续训"这种完全错误的结论。

**3. 沙箱里编 mingw 测试程序的两个坎（不是代码问题）**

写非 torch 的独立测试时用 `tools\mingw64\bin\g++.exe`，踩了两个都会伪装成"代码有问题"的错：

- `Cannot create temporary file in C:\Users\...\Temp\: Permission denied` ⇒ g++ 的临时目录不可写。**必须把 `$env:TMP`/`$env:TEMP` 指到项目内可写目录**（`C++\build\tmp`）。指到 `tsest\tmp` 也失败，`C:\...\Temp` 更不行。
- 编出来能跑但**退出码 `-1073741515`（0xC0000135，DLL not found）且一个字不打印** ⇒ 缺 `libstdc++-6.dll`/`libwinpthread-1.dll`。把 `tools\mingw64\bin` 加进 `$env:PATH` 即可。这和 §8-11 的 torch DLL 问题是同一类症状、不同成因。

**4. MCTS 那侧不会因为 `vcf_find_move` 的 bug 落错子——所以这个 bug 一直没被发现**

`mcts.cpp:719-723` 里 `cell` 初值 −1，且命中后还要过 `cell >= 0 && cell < N_CELLS`。所以"返回 1 但没写 `out_cell`"只表现为**这次命中被静默丢掉**（白算一次 VCF），棋力上完全看不出来。这类"契约坏但调用方有兜底"的 bug 只能靠**独立测试直接断言返回值语义**才抓得到 —— 顺带说明：只看对局结果做 A/B，是查不出这类问题的。

**5. 同权重 A/B 的颜色分裂是正常的，不是 bug**

`--cand-vcf-root` 同权重自打 400 局，候选执黑 71% / 执白 28%。一开始怀疑 `--hw/--pw` 是分给两边的，查 `gate.cpp:89-90` 后确认两侧 `make_opts()` 传的是**同一组** `heuristic_weight`/`policy_weight`——分裂来自五子棋无禁手的先手优势（§15 已记 800 模拟下先手胜率 89%）。所以**同权重 A/B 的基线是 0.5，别拿颜色拆分当异常去查**。

**6. `ninja: no work to do` 不等于"源码没变"——头文件改动不会触发重编**

这次把 VCF 修复编进去时踩到。`vcf.cpp` 改过，所以它自己重编了、`gk.exe` 也重新链接了，看起来"编译成功"；
但同一个 `config.h` 的改动**没有**让 `main.cpp.obj` / `pipeline.cpp.obj` 重编（它们都 include 了 config.h）。

判据（不用跑任何东西，看时间戳就行）：

```powershell
Get-ChildItem "C++\build\CMakeFiles\gk_core.dir\src\*.obj" | Sort-Object LastWriteTime |
  Select-Object -First 3 Name,LastWriteTime      # 最早的那个 obj 必须晚于你最晚改过的头文件
```

本次 config.h 只改了注释，所以结果没错；但如果改的是值，就会得到一个静默的旧二进制。
**规则：动头文件 → `build.bat clean` → `build.bat`。** 详见 §8-25。

**7. 删掉"被否决的几代"时，`train_log.jsonl` 也得处理，否则看板不认账**

把 gen_076–081 的权重从 `run_v2` 移走之后，看板仍然显示「第 81 代」——因为 `serve_dashboard.py`
的 `cpp_state()` 里 `last_gen` 取的是 **`train_log.jsonl` 最后一条的 `gen` 字段**，不是 `gen_*.bin` 的存在性：

```python
last_gen = (entries[-1].get("gen") if entries else (gens[-1] if gens else None))
```

更麻烦的是：新训练会重新产出同编号的 gen_076…080，于是 `train_log` 里同一个代会出现**两条记录**，
而损失曲线的 x 轴是按代数画的（§14.5）⇒ 同一个 x 上两个点，"逐代表"也会出现重复行。

处理（2026-10-31）：把那 6 行**移到旁路文件** `train_log_rejected_recipe5.jsonl`（6 行、1975 字节，数据不丢），
主日志回到 75 行、最后一条 `gen: 75`。看板刷新后即显示 75。

⇒ 以后凡是"废弃某几代"，三个地方都要看：`gen_*.bin`、`train_log.jsonl`、`best.bin`（用哈希核对是哪一代）。
顺带：`elo.json` 这次没有 76–81（它停在 gen 61），所以棋力曲线不受影响——但换台机器上未必如此。

**8. 批处理窗口：批越大越慢，"凑大 batch"这条路彻底否决**

`batcher.cpp:112` 的合并条件是 `states + r->n <= max_states_`（= `--eval-batch`），而每个请求固定
`batch_mcts=16` 个状态。默认 `--eval-batch 24` ⇒ `16+16=32>24`，**第二个请求永远装不下**，
所以平均批恒为 16（实测 15.9）。看起来是明显的浪费，但实测证明**把它修好反而更慢**：

| eval-batch / window-ms | 前向次数 | 平均 batch | p50 单批 | 手/s | 总耗时 |
|---|---|---|---|---|---|
| **24 / 0.4（现状）** | 9700 | 15.9 | 3.70 ms | **26.1** | **26.1 s** |
| 64 / 1.0 | 6026 | 26.5 | 2.19 ms | 18.5 | 36.5 s（−29%） |
| 96 / 2.0 | 5096 | 40.2 | 5.28 ms | 20.0 | 42.2 s（−38%） |

（16 局 × 600 sims × 20 workers，`gk.exe selfplay --net ...`，脚本 `tsest\ab_batch.ps1`）

批确实变大、前向次数确实少了一半，但总吞吐反而下降 29–38%。原因是 **GPU 本来就接近满载**：
`[net] service 71.6 s (274.7% wall)`、`sum-wait 351.7 s`（每局 22 s，占墙钟 84%）⇒
三条评估流合计约 13k 状态/s，正好是 §10B 的单流峰值 12k。批凑大后单批更快，但**尾部延迟**从
p95 5.3 ms 涨到 22–28 ms，每个对局都要等自己那一批，尾部把收益吃光。

⇒ 结论：`--eval-batch 24 --window-ms 0.4` 是最优，别改。同时这也解释了历史上 fp16@b64、
fp32@b64 的负结果——不是实现问题，是"批越大越慢"本身。**每代 ~7 分钟是这台机器的下限**，
再快只能减少工作量（games / sims），那是改棋力口径，不是提速。

### 16.5 配方 5 的坑（原 `peifang\配方05-训练再加档.json` 已移出 peifang，本节是唯一活文档）

**从 `peifang\` 移走了什么**：配方档案 json（8.7 KB，留档在 `_archive\peifang\`，看板「配方档案」下拉现在只列 01–04）；
它训出来的 gen_076–081 权重（41.7 MB）更早已**彻底删除**，不可恢复。
⚠ 现在 `run_v2` 里的 `gen_076–081` 是**配方 4 重训**的，与配方 5 无关，别搞混（见 `model\RUNS.md`）。

**一句话结论**：`--min-steps 800 → 1600` 是过拟合，gen_081 比 gen_075 **弱 88.7 Elo**。
**min_steps 的甜点是 800，不要再往上加。**

本配方想一次验两个正交变量，结果两个都被否决：

| 变量 | 从 → 到 | 当时的依据 |
|---|---|---|
| `--min-steps` | 800 → 1600 | 160→320 时 +1.3→+3.9、320→800 时 +3.9→+10.95 Elo/代，**两次加档都没见拐点**；800 步只占单代 8%，翻倍到 15% |
| `GK_HALF=1` + `--eval-batch 64` | — | 自对弈是 100% GPU 前向瓶颈（gen 66：前向 248,152 次 × ~90 μs ≈ 400 s ≈ 那代自对弈 396.5 s） |

**坑 1（最重要）：损失更低 ≠ 棋力更强**

`mse` 从 0.45 一路降到 **0.39**、`ce` 也创新低，看着像大进步，实际 −88.7 Elo。
损失下降来自对回放池旧分布的记忆，不是泛化。
⇒ **此后任何配方验收只认定点对打得分率；`mse` 只用来排查"训练崩没崩"。**

**坑 2：`--min-steps` 不是越大越好，800 是拐点**

| min_steps | 占单代耗时 | 实测速率 |
|---|---|---|
| 160（事故值） | 3% | +1.3 Elo/代 |
| 320 | 6% | +3.9 Elo/代 |
| **800** | 8% | **+10.95 Elo/代（甜点）** |
| 1600 | 15% | **负（6 代 −88.7 Elo）** |

曲线在 800 前后掉头。再加档不是"训练更充分"，是"把回放池背下来"。

**坑 3：防守能力先崩，只看合并得分率看不出来**

执黑 100/200 = 50%（正常），执白 **50/200 = 25%**；配方 4 时 gen_075 打 gen_065 执白能赢 **64%**。
⇒ **定点对打必须拆分执黑/执白**，否则会漏掉"先手还凑合、后手已经不会守"这种结构性退化。

**坑 4：fp16 在自对弈口径是负优化（0.899×）**

| | gen_075（fp32, b24） | gen_076（fp16, b64） |
|---|---|---|
| 自对弈 | 480.3 s | **534.5 s（0.899×）** |
| 平均 batch | 17.9 | 51.3 |
| 前向次数 | 248,152 | 84,367（−66%） |

批确实凑大了、调用次数降了 66%，总耗时反而 +54 s：fp16 的 half→float 转换开销在 20 worker 并发下
吃掉了批量收益还倒贴。
⇒ **`gk analyze` 单线程口径的加速比不能外推到自对弈**——自对弈靠 20 worker 天然凑批，
analyze 那种"批太小喂不饱张量核"的困境在并发下根本不存在。

**坑 5：`GK_HALF` 是环境变量，不是命令行开关**

`net_torch.cpp:489` 用 `std::getenv` 读。写进配方 `command` 里完全无效，必须 `$env:GK_HALF=1`；
而且**会被 shell 继承**——不想要 fp16 时必须显式清掉。

**坑 6：fp16 会改变搜索轨迹，不是纯数值误差**

SPEED_HANDOFF §3.6 一致性表：fp16@b64 的 `root_win` 漂移 0.123，与 fp32@b64 的 0.101 同量级
（都属于"改了搜索形状"）。⇒ 它会影响自对弈数据质量，所以**不能把 fp16 和 min_steps 当两个独立实验**。
这也是本次 −88.7 Elo 里"fp16 血统"与"过拟合"分不开的原因（gen_076 是 fp16 那批，gen_077–081 从它续训）。
当时决定不做区分实验：fp16 已独立否决、不会再碰，且"更弱就是更弱"。

**坑 7：别上 `--eval-batch 128`**

SPEED_HANDOFF §3.6 结论 2：b128 比 b64 只多 0.3%，但一致性明显变差（漂移 0.173、top-1 4/7）。

**坑 8（流程）：止损点救回了 88 分钟**

只跑到 gen_081 就停下来做定点对打，没跑满 10 代。若跑满就是白烧约 88 分钟（8 代 × 575 s）
去造一个更弱的模型。**「每 5 代必做一次定点对打」这条纪律保留。**

**当时的完整命令行（留档，勿用）**：

```
gk.exe loop --dir model/run_v2 --recipe 5 --gens 10 --games 320 --workers 20 --min-steps 1600 \
  --eval-games 60 --eval-every 2 --sims 600 --batch-mcts 16 --batch 256 --lr 0.0003 \
  --lr-gen-decay 0.97 --lr-floor 1e-4 --fresh-ratio 0.5 --eval-batch 24 --eval-threads 3 \
  --half-life 20 --buffer-gens 30 --opening-plies 2 --procs 0 --keep-gens 0 \
  --keep-buffer-gens 12 --hw 0.5 --pw 1 --gate-threshold 0.52 --seed 12345 --nice
```

fp16 变体 = 上面 + `$env:GK_HALF=1` + `--eval-batch 64`（实测更慢，勿用）。

**推理侧提速这条路到本配方为止基本走完了**：BN 折叠 1.02×（噪声内）、fp16 负面、
fp32 加大 batch 也是负面（0.915×）。剩下只有 CUDA Graph / 更大图桶，收益未知。
**自对弈 ~460 s/代 当作固定成本。**

### 16.6 配方 2 的坑（160 局事故，原 `peifang\配方02-事故160局.json` 已移出 peifang）

**它不是设计出来的配方，是事故**：看板「从上次参数填充」把 PYTHON V4 的 160 局 / 320 步灌了进来，
还带了个不存在的 `--buffer-decay-half-life`（被静默忽略）。

**实际跑坏的是 `gen_011` / `gen_012` 两代** —— 证据是 `run_v2\_bad160\` 里躺着的
`gen_011.bin` / `gen_012.bin` + `train_log_full_20261027.jsonl`。
⚠ 原档案 `status` 写的是「gen 5–7」，**那是错的**，以 `_bad160\` 里的实物和 `model\RUNS.md` 为准。

**结果**：单代样本 ~10,000（配方 1 是 ~20,500，少一半），`mse` 直接跳到 0.57；gen 6 门控 9/20 被拒。
已回滚：两代权重与 buffer 移入 `_bad160\`，`train_log` 截到 gen 10，主线从 **gen_010** 重跑。

**三个坑：**

1. **样本量减半 ⇒ 单代梯度噪声翻倍**。同样步数下，过采样倍数从 2.1 遍变成 4.1 遍 —— 这已经不是
   "同一个配方少跑点"，而是换了个配方。
2. **参数覆盖是一条事故传播链**：错误参数先污染 `last_train_params.json`，再被看板「从上次参数填充」
   回填给下一次启动 ⇒ 只要那个文件还是错的，**每次新建训练都会继承它**。见 §8-21。
3. **不认识的长选项必须报错，不能静默忽略**。`--buffer-decay-half-life` 的真名是 `--half-life`，
   写错了既不报错也不生效，事故现场完全看不出来。

**为什么这份也要移出 `peifang\`**：看板会把它列进「配方档案」下拉，而它的 `args` 仍然是 160 局 / 320 步
—— 一旦有人点「填入配方」，就把这次事故原样重演一遍。留在归档里当反例，但**不给"一键复现"的入口**。

## 17. 本期工作记录：代码审查与测量口径修复（2026-10-31 下午）

### 17.0 改动清单

审查范围：`tsest\` 下 6 个脚本/测试 + `tests\test_vcf.cpp`。引擎本体（`mcts.cpp` / `batcher.cpp` /
`net_torch.cpp`）**未审**，只读了 `board.h` / `board.cpp` / `vcf.h` 用于核对接口语义。

| 文件 | 改动 | 性质 |
|---|---|---|
| `tests\test_vcf.cpp` | `board_has_five()` 重写（原实现**恒返回 false**，见 §17.1）；`main()` 参数化（原本无参，`400` 被静默丢弃）；随机验证打印「迭代 → 有效局面」损耗 | 正确性 |
| `tsest\test_vcf.cpp` | `verify_line()` 的"对手反杀"检测由「落子后 `check_win_after`」改为用 `place()` 返回值；删 3 处死代码；`ngames<=0` 直接退出而非除零；`VCFOptions` 改具名字段赋值（不再依赖聚合体字段顺序） | 正确性 |
| `tsest\elo_fit.py` | bootstrap 三口径对齐（重采样后重算权重 / `BOOT_ITERS` 与主拟合一致 / 剔除孤立点）；`wls()` 加 `Sw=0` 保护；`dict.get` 默认值被急切求值的隐患；热循环预映射整数下标；新增 `max\|grad\|` 收敛打印与数据充足性警告 | 统计口径 |
| `tsest\ab_batch.ps1` | 加 warmup；每组 3 次取中位；`$LASTEXITCODE` 门禁；`Push/Pop-Location`；对齐 `[Console]::OutputEncoding`；`ErrorActionPreference` 由 `Stop` 改 `Continue`（§17.4-4） | 测量有效性 |
| `tsest\gate_080_vs_075.ps1` | `$Cand`/`$Best` 加 `ValidatePattern`（原先可路径穿越 + 任意路径覆盖写）；exit code 门禁；编码对齐 | 安全 |
| `tsest\read_gklog.ps1` | 改为先整文件严格试解码 UTF-8（原实现无条件 `gbk→utf8`，会**破坏本来就正确的日志且不可逆**）；原生 `-Tail`；`@()` 强制数组；还原失败保留原行 | 正确性 |
| `tsest\run_vcf_test.ps1` | 加 `-Games` 透传；编码对齐 | 可用性 |

`elo_fit.py` 的数值参数（`lr=40` / momentum 0.9）**未动**——动了会让已有跑分不可比。

### 17.1 `check_win_after()` 只对空点有效 —— 被误用成死代码，两处都踩

`board.h:103-105`：

```cpp
inline bool Pos::check_win_after(int idx, int who) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    if (p[base] != 0) return false;   // ← 关键：已占点直接 false
```

语义是「**该点（必须为空）落子后** who 是否立刻五连」。它**不能**用来判断"盘上已有一颗 who 的棋子是否构成五连"。

两处踩法：

1. **`tests\test_vcf.cpp` 的 `board_has_five()`** 写成 `if (p[..] == who && check_win_after(..))`。前半句保证 `p[..] != 0`，后半句因此恒 false ⇒ **整个函数永远返回 false**，第 295 行「已分胜负的局面没有意义」的过滤**静默失效**，这些盘面照旧进了双向交叉验证。已改为直接扫 5 连窗口。
2. **`tsest\test_vcf.cpp` 的 `verify_line()`** 把"对手被迫挡子若自己先成五 → 这条链对他赢"的保护写在 `place()` **之后**，同理恒不触发。改用 `bool place()` 的返回值（它就是"这手是否成五"，见 §8-6）。

**危害方向**：只漏报、不误报 —— 真·假必胜会被判成"验证通过"。所以 §16.2 里那句
「首手有效 1334 个、链验证 1334 通过、**假必胜 0**」是**乐观偏估**，真实假必胜率只会更高。

修复后实测（`gk.exe` 重编 + `run_vcf_test.ps1 -Games 2000`）：

```
random: 迭代 2000 次 → 有效 998 局面（跳过 1002）, brute 报胜 236, 不一致 0
ALL PASS
```

998 个局面、其中 236 个真必胜，`brute_vcf` 与 `vcf_find_move` **双向零分歧**。

限定：`brute_vcf` 本身也是简化模型（攻方只看冲四点、守方只走唯一挡点），所以这个 0 证明的是
「两者一致」，不等于「绝对正确」——两边若有共同误解，这里查不出来。

### 17.2 `elo.json` 的样本量决定它只能当记录，不能当趋势

查证 `model\run_v2\elo.json`（2026-10-30T21:00 生成）：

| 项 | 值 |
|---|---|
| 边 | 118 = 61 条相邻代（差 1）+ 57 条「每代 vs gen_010」 |
| **每条边的对局数** | **12**（无一例外） |
| 每版本总局数 | 36（12 上行 + 12 下行 + 12 vs gen_010） |
| `gen_010` | **732 局**（星型 hub，它是 57 条边的公共端） |

后果：**逐代 Elo 是噪声的形状**。`elo_fit.py`（无正则 MLE）拟合出 `gen_006 = −166`、`gen_043 = 36.5`、
`gen_056 = 371`，跨度 537 Elo。

而 `elo.json` 自带的 `lo/hi` 半宽只有 ±110–140（如 `gen_016` 是 279.4，CI [159.8, 437.3]），
说明 **C++ 侧 `elo.cpp` 是带先验/收缩的**；`elo_fit.py` 是**纯 MLE 无正则**。两套数不可直接相减。

这解释了 §6.5 那一类「曲线拟合 vs 实测」的符号矛盾：

| 对比 | 实测 | 曲线拟合 |
|---|---|---|
| gen_016 → gen_025 | +50.7 | **−159.1** |
| gen_025 → gen_050 | +33.1 | +115.1 |
| gen_016 → gen_050 | +96.2 | **−44.0** |
| gen_050 → gen_061 | +42.8 | −115.7 |

四个对比两个反号 —— 不是谁算错了，是拿 ±150 量级的估计去比 ±25 量级的估计。

⇒ **结论（与 §6.7 并列）：**

- 趋势判断**只认 200 局以上的定点对打**。`gen_016 → gen_061 = +139 Elo / 45 代 = +3.09 Elo/代`，这个可用。
- `elo.json` 只当「通关记录」，不要用来比较相邻代，更不要用「全段 +4.56 Elo/代」这种数
  （它被 gen 2–20 的 +12.25 拉起来；`gen 40–61` 反而是 −0.73）。
- 想救它只能重跑带更多对局的 elo 采集（每条边 12 → 100 局 = 8 倍采集量）。**加正则只能抹平噪声，不会变出信息量。**

`elo_fit.py` 已加数据充足性警告（边中位局数 < 50 时打印），防止下次再把双列数字读成棋力。

### 17.3 门控局数：只影响 `best.bin`，不影响训练

问「gate 局数会不会影响模型训练」，必须分两个 gate：

**（a）手动验收（`tsest\gate_080_vs_075.ps1`）—— 完全不影响。**
它是 `gk.exe gate` 子命令：读两个权重、跑对局、打日志。不写 `best.bin`、不改 lr、不回写任何训练状态。
加到 5000 局也只是多花时间。

**（b）训练内门控（`pipeline.cpp:558-623`，`--eval-every 2`）—— 影响，但只有一条半通路：**

| 通路 | 位置 | 是否受噪声影响 |
|---|---|---|
| `best.bin` 更新 | 611-615 → 643 每代落盘 | **是**，唯一实质影响 |
| lr 衰减 | 616-621，需 `win_rate < 0.30` | 实际是死的：60 局下噪声触发概率 ≈ 0.1%（`config.h:114` 注释「43 代一次没触发」） |
| 回滚训练权重 | — | **没有**。`gen_XXX.bin` 存的是原始 `net`（642 行），下一代继续在 `net` 上训 |

⇒ **被门控拒绝 ≠ 退回去重训**，只是 `best.bin` 不更新。这也和 §16.4-1（续训用编号最大的 `gen_*.bin`）自洽。

**局数只决定 `best.bin` 的误采率**（阈值 0.52 = 配方 4/5 的值，候选与 best 真实同强度时）：

| 局数 | se | 误采率 |
|---|---|---|
| **60（现状）** | 0.065 | **~35%** |
| 200 | 0.035 | ~28% |
| 1000 | 0.016 | ~10% |

阈值 0.52 离 0.5 只有 0.02，所以**堆局数收益很慢**。要让「通过」真正有意义得配 `--gate-ci-ll 0.5`
（Wilson 下界，现默认 `0.0` = 关闭）：`200 局 + ci_ll 0.5` ⇒ 误采率约 2.5%，真 +42 Elo 的检出率约 55%。

墙钟代价（配方 03 已实测 20→60 局 = 23 s→70 s/次）：60→200 约 +160 s/次，`eval_every=2` 摊到每代 +80 s，
即 8.4 → 约 9.7 分钟/代（**+15%**）。

**但结论是：不加也没关系。** 配方 04 note 已写「门控分辨率 ±45 Elo，单代进步 3–4 Elo」，
配方 05 note 已写「定点对打，不用门控判定」。信噪比 1:15 —— **任何阈值、任何局数都不可能分辨单代进步**。
门控的正确用途只有一个：挡掉 `best.bin` 被明显崩坏的权重污染（配方 5 的 −88 Elo 那种量级）。按这个定位，60 局够用。

顺带：`tsest\ab_batch.ps1` 想验的「加大 `--eval-batch`」在 §16.4-8 已经测完并否决（批越大越慢）。
本次对该脚本的改动（warmup / 中位 / exit code）**不改变那个结论**——原来的计时把冷启动全算在基准组 A 头上，
是**对 A 不利**的偏差，而 A 仍然最快 ⇒ 修正偏差只会让 A 赢得更多，§16.4-8 的结论是稳健的。

### 17.4 本期新增的踩坑

**1. `check_win_after()` 只对空点有效** —— 见 §17.1。判据：函数体第一句就是 `if (p[base] != 0) return false;`。
要判「盘上已有五连」必须自己扫 5 连窗口。写成 `p[..]==who && check_win_after(..)` 得到的是一个**恒 false 的死条件**，
不报错、不告警，只是让那段逻辑永远不执行。

**2. `main()` 无参时，命令行传的参数被静默忽略** —— `test_vcf.exe 400` 里的 `400` 从头到尾没被读，
实际跑的是硬编码的 300 次迭代（有效 155 局面）。症状：**参数加了但结果一模一样**。
判据：看程序自己打印的计数有没有跟着变；别只看"进程起来了、有输出"。

**3. `build\` 里没有任何 DLL，`test_vcf.exe` 依赖 torch** —— 忘了 PATH 时加载器报 `0xC0000135`，
**一个字都不打印**（因为加载器失败发生在 `main()` 之前），极易误判成"代码被我改崩了"。
同 §8-11，但那次还能看到退出码；这次连退出码都被管道吞了，只能靠 `; "exit=$LASTEXITCODE"` 显式打出来。

**4. ⚠ PowerShell 5.1 下 `$ErrorActionPreference='Stop'` + 外部命令 `*>&1` 会抛 NativeCommandError**
5.1 把外部程序的 stderr 包成 `ErrorRecord`，配 `Stop` 直接终止脚本（PS7 才没这问题）。
`gk.exe` 只要往 stderr 写一个字，A/B 就会在第 2 组中途静默死掉，而且看起来像"跑完了"。
**跑外部程序一律用 `Continue` + 显式检查 `$LASTEXITCODE`。**

**5. 本机交互式控制台是 Windows PowerShell 5.1，不是 pwsh 7** —— §8-16 的编码修法要在 5.1 下验证；
`[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)` 在 5.1 有效（`::new()` 可用）。
症状对比：`[ ok ] 空盘 → 0`（已对齐）vs `[ ok ] 绌虹洏 鈫?0`（未对齐）。

**6. `read_gklog.ps1` 的还原逻辑不幂等** —— 原实现「行内有非 ASCII 就 `gbk→utf8`」，
对**本来就正确的** UTF-8 日志是破坏性的（且「）」会丢成 `?`），每跑一次坏一点，且不可逆。
判据应改为「整文件能否按 UTF-8 严格解码」：`[Text.UTF8Encoding]::new($false,$true)` 试解码，抛异常才走还原。
另外 `Get-Content` 对单行文件返回的是 `string`，`$lines[0..0]` 取到的是 **char** —— 用 `@()` 强制数组。

## 18. 本期工作记录：配方 4 续训 10 代（gen_076–085，2026-10-31 下午）

背景：配方 5（`min_steps 1600`）否决后，把 gen_076–081 移出 `run_v2` 让 `find_last_gen()` 回到 75，
然后用**配方 4 原参数**（`320 局 / 600 模拟 / 800 步 / lr 3e-4 / 门控 2 代一次`）续训两批各 5 代。

### 18.1 两次定点对打（600 sims / 配对开局 2 手 / hw 0.5 / pw 1 / 两 seed 各 200 局）

| 对比 | 结果 | 得分率 | Elo（95%CI） | 执黑 | 执白 |
|---|---|---|---|---|---|
| `gen_080 vs gen_075` | 213胜187负 | 0.5325 | **+22.6** [−11.4, +56.7] | 80% | **26%** |
| `gen_085 vs gen_075` | 225胜175负 | **0.5625** | **+43.7** [+9.6, +77.7] | 80% | **33%** |

- 10 代 **+43.7 Elo，CI 已排除 0**（真进步，不是倾向）。两 seed 分别 0.560 / 0.565，稳。
- 反推 `gen_085 vs gen_080` ≈ **+21 Elo**（43.7 − 22.6），即两个 5 代窗口速率一致 ≈ **4.4 Elo/代**。
- 同权重自打基准是执黑 71% / 执白 28%（§16.1 的 VCF A/B 实测）。

### 18.2 关键发现：后 5 代的进步全部来自**白方**

| 窗口 | 执黑 | 执白 |
|---|---|---|
| 075→080 | 80% | 26% |
| 075→085 | **80%（没动）** | **33%（+7pp）** |

⇒ **"白方练不出来"这个判断是错的，它只是慢。** §16.3 之后一度想上"随机开局"（`--sp-open-plies`），
现在看不必急：白方在第二个 5 代窗口里自己动起来了。

### 18.3 待验证的核心问题：速度掉一半，是 lr 退火还是饱和？

| 时点 | lr | 实测速率 |
|---|---|---|
| gen 66（配方 4 起点） | 3.00e-4 | 10.95 Elo/代 |
| gen 85（现在） | 1.63e-4 | 4.4 Elo/代 |

lr 衰减 46%，速率掉 60%。形态上是**稳定地慢**，不像饱和（饱和通常是先变平再抖动）。

**判定实验（下一步）**：参数一字不改，只把 `--recipe 4` 改成 `--recipe 6` ——
`pipeline.cpp:439-447` 遇到配方变更会把 lr 重设回 `--lr 3e-4`，等价于一次 cosine restart。跑 5 代后：

- `gen_090 vs gen_085` ≥ 0.55（≈+35 Elo）⇒ 是 lr 拖慢的，采用"每 10 代暖重启"的节奏
- 仍在 ~0.53（≈+21）⇒ 真饱和，那时才动结构（随机开局 / 加容量 / 加输入特征）

### 18.4 速度侧（同期结论）

- 每代 **7.0 分钟**（自对弈 364–514 s 波动，训练 37–44 s，门控 60 局摊 38 s/代）
- 批处理窗口、fp16、CUDA Graph、多进程、workers/eval-threads **全部已否决**（§16.4-8、§10B）
- ⇒ 这台机器在当前参数下的下限就是 ~7 分钟/代，再快只能降 games/sims（那是改棋力口径）

### 18.5 新增的休眠开关：`--sp-open-plies`（默认 0 = 旧行为）

`selfplay.cpp` 的 `play_game` 在 `set_root` 之前先走 N 手**均匀随机的合法点**，这几手**不进训练样本**
（没有搜索访问数，policy 目标是无意义的随机分布）。动机是削弱无禁手的先手优势（黑胜约 89%）。

**做不到"给白方补一子"**：`board.cpp:62` 的 `place()` 假定 `who == player`，在非行棋方落子会把 hash 的
行棋方分量搞乱（hash 是价值缓存与树复用的键），而重算哈希用的 `zobrist()` 是 `board.cpp` 的文件内静态，外部拿不到。

标定方法（不要拍数字）：先跑 `gk.exe selfplay --net <最新代> --games 16 --sims 600 --workers 20
--blocks 6 --channels 128 --sp-open-plies N`，看输出的「黑 X / 白 Y」，目标是让黑胜落到 **60–65%**。

## 19. Elo 曲线与测量机制的重做（2026-10-31 傍晚）

### 19.1 先钉死问题：这条曲线测不出"棋力随代数上升"

2026-10-31 把曲线从 gen_061 延伸到 gen_085 之后，图上出现两处反常：

| 现象 | 曲线说 | 400 局定点对打实测 |
|---|---|---|
| gen_075 vs gen_065 | **−100.4**（8.4 vs 108.8） | **+109.5** |
| gen_075 → gen_085 | +84.1 | +43.7 |

同一个对比差了 200 Elo，说明不是"噪声大"，是**这段没有分辨力**。根因两条：

1. **每条边只有 12 局**。得分的抽样误差 ±14% ⇒ 换算成 Elo 是 **±50**。链式拟合 24 步，累积漂移
   可以到 ±200（随机游走）。`elo_fit.py` 自己就会打印警告：「边上中位局数 < 50 时，逐代差异不可分辨」。
2. **远锚点饱和**。每代都要跟锚点 gen_010 打 12 局，而 61 代之后全是 12:0 —— 这条边**得分恒为 1，
   提供 0 信息**，白烧一半对局；拟合的绝对刻度反而被它拽住。`§6.5` 那次"曲线把 gen_016/025 排进
   前三、实测顺序是反的"，同一个病。

### 19.2 改动清单

**C++（`main.cpp: cmd_elo`）—— 需要 `build.bat clean` 后重编**

| 改动 | 说明 |
|---|---|
| 默认锚点 = 窗口内最早一代 | 不再默认 gen_010。参照边跨度降到十几代（≈40–60 Elo，得分率 0.55–0.6），才有分辨力 |
| `--max-ref-span`（默认 30 代） | 跨度超过 30 代的参照边**直接不打**，省下必然饱和的局数；链式边仍保证连通 |
| 锚点过远告警 | 距最远版本 > 30 代时打印警告，提示"这些对局不提供信息" |
| 饱和边统计 | 跑完统计"全胜/全负"的边数并打印——它们只把两端拉开、再由拟合硬压回去，是曲线乱跳的直接来源 |

**Python（`serve_dashboard.py`）与模板（`watch_progress.ps1`）—— 需要重启看板**

| 改动 | 说明 |
|---|---|
| `eta` 改为按已完成对数外推 | 原来只在启动时写一次静态估算，实际比估算慢时页面显示"预计还要 0 秒" |
| `est_seconds` 0.75 → 1.6 s/局 | 0.75 是早期口径；最近这些代对局更长、棋力更强，实测约 1.7 s/局 |
| 锚点自动降级 | 面板/接口给出的锚点距最新代 > 30 代时，自动改用窗口最早一代，并在预览里提示原因 |
| 合并改为**平移对齐** | 锚点不同不再直接拒绝：用共有代的中位偏移把新段整体平移，消除拼接处的 +48 跳变；没有共有代才拒绝 |
| 面板默认每对局数 12 → 30 | 12 局的单边误差 ±50 Elo，是"曲线乱跳"最直接的来源 |

### 19.3 正确用法（重要）

```
窗口 = [旧曲线末代 − 2, 最新代]      锚点 = 窗口最早一代      每对 ≥ 20 局
```

窗口必须**与旧曲线重叠至少 2 代**，否则两边没有共有代、无法对齐刻度（合并会被拒绝）。
每 5 代更新一次的话，窗口约 7–8 代 ⇒ 14 对 × 20 局 = 280 局 ≈ 8 分钟，这是可接受的口径。

**当前状态**：曲线停在 gen_061，而最新是 gen_085，**落后 24 代、没有重叠**。要接上只能做一次
"桥接"：窗口 61–85、锚点 61、每对 20 局 = 48 对 × 20 = 960 局 ≈ 25 分钟。做完之后就能按上面
的滚动口径增量更新。

### 19.4 踩坑：直接改 `progress_live.html` 是白费

它是**生成物**——`watch_progress.ps1` 每 3 秒按模板重写一遍。改了 30 秒内就被覆盖，
而且症状很迷惑（编辑器里明明改过、浏览器里还是旧的）。**要改就改 `watch_progress.ps1` 里的模板**，
然后重启看板。`§7` 那句"模板是启动时解析的"只说了一半，这次补全。

### 19.5 滚动锚点的第一次实测（桥接 gen_061 → 085）

用"粗粒度阶梯"而不是密链：`--gens 61,66,71,76,81,85 --eval-games 40 --anchor 61 --out elo.json.new`，
10 对 × 40 局 = 400 局、543 s。每对跨度 5 代（≈20–50 Elo，得分率 0.5–0.6，正好在分辨力最好的区间）。

| 代 | 拟合 Elo（锚 gen_061=0） | 95% 区间 | 说明 |
|---|---|---|---|
| 061 | 0.0 | — | 锚点 |
| 066 | +15.2 | [−42.5, +66.8] | 配方 4 首代，链式边对 61 是 **0.575（61 赢）**——lr 重置的回落 |
| 071 | +27.5 | [−32.0, +79.8] | |
| 076 | +14.7 | [−52.9, +83.1] | 回放池被配方 5 污染后的首代，对 71 是 **0.400** |
| 081 | +50.3 | [−3.6, +116.9] | |
| 085 | +48.6 | [−17.9, +137.2] | 对 81 是 0.550（81 略强） |

**关键验证：61→85 = +48.6 Elo，而独立跑的 400 局定点对打（075→085）是 +43.7 —— 差 5 Elo。**
换测法之后曲线终于和实测对得上了。形状也合理：71 之后回落（76 那个低谷）、81 到顶、85 持平。

合并用一个独立脚本 `tsest\elo_merge.py`（`_elo_merge` 的离线版）：按共有代（这里是 gen_061）
的中位偏移 **+128.2** 把新段整体平移，写回 `elo.json`（旧文件存 `elo.json.before_merge`）。
合并后 65 代（2–61 + 66/71/76/81/85）、127 条边。锚点那一代的 CI 是退化的 `[0,0]`，
合并时**跳过覆盖**，保留旧曲线的真实区间。

**已知不足**：2–61 段仍是旧的 36 局/点，`gen_056 = 372.2 [247.8, 591.9]` 这种明显是假峰。
要修就用同一套粗粒度阶梯重测（`--gens 2,10,20,30,40,50,61 --eval-games 40`，约 9 分钟），
否则读那段时只看趋势、不看单点。

### 19.6 老段重测 + 最终形态（12 个点，全程口径统一）

按上面的办法重测 2–61：`--gens 2,10,20,30,40,50,61 --anchor 30 --eval-games 40`（11 对，683 s）。
锚点取 **gen_030** 而不是 61——这样各点跨度 ≤30 代；`gen_061` 那条参照边（跨 31 代）被
`--max-ref-span` **自动跳过**，改由链式边 `50→61` 连上（新逻辑的第一次实战，按预期工作）。

新拟合（锚 gen_030=0）：`002 −89.6 / 010 −102.8 / 020 −16.3 / 030 0 / 040 −2.0 / 050 +19.4 / 061 +68.7`。

**关键判断：新旧两套的差值散在 +59.5 ~ +218，不是常数平移。** 也就是说旧曲线 2–61 段的**形状**
本身就是噪声（每点 12–36 局），不能靠平移对齐、也不该留着。所以合并时：

- 用 `--prune-unmeasured` **剔除该区间内本次没重测的 53 个旧点**（连带 111 条边），
  否则新老精度不同的点会在图上拼出锯齿；
- 用 `--align-gen 61` 钉在 gen_061 对齐（偏移 +59.5）——它是与"已确认段 66–85"共享的节点，
  按中位数对齐（+190）会把 61 顶到 258、与 66 的 143 之间凭空掉 80 Elo。

**最终曲线 12 个点**（每点 40 局以上）：

| gen | 002 | 010 | 020 | 030 | 040 | 050 | 061 | 066 | 071 | 076 | 081 | 085 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Elo | −30 | −43 | +43 | +60 | +58 | +79 | +128 | +143 | +156 | +143 | **+178** | +177 |

全程 **+206.8 Elo / 83 代 ≈ +2.5 Elo/代**；`gen_076` 那个小回落对应回放池被配方 5 污染那代。
`elo.json` 现在只有这 12 点（旧的 60 点已剔除，备份在各次 `elo.json.before_merge` / `.bak_*`）。

**踩坑**：合并时"锚点那一代跳过覆盖"是错的——旧刻度的值会在新刻度里变成尖峰
（`gen_030`：旧 194.3 vs 新段应为 59.5）。正确做法是**用新的中心、借旧区间的宽度**。
`elo_merge.py` 已按此实现。

---

### 19.7 边库模式：把「分段平移拼接」换成「全局拟合」（2026-10-31 夜）

上面 19.1–19.6 那套做法（每次测一个窗口 → 平移接到旧曲线上）**已经被替换**。留着
是因为它记录了踩过的坑，但**不要再照它操作**。

#### 为什么必须换

平移对齐有个致命前提：新旧两段之间要存在一个常数偏移。实测差值散在 **+59.5 ~ +218**，
根本不存在这样的常数 ⇒ 拼出来的形状取决于选哪个共有代做锚，等于在猜。而且每接一段
累积一次 ±45 Elo 的误差。

更直接的证据：旧 `elo.json` 里 `gen_002 vs gen_010` 那条边是 **21.5/40（002 得分率 0.537，
即 002 高 26 Elo）**，但旧曲线给的是 `gen_002 = −30.1`、`gen_010 = −43.3`（002 只高 13）。
**曲线和它自己的对局数据不自洽** —— 那些值是拼接算出来的，不是拟合出来的。

#### 新机制

| 文件 | 作用 |
|---|---|
| `<run>/elo_edges.jsonl` | **持久边库**：所有打过的对局边，一行一条，只追加。同一对再打则局数累加 |
| `<run>/elo.json` | 每次用**边库全部边**重新全局 BT 拟合的结果（永久是全量曲线） |

新增源码：`src/elo_store.{h,cpp}`（边库读写 + 并查集连通分量 + 全局拟合，不依赖 torch）、
`src/elo_run.{h,cpp}`（打对局 + 收口写 json）。`cmd_elo` 与训练循环共用同一段逻辑。

**锚点**：默认取边库里**最老的一代**，曲线向上 = 变强。边库模式下远锚点不会饱和
（拟合是全量的，锚点只是平移自由度，不需要它参与对局）。

#### 命令

```bash
# 只重拟合，不打对局（导入边库后重建曲线 / 手改边库后重算）
gk.exe elo --dir model/run_v2 --refit-only

# 跨段标定：老代直打新代 400 局。加进边库后全局拟合会自动校正中间各代
gk.exe elo --dir model/run_v2 --gens 85,100 --eval-games 400

# 一次性测量，不污染边库（不写 elo.json，除非显式 --out）
gk.exe elo --dir model/run_v2 --gens 61,85 --eval-games 40 --no-store

# 看板送的窗口命令（--skip-existing 让已打够局数的对跳过，别重复烧 GPU）
gk.exe elo --dir model/run_v2 --gens 81,82,83,84,85 --eval-games 30 --skip-existing 30
```

历史边用 `tsest\elo_seed_store.py --dir <run> --write` 导入（旧 `elo.json` 的 25 条边 →
21 条，重复对按方向换算后累加）。导入后 `--refit-only` 应能复现旧曲线的形状。

导入 + 重拟合后的主线曲线（锚点 `gen_002 = 0`，21 条边 / 12 个版本）：

| gen | 002 | 010 | 020 | 030 | 040 | 050 | 061 | 066 | 071 | 076 | 081 | 085 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Elo | 0 | −24 | +86 | +100 | +103 | +119 | +149 | +164 | +177 | +164 | +200 | **+198** |

全程 **+198 Elo / 83 代 ≈ +2.4 Elo/代**，与 19.6 的 +206.8 是同一条形状（只是零点从
`gen_010` 挪到 `gen_002`，整体平移 23.9）。`gen_010` 那个 −24 不可信：
`gen_002 vs gen_010` 只有 40 局，CI 是 [−105, +57]，覆盖 0。

#### 两种自动模式

| 模式 | 开关 | 触发 | 打什么 | 代价 |
|---|---|---|---|---|
| **采纳驱动**（默认，推荐） | `--elo-on-accept 1` | 每 **2** 次门控采纳 | 「本代 vs 上一个**被测量**的代」× **150** 局 | 约每 2 次采纳 4.3 分钟 |
| 每 N 代滚动 | `--elo-every 5` | `gen % 5 == 0` | 最近 5 代窗口 × 30 局 | 每 5 代 ~8 分钟 |

**为什么是 150 局 / 每 2 次采纳**（2026-10-31 调整，原先 400 局 / 每次）：
+10~+20 的进步**400 局也测不出来**（`075→080` 实测 +22.6，已贴着 ±17 的噪声边），
所以 400 局买到的主要是"发现配方崩坏"（配方 5 那次 −88）——150 局 ±28 抓 −50 以上的退化
绰绰有余。要对外结论时单独打 600 局。跨 2 次采纳打一条边仍是**直接对打**，误差不累积，
只是曲线上的点稀一些；Elo 开销从 +43% 降到 +20%。

**采纳驱动为什么是默认**：门控只回答"有没有更强"（二元，>0.55 就过），给不出"强多少"。
补打同一对后：① 曲线上每段都是**直接对打**的结果，不是链式推的 ⇒ 累积误差消失，
**跨段标定这件事本身就不存在了**；② 只在采纳时跑，被拒的代不浪费 GPU（43 代里 19 代被拒）；
③ 与门控测同一对模型，"采纳"和"涨了多少"不会互相矛盾（旧曲线 `gen_076` 比 `gen_071`
低 13 Elo 那种矛盾就是这么来的）。

代价：曲线只画**被采纳的代**。没被采纳 = 没比 best 强，画上去本来也是噪声。

要字面上"每 5 代一个点"就 `--elo-every 5`（两点都想要就两个开关都开，代价叠加）。

#### 踩坑

* **采纳驱动的 Elo 必须放在落盘之后**。门控在落盘之前跑完，那时本代 `gen_%03d.bin`
  还没写盘，`elo_play` 只剩"上一个被采纳的代"一个版本，会直接跳过
  （实测打出 `[elo] 可用版本不足 2 个`）。
* **左端点必须取"边库里最大的代"，不能拿 `last_gen` 近似**。续训时 `best.bin` 不带代数，
  用续训起点会把一个**从没测过**的代当左端点，那条边落库后与主体不连通（拟合直接剔除，白打）。
* **「每 N 次采纳测一次」会在低采纳期饿死**。实测 88–96 九代只采纳 gen_092 一代
  （32/60=0.533），计数器没到 2 ⇒ 整轮一条边都没打，曲线停在 gen_086。
  已加**收尾 flush**：本轮结束时若「最后采纳代 > 最后测量代」，补一条从上次测量代到
  最后采纳代的直边（跨度 2~4 代仍是直接对打）。
* **看板不能再 `--out elo.json.new` + 平移合并**：边库模式下 gk 直接写 `elo.json`，
  写 `.new` 会导致"边库更新了、曲线没换"。`serve_dashboard.py` 的 `_elo_merge` 已停用
  （保留存档）。旧曲线快照必须在**启动 gk 之前**拍，否则跑完再读拿到的是新文件
  ⇒ 同代相减恒为 0。
* 边库里版本权重文件被 `keep_gens` 清掉时，该版本连同它的边一起剔除，并打印条数；
  与主体不连通的版本同样剔除（BT 对不同分量没有可比刻度）。










