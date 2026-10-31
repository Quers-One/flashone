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

**当前（2026-11-01 00:20）**：`model/run_v2` 已跑到 **gen_096**，`best.bin` 内容 = **gen_092**
（md5 核对）。配方 **4**（`--min-steps 800` + `--gate-threshold 0.52` + `--eval-games 60` + `--eval-every 2`）。
目录与配方沿革明细见 §6 与 `C++/model/RUNS.md`。

**棋力**：Elo 已改成「持久边库 + 全局重拟合」（实现见 `C++/src/elo_store.*`，用法见 §12.2），当前 14 个版本 / 23 条边，锚点 `gen_002 = 0`，
最新点 **`gen_092 = +200 Elo`**。每段都是同一对模型的直接对打（150 局 ±28、400 局 ±17），
不存在分段拼接的累积误差。近期（88–96 代）门控胜率贴平在 0.483，尚未分清是学习率退火压死
还是网络容量到顶。
⇒ **判断进展用 ≥200 局的定点对打；门控只用来防止 `best.bin` 被崩坏权重污染。**

历史（保留原因）：gen 25–43 曾连续 19 代没过门控，根因是 20 局门控的分辨率（±100 Elo）
比后期真实速度（+1.3 Elo/代）粗两个数量级 —— 这是配方 3 把门控提到 60 局 + 配对开局的起因。
但即便如此，60 局也只够「防止 `best.bin` 被崩坏权重污染」，不够判断单代强弱。

2026-10-26 下午那一段：① CUDA Graph 化推理（当时生产形态 +10%，见第 12 节；**10-28 已改默认关闭**，见第 14.2 节）；
② 借鉴 Rapfi 实现 VCF 静态搜索与「强制着法不剪枝」（**两项实测均无收益，已默认关闭**，见第 13 节）；
③ 结论是**当前引擎的瓶颈不在搜索算法，而在并行形态**（第 13.4 节）。

---

## 1. 目录与文件

```
README.md  LICENSE  .gitignore
docs/        DEVLOG.md（本文）· WEIGHTS.md（权重沿革）
C++/         CMakeLists.txt · build.bat · README.md（构建、子命令、性能表）
  src/       33 个文件：规则 / MCTS / VCF / 自对弈 / 训练 / 门控 / Elo 边库
  tests/     13 个测试程序（1.4 万项检查）
  tools/     convert_ckpt.py（.pt ↔ .bin 互转）
  peifang/   3 份配方档案（假设 / 改动 / 结果 / 教训）
PYTHON/      53 个文件：看板服务 · 对弈页 · 旧版 Python 原型（train.py / mcts.py / network.py）
```

模型权重、回放池、构建产物都不在仓库里（见 `.gitignore`）；权重通过 Release 分发。

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
生产上实际用的是 `workers 20 / eval-batch 24 / eval-threads 3`（见 §5 第 128 行），配方档案见 `C++/peifang/`。

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

## 7.1 改 watch_progress.ps1 的坑（模板被常驻进程覆写）

**踩到的坑（看板改了不生效，浪费了一圈排查）**：
`progress_live.html` 是 `watch_progress.ps1` 生成的**静态产物**，而 `Flash UI.cmd` 会拉起一个**常驻的
`pwsh watch_progress.ps1 -Html progress_live.html -Interval 5`**，每 5 秒整份重写一次。
只改脚本、不重启那个常驻进程 ⇒ 手动生成的新页面在 5 秒内被旧进程覆盖回旧版本，
现象是「源码里有新代码、页面里没有」，极易误判为自己改错。

**正确做法**：改完 `watch_progress.ps1` 必须
①`Stop-Process` 杀掉常驻 watch 进程 → ②重新拉起（或用 `Flash UI.cmd -Restart`）→ ③确认
`Get-CimInstance Win32_Process | ? CommandLine -match 'watch_progress\.ps1'` **只有一个**。
多个 watch 进程会互踩（实测同时出现 2 个，页面在两种版本间跳）。

## 7.2 Elo 对战面板（曾把 elo.json 碾成 3 个点）

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

## 7.3 看板结构、参数回填与重启

- 看板：`http://127.0.0.1:8791`（`PYTHON\serve_dashboard.py`）
- 版本切换：`#verBar`（Python / C++），**C++ 面板现在 7 张卡**：概览 / 损失曲线（ce·mse·配方断点）/
  **棋力曲线（Elo）** / 逐代表 / 各 run 训练量 / champion / stdout 尾巴
- 训练引擎下拉 `#p_Engine`：`python` → `train.py`；`cpp` → `gk.exe loop`
  - C++ 参数来自 `gk.exe defaults`（即 `config.h`），启动前走 dry-run 拿服务端拼好的命令行再确认
  - **配方档案**下拉 + [填入配方]：读 `C++\peifang\*.json`，一键把整套参数回填（见 §6）
  - C++ 运行目录是**下拉**（`#p_CppDirSel`，显示 `目录 · 角色 · 全局第几代`，来自各目录 `run.json`），默认选主线
  - **空目录保护**：目标目录没有 `gen_*.bin`/`best.bin` 时服务端**拒绝启动**（`error_kind: "scratch"`），避免拿随机权重从零开训
- 对弈页：`web_play.py`（8770），通过 iframe 嵌进看板；引擎可选 C++ / Python，C++ 走 `gk.exe play` 常驻进程（行协议 `GO/MOVE/INFO/QUIT`），树复用后单步 30 ms
- 实时同步：`GET /api/stream`（SSE，chunked 手写），页面 `EventSource` + 120 ms 防抖 + 原地 patch（`patchNode/patchInto`），不再整页刷新

**10-28 新增/修好的看板行为**：损失曲线 x 轴按**代数**（原来按行号，断代被抹平）+ 配方切换竖虚线；
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
   **恒 false 的死条件**，不报错、不告警，只是那段逻辑永远不执行。已在两处测试里踩到，见 §12 表格的「代码审查」一行。
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
    无论命令行怎么写都会载入最差的那一代，而日志只有一行「[续训] 载入 gen_XXX」，极容易看漏。判据是 md5 比对，不是时间戳（见 §8）。
23. **`best.bin` 是哪一代只能哈希对，别信时间戳**：`md5sum gen_*.bin | awk '$1=="<best 的 md5>"{print $2}'`。
    本次靠它才发现 best 冻在 gen_074（配方 5 六代全被拒），而时间戳指向 gen_081。
24. **沙箱里编 mingw 独立测试：`$env:TMP` 必须指到项目内可写目录**（否则 `Cannot create temporary file`），
    运行时 `tools\mingw64\bin` 必须在 `PATH` 里（否则 `0xC0000135` DLL not found、且不打印任何东西）。
    另外 `gk.exe` 里的 MSVC 编译不受影响——两套工具链各自独立。
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
    §8-16 的编码修法要用 5.1 验证。

> §9、§10 已从公开版移除：原「当前模型状态」极易过期（当时写的是 `gen_083`，后来已经跑到 96 代），
> 原「下一步该干什么」是个人优先级清单，不适合当公开文档。
> 要查真实状态请看 `C++/model/RUNS.md` 与 `model/<run>/elo.json`。

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

## 12. 历史工作记录（结论压缩版）

> 2026-10-26 → 11-01 这十几轮实验的完整流水账（每步改了什么、跑了多少局、当时的原始输出）
> 留在开发者的本地记录里，不随仓库分发。这里只保留**能复用的判断**。

| 主题 | 做了什么 | 结论 |
|---|---|---|
| CUDA Graph 化推理 | 把逐步前向捕获成图 | 生产形态 +10%（22.31 → 24.57 手/s）；但每张图独占一个显存池，12 张图峰值 12.8 GB，**10-28 改回默认关闭** |
| 借鉴 Rapfi 的两项 | VCF 静态搜索、强制着法不剪枝 | **两项实测均无收益**，默认关闭。由此确认：瓶颈不在搜索算法，在并行形态 |
| 配方 3：门控解冻 | 门控 20 局 → 60 局，并加配对开局 | 43 代平台期的真因是 20 局门控的分辨率（±100 Elo）比真实速度（+1.3 Elo/代）粗两个数量级 |
| Elo 覆写事故 | 第一版直接 `os.replace(elo.json.new, elo.json)` | 「最近 5 代」的一次测量把 40 代曲线碾成 3 个点，靠 60 个 `gen_*.bin` 全量重跑 1431 s 才恢复 ⇒ 写曲线必须先合并再替换 |
| 损失平坦的误判 | 曾把 mse 0.52 当成贝叶斯误差限 | **损失平坦 ≠ 棋力停滞**；`mse` 只用来排查"训练崩没崩" |
| 门控自锁 | `best.bin` 冻了 24 代没更新 | lr 退火只在胜率 < 0.30 时触发，而胜率长期在 0.45–0.55 ⇒ lr 43 代一次没降 |
| 训练侧欠量 | 每代训练只占单代的 3–4% | 提到 800 步（配方 4）后门控通过率恢复；训练是当时唯一还没被榨干的部分 |
| fp16 提速 | 自对弈实测 | **0.899×（反而慢 11%）**；单线程 `gk analyze` 的加速比不能外推到 20 worker 的自对弈 |
| 配方 5：`min_steps 1600` | 跑 6 代 | **−88.7 Elo**（400 局、两 seed 同向），白方胜率 64% → 25%；mse 降到 0.39 是拟合旧数据的假象 |
| 根节点 VCF | 400 局实测 | **中性**（0.495，CI 跨 0）；600 sims 下 MCTS 自己就能找到这些强制胜，VCF 是冗余 |
| 代码审查 | `check_win_after()` 只对空点有效 | 两处误用成死代码（含 VCF 的「对手反杀」检测）；修完 VCF **没有假必胜** |
| 配方 4 续训 10 代 | gen 76–85 | **+43.7 Elo**（400 局）；进步**全部来自白方**——"白方练不出来"是错的，它只是慢 |
| Elo 机制重做 | 分段平移拼接 → 持久边库 + 全局重拟合 | 每段都是**同一对模型直接对打**，「跨段标定」这件事从此不存在（实现见 `src/elo_store.*`，用法见 §12.2） |

### 12.1 仍然有效的五条纪律

1. **判断进展用 ≥200 局定点对打**，别读门控胜率——60 局只够防止 `best.bin` 被崩坏权重污染。
2. **定点对打必须拆执黑/执白**，否则会漏掉"先手还凑合、后手已经不会守"这种结构性退化。
3. **配方验收只认对打得分率**，`mse` 只用来判断训练有没有崩。
4. **每 5 代必做一次定点对打**：只跑到第 6 代就停下，能省掉 88 分钟去造一个更弱的模型。
5. **改 `watch_progress.ps1` 后必须重启那个常驻写入进程**，否则新页面 5 秒内被旧进程覆写（见 §7.1）。

### 12.2 Elo 边库的用法

```bat
REM 打几条直边（只读权重，不改权重）
gk.exe elo --dir model/run_v2 --gens 86,92 --eval-games 150

REM 不打局，只用现有边库全量重拟合（用来复现 elo.json）
gk.exe elo --dir model/run_v2 --refit-only

REM 训练时自动测：门控采纳后补打一条
gk.exe loop ... --elo-on-accept 1 --elo-calib-games 150 --elo-every-accepts 2
```

- 每条边记 `(a, b, games, score_a)`，只追加；`elo.json` 永远是**全部边**重拟合的结果，
  不存在"拟合了这一段、拼接了那一段"。
- 锚点默认取边库里**最早的代**，所以曲线向上 = 变强。
- 分辨率：150 局 ±28 Elo、400 局 ±17、600 局 ±14。小锯齿是采样噪声，**看趋势，别读单点**。
- 「每 N 次采纳测一次」在低采纳期会饿死，所以循环结束时会补一次 flush（本轮欠的边当场结清）。
