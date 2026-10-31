"""
train.py
========
五子棋强化学习训练模块：自我对弈 → 训练 → 评估 的完整闭环
（参考 AlphaGo Zero / KataGo 训练流程）。

流程概述：
    1) 自我对弈：当前模型经 MCTS 对弈生成对局数据
       每步保存 (局面张量, MCTS 策略概率, 最终胜负标签)
       数据增强：旋转+翻转 8 倍
    2) 训练：策略交叉熵 + 价值 MSE，权重 1:1，Adam 优化器
    3) 评估：新模型 vs 旧模型 多局对弈
       - 胜率 > 阈值 → 采纳新模型，进入下一轮
       - 胜率不足   → 降学习率重新训练
    4) 保存：模型权重、训练日志、每代数据

本轮修复（对应"阶段 0/1"优化）
--------------------------------------------------------------------------
* [自对弈温度采样] 旧实现用 temp=1.0 拿到访问概率后又取 argmax，温度完全失效，
  导致同一模型下每局逐手完全相同（实测 3 局都是 241 手同一序列）。
  现改为 sample_move()：按 p ∝ n^(1/T) **真正采样**，并加入根节点 Dirichlet 噪声。
* [并行 worker 崩溃] 旧实现 _worker_run_games 引用了不存在的全局 cfg，
  而 Config.n_workers 默认 4，一跑并行分支就 NameError。
  现把 cfg 作为参数显式传入（用 pickle 安全的普通 dict）。
* [MCTS 参数同步] 与 mcts.py 的新默认值对齐（c_puct=1.0, virtual_loss=3.0,
  max_children=6, root_max_children=16, 规则先验权重 6.0）。
* [门控与日志] 评估局数、阈值、学习率衰减等参数已按实测重设，
  并在日志中记录关键 MCTS 指标，便于判断"搜索引擎是否正常"。
"""

import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch

from gomoku_env import Board
from network import GomokuNet
from mcts import MCTS


# ----------------------------------------------------------------------
# 全局配置（参数含义与调优方向见注释）
# ----------------------------------------------------------------------
class Config:
    """训练配置。各参数含义与调优方向见注释。"""

    # ---- 数据生成 ----
    games_per_epoch: int = 40          # 每轮迭代生成的对局数。调优：越大数据越丰富但更耗时
    c_puct: float = 1.0                # PUCT 探索系数。实测 1.0 比 2.5 更能集中预算
    n_sim: int = 400                   # 每步 MCTS 模拟轮数。None=按深度动态
    virtual_loss: float = 3.0          # 虚拟损失。批量推理下必须 > 0（旧默认 0 = 批量无效）
    batch_size_mcts: int = 16          # MCTS 批量推理大小
    max_children: int = 6              # 非根节点展开宽度
    root_max_children: int = 16        # 根节点展开宽度（含全部强制点）
    n_workers: int = 1                 # 并行 worker 数。GPU 是瓶颈时保持 1

    # ---- 自对弈探索（修复：旧实现完全没有探索）----
    temperature: float = None          # None=按手数退火(1.0/0.5/0.0)
    temp_threshold: int = 12           # 前 N 手用温度 1.0
    root_noise_alpha: float = 0.15     # 根节点 Dirichlet 噪声 alpha（19×19 建议 0.1~0.2）
    root_noise_eps: float = 0.25       # 噪声混合比例
    resign_threshold: float = -0.95    # 价值头低于此值且持续 N 手则认输（None 关闭）
    resign_plies: int = 3              # 认输判定所需的连续手数

    # ---- 规则先验权重（随网络变强逐步退火）----
    #   前 20 代: 6.0 / 0.3    网络≈噪声，主要靠规则
    #   20~40 代: 线性 6.0->2.0 / 0.3->1.0
    #   40~60 代: 线性 2.0->0.5 / 1.0
    #   60 代之后: 0.5 / 1.0   让网络主导，规则只兜底成五/必挡
    #
    # 【为什么必须退火】heuristic_weight=6.0 是"带拐杖"的启动配置：随机网络的
    # 策略先验是噪声，靠规则棋型分才能下出有意义对局。但如果 80 代都固定 6.0，
    # 规则会一直压住网络的策略学习 —— 自对弈的落点几乎完全由规则决定，
    # 策略头的训练目标里没有网络自己的信息，网络就永远学不动（表现是
    # win_rate 长期在 0.5 附近震荡、强度停在"规则引擎"水平）。
    # 注意：成五/必挡的**强制覆盖与权重无关**（永远生效），所以退火不会
    # 牺牲基本战术正确性 —— test_tactics.py 在任意权重下都是 19/19。
    heuristic_weight: float = 6.0      # 起始值（anneal_enabled=False 时就是它）
    policy_weight: float = 0.3
    anneal_enabled: bool = True
    anneal_start: int = 20             # 该代及以前保持起始权重
    anneal_mid: int = 40               # 线性插值到"中段"权重的代数
    anneal_end: int = 60               # 线性插值到"末尾"权重的代数
    anneal_h_start: float = 6.0
    anneal_h_mid: float = 2.0
    anneal_h_end: float = 0.5
    anneal_p_start: float = 0.3
    anneal_p_mid: float = 1.0
    anneal_p_end: float = 1.0

    # ---- 数据增强 ----
    augment_factor: int = 8            # 增强倍数(旋转+翻转)。调优：1~8，8 为全对称群
    # 注意：增强只作用于局面/策略，胜负标签不变(对称不变量)

    # ---- 经验回放 (Replay Buffer) ----
    # 旧实现：每代只生成十几局、然后拿这十几局反复训 50 个 epoch —— 典型过拟合，
    # 而且每代数据用后即弃，网络会"遗忘"上一代学到的局面。
    # AlphaZero 的做法是保留最近若干代的样本混合采样，这里照此实现。
    buffer_size: int = 500_000        # 回放池容量（样本条数）。12GB 显存可用 30~50 万
    buffer_gen_limit: int = 30         # 最多保留多少代的样本
    fresh_ratio: float = 0.7           # 每个 batch 中"当代新数据"的比例，其余取自历史
    # ---- 回放池采样整形（任务 4，2026-10-13）----
    # 两个开关**默认都关** ⇒ 采样与旧版逐位一致（按样本数比例选代 ≡ 全体样本均匀采样）。
    #   buffer_decay_half_life > 0：按"代龄"做几何衰减，代权重 = 2^(-代龄/半衰期)。
    #       半衰期 H 的定义是"相隔 H 代的两代，采样权重相差 2 倍"。取 H=20 时，
    #       60 代池里最近 20 代合计约占 55% 的抽样权重（均匀本应是 32%）。
    #   buffer_entropy_weight > 0：代内按策略目标的信息熵加权，权重 = exp(β·H/Hmax)
    #       ⇒ 搜索"还没判明"的局面被多采（任务卡说的"新颖度"）。
    # 未实现的两条（都超出"只动 train.py"，见 NOTES §20）：
    #   价值误差：要给整个池做前向才拿得到，代价 ≫ 收益；且要往池里存预测值。
    #   对局长度：池里没有局号/局长，要动 self_play_game / worker / npz 格式。
    buffer_decay_half_life: float = 0.0
    buffer_entropy_weight: float = 0.0
    samples_per_gen: float = 2.0       # 每代训练步数 = 新数据量 × 该系数（配合 batch_size）
    min_samples_per_gen: int = 400     # 每代最少训练步数（防止新数据太少时训练不足）
    on_the_fly_augment: bool = True    # 训练时在线随机增强（而不是提前把数据复制 8 份）

    # ---- 网络骨干规模 ----
    # 【放大网络】实测（probe_scale.py，§14）：
    #   3×64  = 24.7 万参数，前向 0.95 ms(b16)、训练步 ~20 ms
    #   6×128 = 182 万参数（7.4×），前向 1.69 ms、训练步 ~? ms
    # 只有**新建网络**时这两个值生效；给了 `--init-weights` 就以权重文件的
    # 架构为准（否则会把权重全丢掉、得到一个随机网络 —— 见 network.infer_arch）。
    n_blocks: int = 3                  # 残差块数
    channels: int = 64                 # 通道数
    value_hidden: int = 64             # 价值头隐层宽度

    # ---- 训练 ----
    batch_size: int = 256              # 小批量大小。调优：128~512，显存受限则减小
    epochs: int = 4                    # 每轮训练轮数。数据量上去后不需要反复拟合
    learning_rate: float = 1e-3        # 初始学习率。调优：1e-4~1e-2
    weight_policy: float = 1.0         # 策略损失权重。调优：1.0 均衡，>1 偏策略
    weight_value: float = 1.0          # 价值损失权重。调优：1.0 均衡
    max_grad_norm: float = 5.0         # 梯度裁剪阈值。调优：1~10，防梯度爆炸

    # ---- 【任务 6】辅助头（auxiliary heads）----
    #
    # 目的：给主干加"更密的监督信号"。策略目标是 361 维里挑 1 个点（很稀疏），
    # 价值目标一盘一标量，二者都只在**对局结束时**才有信息。辅助头把棋盘空间上
    # 本来就算得出来的事实（哪些点一步成五 = threat、每点最终归属 = ownership）
    # 变成逐点的密集目标，逼主干特征把战术与地盘信息显式编码出来。
    #
    # 【默认全关】`aux_heads=""` ⇒ 不建任何辅助头，网络结构与旧权重逐字节兼容，
    # 所有历史 ckpt / 门控 / 自对弈行为一字不变。
    #
    # 【不动 MCTS】辅助头只在 `GomokuNet.forward_with_aux()` 里算，推理路径
    # （`forward()`）不碰它们 ⇒ 加辅助头**不增加自对弈时间**，只增加训练步成本。
    #
    # 【支持热启动】从 `--init-weights models/V3/gen_110.pt` 这类无辅助头的
    # ckpt 继续训：主干权重原样保留，只多挂两个小头（`workflow`: 见
    # `GomokuNet.add_aux_heads`）。因此不必从零重训、不必丢历史数据。
    aux_heads: str = ""                # 逗号分隔：threat / ownership（空=关闭）
    aux_ch: int = 32                   # 辅助头内部通道数（1×1 卷积降维）
    aux_weight_threat: float = 0.15    # 威胁头损失权重（BCEWithLogits）
    aux_weight_ownership: float = 0.15  # 归属头损失权重（MSE）
    aux_threat_pos_weight: float = 20.0  # 威胁正样本权重。正点极稀疏（实测约
    #   0.3%~1%），不加权会退化成"整盘输出 0"——loss 照样下降，但什么都没学到。
    #   取 20 让正负样本的总权重同量级；这是**只影响辅助头**的旋钮。

    # ---- EMA / SWA（权重滑动平均，显著降低门控抖动）----
    ema_enabled: bool = True           # 是否启用 EMA 权重
    ema_decay: float = 0.999           # EMA 衰减。调优：0.99~0.9999，越大越平滑
    eval_both: bool = True             # 是否同时评估"原始权重"和"EMA 权重"，取较优者

    # ---- 评估与更新 ----
    # 【2026-10-13 任务 1：20 → 100】20 局在"新旧真实相当"(p=0.5) 时假拒率
    # 74.8%，而阈值 0.55 实际要求 ≥12/20=0.60 才算采纳 —— 这条 20 局的门控
    # 【2026-10-13 定论：20 局是**刻意选择**，不是"还没调大"】
    # 试过 100 局并实测过它的收益与代价：
    #   * 它**没有降低门槛** —— "胜率>0.55 且 Wilson下界>0.50"在 n=100 下
    #     等价于 ≥60/100，与旧的 ≥12/20 同为 0.60；
    #   * 它只把"同一门槛下的判定误差"压小（p=0.65 时误拒 23.8%→12.5%，
    #     误纳 p=0.50 时 13.6%→2.8%）；
    #   * 代价是门控局量从自对弈的 **1/8 涨到 5/8**、每代墙钟 **+45%**，
    #     而棋力收益为 0（门控不训练模型，它只决定 best_net 换不换）。
    # ⇒ 局数不改。要更准就整轮只跑一次的晋升闸（auto_eval.ps1）去花局数，
    #   那里的每一局都比这里值钱。
    # 完整功效表、以及三条验收标准为何互相矛盾：见 NOTES.md §18
    # 与 `python verify_gate_protocol.py`（离线，不跑对局）。
    eval_games: int = 20
    # 门控对局的并行进程数。门控一次 = 2×eval_games 局，"隔代门控"时它与训练步
    # 一起构成新瓶颈（自对弈已并行化，见 §15）。颜色按局号分配，并行不偏置判定。
    #
    # 【2026-10-11 实测后 4 → 8】门控阶段实测 CPU 只用到 15%、GPU-Util 47%、功耗 30W/120W
    # —— 两头都闲着（因为门控只有 4 个进程在打，机器其余部分是空的）。提到 8 让 GPU
    # 利用率和功耗上去（预测 ~95%/38W），门控那一代的墙钟可望接近减半。
    # 注意：门控局种子来自时钟（本来就不可复现），所以改并行度**不会**损失任何可复现性。
    #
    # ⚠️ 【别跟着 eval_games 一起把 eval_workers 调大】每个 torch worker 提交
    # ~3.9 GB 内存，16 个 ≈62 GB，会把机器顶到 OOM（2026-10-12 的崩溃循环就是这个）。
    eval_workers: int = 8
    # 每几代门控一次。门控一次 = eval_games 局（raw）+ eval_games 局（EMA），
    # 与一代自对弈的耗时同量级，所以 eval_every=2 直接省掉约 1/4 的总时长。
    # 注意：不门控的代**仍然正常训练并保存快照**，只是不更新 best_net、不降 lr。
    eval_every: int = 1
    # 评估用的根噪声。**必须 > 0**：评估走 temp=0 的贪心，没有噪声时同一先后手
    # 会走出完全相同的对局，eval_games=20 实际只有 2 个独立样本。
    eval_noise_alpha: float = 0.3
    eval_noise_eps: float = 0.25
    win_rate_threshold: float = 0.55   # 采纳阈值（点估计）。调优：0.5~0.6，高更保守
    # 可选的第二道条件：采纳还要求 **Wilson 95% 下界 > 此值**（2026-10-13 新增）。
    # **默认 0.0 = 关闭** ⇒ 判定与历史完全一致（只看点估计 > win_rate_threshold）。
    #
    # 打开它要付代价，别被规则名骗了 —— 两条是"且"关系，门槛会被顶高：
    #   * n=20（当前默认）下 0.50 ⇒ 至少要 15/20 = 0.75（14/20 的下界是 0.481）；
    #   * n=100 下 0.50 ⇒ 至少要 60/100 = 0.60（与旧的 ≥12/20 相同）。
    # 也就是说在 20 局上"加 CI"是**把门槛从 0.60 抬到 0.75**，不是"更严谨一点"。
    # 要用就同时把 --eval-games 提上去，并接受 +45% 墙钟（见上）。
    gate_ci_ll: float = 0.0
    lr_decay: float = 0.8              # 未采纳时的学习率衰减因子（旧值 0.5 衰减太快）
    # 只有胜率低于此值才降 lr（真正的退化），而不是"没打赢阈值就降"。
    # 依据：阈值 0.55 + 20 局时，新旧真实相当的模型仍有 74.8% 概率被判拒绝，
    # 若每次都降 lr，平台期会被纯统计噪声把学习率压死。
    lr_decay_min_win_rate: float = 0.30
    # 【lr 棘轮】默认 True = 旧行为（胜率低于阈值就降一次 lr）。
    # 实测代价（§10.6 + run3）：门控在 20 局下假拒率 74.8%，平台期被统计噪声
    # 一路降 lr —— run3 降了 13 次到初始值的 5.5%，越到后面越难翻身。
    # 用 `--no-lr-ratchet` 改成"必须连续 lr_decay_streak 次显著退化才降一次"。
    lr_ratchet: bool = True
    lr_decay_streak: int = 3           # --no-lr-ratchet 时的连续退化次数门槛
    lr_floor: float = 1e-4             # 学习率下限，别让它掉到无意义的量级
    max_lr_decay: int = 20             # 最多降学习率次数
    max_epochs: int = 80               # 最大迭代轮数

    # ---- 保存 ----
    # 故意留一个"显然不该用"的占位值：真正的目录必须由 `--save-dir` 显式给出
    # （2026-10-11 目录归位后，训练产物一律写 `models/<版本>/`）
    save_dir: str = "models/UNSPECIFIED"
    # 【代数接着数】本 run 的第 1 代对外叫 gen_{start_gen}。
    # 例：run3 到 gen_60，则新一轮 `--start-gen 61` → 快照叫 gen_61、gen_62…
    # 注意：**退火曲线/门控节奏仍按本 run 内部的第几代算**（否则一开跑就处于退火末端）。
    start_gen: int = 1
    # 【断点续训】每代写一份 resume.pt，含优化器动量 / EMA / 回放池 / 代数 /
    # 学习率衰减计数 / RNG。没有它，长跑被中断（实测被余额中断过一次）就只能
    # 从 gen_N.pt 重起 —— 网络权重在，但"训练进程的记忆"全丢。
    resume_save: bool = True
    resume_every: int = 1              # 每几代写一次
    save_data: bool = True             # 是否保存每代数据

    def to_dict(self) -> dict:
        """
        导出为普通 dict（pickle 安全，供多进程 worker 使用）。

        【踩过的坑】旧实现写的是 `vars(type(self)).items()` —— 那读的是**类属性**
        （也就是 dataclass 的默认值），实例上被改过的字段（例如命令行 `--batch-size`
        或逐代退火的权重）根本传不进 worker。改成逐字段 `getattr(self, k)`：
        实例值优先，未覆盖时自然回落到类默认值。
        """
        out = {}
        for k in vars(type(self)):
            if k.startswith("_"):
                continue
            if callable(getattr(type(self), k, None)):
                continue
            out[k] = getattr(self, k)
        return out

    def with_weights(self, heuristic_weight: float, policy_weight: float) -> "Config":
        """浅拷贝一份并覆盖规则先验权重（供逐代退火时传给 worker）。"""
        import copy as _copy
        c = _copy.copy(self)
        c.heuristic_weight = float(heuristic_weight)
        c.policy_weight = float(policy_weight)
        return c


def annealed_weights(gen: int, cfg: "Config"):
    """
    按代数返回 (heuristic_weight, policy_weight) 的退火值。

    分段线性：
        gen <= anneal_start            -> (h_start, p_start)
        anneal_start..anneal_mid       -> 线性插值到 (h_mid, p_mid)
        anneal_mid..anneal_end         -> 线性插值到 (h_end, p_end)
        gen >= anneal_end              -> (h_end, p_end)

    为什么必须做（见 Config 里 heuristic_weight 的说明）：固定的高规则权重会
    长期压住网络的策略学习，让 80 代训练停在"规则引擎"水平。
    """
    if not cfg.anneal_enabled:
        return cfg.heuristic_weight, cfg.policy_weight
    g0, g1, g2 = cfg.anneal_start, cfg.anneal_mid, cfg.anneal_end
    if gen <= g0 or g1 <= g0:
        return cfg.anneal_h_start, cfg.anneal_p_start
    if gen <= g1:
        t = (gen - g0) / float(g1 - g0)
        return (cfg.anneal_h_start + t * (cfg.anneal_h_mid - cfg.anneal_h_start),
                cfg.anneal_p_start + t * (cfg.anneal_p_mid - cfg.anneal_p_start))
    if g2 <= g1 or gen <= g2:
        if g2 <= g1:
            return cfg.anneal_h_mid, cfg.anneal_p_mid
        t = (gen - g1) / float(g2 - g1)
        return (cfg.anneal_h_mid + t * (cfg.anneal_h_end - cfg.anneal_h_mid),
                cfg.anneal_p_mid + t * (cfg.anneal_p_end - cfg.anneal_p_mid))
    return cfg.anneal_h_end, cfg.anneal_p_end


# ----------------------------------------------------------------------
# 经验回放池（阶段 2 新增）
# ----------------------------------------------------------------------
def _policy_entropy_weights(policies: np.ndarray, beta: float) -> np.ndarray:
    """
    【任务 4】按策略目标的信息熵给每条样本一个采样权重。

    动机（任务卡的"新颖度"）：MCTS 输出的策略目标熵低 ⇒ 这个局面已被搜索判明、
    或接近强制手，能提供的梯度信号少；熵高 ⇒ 搜索"还没想清楚"，是真正还能改进
    策略的样本。权重 w = exp(beta * H/Hmax)，Hmax = ln(361)（361 个点的均匀分布）。
    beta=0 ⇒ 全 1（等价于均匀）；beta=1 ⇒ 权重跨度 1~e≈2.72。

    只用池里已经存着的 `policies` 现算（O(N×361)，每代一次，可忽略），
    不需要动数据管线、npz 格式或 self_play_game。
    """
    p = np.asarray(policies, dtype=np.float64)
    pos = p > 0.0
    logp = np.zeros_like(p)
    logp[pos] = np.log(p[pos])
    h = -(p * logp).sum(axis=1)
    h = np.clip(h / float(np.log(p.shape[1])), 0.0, 1.0)
    return np.exp(float(beta) * h).astype(np.float32)


class ReplayBuffer:
    """
    跨代经验回放池。

    为什么必须要有它
    ----------------
    旧流程是"每代生成 10 局 → 用这十几局反复训 50 个 epoch → 丢弃"，
    有两个硬伤：
        1) 数据量极小却反复拟合 → 严重过拟合；
        2) 数据用后即弃 → 网络会遗忘前几代学到的局面，训练呈"锯齿状"。
    AlphaZero 的做法是保留最近若干代的样本混合采样，这里照此实现。

    存储设计
    --------
        states   : uint8   (0/1 的棋盘通道，比 float32 省 4 倍内存)
        policies : float32 (训练目标分布；必须保留原精度)
        values   : float32
    增强策略
    --------
        池里只存**未增强**的样本，训练时对采样出的小批量做**在线随机增强**。
        好处：内存不 ×8；每个 epoch 看到的是不同的对称变换，等价于无限增强。
        （旧实现把 8 个变换全部物化进数据集，内存放大 8 倍且每轮的增强完全相同。）

    容量
    ----
        buffer_size 条上限 + buffer_gen_limit 代上限，先到先淘汰最早的一代。
        50 万条 ≈ 500MB 内存（uint8 状态占大头），对 12GB 显存机器完全可行。
    """

    def __init__(self, capacity: int = 500_000, gen_limit: int = 30,
                 rng: Optional[np.random.Generator] = None,
                 decay_half_life: float = 0.0,
                 entropy_weight: float = 0.0):
        self.capacity = int(capacity)
        self.gen_limit = int(gen_limit)
        self.rng = rng or np.random.default_rng()
        # 【任务 4】两个整形开关；都为 0 时 sample() 走与旧版完全相同的代码路径
        self.decay_half_life = max(0.0, float(decay_half_life or 0.0))
        self.entropy_weight = max(0.0, float(entropy_weight or 0.0))
        self.shape_enabled = (self.decay_half_life > 0.0
                              or self.entropy_weight > 0.0)
        self.last_sample_stats: dict = {}   # 【任务 4】最近一次 sample() 的分布记录
        # 每一代作为一个独立条目，便于按代淘汰
        self.generations: list = []      # [{"states","policies","values","gen"}]
        self.total = 0

    # ------------------------------------------------------------------
    def add_generation(self, states, policies, values, gen: int) -> None:
        """把一代自对弈数据加入回放池。"""
        if len(states) == 0:
            return
        self.generations.append({
            "gen": int(gen),
            "states": np.asarray(states, dtype=np.uint8),
            "policies": np.asarray(policies, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
        })
        if self.entropy_weight > 0.0:
            # 【任务 4】新颖度权重在这里算一次并缓存（每代一次，O(N×361)）
            self._ensure_sample_weights(self.generations[-1])
        self.total += len(states)
        self._evict()

    def _ensure_sample_weights(self, g: dict) -> None:
        """给一代算好"新颖度"逐样本权重与前缀和（缓存进该代条目，只算一次）。"""
        w = _policy_entropy_weights(g["policies"], self.entropy_weight)
        g["sample_w"] = w
        g["sample_cumw"] = np.cumsum(w, dtype=np.float64)

    def _evict(self) -> None:
        """按"代上限"和"容量上限"淘汰最旧的数据。"""
        while len(self.generations) > self.gen_limit:
            self._pop_oldest()
        while self.total > self.capacity and len(self.generations) > 1:
            self._pop_oldest()

    def _pop_oldest(self) -> None:
        g = self.generations.pop(0)
        self.total -= len(g["states"])

    # ------------------------------------------------------------------
    @property
    def n_generations(self) -> int:
        return len(self.generations)

    def counts(self) -> list:
        """每代样本数（用于日志）。"""
        return [(g["gen"], len(g["states"])) for g in self.generations]

    # ------------------------------------------------------------------
    def gen_labels_monotone(self) -> bool:
        """
        回放池里的 `gen` 标签是否严格递增。

        不递增 = 标签撞号了（典型场景：`prepare_run.ps1` 把继承来的池子重编号成
        1..N，而新 run 又从 1 重新计数）。**采样本身不依赖这个标签**（代龄按池中
        位置算，见 `_ages()`），但撞号说明"日志里的 gen N"与"本 run 的第 N 代"
        不是一回事，看日志时要当心。
        """
        gens = [int(g["gen"]) for g in self.generations]
        return all(b > a for a, b in zip(gens, gens[1:]))

    def _ages(self) -> np.ndarray:
        """
        整池每一条的**代龄**：最新一代 = 0、每往前一代 +1。

        刻意按"池中位置"算，**不看 `g["gen"]` 标签** —— 那个标签会撞号：
        `prepare_run.ps1` 把继承来的回放池代数重编号成 1..N（脚本里那段
        `g['gen'] = i + 1`），而新 run 又从 1 重新计数，于是 `最新gen - 本条gen`
        变成**负数**，`2^(-代龄/半衰期)` 反过来把**最老**的数据抬到最高权重。

        2026-10-13 的 V4 就是这样白跑了 15 代：日志 `mean_age_exp` 从 -37.63 爬到
        -23.69、`share_recent20_seen` 恒为 1.0（设计值 ≈0.55），等于把"偏向最老
        数据"当成了"偏向最新数据"跑。回归测试见 `test_buffer_sample.py`。
        """
        n = len(self.generations)
        return np.arange(n - 1, -1, -1, dtype=np.float64)

    def sample(self, n: int, exclude_latest: bool = False):
        """
        采样 n 条样本（可重复）。

        参数:
            n              : 采样条数
            exclude_latest : True 时排除最新一代（用于取"历史"部分）
        返回:
            (states_uint8, policies_float32, values_float32)
            池为空时返回 None

        采样分布
        --------
        默认（两个整形开关都关）：先按"代"采样、概率正比于该代样本数，再在代内均匀
        采样。概率正比于样本数 ⇒ **等价于在整个池上按样本均匀采样**。（所以旧注释里
        说的"避免某代样本多就垄断 batch"其实并不成立；真正要防的是"旧代把新代淹没"，
        那得靠 buffer_decay_half_life。）
        【任务 4】打开整形后：代权重 = 样本数 × 2^(-代龄/半衰期)（最近性）；
        代内权重 = exp(β·H/Hmax)（新颖度，H = 策略目标的信息熵）。
        本次抽样的分布记录在 self.last_sample_stats。
        """
        pool = self.generations[:-1] if exclude_latest else self.generations
        if not pool or n <= 0:
            return None
        sizes = np.array([len(g["states"]) for g in pool], dtype=np.float64)
        # 代龄 = 池中位置（pool 一定是整池的前缀：要么全池、要么去掉最后一代）。
        # 不用 g["gen"] 标签，理由见 _ages()。
        ages = self._ages()[:len(pool)]
        if self.shape_enabled:
            w = sizes.copy()
            if self.decay_half_life > 0.0:
                w *= np.power(2.0, -ages / self.decay_half_life)
            probs = w / w.sum()
        else:
            # 关掉整形时保持旧代码逐位一致（同样的 probs、同样的 rng 调用序列）
            probs = sizes / sizes.sum()
        gen_idx = self.rng.choice(len(pool), size=n, p=probs)
        out_s = np.empty((n, 3, 19, 19), dtype=np.uint8)
        out_p = np.empty((n, 361), dtype=np.float32)
        out_v = np.empty((n,), dtype=np.float32)
        for gi in np.unique(gen_idx):
            mask = gen_idx == gi
            k = int(mask.sum())
            g = pool[int(gi)]
            if self.entropy_weight > 0.0:
                if "sample_cumw" not in g:
                    # 旧 resume.pt 里的代没有缓存权重 ⇒ 当场补算（每代只发生一次）
                    self._ensure_sample_weights(g)
                cumw = g["sample_cumw"]
                # 前缀和 + searchsorted：O(k·log N)，比 rng.choice(p=...) 快很多
                u = self.rng.random(k) * float(cumw[-1])
                sel = np.searchsorted(cumw, u, side="right")
                np.clip(sel, 0, len(g["states"]) - 1, out=sel)
            else:
                sel = self.rng.integers(0, len(g["states"]), size=k)
            out_s[mask] = g["states"][sel]
            out_p[mask] = g["policies"][sel]
            out_v[mask] = g["values"][sel]
        if self.shape_enabled:
            self._record_sample_stats(n, pool, probs, ages, gen_idx)
        return out_s, out_p, out_v

    def _record_sample_stats(self, n, pool, probs, ages, gen_idx) -> None:
        """【任务 4】记录最近一次抽样的代龄分布（期望值 + 实际抽到值）。"""
        sel_gen = np.array([pool[int(i)]["gen"] for i in gen_idx], dtype=np.int64)
        # 实际抽到的代的代龄：直接按池中位置取，不要再从 gen 标签反推（会撞号）
        seen_age = ages[gen_idx]
        self.last_sample_stats = {
            "half_life": self.decay_half_life,
            "entropy_weight": self.entropy_weight,
            "n_hist": int(n),
            "gens_drawn": int(len(np.unique(sel_gen))),
            "mean_age_exp": round(float((probs * ages).sum()), 2),
            "mean_age_seen": round(float(seen_age.mean()), 2),
            "share_recent20_exp": round(float(probs[ages < 20.0].sum()), 4),
            "share_recent20_seen": round(float((seen_age < 20).mean()), 4),
        }

    def weight_distribution(self) -> list:
        """【任务 4】当前"每代采样权重占比"（诊断用；未开整形时即按样本数占比）。"""
        if not self.generations:
            return []
        sizes = np.array([len(g["states"]) for g in self.generations],
                         dtype=np.float64)
        if self.shape_enabled and self.decay_half_life > 0.0:
            sizes = sizes * np.power(2.0, -self._ages() / self.decay_half_life)
        share = sizes / sizes.sum()
        # 不四舍五入：这里返回的是真权重（调用方要显示自己 round），
        # 否则 60 个 4 位小数相加会偏离 1 达 3e-3。
        return [(int(g["gen"]), float(x))
                for g, x in zip(self.generations, share)]


def augment_batch(states: np.ndarray, policies: np.ndarray,
                  rng: np.random.Generator, augment: bool = True):
    """
    对一个小批量做**在线随机**对称增强（D4 群 8 个变换，逐样本独立随机）。

    参数:
        states  : (B, 3, 19, 19)
        policies: (B, 361)  索引约定 idx = y*19 + x
    返回:
        (states_aug, policies_aug)

    几何说明：局面张量索引是 [y, x]，策略 reshape(19,19) 后同样是 [y, x]，
    因此对两者施加同一个 np.rot90/fliplr 变换在几何上完全对应。
    """
    if not augment:
        return states, policies
    b = len(states)
    k = rng.integers(0, 4, size=b)          # 旋转次数 0~3
    flip = rng.integers(0, 2, size=b)       # 是否左右翻转
    out_s = np.empty_like(states)
    out_p = np.empty_like(policies)
    p2d = policies.reshape(b, 19, 19)
    for i in range(b):
        s = np.rot90(states[i], k=int(k[i]), axes=(1, 2))
        p = np.rot90(p2d[i], k=int(k[i]))
        if flip[i]:
            s = s[:, :, ::-1]
            p = p[:, ::-1]
        out_s[i] = s
        out_p[i] = p.reshape(-1)
    return out_s, out_p


# ----------------------------------------------------------------------
# EMA（指数滑动平均权重，阶段 2 新增）
# ----------------------------------------------------------------------
class EMAWeights:
    """
    权重的指数滑动平均（EMA）。

    作用：门控抖动的主要来源是"单代训练后权重突变"。对权重做滑动平均后，
    评估用的模型更平滑，胜率曲线更稳定，通常也能白拿一点强度。
    实现只保存一份影子参数，开销可忽略（4.4M 参数）。
    """

    def __init__(self, net, decay: float = 0.999):
        import copy
        self.decay = float(decay)
        self.shadow = copy.deepcopy(net).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, net) -> None:
        """用当前网络权重更新影子权重：s = d*s + (1-d)*p"""
        d = self.decay
        with __import__("torch").no_grad():
            for s, p in zip(self.shadow.parameters(), net.parameters()):
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            # 同步 buffer 统计量（BN 的 running stats 不在 parameters 里）
            for s, p in zip(self.shadow.buffers(), net.buffers()):
                s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        self.shadow.load_state_dict(sd, strict=strict)


# ----------------------------------------------------------------------
# 数据增强（兼容旧接口）
# ----------------------------------------------------------------------
def _apply_transform(arr: np.ndarray, idx: int) -> np.ndarray:
    """
    按索引 idx(0~7) 应用对称变换：idx//2 次旋转 + idx%2 翻转。
    共 8 个变换 = 4 旋转 x 2 翻转(二面体群 D4)。
    """
    rotated = np.rot90(arr, k=idx // 2)
    if idx % 2 == 1:
        rotated = np.fliplr(rotated)
    return rotated


def augment_state_policy(state: np.ndarray, policy: np.ndarray) -> list:
    """
    对 (局面张量, 策略概率) 生成增强数据。

    几何正确性说明（已实测校验）：
        局面张量是 numpy 数组 (3,19,19)，形状与棋盘数组一致，
        索引是 [y, x]；策略是 361 维，索引是 y*19+x，reshape(19,19) 后
        同样是 [y, x]。因此对两者施加**同一个** np.rot90/fliplr 变换，
        几何上完全对应（数组变换会同时作用于两个轴，与棋盘的几何旋/翻一致）。
        曾经担心"转置导致 x/y 互换"，但那是对 (x,y) 元组手工变换才会有的问题，
        对数组做 rot90/fliplr 不存在该问题。

    参数:
        state:  (3, 19, 19) 局面张量
        policy: (361,) 策略概率 (索引约定 idx = y*19 + x)
        augment_factor: 只取前 N 个变换（1 表示不增强；>1 时生效）
    返回:
        N 个 (state_aug, policy_aug) 元组
    """
    policy_2d = policy.reshape(19, 19)
    results = []
    for i in range(8):
        state_aug = np.stack([_apply_transform(state[c], i) for c in range(3)], axis=0)
        policy_aug = _apply_transform(policy_2d, i).flatten()
        results.append((state_aug, policy_aug))
    return results


def augment_pair(state: np.ndarray, policy: np.ndarray, idx: int) -> tuple:
    """
    只生成第 idx 个对称变换（用于按 augment_factor 限制增强倍数，
    避免一次性生成 8 份再丢弃）。
    """
    policy_2d = policy.reshape(19, 19)
    state_aug = np.stack([_apply_transform(state[c], idx) for c in range(3)], axis=0)
    policy_aug = _apply_transform(policy_2d, idx).flatten()
    return state_aug, policy_aug


# ----------------------------------------------------------------------
# 自我对弈数据生成
# ----------------------------------------------------------------------
def _policy_from_probs(probs: dict) -> np.ndarray:
    """
    将 MCTS 概率字典 {(x,y): p} 转为 361 维策略数组。
    索引约定：idx = y*19 + x
    """
    policy = np.zeros(361, dtype=np.float32)
    for (x, y), p in probs.items():
        policy[int(y) * 19 + int(x)] = float(p)
    return policy


def self_play_game(mcts: MCTS, net: GomokuNet,
                   board_init: Board = None,
                   temperature=None, temp_threshold: int = 12,
                   noise_alpha: float = None, noise_eps: float = 0.25,
                   resign_threshold=None, resign_plies: int = 3) -> list:
    """
    用当前模型 MCTS 自我对弈一局，返回每步数据。

    参数:
        mcts:      MCTS 搜索器
        net:       当前模型
        board_init: 初始棋盘(默认空棋盘)
        temperature: 采样温度覆盖值；None 表示按手数退火
        temp_threshold: 前 N 手使用 temperature（若给出）
        noise_alpha: 根节点 Dirichlet 噪声 alpha（None/<=0 关闭）
        noise_eps:   噪声混合比例
        resign_threshold: 价值低于该值且连续 resign_plies 手则认输（None 关闭）
        resign_plies:    认输判定所需连续手数
    返回:
        列表 [(state, policy, value), ...]
            state:  (3, 19, 19)
            policy: (361,)  MCTS 策略概率
            value:  ±1 或 0  该步执子方视角的最终胜负标签

    修复要点：
        * [温度采样] 旧实现 `best = max(probs, key=probs.get)` 对 temp=1.0 的
          访问概率直接取 argmax，等于把温度完全作废 —— 同一模型下每局逐手相同，
          训练数据毫无多样性。现改为 mcts.sample_move() 真正按温度采样。
        * [根噪声] 加入 Dirichlet 噪声，保证同一局面也能走出不同变化。
        * [认输] 价值头持续看衰则提前认输，省掉大量无意义的百手残局；
          认输局的数据**不参与价值训练**（用 value=None 标记），避免网络学会乱认输。
    """
    board = Board() if board_init is None else board_init.copy()

    steps = []          # 暂存 (state, policy, player)
    resign_step = None  # 认输发生的步数（该步之后的数据不用于价值训练）
    low_value_streak = 0

    while True:
        state = board.get_state_tensor()

        n_plies = len(board.history)
        if temperature is None:
            # 温度退火：开局随机、中盘半随机、后盘贪心
            if n_plies < temp_threshold:
                temp = 1.0
            elif n_plies < 30:
                temp = 0.5
            else:
                temp = 0.0
        else:
            temp = temperature if n_plies < temp_threshold else 0.0

        mv, policy = mcts.sample_move_with_policy(
            board, temp=temp, noise_alpha=noise_alpha, noise_eps=noise_eps)
        if mv is None:
            break

        steps.append((state, policy, board.current_player))

        # 认输判定（用当前执子方视角的网络价值）
        if resign_threshold is not None:
            _, v = net.policy_value(state)
            if v < resign_threshold:
                low_value_streak += 1
            else:
                low_value_streak = 0
            if low_value_streak >= resign_plies:
                resign_step = len(steps) - 1
                break

        x, y = int(mv[0]), int(mv[1])
        if not board.move(x, y, board.current_player):
            break
        mcts.set_root_move((x, y))

        if board.is_game_over()[0]:
            break

    over, winner = board.is_game_over()

    # 认输时把"认输方"判为负
    resigned_player = None
    if resign_step is not None and not over:
        resigned_player = steps[resign_step][2]
        winner = 3 - resigned_player

    result = []
    for i, (state, policy, player) in enumerate(steps):
        if winner == 0:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        # 认输之后（含认输那步）的数据价值标签不可信，标记为无效
        usable = (resign_step is None) or (i < resign_step)
        result.append((state, policy, value, usable))
    return result


def _search_policy(mcts: MCTS, board: Board, chosen_move) -> np.ndarray:
    """
    取当前搜索的访问次数分布作为策略训练目标。

    说明：sample_move 内部已经跑完搜索，根节点的访问次数就在树上，
    因此这里直接读取根的访问分布，避免像旧实现那样为了拿分布再搜一次
    （旧实现是"同一次搜索结果既当策略又取 argmax"，本身没问题，
     但新接口 sample_move 只返回落点，所以这里补一个读取函数）。
    """
    root = mcts._root
    policy = np.zeros(361, dtype=np.float32)
    if root is None or not root.children:
        if chosen_move is not None:
            policy[int(chosen_move[1]) * 19 + int(chosen_move[0])] = 1.0
        return policy
    counts = {mv: c.n for mv, c in root.children.items() if c.n > 0}
    total = sum(counts.values())
    if total <= 0:
        if chosen_move is not None:
            policy[int(chosen_move[1]) * 19 + int(chosen_move[0])] = 1.0
        return policy
    for (x, y), n in counts.items():
        policy[int(y) * 19 + int(x)] = n / total
    return policy


def _worker_run_games(args):
    """
    多进程 worker：独立进程内生成若干局自对弈数据。

    修复：旧实现直接引用模块级 `cfg`，但子进程不会继承调用方的局部变量，
    因此 `cfg` 未定义 -> NameError，而 Config.n_workers 默认 4 会直接走这条
    分支，等于并行数据生成一开始就是坏的。
    现在 cfg 以普通 dict 显式通过 args 传入（pickle 安全）。
    """
    import torch
    from gomoku_env import Board
    from network import GomokuNet
    from mcts import MCTS
    import numpy as np

    (net_path, n_games, cfg, augment, augment_factor, seed, mcts_kw) = args
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    net = load_checkpoint(net_path, verbose=False)
    net.eval()

    mcts = MCTS(net, n_sim=cfg["n_sim"], c_puct=cfg["c_puct"],
                virtual_loss=cfg["virtual_loss"],
                batch_size=cfg["batch_size_mcts"],
                max_children=cfg["max_children"],
                root_max_children=cfg["root_max_children"],
                heuristic_weight=cfg["heuristic_weight"],
                policy_weight=cfg["policy_weight"], seed=seed)
    all_states, all_policies, all_values = [], [], []
    for gi in range(n_games):
        mcts.reset_tree()
        game = self_play_game(
            mcts, net,
            temperature=cfg["temperature"],
            temp_threshold=cfg["temp_threshold"],
            noise_alpha=cfg["root_noise_alpha"],
            noise_eps=cfg["root_noise_eps"],
            resign_threshold=cfg["resign_threshold"],
            resign_plies=cfg["resign_plies"])
        for state, policy, value, usable in game:
            if not usable and cfg["resign_threshold"] is not None:
                # 认输后的数据不参与价值训练：用 value=0 且权重靠 mask 处理，
                # 这里简化为直接丢弃（等价于不使用该段数据）
                continue
            if augment:
                for i in range(augment_factor):
                    s_aug, p_aug = augment_pair(state, policy, i)
                    all_states.append(s_aug)
                    all_policies.append(p_aug)
                    all_values.append(value)
            else:
                all_states.append(state)
                all_policies.append(policy)
                all_values.append(value)
    if not all_states:
        return (np.zeros((0, 3, 19, 19), dtype=np.uint8),
                np.zeros((0, 361), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    # states 只含 0/1，用 uint8（内存省 4 倍）；训练时在 train_step 里转 float32
    return (np.array(all_states, dtype=np.uint8),
            np.array(all_policies, dtype=np.float32),
            np.array(all_values, dtype=np.float32))


def collect_training_data(mcts: MCTS, net: GomokuNet,
                          n_games: int, cfg: Optional[Config] = None,
                          augment: bool = True,
                          augment_factor: int = 8, n_workers: int = 1,
                          seed: int = None) -> tuple:
    """
    收集 n_games 局自对弈数据，做数据增强，返回堆叠后的张量。

    参数:
        cfg: 配置对象（并行路径需要，用于把参数传给子进程）。
             旧实现在 worker 里引用未定义的全局 cfg，导致并行路径必然崩溃。

    优化点:
        * 每局打印进度
        * 每局结束后重置 MCTS 树（新对局新树）
        * [Parallel] 多 worker 并行生成对局数据（GPU 是瓶颈时建议 n_workers=1）
    """
    if cfg is None:
        cfg = Config()

    def _iter_games(n: int, m: MCTS):
        """单进程生成 n 局的数据（增强后 yield）。"""
        for gi in range(n):
            m.reset_tree()
            if gi % max(1, n // 10) == 0 or gi == n - 1:
                pct = (gi + 1) * 100 // n
                bar_len = 30
                filled = int(bar_len * gi / n)
                bar = "#" * filled + "." * (bar_len - filled)
                print(f"    对局进度: [{bar}] {gi + 1}/{n} ({pct}%)",
                      end="\r", flush=True)
            game = self_play_game(
                m, net,
                temperature=cfg.temperature,
                temp_threshold=cfg.temp_threshold,
                noise_alpha=cfg.root_noise_alpha,
                noise_eps=cfg.root_noise_eps,
                resign_threshold=cfg.resign_threshold,
                resign_plies=cfg.resign_plies)
            for state, policy, value, usable in game:
                if not usable and cfg.resign_threshold is not None:
                    continue
                if augment:
                    for i in range(augment_factor):
                        yield augment_pair(state, policy, i) + (value,)
                else:
                    yield state, policy, value

    if n_workers <= 1 or n_games < 2:
        all_states, all_policies, all_values = [], [], []
        for s, p, v in _iter_games(n_games, mcts):
            all_states.append(s)
            all_policies.append(p)
            all_values.append(v)
        print(f"    对局进度: [{'#' * 30}] {n_games}/{n_games} (100%)", flush=True)
        if not all_states:
            return (np.zeros((0, 3, 19, 19), dtype=np.uint8),
                    np.zeros((0, 361), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        # states 只含 0/1，用 uint8 存储（内存/磁盘省 4 倍），训练时再转 float32
        return (np.array(all_states, dtype=np.uint8),
                np.array(all_policies, dtype=np.float32),
                np.array(all_values, dtype=np.float32))

    import multiprocessing
    import tempfile
    import os as _os

    # 用 NamedTemporaryFile 避免 mktemp 的竞态，并保证异常时也能清理
    fd, tmp_net_path = tempfile.mkstemp(suffix=".pt")
    _os.close(fd)
    try:
        torch.save({"state_dict": net.state_dict()}, tmp_net_path)
        cfg_dict = cfg.to_dict()
        games_per_worker = n_games // n_workers
        remainder = n_games % n_workers
        base_seed = seed if seed is not None else int(time.time()) % 100000
        worker_args = []
        for wi in range(n_workers):
            ng = games_per_worker + (1 if wi < remainder else 0)
            if ng > 0:
                # 每个 worker 用不同种子，避免生成完全相同的对局
                worker_args.append((tmp_net_path, ng, cfg_dict, augment,
                                    augment_factor, base_seed + wi * 7919, None))
        print(f"    并行生成: {len(worker_args)} workers × "
              f"{games_per_worker}~{games_per_worker + 1} 局")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_run_games, worker_args)
    finally:
        if _os.path.exists(tmp_net_path):
            _os.remove(tmp_net_path)

    all_states, all_policies, all_values = [], [], []
    for states, policies, values in results:
        if len(states):
            all_states.append(states)
            all_policies.append(policies)
            all_values.append(values)
    if not all_states:
        return (np.zeros((0, 3, 19, 19), dtype=np.uint8),
                np.zeros((0, 361), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    all_states = np.concatenate(all_states, axis=0)
    all_policies = np.concatenate(all_policies, axis=0)
    all_values = np.concatenate(all_values, axis=0)
    print(f"    对局完成: {len(all_states)} 条数据", flush=True)
    return (all_states, all_policies, all_values)


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
def _to_tensor(a: np.ndarray):
    """numpy -> torch tensor (float32)。"""
    import torch
    return torch.from_numpy(a).float()


def ce_loss(policy: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    策略交叉熵损失。
    target 为 361 维概率分布(可能含 0)，加 epsilon 防 log(0)，并归一化。
    """
    import torch
    target = target + 1e-8
    target = target / target.sum(dim=1, keepdim=True)
    return -torch.sum(target * torch.log(policy + 1e-8), dim=1).mean()


def mse_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """价值均方误差损失。"""
    import torch
    return torch.mean((pred - target) ** 2)


# ----------------------------------------------------------------------
# 【任务 6】辅助头的监督目标
# ----------------------------------------------------------------------
#
# 【关键设计：目标在训练步里现算，不改数据格式】
# 自对弈产出 `(state, policy, value)` 三元组，并在 `save_generation_data` 里写进
# npz；回放池 `ReplayBuffer` 按代存 `states/policies/values`。辅助头的目标全部
# 可以**从 state + value 推出来**，所以：
#   * 不用动 `self_play_game`、不用动并行 worker 的返回协议、不用动 npz 格式、
#     不用动回放池 —— 老数据（含磁盘上已存的 21,000 局）**天然带标签**，
#     热启动后立刻可用；
#   * 目标在 `augment_batch()` **之后**算，与增强后的棋盘几何完全一致
#     （若提前算好再增强，旋转/翻转会让标签和棋盘错位）。
# 代价是每个 batch 多花几次 numpy 切片（19×19×8，毫秒级），可以忽略。
def _win_points(stones: np.ndarray) -> np.ndarray:
    """
    给定某一方的棋子布局 (B,19,19)（0/1），返回"该方在此落子即连五"的空点掩码
    (B,19,19) bool。

    判据：空点 (y,x) 放上该方棋子后，四个方向中存在一个方向使连子数 ≥5，即该点
    在某个方向上**紧邻同色子数 ≥4**（两侧合计）。实现：四周 pad 4 圈 0，于是任意
    方向偏移都能表达成整块切片，一次算完整批（8 组切片，毫秒级）。

    ⚠️ 必须是"**某个方向** ≥4"，不能把四个方向的计数加起来 —— 四个方向各 1 个
    同色子加起来也是 4，但那种棋型一步根本连不成五。
    """
    b = stones.shape[0]
    s = (stones > 0).astype(np.uint8)
    P = np.pad(s, ((0, 0), (4, 4), (4, 4)))
    win = np.zeros((b, 19, 19), dtype=bool)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        run = np.zeros((b, 19, 19), dtype=np.int8)
        for sign in (-1, +1):
            acc = np.ones((b, 19, 19), dtype=bool)
            for t in (1, 2, 3, 4):
                y0 = 4 + sign * t * dy
                x0 = 4 + sign * t * dx
                acc &= P[:, y0:y0 + 19, x0:x0 + 19] > 0
                run += acc
        win |= (run >= 4)
    return win & (s == 0)


def aux_targets(states: np.ndarray, values: np.ndarray, names) -> dict:
    """
    从局面张量与对局结果算出辅助头的监督目标。

    参数:
        states: (B,3,19,19) uint8（通道约定见 `gomoku_env.Board.get_state_tensor`：
                ch0=黑子, ch1=白子, ch2=行棋方平面）
        values: (B,) float32，**行棋方视角**的结果（+1 赢 / -1 输 / 0 和）——
                与 `self_play_game` 写进数据的 value 同义。
        names : 需要哪些目标（"threat" / "ownership" 的子集）
    返回:
        {"threat": (B,2,19,19) float32, "ownership": (B,1,19,19) float32}

    threat（**精确标签**，无近似）：
        ch0 = 行棋方落此处即连五（一步成五）；ch1 = 对方落此处即连五（必挡点）。
        两者都直接由棋盘算出，是这个局面里可以 100% 确证的事实。

    ownership（**代理标签**，非 KataGo 式终局归属）：
        目标 = sign(value) × (己方子 +1 / 对方子 −1 / 空点 0)。
        含义是"从价值头的结论往回推：谁赢了，赢家占的地方就是他的地盘"。
        为什么用代理：真正的终局归属需要**终局棋盘**，而 npz 里只存了中间局面，
        要拿到就得改自对弈 → worker → 回放池的整条链路（任务要求只改两个文件）。
        代理的信息量低于终局归属（空点上的争夺完全没体现），但足以逼主干学出
        "这盘棋的地盘长什么样"，且零链路改动。**若将来要上真标签，只需在这里
        换目标函数，不改任何调用方。**
    """
    out = {}
    if not names:
        return out
    s = np.asarray(states)
    v = np.asarray(values, dtype=np.float32).reshape(-1)

    black = s[:, 0] > 0                          # (B,19,19)
    white = s[:, 1] > 0
    # 行棋方平面：任意一点上该通道 >0 即表示"轮黑走"（见 gomoku_env）
    black_to_move = s[:, 2, 0, 0] > 0            # (B,)
    me = np.where(black_to_move[:, None, None], black, white)
    opp = np.where(black_to_move[:, None, None], white, black)

    if "threat" in names:
        t = np.stack([_win_points(me), _win_points(opp)], axis=1)   # (B,2,19,19)
        out["threat"] = t.astype(np.float32)

    if "ownership" in names:
        sign = np.sign(v).astype(np.float32)                        # (B,)
        own = (me.astype(np.float32) - opp.astype(np.float32))
        out["ownership"] = (own * sign[:, None, None])[:, None, :, :]

    return out


def train_step(net: GomokuNet, fresh: tuple, cfg: Config, optimizer,
               buffer: "ReplayBuffer" = None, rng=None,
               ema: "EMAWeights" = None) -> tuple:
    """
    用"当代新数据 + 回放池历史数据"混合训练一个 step。

    参数:
        fresh   : (states, policies, values) 当代新数据（未增强）
        buffer  : 回放池；为 None 时只用新数据（退化为旧行为）
        rng     : numpy Generator（在线增强与采样用）
        ema     : EMA 影子权重；不为 None 时每个 step 后更新
    返回:
        loss_policy, loss_value, loss_total

    与旧实现的关键差别：
        * 旧实现把整个数据集按 batch_size 切一遍算作一个 epoch，反复跑 epochs 次；
          现在改为"固定步数的随机小批量采样"，可以无缝混合历史数据。
        * 增强从"提前物化 8 份"改为"每个 batch 在线随机变换"。
    """
    import torch
    net.train()
    rng = rng or np.random.default_rng()
    f_states, f_policies, f_values = fresh
    b = cfg.batch_size

    n_fresh = 0
    n_hist = 0
    if buffer is not None and buffer.n_generations > 0 and len(f_states) > 0:
        n_fresh = int(round(b * cfg.fresh_ratio))
        n_hist = b - n_fresh
        # 历史样本：排除最新一代（即刚加入的 fresh），避免同一批重复计入
        hist = buffer.sample(n_hist, exclude_latest=True)
        if hist is None:
            n_fresh, n_hist = b, 0
    else:
        n_fresh = b

    parts_s, parts_p, parts_v = [], [], []
    if n_fresh > 0 and len(f_states) > 0:
        sel = rng.integers(0, len(f_states), size=n_fresh)
        parts_s.append(f_states[sel])
        parts_p.append(f_policies[sel])
        parts_v.append(f_values[sel])
    if n_hist > 0:
        hs, hp, hv = hist
        parts_s.append(hs)
        parts_p.append(hp)
        parts_v.append(hv)

    s_np = np.concatenate(parts_s, axis=0)
    p_np = np.concatenate(parts_p, axis=0).astype(np.float32)
    v_np = np.concatenate(parts_v, axis=0).astype(np.float32)

    # 在线随机增强
    s_np, p_np = augment_batch(s_np, p_np, rng, augment=cfg.on_the_fly_augment)

    s = torch.from_numpy(np.ascontiguousarray(s_np)).float().to(net.device)
    p = torch.from_numpy(np.ascontiguousarray(p_np)).float().to(net.device)
    v = torch.from_numpy(np.ascontiguousarray(v_np)).float().to(net.device)

    # 【任务 6】有辅助头时走"一次骨干前向、三路输出"的训练路径；没有辅助头时
    # 就是原来的两路 —— 老配置的行为逐字节不变。
    aux_names = tuple(getattr(net, "aux_names", ()))
    aux_out = {}
    if aux_names:
        policy, value, aux_out = net.forward_with_aux(s)
    else:
        policy, value = net.forward(s)

    loss_policy = ce_loss(policy, p)
    loss_value = mse_loss(value, v)
    loss = cfg.weight_policy * loss_policy + cfg.weight_value * loss_value

    # 辅助损失：默认权重 0.15，**刻意远小于主损失**。辅助头是表示学习/正则手段，
    # 不是训练目标本身；权重调大会把主干拽去拟合辅助任务而伤棋力。
    # 目标用 s_np（**已增强**的局面）现算，见 aux_targets 的说明。
    aux_log = {}
    if aux_out:
        import torch.nn.functional as F
        tgt = aux_targets(s_np, v_np, tuple(aux_out.keys()))
        if "threat" in aux_out:
            tw = torch.from_numpy(np.ascontiguousarray(tgt["threat"]))
            tw = tw.float().to(net.device)
            pos_w = torch.tensor(float(cfg.aux_threat_pos_weight),
                                 device=net.device)
            loss_threat = F.binary_cross_entropy_with_logits(
                aux_out["threat"], tw, pos_weight=pos_w)
            loss = loss + cfg.aux_weight_threat * loss_threat
            aux_log["loss_threat"] = float(loss_threat.item())
        if "ownership" in aux_out:
            ow = torch.from_numpy(np.ascontiguousarray(tgt["ownership"]))
            ow = ow.float().to(net.device)
            loss_own = mse_loss(aux_out["ownership"], ow)
            loss = loss + cfg.aux_weight_ownership * loss_own
            aux_log["loss_ownership"] = float(loss_own.item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
    optimizer.step()

    if ema is not None:
        ema.update(net)

    # 供训练循环记录（仅最后一步的值；循环里再做 running mean）
    net.last_aux = aux_log

    return loss_policy.item(), loss_value.item(), loss.item()


# ----------------------------------------------------------------------
# 评估：新模型 vs 旧模型 对弈
# ----------------------------------------------------------------------
def play_match(mcts_a: MCTS, mcts_b: MCTS) -> int:
    """
    两模型对弈一局，返回胜者(1黑/2白/0平)。
    mcts_a 执黑先手, mcts_b 执白后手。

    注意：双方共用同一块棋盘，但各自持有独立的搜索树。
    每手落子后调用 set_root_move 复用该方的子树（命中则省一次搜索）。
    """
    board = Board()
    mcts_a.reset_tree()
    mcts_b.reset_tree()

    while True:
        over, winner = board.is_game_over()
        if over:
            return winner
        me = board.current_player
        m = mcts_a if me == 1 else mcts_b
        best = m.get_best_action(board)
        if best is None:
            return 0
        x, y = int(best[0]), int(best[1])
        if not board.move(x, y, me):
            return 0
        m.set_root_move((x, y))


def _mcts_kwargs_from(m: MCTS) -> dict:
    """从已有的 MCTS 实例抽出构造参数（供子进程重建同参数的搜索器）。"""
    return {
        "n_sim": m.n_sim, "c_puct": m.c_puct, "virtual_loss": m.virtual_loss,
        "batch_size": m.batch_size, "max_children": m.max_children,
        "root_max_children": m.root_max_children,
        "fpu_reduction": m.fpu_reduction,
        "prior_fpu_weight": m.prior_fpu_weight,
        "root_noise_alpha": m.root_noise_alpha,
        "root_noise_eps": m.root_noise_eps,
        "use_heuristics": m.use_heuristics,
        "heuristic_weight": m.heuristic_weight,
        "policy_weight": m.policy_weight,
        "terminal_blend": m.terminal_blend,
        "open_four_win": m.open_four_win,
        "immediate_win_loss": m.immediate_win_loss,
    }


# ----------------------------------------------------------------------
# 门控统计：Wilson 区间 / Elo 换算 / 颜色记账
# ----------------------------------------------------------------------
# 【为什么和 ladder.py 用同一套公式】项目里已经有 `ladder.py` 的 wilson()/
# elo_from_score()/elo_ci()。门控若自己另写一套、口径稍有差别，就会出现
# "同一个模型在门控里 +40 分、在标尺里 -10 分"这种没法解释的读数。
# 下面几个函数与 ladder.py **同式**，改任何一个都要同步另一个；
# `verify_gate_protocol.py` 会拿 ladder.py 做数值一致性抽查。
GATE_Z = 1.96


def wilson_interval(k, n, z=GATE_Z):
    """二项比例 k/n 的 Wilson 95% 区间（小样本比正态近似稳）。n=0 → (0,1)。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (c - half) / d), min(1.0, (c + half) / d)


def elo_from_score(s):
    """得分率 -> Elo 差（400 刻度）。0/1 换算不出（返回 None）。"""
    if s is None or s <= 0.0 or s >= 1.0:
        return None
    return 400.0 * math.log10(s / (1.0 - s))


def elo_from_ci(k, n):
    """由得分率与其 Wilson 区间给出 (elo, ci_lo, ci_hi)。

    端点用 1/(2n) 的连续性修正顶开（与 ladder.elo_ci 同法），
    否则"全胜/全败"会给出 ±inf，把日志和表格撑坏。
    """
    if n <= 0:
        return None, None, None
    lo, hi = wilson_interval(k, n)
    eps = 1.0 / (2.0 * n)
    lo_c = min(max(lo, eps), 1.0 - eps)
    hi_c = min(max(hi, eps), 1.0 - eps)
    return (elo_from_score(k / n) if 0 < k < n else None,
            elo_from_score(lo_c), elo_from_score(hi_c))


def color_bias_elo(wins_black, n_black, wins_white, n_white):
    """颜色偏差 = "新模型执黑得分率"与"执白得分率"相差多少 Elo。

    【它衡量什么、不衡量什么】黑白各半时**总胜率本来就是无偏的**，
    所以这个数不是"偏差有没有影响结论"，而是用来抓两类问题：
      * 颜色记账写反了（某一颜色恒 0/1）；
      * 先后手极端不对称 —— 15×15 五子棋先手优势真实存在，会在这里显形。
    端点和 elo_from_ci 一样用 1/(2n) 顶开，避免 ±inf。
    """
    if n_black <= 0 or n_white <= 0:
        return None
    eps_b = 1.0 / (2.0 * n_black)
    eps_w = 1.0 / (2.0 * n_white)
    sb = min(max(wins_black / n_black, eps_b), 1.0 - eps_b)
    sw = min(max(wins_white / n_white, eps_w), 1.0 - eps_w)
    eb, ew = elo_from_score(sb), elo_from_score(sw)
    if eb is None or ew is None:
        return None
    return eb - ew


def make_gate_stats(wins, n, wins_black, n_black, wins_white, n_white, draws):
    """把颜色记账拼成一份门控统计（不含阈值判定，判定见 gate_verdict()）。"""
    lo, hi = wilson_interval(wins, n)
    elo, elo_lo, elo_hi = elo_from_ci(wins, n)
    # 和棋算半分只用于 Elo 换算；判定仍用 win_rate（与历史日志同口径）
    score_half = ((wins + 0.5 * draws) / n) if n else 0.0
    return {
        "games": n, "wins": wins, "draws": draws,
        "losses": max(0, n - wins - draws),
        "win_rate": (wins / n) if n else 0.0,
        "score_half": score_half,
        "wilson_lo": lo, "wilson_hi": hi,
        "elo": elo, "elo_lo": elo_lo, "elo_hi": elo_hi,
        "games_black": n_black, "wins_black": wins_black,
        "games_white": n_white, "wins_white": wins_white,
        "score_black": (wins_black / n_black) if n_black else None,
        "score_white": (wins_white / n_white) if n_white else None,
        "color_bias_elo": color_bias_elo(wins_black, n_black,
                                         wins_white, n_white),
    }


def gate_verdict(stats, threshold, ci_ll):
    """门控判定：**胜率严格 > threshold**；ci_ll > 0 时**还要求 Wilson 下界 > ci_ll**。

    ci_ll 默认 0.0 = 第二道条件**关闭** ⇒ 判定与历史完全一致（只看点估计）。
    【一旦打开，别被规则名骗了 —— 两条是"且"，门槛会被顶高】
      * 点估计 >0.55      ⇒ n=20 至少 12/20、n=100 至少 56/100；
      * Wilson 下界 >0.50 ⇒ n=20 至少 **15/20 = 0.75**（14/20 的下界是 0.481）、
                            n=100 至少 60/100。
    所以在 20 局上打开它 = 把门槛从 0.60 抬到 0.75（几乎全拒）；在 100 局上它
    给出的是 0.60 —— 与旧的 ≥12/20 相同。也就是说**加局数买到的是"同门槛下
    少误判"，不是"更低的门槛"**（p=0.50 时拒绝概率反而从 74.8% 涨到 97.2%）。
    `verify_gate_protocol.py` 会把有效阈值与功效表算出来，别靠字面理解。
    """
    st = dict(stats)
    st["threshold"] = threshold
    st["ci_ll"] = ci_ll
    st["pass_score"] = bool(st["win_rate"] > threshold)
    st["pass_ci"] = bool(ci_ll <= 0 or st["wilson_lo"] > ci_ll)   # ci_ll<=0 = 关闭该条件
    st["accepted"] = bool(st["pass_score"] and st["pass_ci"])
    return st


def fmt_gate_stats(stats):
    """一行人类可读的门控摘要（打印 + 日志都用它，保证两边一致）。"""
    def _f(v):
        return "n/a" if v is None else f"{v:+.1f}"
    return (f"胜率 {stats['win_rate']:.3f} ({stats['wins']}/{stats['games']}) "
            f"Wilson[{stats['wilson_lo']:.3f},{stats['wilson_hi']:.3f}] "
            f"Elo {_f(stats['elo'])}"
            f"[{_f(stats['elo_lo'])},{_f(stats['elo_hi'])}] "
            f"平{stats['draws']} "
            f"黑{stats['wins_black']}/{stats['games_black']} "
            f"白{stats['wins_white']}/{stats['games_white']} "
            f"色偏{_f(stats['color_bias_elo'])}")


def _worker_eval_games(args):
    """
    多进程跑一批"新 vs 旧"评估对局（门控用）。

    返回 **(胜局, 局数, 执黑胜, 执黑数, 执白胜, 执白数, 和棋)** ——
    必须分颜色记账：只回一个总数就查不出"颜色写反了"，
    而项目历史上正好在这上面栽过一次（见 evaluate_models_stats 的踩坑说明）。

    【为什么并行不偏置判定】
      * `play_match` 每局内部都 `reset_tree()`，对局之间本来就互不影响；
      * 颜色分配**按局号**决定（偶数局新模型执黑、奇数局执白），与串行版一致，
        子进程只是把不同局号分给不同进程 —— 不会出现"某个颜色全胜"的记账错；
      * 门控的 MCTS 本来就是 `seed=None`（每次运行随机初值，见 NOTES §13.4），
        所以并行**没有**破坏可复现性 —— 它本来就不具备可复现性。
    """
    (tmp_new, tmp_old, game_ids, kw, seed) = args
    import numpy as np                     # noqa: F401  (子进程需自行 import)
    import torch                           # noqa: F401
    net_new = load_checkpoint(tmp_new, verbose=False)
    net_old = load_checkpoint(tmp_old, verbose=False)
    net_new.eval()
    net_old.eval()
    m_new = MCTS(net_new, seed=seed, **kw)
    m_old = MCTS(net_old, seed=seed + 7919, **kw)
    wins = draws = 0
    wins_black = n_black = wins_white = n_white = 0
    for g in game_ids:
        if g % 2 == 0:
            winner = play_match(m_new, m_old)
            color = 1
        else:
            winner = play_match(m_old, m_new)
            color = 2
        if winner == 0:
            draws += 1
        if color == 1:
            n_black += 1
            if winner == color:
                wins_black += 1
        else:
            n_white += 1
            if winner == color:
                wins_white += 1
        if winner == color:
            wins += 1
    return (wins, len(game_ids), wins_black, n_black,
            wins_white, n_white, draws)


def _evaluate_models_parallel(net_new, net_old, n_games, mcts_new,
                              n_workers) -> dict:
    """
    把 n_games 局按局号轮分给 n_workers 个进程（见 `_worker_eval_games`），
    返回汇总后的门控统计字典。

    实测收益：门控一次 = `eval_games` 局（raw）+ `eval_games` 局（EMA），
    默认 20+20 局，串行要 3 分钟左右；并行后与自对弈同量级地缩到几十秒。
    """
    import multiprocessing
    import tempfile

    kw = _mcts_kwargs_from(mcts_new)
    fds = []
    try:
        paths = []
        for net in (net_new, net_old):
            fd, p = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            torch.save({"state_dict": net.state_dict(),
                        "arch": net.arch()}, p)
            fds.append(p)
            paths.append(p)
        base_seed = int(time.time()) % 100000
        jobs = []
        for wi in range(n_workers):
            ids = [g for g in range(n_games) if g % n_workers == wi]
            if ids:
                jobs.append((paths[0], paths[1], ids, kw,
                             base_seed + wi * 7919))
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(jobs)) as pool:
            results = pool.map(_worker_eval_games, jobs)
        wins = sum(r[0] for r in results)
        n = sum(r[1] for r in results)
        wb = sum(r[2] for r in results)
        nb = sum(r[3] for r in results)
        ww = sum(r[4] for r in results)
        nw = sum(r[5] for r in results)
        dr = sum(r[6] for r in results)
        print(f"    门控并行: {len(jobs)} 进程 × "
              f"{'/'.join(str(r[1]) for r in results)} 局", flush=True)
        return make_gate_stats(wins, n, wb, nb, ww, nw, dr)
    finally:
        for p in fds:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


def evaluate_models_stats(net_new: GomokuNet, net_old: GomokuNet,
                          n_games: int, mcts_new: MCTS,
                          mcts_old: MCTS = None,
                          n_workers: int = 1) -> dict:
    """
    新模型 vs 旧模型 对弈，返回**完整统计字典**（胜率 / Wilson 区间 / Elo /
    颜色记账 / 和棋），而不是一个光秃秃的标量。

    新模型先手/后手交替（偶数局执黑、奇数局执白），**黑白各半**；
    局数为奇数时自动 +1 局跑 —— 黑白无法各半的局数会污染颜色记账。

    返回键：games/wins/losses/draws/win_rate/score_half/wilson_lo/wilson_hi/
    elo/elo_lo/elo_hi/games_black/wins_black/games_white/wins_white/
    score_black/score_white/color_bias_elo。

    【踩过的大坑 —— 这个 bug 会让门控完全失效】
    旧实现两个分支都写 `new_won = (winner == 1)`：

        if g % 2 == 0:
            winner = play_match(mcts_new, mcts_old)
            new_won = (winner == 1)        # 新模型执黑，正确
        else:
            winner = play_match(mcts_old, mcts_new)
            new_won = (winner == 1)        # ← BUG：这一局新模型执白！

    而 `play_match(mcts_a, mcts_b)` 里 **mcts_a 永远执黑**。所以后半数对局
    记的是"黑方（也就是旧模型）有没有赢"，跟新模型毫无关系。
    实测后果（这次长跑的真实日志）：

        gen 2  raw=0.500  -> 拒绝
        gen 4  raw=0.000  -> 拒绝     ← 不是"新模型全败"，是"白方全胜"
        gen 6  raw=0.000  -> 拒绝
        gen 8  raw=0.500  -> 拒绝

    0.000 的真实含义是"20 局全是白胜"：偶数局新模型执黑输了不计分，
    奇数局旧模型执黑输了也不计分 —— 于是新模型一分没拿到。
    也就是说门控一直在用噪声做决定，还顺手把学习率降了 4 次。

    教训：**"胜率"这类聚合指标必须按"谁执哪一方"显式记账**，
    两个分支的判定条件不可能相同 —— 看到两行一模一样的比较就该警觉。
    （上面的分颜色记账就是为了让这个 bug 再也不可能悄悄复现。）
    """
    if n_games % 2 == 1:
        print(f"    [提示] 门控局数为奇数 {n_games}，黑白无法各半 "
              f"⇒ 按 {n_games + 1} 局跑", flush=True)
        n_games += 1
    if n_workers and n_workers > 1 and n_games >= n_workers:
        return _evaluate_models_parallel(net_new, net_old, n_games,
                                         mcts_new, n_workers)
    if mcts_old is None:
        raise ValueError("串行评估需要 mcts_old（旧模型执白的那一方）")
    wins = draws = 0
    wins_black = n_black = wins_white = n_white = 0
    for g in range(n_games):
        if g % 2 == 0:
            winner = play_match(mcts_new, mcts_old)
            new_color = 1              # 新模型执黑
        else:
            winner = play_match(mcts_old, mcts_new)
            new_color = 2              # 新模型执白
        if winner == 0:
            draws += 1
        if new_color == 1:
            n_black += 1
            if winner == new_color:
                wins_black += 1
        else:
            n_white += 1
            if winner == new_color:
                wins_white += 1
        if winner == new_color:
            wins += 1
        # 进度显示
        pct = (g + 1) * 100 // n_games
        bar_len = 20
        filled = int(bar_len * g / n_games)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"    评估进度: [{bar}] {g + 1}/{n_games} ({pct}%)", end="\r", flush=True)

    print(f"    评估进度: [{'#' * 20}] {n_games}/{n_games} (100%)", flush=True)
    return make_gate_stats(wins, n_games, wins_black, n_black,
                           wins_white, n_white, draws)


def evaluate_models(net_new: GomokuNet, net_old: GomokuNet,
                    n_games: int, mcts_new: MCTS, mcts_old: MCTS = None,
                    n_workers: int = 1) -> float:
    """向后兼容的薄封装：**只返回胜率**（`test_eval_parallel.py` 依赖 float）。

    需要 CI / Elo / 颜色记账请直接用 `evaluate_models_stats()`。
    """
    return evaluate_models_stats(net_new, net_old, n_games, mcts_new,
                                 mcts_old, n_workers)["win_rate"]


# ----------------------------------------------------------------------
# 保存 / 加载
# ----------------------------------------------------------------------
def _engine_fp():
    """当前引擎指纹（mcts.py 前 8 位）——写进登记表，便于日后辨认数字出自哪把尺子。"""
    import hashlib
    try:
        h = hashlib.sha1()
        with open("mcts.py", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def append_registry(save_dir: str, name: str, model_path: str, net,
                    local_gen: int = 0):
    """
    往 <save_dir>/../registry.csv 追加一行"这个 gen 号是谁"。

    【为什么需要】用户的诉求是"代数接着上一轮数"（gen_61、gen_62…），但不同 run
    的模型放在不同目录、网络规模也可能不同。有了这张表，任何 gen 号都能查到：
    它在哪个 run、什么架构、什么时候产出、引擎是哪版 —— 不会再有"哪个 gen_61"的歧义。
    """
    # 【路径修正 2026-10-11】登记表固定在 <项目根>/checkpoints/registry.csv。
    # 原写法用 dirname(save_dir)，在 save_dir=checkpoints/pilot/big 这种嵌套目录下
    # 会写到 checkpoints/pilot/registry.csv —— 表就"丢了"，gen 号也没处查。
    _root = os.path.dirname(os.path.abspath(__file__))
    reg = os.path.join(_root, "checkpoints", "registry.csv")
    ver = ""
    try:
        vp = os.path.join(save_dir, "VERSION.txt")
        if os.path.exists(vp):
            ver = open(vp, encoding="utf-8").read().strip()
    except OSError:
        pass
    try:
        new = not os.path.exists(reg)
        with open(reg, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write("version,name,run_dir,model_path,arch,params_wan,"
                        "local_gen,engine_mcts,created_at\n")
            a = net.arch()
            f.write(f"{ver},{name},{save_dir},{model_path},"
                    f"{a['n_blocks']}x{a['channels']},"
                    f"{sum(p.numel() for p in net.parameters()) / 1e4:.1f},"
                    f"{local_gen},{_engine_fp()},"
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError as e:
        print(f"    [提示] registry 写入失败（不影响训练）: {e}")


def save_checkpoint(net: GomokuNet, gen: int, cfg: Config,
                    lr: float, win_rate: float) -> str:
    """保存模型权重到 `<save_dir>/gen_{gen}.pt`（约定 save_dir = models/<版本>）。"""
    import torch
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, f"gen_{gen}.pt")
    torch.save({
        "state_dict": net.state_dict(),
        "gen": gen,
        "lr": lr,
        "win_rate": win_rate,
        "arch": net.arch(),          # 【放大网络】让加载方知道该建多大的网
    }, path)
    return path


# ---------------------------------------------------------------- 回放池的压缩存储
# 【2026-10-16】resume.pt 曾经 97% 是回放池、而且是**未压缩**存的：numpy 数组会被
# pickle 当成 bytes 内联进 `data.pkl`，而 torch.save 的 zip 默认是 ZIP_STORED（不压）
# ⇒ 58 代 / 49.4 万条 = 1249.9 MB（states uint8 + policies float32），每代重写一遍
# ≈1.23 GB（12 代/小时 ⇒ ≈350 GB/天）。
#
# 而同一批样本早就用 `np.savez_compressed` 落在 `data/gen_N.npz` 里：0.44 MB/代，
# 52× 压缩比。这里就照那个办法存 —— **一代一个 npz 压缩块**，并把压好的字节缓存回
# 池条目，于是只有"新加的那一代"需要压（≈23 MB 原始、0.2 s 量级），写盘从 ≈1231 MB/代
# 降到 ≈60 MB/代（池 ≈25 MB + 网络/EMA/best/优化器 36 MB）。
#
# 为什么不干脆不存池：见 save_resume_state 的 docstring —— 丢了池，续训就退化成
# "换个起点从头再来"（过拟合 + 遗忘，训练曲线呈锯齿状）。
BUFFER_BLOB_KEY = "_npz_blob"          # 池条目上的压缩块缓存（私有键，采样逻辑不看它）
BUFFER_BLOB_KEYS = "_npz_blob_keys"    # 该缓存覆盖的字段集合：条目后来多出字段就重压


def pack_generation(g: dict) -> bytes:
    """
    把一代样本压成 npz 字节；同一代只压一次（结果缓存在条目里复用）。

    存**条目里所有 ndarray 字段**（`states`/`policies`/`values`，开启新颖度轴时还会
    多出 `sample_w`/`sample_cumw`），dtype 原样保留；`gen` 这类标量留在外层。

    缓存按**字段集合**校验：`sample()` 会给缺权重的旧条目当场补上 `sample_w`/
    `sample_cumw`（`_ensure_sample_weights`），补完字段集就变了，必须重压 —— 否则新
    字段会被旧缓存悄悄吃掉。
    """
    import io
    arrays = {k: np.asarray(v) for k, v in g.items()
              if not k.startswith("_") and isinstance(v, np.ndarray)}
    if "states" not in arrays:
        raise ValueError("回放池条目缺少 states 数组，无法压缩存储")
    keys = frozenset(arrays)
    blob = g.get(BUFFER_BLOB_KEY)
    if isinstance(blob, (bytes, bytearray)) and g.get(BUFFER_BLOB_KEYS) == keys:
        return bytes(blob)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    blob = buf.getvalue()
    g[BUFFER_BLOB_KEY] = blob
    g[BUFFER_BLOB_KEYS] = keys
    return blob


def pack_buffer(buffer: "ReplayBuffer") -> list:
    """把整个回放池压成 `[{"gen", "n", "blob"}, …]`（resume.pt 的现行池格式）。"""
    return [{"gen": int(g.get("gen", 0)), "n": int(len(g["states"])),
             "blob": pack_generation(g)}
            for g in buffer.generations]


def unpack_buffer(packed: list) -> list:
    """
    `pack_buffer` 的逆操作：还原成 `buffer.generations`（`[{"states", …}, …]`）。

    顺手把压缩块当缓存塞回条目 ⇒ 刚 resume 完又要写盘时，不必把整池重压一遍。
    """
    import io
    gens = []
    for item in packed:
        with np.load(io.BytesIO(item["blob"])) as z:
            files = list(z.files)
            d = {k: z[k] for k in files}
        d["gen"] = int(item.get("gen", 0))
        d[BUFFER_BLOB_KEY] = bytes(item["blob"])
        d[BUFFER_BLOB_KEYS] = frozenset(files)
        gens.append(d)
    return gens


def save_resume_state(path: str, *, gen: int, net: GomokuNet, ema, best_net,
                      optimizer, lr: float, lr_decay_count: int, rng,
                      buffer: "ReplayBuffer", summary: dict,
                      lr_bad_streak: int = 0, name_offset: int = 1) -> str:
    """
    写出"能接着跑"的完整训练状态（`--resume` 用它续训）。

    【为什么需要它】本项目长跑曾因余额中断。原先只存 `gen_N.pt`（网络权重），
    丢掉的是：优化器动量、EMA 影子权重、回放池（6 万条）、代数计数、
    学习率衰减计数、RNG —— 于是"续训"实际是"换个起点从头再来"，退火曲线也会
    重走一遍。这个文件把这些全带上。

    写入用"临时文件 + 原子替换"：即使正好在写盘时被 kill，也不会毁掉上一份。

    【2026-10-16】回放池改成"一代一个 npz 压缩块"存（见 pack_buffer）：同样这个池
    从 1249.9 MB 降到 ≈25 MB，整个 resume.pt 从 1231 MB/代 降到 ≈60 MB/代。新旧
    格式双向兼容 —— `load_resume_state` 两种都读，旧文件不需要转换。
    """
    import torch
    state = {
        "gen": int(gen),
        "arch": net.arch(),          # 【放大网络】续训时据此重建同架构的网络
        "net": net.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "best_net": best_net.state_dict() if best_net is not None else None,
        "optimizer": optimizer.state_dict(),
        "lr": float(lr),
        "lr_decay_count": int(lr_decay_count),
        "lr_bad_streak": int(lr_bad_streak),
        "name_offset": int(name_offset),
        "rng": rng.bit_generator.state,
        "buffer": pack_buffer(buffer),       # 【2026-10-16】压缩存（旧格式读端兼容）
        "buffer_packed": 2,                  # 池格式标记：每个元素是 npz 压缩块
        "summary": dict(summary),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = path + ".tmp"
    _t0 = time.time()
    torch.save(state, tmp)
    os.replace(tmp, path)
    _mb = os.path.getsize(path) / 1e6
    print(f"    [续训] resume.pt 已写: {_mb:.1f} MB / {time.time() - _t0:.1f}s"
          f"（池 {len(buffer.generations)} 代压缩存）")
    return path


def load_resume_state(path: str, net: GomokuNet, ema, optimizer,
                      buffer: "ReplayBuffer", rng):
    """
    读回 `save_resume_state` 写出的状态，返回 state（调用方再用 gen/summary 等）。

    只负责"把状态灌回对象"；`best_net` 由调用方自行重建（它是独立网络实例）。
    """
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    arch = st.get("arch")
    if arch:
        cur = (int(net.n_blocks), int(net.channels))
        got = (int(arch.get("n_blocks", 3)), int(arch.get("channels", 64)))
        if cur != got:
            raise ValueError(
                f"resume.pt 的架构 {got[0]}×{got[1]} 与当前网络 "
                f"{cur[0]}×{cur[1]} 不一致 —— 续训必须用相同的骨干规模"
                f"（train() 会自动按 resume.pt 的 arch 建网，看到这条说明"
                f"是手动构造的网络传进来的）")
    # 【任务 6】辅助头允许"新挂"或"缺少"：主干预训练 + 新加小头正是任务 6 的核心
    # 用法。此时 strict=True 会为 `aux_heads.*` 的键对不上而报错，所以只在辅助头
    # 集合不一致时放开 strict（主干仍按键逐一对齐，规模不一致照样报）。
    _sd = st["net"]
    _aux_info = GomokuNet.infer_aux(_sd)
    _need = {x for x in _aux_info["aux"].split(",") if x}
    if _need - set(net.aux_names):
        _added = net.add_aux_heads(_aux_info["aux"], _aux_info["aux_ch"])
        if _added:
            print(f"    [架构] resume.pt 带辅助头，就地补建：{','.join(_added)}"
                  f"（主干权重照常加载；新头从随机初始化开始）")
    net.load_state_dict(_sd, strict=(set(net.aux_names) == _need))
    if ema is not None and st.get("ema"):
        # 【任务 6 缺口修复】影子权重是按**旧网络**存的：新挂辅助头时它的键集合里
        # 没有 `aux_heads.*`，原来的 strict=True 会直接报错、让整个续训起不来。
        # 键集合一致时保持原来的严格加载（老路径行为不变）。
        _esd = st["ema"]
        _cur_keys = set(ema.state_dict().keys())
        if set(_esd.keys()) == _cur_keys:
            ema.load_state_dict(_esd)
        else:
            _miss = sorted(_cur_keys - set(_esd.keys()))
            ema.load_state_dict(_esd, strict=False)
            print(f"    [架构] EMA 影子权重按旧网络加载（缺 {len(_miss)} 个新键："
                  f"{_miss[0] if _miss else '-'}）—— 新增的辅助头从当前值开始平滑")
    if st.get("optimizer"):
        _osd = st["optimizer"]
        _og, _live = _osd.get("param_groups", []), optimizer.param_groups
        # 【为什么不再"参数个数相同就直接 load_state_dict"】torch 的
        # `optimizer.state_dict()["state"]` 是**按位置**索引的，所以"个数相同"
        # 并不等于"顺序相同"。曾经在这里先判个数、相同就走原生加载，结果：
        # 存档的辅助头顺序是 `threat,ownership`、而重建网络时 `infer_aux` 按
        # 规范顺序回来 `ownership,threat` ⇒ 39 对 39、个数相同，位置全错位，
        # 训练第一步就在 `_multi_tensor_adam` 里抛
        # `RuntimeError: output with shape [1, 32, 1, 1] doesn't match the
        # broadcast shape [2, 32, 1, 1]`。现在**一律按名字对齐**（顺序相同时
        # 它退化成恒等映射，与原生加载等价），不再有任何按位置走的捷径。
        #
        # 另一半修法在 `GomokuNet.infer_aux`：它现在逐字保留存档里的辅助头
        # 顺序，这样"反推配置→重建网络"也得到同一个参数顺序。两边都修，是因为
        # 只要有一边按位置对齐，这种错位就是**静默**的（形状巧合相同的张量不会
        # 报错，只会让动量变脏）。
        _cur_pnames = [n for n, _ in net.named_parameters()]
        _pos_of = {n: i for i, n in enumerate(_cur_pnames)}
        _old_pnames = [k for k in _sd.keys() if k in _pos_of]
        _new = optimizer.state_dict()          # 当前 param_groups（已含辅助头）
        _flat = [p for g in _live for p in g["params"]]
        _kept, _skip = 0, 0
        for _j, _oid in enumerate(
                [i for g in _og for i in g.get("params", [])]):
            _ost = _osd.get("state", {}).get(_oid)
            if not _ost or _j >= len(_old_pnames):
                continue
            _ni = _pos_of[_old_pnames[_j]]
            _ea = _ost.get("exp_avg")
            if (_ea is not None and _ni < len(_flat)
                    and tuple(_ea.shape) != tuple(_flat[_ni].shape)):
                _skip += 1
                continue
            _new["state"][_ni] = _ost
            _kept += 1
        optimizer.load_state_dict(_new)
        print(f"    [架构] 优化器状态按参数名对齐、部分加载：{_kept} 个参数沿用旧动量"
              + (f"，{_skip} 个形状不符被跳过" if _skip else "")
              + "，新增参数（辅助头）从零动量开始")
    if st.get("buffer"):
        # 【2026-10-16】池有两种落盘格式，两种都要能读：
        #   新格式：每代是 {"gen", "n", "blob"}（npz 压缩块，见 pack_buffer）
        #   旧格式：每代直接是 {"gen", "states", "policies", "values"} 原始数组
        # 判据看第一条。现存 models/*/resume.pt 全是旧格式，读坏了等于毁掉全部可续训
        # 状态。顺序必须原样保留 —— 代龄按**池中位置**算（`_ages()`），不认 gen 标签。
        _pool = st["buffer"]
        if _pool and isinstance(_pool[0], dict) and "blob" in _pool[0]:
            buffer.generations = unpack_buffer(_pool)
        else:
            buffer.generations = list(_pool)
        buffer.total = int(sum(len(g["states"]) for g in buffer.generations))
    if st.get("rng"):
        rng.bit_generator.state = st["rng"]
    return st


def load_best_net_into(best_net, sd) -> int:
    """
    把 resume.pt 里存的 `best_net`（门控对手）灌回一个**按当前结构新建**的网络。

    【任务 6 缺口修复，实测复现过】`best_net` 由 `_new_net()` 按当前网络重建，
    所以新挂辅助头后它比旧存档多 6 个 `aux_heads.*` 键，直接严格加载会抛
    `RuntimeError: Missing key(s) in state_dict: "aux_heads.threat.0.weight"…`
    —— 续训根本起不来（和优化器/EMA 那条是同一类问题）。

    这里**只容忍辅助头**：差异键里只要有一个不是 `aux_heads.` 开头，就说明骨干
    结构真的不一致，照旧抛 ValueError（绝不允许静默降级，见 test_arch.py 的立场）。
    返回"未加载的新键个数"。
    """
    cur = best_net.state_dict()
    diff = sorted((set(cur) - set(sd)) | (set(sd) - set(cur)))
    if not diff:
        best_net.load_state_dict(sd)
        return 0
    bad = [k for k in diff if not k.startswith("aux_heads.")]
    if bad:
        raise ValueError(
            f"resume.pt 里的 best_net 与当前网络结构不一致：{bad[:4]}")
    best_net.load_state_dict(sd, strict=False)
    print(f"    [架构] best_net（门控对手）按旧网络加载：{len(diff)} 个新挂的辅助头键"
          f"未加载，从随机初始化开始")
    return len(diff)


def load_checkpoint_ex(path: str, strict: bool = False, verbose: bool = True):
    """
    加载模型权重，返回 `(net, ckpt)`。

    参数:
        strict : True 时要求权重文件与当前网络结构完全一致。
                 默认 False —— 因为 network.py 的策略/价值头结构改过
                 （旧版 policy_fc/value_fc 已被全卷积 / 全局池化头替代），
                 旧 checkpoint 无法直接加载。默认行为是"能加载多少加载多少"，
                 并把不匹配的键明确报出来，而不是直接抛异常。

    【放大网络】网络规模**按权重反推**（`network.GomokuNet.infer_arch`），
    而不是写死 `GomokuNet()`。否则 6×128 的 checkpoint 会被静默加载成
    随机初始化的 3×64 —— 工具照样跑、比分照样打印，但结论全是假的。

    【任务 6】辅助头同样按权重反推（`network.GomokuNet.infer_aux`）：带辅助头的
    checkpoint 会被还原成带辅助头的网络（`strict=True` 也能对上）。**没有**辅助头
    的旧 checkpoint 依旧建成无辅助头的网络，行为与历史完全一致。
    """
    import torch
    net = GomokuNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    nb, ch, vh = GomokuNet.infer_arch(sd)
    aux_info = GomokuNet.infer_aux(sd)
    aux, aux_ch = aux_info["aux"], aux_info["aux_ch"]
    if ((nb, ch, vh) != (net.n_blocks, net.channels,
                         int(net.value_fc1.weight.shape[0]))
            or aux != ",".join(net.aux_names)
            or (aux and aux_ch != net.aux_ch)):
        net = GomokuNet(n_blocks=nb, channels=ch, value_hidden=vh,
                        aux=aux, aux_ch=aux_ch)
        if verbose:
            n_par = sum(p.numel() for p in net.parameters())
            print(f"    [架构] 按权重反推：{nb} 块 × {ch} 通道"
                  f"（value_hidden={vh}，{n_par / 1e4:.1f} 万参数）")
            if aux:
                print(f"    [架构] 辅助头：{aux}（aux_ch={aux_ch}）")
    missing, unexpected = net.load_state_dict(sd, strict=strict)
    if verbose and not strict and (missing or unexpected):
        print(f"    [提示] {os.path.basename(path)} 与当前网络结构不完全一致：")
        if missing:
            print(f"           未加载的层 {len(missing)} 个（将以随机初始化使用），"
                  f"例如 {list(missing)[:4]}")
        if unexpected:
            print(f"           忽略的多余层 {len(unexpected)} 个，"
                  f"例如 {list(unexpected)[:4]}")
        if missing and any(k.startswith(("policy_", "value_")) for k in missing):
            print("           （头部结构已升级为全卷积/全局池化，"
                  "建议用 --fresh 重新训练）")
    return net, ckpt


def load_checkpoint(path: str, strict: bool = False, verbose: bool = True) -> GomokuNet:
    """加载模型权重，只返回 net（细节见 `load_checkpoint_ex`）。"""
    return load_checkpoint_ex(path, strict=strict, verbose=verbose)[0]


def save_log(log_path: str, record: dict):
    """追加训练日志(每代一行 JSON)。"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_generation_data(dirpath: str, states: np.ndarray,
                         policies: np.ndarray, values: np.ndarray, gen: int):
    """保存每代数据到 `<save_dir>/data/gen_{gen}.npz`。"""
    os.makedirs(os.path.join(dirpath, "data"), exist_ok=True)
    np.savez_compressed(os.path.join(dirpath, "data", f"gen_{gen}.npz"),
                        states=states, policies=policies, values=values)


# ----------------------------------------------------------------------
# 主训练循环
# ----------------------------------------------------------------------
def lr_decay_decision(win_rate, accepted, cfg, bad_streak):
    """
    门控的 lr 决策，返回 `(是否降 lr, 新的连续退化计数)`。

    【为什么单独抽成纯函数】这是"训练会不会把自己锁死"的关键开关，历史上踩过：
    run3 的门控在 20 局下假拒率 74.8%，平台期里 lr 被统计噪声一路压低
    （13 次 → 初始值的 5.5%），而同期独立对战证明棋力其实还在涨。

    两种模式：
      * `cfg.lr_ratchet=True`（默认，与历史行为一致）：胜率 < 阈值就降一次；
      * `cfg.lr_ratchet=False`（`--no-lr-ratchet`）：必须**连续**
        `cfg.lr_decay_streak` 次低于阈值才降一次，中间任意一次不满足就清零
        —— 孤立的一次坏运气不再永久压低学习率。
    降一次之后计数归零，重新累计。
    """
    if accepted:
        return False, 0
    if win_rate is None:                  # 不门控的那一代：什么都不动
        return False, bad_streak
    if win_rate >= cfg.lr_decay_min_win_rate:
        return False, 0
    streak = bad_streak + 1
    if cfg.lr_ratchet or streak >= cfg.lr_decay_streak:
        return True, 0
    return False, streak


def train(cfg: Config = None, init_weights: str = None,
          resume_path: str = None) -> dict:
    """
    执行完整训练闭环。

    参数:
        cfg:         配置
        init_weights: 初始化模型权重路径(可选，否则随机初始化)
        resume_path:  断点续训状态文件（`resume.pt`）。给了就从中恢复
                      优化器/EMA/回放池/代数/学习率计数，并从 next_gen 继续
    返回:
        dict 训练摘要
    """
    import torch
    if cfg is None:
        cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, "train_log.jsonl")

    # 初始化网络
    # 【放大网络】架构来源优先级：--init-weights 的权重 > resume.pt 的 arch >
    # Config/命令行。写死 GomokuNet() 是错的：给了大网络的权重也会被丢掉。
    resume_arch = None
    if resume_path and os.path.exists(resume_path):
        try:
            _st = torch.load(resume_path, map_location="cpu", weights_only=False)
            resume_arch = _st.get("arch")
        except Exception as e:      # 读不出来就当没有，后面的 load_resume_state 会报
            print(f"    [提示] 读 resume.pt 的架构失败（{type(e).__name__}），"
                  f"按 Config 建网")
    if init_weights:
        net, _ck0 = load_checkpoint_ex(init_weights)
        if (net.n_blocks, net.channels) != (cfg.n_blocks, cfg.channels):
            print(f"    [架构] 以 --init-weights 为准：{init_weights} = "
                  f"{net.n_blocks} 块 × {net.channels} 通道"
                  f"（Config/命令行的 {cfg.n_blocks}×{cfg.channels} 被忽略）")
        # 【任务 6】热启动：权重文件里没有辅助头时就地补建 —— 主干权重原样保留，
        # 只是多挂两个小头，所以**不必从零重训、不丢历史数据**。
        if cfg.aux_heads:
            added = net.add_aux_heads(cfg.aux_heads, cfg.aux_ch)
            if added:
                print(f"    [架构] 热启动补建辅助头：{','.join(added)}"
                      f"（aux_ch={cfg.aux_ch}；主干权重保留，新头随机初始化）")
    else:
        nb, ch, vh = cfg.n_blocks, cfg.channels, cfg.value_hidden
        if resume_arch:
            nb = int(resume_arch.get("n_blocks", nb))
            ch = int(resume_arch.get("channels", ch))
            vh = int(resume_arch.get("value_hidden", vh))
            print(f"    [架构] 按 resume.pt 恢复：{nb} 块 × {ch} 通道")
        net = GomokuNet(n_blocks=nb, channels=ch, value_hidden=vh,
                        aux=cfg.aux_heads, aux_ch=cfg.aux_ch)
        print(f"    [架构] 新建网络：{net.n_blocks} 块 × {net.channels} 通道"
              f"（{sum(p.numel() for p in net.parameters()) / 1e4:.1f} 万参数）"
              + (f"，辅助头 {','.join(net.aux_names)}" if net.aux_names else ""))
    net.eval()

    def _new_net():
        """按当前 net 的架构新建一个空网络（best_net 等必须同架构）。"""
        return GomokuNet(n_blocks=net.n_blocks, channels=net.channels,
                         value_hidden=int(net.value_fc1.weight.shape[0]),
                         aux=",".join(net.aux_names), aux_ch=net.aux_ch)

    # MCTS 参数与 mcts.py 的新默认值对齐
    def _make_mcts(model, hw=None, pw=None, noise_alpha=None,
                   noise_eps=0.25):
        return MCTS(model, n_sim=cfg.n_sim, c_puct=cfg.c_puct,
                    virtual_loss=cfg.virtual_loss,
                    batch_size=cfg.batch_size_mcts,
                    max_children=cfg.max_children,
                    root_max_children=cfg.root_max_children,
                    heuristic_weight=(cfg.heuristic_weight if hw is None else hw),
                    policy_weight=(cfg.policy_weight if pw is None else pw),
                    root_noise_alpha=noise_alpha,
                    root_noise_eps=noise_eps)

    mcts = _make_mcts(net)
    lr = cfg.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_decay_count = 0
    bad_streak = 0                     # --no-lr-ratchet 用的连续退化计数
    rng = np.random.default_rng(12345)

    # ---- 阶段 2：经验回放池 + EMA 权重 ----
    buffer = ReplayBuffer(capacity=cfg.buffer_size,
                          gen_limit=cfg.buffer_gen_limit, rng=rng,
                          decay_half_life=cfg.buffer_decay_half_life,
                          entropy_weight=cfg.buffer_entropy_weight)
    if buffer.shape_enabled:
        print(f"[回放池] 采样整形已开启：半衰期={buffer.decay_half_life} 代，"
              f"新颖度权重 β={buffer.entropy_weight}"
              f"（都为 0 时与旧版逐位一致）")
    ema = EMAWeights(net, decay=cfg.ema_decay) if cfg.ema_enabled else None

    # 加载已有 best.pt 作为基线，让第 1 轮也能和旧模型对比
    best_net = None
    if os.path.exists(os.path.join(cfg.save_dir, "best.pt")):
        # 【放大网络】best.pt 也走按权重反推架构的加载器（同架构时行为不变）
        best_net, _ckb = load_checkpoint_ex(
            os.path.join(cfg.save_dir, "best.pt"), verbose=False)
        best_net.eval()
        print(f"    已加载旧最佳模型: {cfg.save_dir}/best.pt"
              f"（{best_net.n_blocks}×{best_net.channels}）")

    # ---- 断点续训：恢复"训练进程的记忆" ----
    # 【注意顺序】这段必须在 `summary` 初始化**之后**：续训要把上一轮的 summary
    # 合并回来（实测第一版插在前面，直接 UnboundLocalError）。
    start_gen = 1
    name_off = int(cfg.start_gen)       # 对外代数偏移（"接着上一轮编号"）
    summary = {"final_gen": 0, "final_lr": lr, "final_win_rate": 0.0,
               "accepted": 0, "rejected": 0}
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"    [警告] 找不到续训状态 {resume_path}，改为从头开始")
        else:
            st = load_resume_state(resume_path, net, ema, optimizer, buffer, rng)
            if buffer.shape_enabled and not buffer.gen_labels_monotone():
                # 【必须是加载之后】这一句原先写在上面建 buffer 的地方，那时池子还是
                # 空的 ⇒ gen_labels_monotone() 空集恒真 ⇒ 警告**从来没打印过**（实测
                # done 了一次 V4 resume 才发现）。池子要到 load_resume_state 才灌进来。
                # 典型场景：prepare_run.ps1 把继承来的池子重编号成 1..N，本 run 又从 1
                # 重新计数（V4 实测 = [16..58] + [1..15]）。采样不受影响（代龄按池中
                # 位置算，见 ReplayBuffer._ages），但日志里的 gen 会撞号。
                print(f"[回放池] 注意：{buffer.n_generations} 代的 gen 标签不是递增的"
                      f"（继承来的池子被 prepare_run.ps1 重编号过，本 run 又从 1 计数）——"
                      f"代龄按池中位置算，采样正确；只是日志里的 gen N 不等于本 run 的第 N 代。")
            if st.get("best_net"):
                best_net = _new_net()
                # 只容忍辅助头差异；骨干不一致照样报错（实现见 load_best_net_into）
                load_best_net_into(best_net, st["best_net"])
                best_net.eval()
            lr = float(st.get("lr", lr))
            for _g in optimizer.param_groups:
                _g["lr"] = lr
            lr_decay_count = int(st.get("lr_decay_count", 0))
            bad_streak = int(st.get("lr_bad_streak", 0))
            if isinstance(st.get("summary"), dict):
                summary.update(st["summary"])
            start_gen = int(st.get("gen", 0)) + 1
            # 名字偏移（"代数接着上一轮数"）随 resume 一起继承，避免续训后改名
            name_off = int(st.get("name_offset", cfg.start_gen))
            print(f"    [续训] 从 {os.path.basename(resume_path)} 恢复："
                  f"上一代={start_gen - 1}，lr={lr:.6f}，"
                  f"回放池={buffer.total} 条/{buffer.n_generations} 代，"
                  f"lr 衰减计数={lr_decay_count}"
                  f"（存于 {st.get('saved_at', '?')}）")
            print(f"    对外代数：本 run 第 1 代 = gen_{name_off}")
            if start_gen > cfg.max_epochs:
                print(f"    [提示] --max-gen={cfg.max_epochs} 已经 <= 已完成的 "
                      f"{start_gen - 1} 代，没有可跑的了。想继续就把 "
                      f"--max-gen 提到 {start_gen - 1 + 10} 之类。")

    for gen in range(start_gen, cfg.max_epochs + 1):
        gname = gen + name_off - 1          # 对外代数（可接着上一轮编号）
        print(f"\n{'='*60}")
        print(f"第 {gen} 轮迭代（对外 gen_{gname}）")
        print(f"{'='*60}")

        # ---- 0) 规则先验权重退火 ----
        # 自对弈与评估必须用**同一套**权重，否则"新模型 vs 旧模型"的比赛里
        # 双方所处的先验环境不同，胜率没有可比性。
        hw, pw = annealed_weights(gen, cfg)
        mcts.heuristic_weight = hw
        mcts.policy_weight = pw
        print(f"[0/4] 规则先验权重: heuristic={hw:.2f} policy={pw:.2f}"
              + ("（退火中）" if cfg.anneal_enabled else "（固定）"))

        # ---- 1) 自我对弈数据生成 ----
        # 注意：新数据一律**不增强**入库，增强放到训练时在线随机做（省 8 倍内存）
        print(f"[1/4] 生成 {cfg.games_per_epoch} 局对局数据 "
              f"(n_sim={cfg.n_sim})...")
        t0 = time.time()
        states, policies, values = collect_training_data(
            mcts, net, cfg.games_per_epoch, cfg=cfg.with_weights(hw, pw),
            augment=False, augment_factor=1,
            n_workers=cfg.n_workers, seed=gen * 1000)
        if len(states) == 0:
            print("    [警告] 本轮未生成任何可用数据，跳过该轮")
            continue
        print(f"    数据: {len(states)} 条 (耗时 {time.time()-t0:.1f}s)  "
              f"回放池: {buffer.total} 条 / {buffer.n_generations} 代")
        if cfg.save_data:
            save_generation_data(cfg.save_dir, states, policies, values, gname)
        buffer.add_generation(states, policies, values, gen)
        fresh = (states, policies, values)

        # ---- 2) 训练（新数据 + 历史数据 混合采样）----
        steps = max(cfg.min_samples_per_gen,
                    int(len(states) * cfg.samples_per_gen / cfg.batch_size))
        print(f"[2/4] 训练 {steps} step "
              f"(batch={cfg.batch_size}, 新数据占比={cfg.fresh_ratio}, "
              f"lr={lr:.6f})...")
        t0 = time.time()
        lp = lv = lt = 0.0
        aux_acc = {}                  # 【任务 6】辅助损失的 running sum
        for step in range(steps):
            lp, lv, lt = train_step(net, fresh, cfg, optimizer,
                                    buffer=buffer, rng=rng, ema=ema)
            for _k, _v in getattr(net, "last_aux", {}).items():
                aux_acc[_k] = aux_acc.get(_k, 0.0) + _v
            if (step + 1) % max(1, steps // 20) == 0 or step == steps - 1:
                pct = (step + 1) * 100 // steps
                bar_len = 30
                filled = int(bar_len * step / steps)
                bar = "#" * filled + "." * (bar_len - filled)
                _extra = "".join(f" {k[5:]}={v/(step+1):.4f}"
                                 for k, v in aux_acc.items())
                print(f"    step {step+1}/{steps} [{bar}] "
                      f"loss_p={lp:.4f} loss_v={lv:.4f} total={lt:.4f}"
                      f"{_extra}",
                      end="\r", flush=True)
        print(f"    step {steps}/{steps} [{'#' * 30}] "
              f"loss_p={lp:.4f} loss_v={lv:.4f} total={lt:.4f}", flush=True)
        print(f"    训练耗时 {time.time()-t0:.1f}s", flush=True)
        # 【任务 4】记录采样分布（关着整形时 last_sample_stats 为空 ⇒ 不打印）
        if buffer.last_sample_stats:
            _bs = buffer.last_sample_stats
            print(f"    [回放池采样] 历史样本 {_bs['n_hist']} 条覆盖 "
                  f"{_bs['gens_drawn']} 代；平均代龄 期望"
                  f"{_bs['mean_age_exp']:.1f}/实际{_bs['mean_age_seen']:.1f}；"
                  f"最近 20 代占比 期望{_bs['share_recent20_exp']:.3f}/实际"
                  f"{_bs['share_recent20_seen']:.3f}", flush=True)
        # 训练用的是 train() 模式（BN 走 batch 统计），评估必须切回 eval()
        # 否则同一份权重在"评估时"和"自对弈搜索时"行为不一致。
        net.eval()

        # ---- 3) 评估：新 vs 旧（原始权重与 EMA 权重都评一遍，取较优者）----
        #
        # 【为什么允许"隔代评估"】门控很贵：20 局 raw + 20 局 EMA = 40 局，
        # 和一代自对弈的 40 局一样慢 —— 也就是说每次门控要花掉整整一代的时间。
        # AlphaZero 原版也不是每代都门控。`eval_every=k` 表示每 k 代评估一次：
        #   * 不评估的代：只训练 + 存快照，**不动** best_net、**不降** lr；
        #   * 评估的代：仍用完整的 eval_games 局（统计量不缩水）。
        # 这比"每代只跑 eval_games/k 局"更好：同样的开销下，
        # 每次决策的证据量不减少，只是决策频率降低。
        cand_kind = "raw"
        stats_raw = None
        stats_ema = None
        stats_chosen = None
        win_rate_raw = None
        win_rate_ema = None
        win_rate = None
        accepted = None
        candidate = None
        baseline_created = best_net is None
        gate_thr = cfg.win_rate_threshold
        gate_ci = cfg.gate_ci_ll

        if baseline_created:
            print("[3/4] 首轮无旧模型，跳过评估，直接建立基线")
            win_rate = 1.0
            accepted = True
            cand_kind = "baseline"
            # 存一份独立副本作为基线：不能直接引用 net，
            # 否则下一轮训练一边更新权重、一边拿它当"旧模型"对比。
            import copy as _copy
            best_net = _copy.deepcopy(net).eval()
        elif gen % max(1, cfg.eval_every) != 0:
            print(f"[3/4] 本轮不门控（每 {cfg.eval_every} 代评估一次），"
                  f"best_net 保持不变")
        else:
            print(f"[3/4] 评估 ({cfg.eval_games} 局/次, 黑白各半, "
                  f"判据 胜率>{gate_thr} 且 Wilson下界>{gate_ci}, "
                  f"根噪声 α={cfg.eval_noise_alpha})...")
            # 【必须加根噪声】评估用的是 `get_best_action`（temp=0 的贪心）。
            # 贪心 + 无噪声 = 完全确定性：同一先后手会走出**逐手完全相同**的对局。
            # 实测 20 局里只有 2 个不同的对局序列（每个颜色各一条），
            # 于是"20 局胜率"实际只有 2 个独立样本，只能取 0 / 0.5 / 1.0 三个值
            # —— 日志里那些 0.000 / 0.500 就是这么来的，和模型强弱毫无关系。
            # 加了根噪声之后，20 局是 20 条不同序列，统计才有意义。
            mcts.root_noise_alpha = cfg.eval_noise_alpha
            mcts.root_noise_eps = cfg.eval_noise_eps
            mcts_old = _make_mcts(best_net, hw, pw,
                                  cfg.eval_noise_alpha, cfg.eval_noise_eps)
            net.eval()
            stats_raw = gate_verdict(
                evaluate_models_stats(net, best_net, cfg.eval_games,
                                      mcts, mcts_old,
                                      n_workers=cfg.eval_workers),
                gate_thr, gate_ci)
            win_rate_raw = stats_raw["win_rate"]
            candidate = net
            if ema is not None and cfg.eval_both:
                ema.shadow.eval()
                stats_ema = gate_verdict(
                    evaluate_models_stats(
                        ema.shadow, best_net, cfg.eval_games,
                        _make_mcts(ema.shadow, hw, pw,
                                   cfg.eval_noise_alpha, cfg.eval_noise_eps),
                        mcts_old, n_workers=cfg.eval_workers),
                    gate_thr, gate_ci)
                win_rate_ema = stats_ema["win_rate"]
                print(f"    raw {fmt_gate_stats(stats_raw)}", flush=True)
                print(f"    ema {fmt_gate_stats(stats_ema)}", flush=True)
            # ---- 选谁：EMA 只作参考，不能绕过门槛 ----
            # 旧写法是"谁的点估计大就用谁"（`win_rate_ema > win_rate_raw`），
            # 那等于让 EMA 权重**绕过采纳门槛**：raw 只赢 0.30、EMA 赢 0.45 时
            # 被"采用"的是 EMA，而 0.45 本身远没过线。现在只有在
            # **它自己也满足判据**时 EMA 才可能被选中；两个都不满足时仍取
            # 点估计大的那个，但只用于 lr 衰减判定与日志留痕，采纳与否看判据。
            if stats_ema is None:
                stats_chosen = stats_raw
            elif stats_ema["accepted"] and not stats_raw["accepted"]:
                stats_chosen, cand_kind, candidate = stats_ema, "ema", ema.shadow
            elif stats_raw["accepted"] and stats_ema["accepted"]:
                if stats_ema["win_rate"] > stats_raw["win_rate"]:
                    stats_chosen, cand_kind, candidate = stats_ema, "ema", ema.shadow
                else:
                    stats_chosen, cand_kind, candidate = stats_raw, "raw", net
            elif stats_ema["win_rate"] > stats_raw["win_rate"]:
                stats_chosen, cand_kind, candidate = stats_ema, "ema", ema.shadow
            else:
                stats_chosen, cand_kind, candidate = stats_raw, "raw", net
            win_rate = stats_chosen["win_rate"]
            accepted = bool(stats_chosen["accepted"])
            print(f"    采用[{cand_kind}] {fmt_gate_stats(stats_chosen)}")
            print(f"    判据: 胜率>{gate_thr} → {stats_chosen['pass_score']} | "
                  f"Wilson下界>{gate_ci} → {stats_chosen['pass_ci']} "
                  f"⇒ {'[采纳]' if accepted else '[拒绝]'}")

        if accepted is None:
            # 本轮不门控：训练继续、快照照存，但不更新 best_net / lr
            pass
        elif accepted:
            if not baseline_created:
                # 一律保存独立快照：绝不能让 best_net 与 net 共享同一份参数，
                # 否则下一轮训练会"一边更新权重、一边拿它当旧模型对比"。
                import copy as _copy
                best_net = _copy.deepcopy(candidate).eval()
            summary["accepted"] += 1
            summary["final_win_rate"] = win_rate
            print(f"    采纳新模型 (gen={gname}, 来源={cand_kind})")
        else:
            summary["rejected"] += 1
            # 【设计修正 —— 有实测依据】只有"确实退化"才降学习率。
            #
            # 原实现是"没打赢阈值就降 lr"。但这个门控在 20 局下的**假拒率极高**：
            # 阈值 0.55 需要 ≥12/20 才采纳，而新旧模型**真实相当**（p=0.5）时
            # P(X≥12) = 0.252 —— 也就是 74.8% 的概率被判"拒绝"。
            # 于是平台期里 lr 完全被统计噪声压着走：实测 18 代内从 1e-3 降到 4.1e-4，
            # 而同期独立对战证明棋力其实涨了约 162 分（p=0.011）。
            # 更糟的是 lr 只降不升，越到后面越难翻身 —— 安全阀
            # `max_lr_decay=20` 按每 2 代门控一次算要到 gen 50 才触发。
            #
            # 现在：胜率 < lr_decay_min_win_rate（默认 0.30，噪声下概率约 5.8%，
            # 是一个真正的单侧检验）才算退化并降 lr；落在中间地带只是"没打赢"，
            # 不更新 best_net、也不动 lr。
            decay, bad_streak = lr_decay_decision(win_rate, accepted, cfg,
                                                  bad_streak)
            if decay:
                lr_decay_count += 1
                if lr_decay_count >= cfg.max_lr_decay:
                    print("    学习率已降至极限，强制采纳最新模型")
                    accepted = True
                    import copy as _copy
                    best_net = _copy.deepcopy(candidate).eval()
                elif lr <= cfg.lr_floor:
                    print(f"    胜率 {win_rate:.3f} 显著退化，但学习率已在"
                          f"下限 {cfg.lr_floor:.1e}，保持不变")
                else:
                    lr = max(cfg.lr_floor, lr * cfg.lr_decay)
                    for gparam in optimizer.param_groups:
                        gparam["lr"] = lr
                    print(f"    胜率 {win_rate:.3f} 低于 {cfg.lr_decay_min_win_rate}"
                          f"（显著退化），学习率降至 {lr:.6f}")
            elif win_rate is not None and win_rate < cfg.lr_decay_min_win_rate:
                print(f"    胜率 {win_rate:.3f} 低于 {cfg.lr_decay_min_win_rate}，"
                      f"但 --no-lr-ratchet 要求连续 {cfg.lr_decay_streak} 次"
                      f"（当前 {bad_streak}）才降 —— 学习率保持 {lr:.6f}")
            else:
                print(f"    胜率 {win_rate:.3f} 未过阈值 {cfg.win_rate_threshold}，"
                      f"但未构成显著退化（需 < {cfg.lr_decay_min_win_rate}）"
                      f"—— best_net 与学习率 {lr:.6f} 均保持不变")

        summary["final_gen"] = gname
        summary["final_lr"] = lr
        # 不门控的那一代 win_rate 是 None，别把上一次的有效值覆盖掉
        if win_rate is not None:
            summary["final_win_rate"] = win_rate

        # ---- 4) 保存 ----
        ckpt_path = save_checkpoint(net, gname, cfg, lr, win_rate)
        append_registry(cfg.save_dir, f"gen_{gname}", ckpt_path, net,
                        local_gen=gen)
        record = {
            "gen": gname,          # 对外代数（可接着上一轮编号）
            "local_gen": gen,      # 本 run 内部第几代（退火/门控节奏用它）
            "lr": lr,
            "win_rate": win_rate,
            "win_rate_raw": win_rate_raw,
            "win_rate_ema": win_rate_ema,
            # ---- 门控统计明细（2026-10-13 任务 1 新增）----
            # 20 局时代只存一个胜率标量，事后**无法**复查"证据够不够、颜色有没有
            # 记反"——于是"这条记录到底是模型强还是运气"只能靠猜。现在把局数、
            # 分颜色战绩、Wilson 区间、Elo 与判据明细都落盘。
            # 键名统一用 gate_ 前缀，旧行没有这些键（读取端一律 .get() 兜底）。
            "gate_games": (stats_chosen or {}).get("games"),
            "gate_wins": (stats_chosen or {}).get("wins"),
            "gate_draws": (stats_chosen or {}).get("draws"),
            "gate_win_rate_raw": win_rate_raw,
            "gate_win_rate_ema": win_rate_ema,
            "gate_wilson_lo": (stats_chosen or {}).get("wilson_lo"),
            "gate_wilson_hi": (stats_chosen or {}).get("wilson_hi"),
            "gate_elo": (stats_chosen or {}).get("elo"),
            "gate_elo_lo": (stats_chosen or {}).get("elo_lo"),
            "gate_elo_hi": (stats_chosen or {}).get("elo_hi"),
            "gate_games_black": (stats_chosen or {}).get("games_black"),
            "gate_wins_black": (stats_chosen or {}).get("wins_black"),
            "gate_games_white": (stats_chosen or {}).get("games_white"),
            "gate_wins_white": (stats_chosen or {}).get("wins_white"),
            "gate_color_bias_elo": (stats_chosen or {}).get("color_bias_elo"),
            "gate_threshold": (stats_chosen or {}).get("threshold"),
            "gate_ci_ll": (stats_chosen or {}).get("ci_ll"),
            "gate_pass_score": (stats_chosen or {}).get("pass_score"),
            "gate_pass_ci": (stats_chosen or {}).get("pass_ci"),
            "gate_rule": "win_rate>thr and wilson_lo>ci_ll",
            "chosen": cand_kind,
            "accepted": accepted,
            "heuristic_weight": round(float(hw), 4),
            "policy_weight": round(float(pw), 4),
            # 损失只当"训练有没有炸"的哨兵，**不要**当成棋力进度指标。
            #
            # 【实测纠正】这里原来写的是"loss 比门控胜率信噪比高得多，能直接看出
            # 模型有没有在学"。18 代数据把这个说法推翻了：
            #     gen vs loss_v 相关系数 = +0.087（无趋势）
            #     gen vs loss_p 相关系数 = +0.420（反而在升）
            # 同时 96 局独立对战证明 gen_2 -> gen_17 涨了约 162 分、p=0.011 显著。
            # 即"两个 loss 都不降，棋力却显著变强"。
            #
            # 原因是 AlphaZero 类自对弈的训练目标是**非平稳**的：策略目标永远是
            # "当前网络自己这轮搜索的输出"，网络变强则目标跟着变难。于是 loss 持平
            # 的真实含义是"网络在跟上一个不断变难的目标"，而不是"没在学"。
            # 参考基准（仅用于判断是否异常）：loss_p 的"完全没学"是 ln(361)=5.89，
            # loss_v 的"一律猜 0"约 0.8~1.0。
            # **判断棋力只能靠 compare.py / elo.py 的独立对战。**
            "loss_policy": round(float(lp), 4),
            "loss_value": round(float(lv), 4),
            "loss_total": round(float(lt), 4),
            # 【任务 6】辅助头：架构写清楚，便于在被撤销/对比时分辨是哪次实验；
            # 辅助损失只用于"头有没有在学"的哨兵，同样**不是棋力指标**。
            "aux_heads": ",".join(getattr(net, "aux_names", ())),
            "aux_loss": {k: round(v / max(1, steps), 4)
                         for k, v in aux_acc.items()},
            "data_samples": int(len(states)),
            "train_steps": int(steps),
            "buffer_total": int(buffer.total),
            "buffer_gens": int(buffer.n_generations),
            # 【任务 4】回放池采样分布（只在实际开了整形时非空；含期望/实际平均代龄
            # 与最近 20 代占比，便于事后判断"旧数据有没有淹没新策略"）
            "buffer_sample": dict(buffer.last_sample_stats),
            "n_games": cfg.games_per_epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 每代写一份 resume.pt（原子替换：写到一半被杀也不会毁掉旧文件）
        # 【2026-10-16】挪到 save_log 之前 —— 这样"这一代写盘写了多少 MB、花了多久"
        # 能和其它指标落进同一行日志（resume_mb / resume_s），一眼看出压缩有没有生效。
        if cfg.resume_save and gen % max(1, cfg.resume_every) == 0:
            _t_rs = time.time()
            try:
                _rpath = save_resume_state(os.path.join(cfg.save_dir, "resume.pt"),
                                           gen=gen, net=net, ema=ema,
                                           best_net=best_net, optimizer=optimizer,
                                           lr=lr, lr_decay_count=lr_decay_count,
                                           rng=rng, buffer=buffer, summary=summary,
                                           lr_bad_streak=bad_streak,
                                           name_offset=name_off)
                record["resume_mb"] = round(os.path.getsize(_rpath) / 1e6, 1)
                record["resume_s"] = round(time.time() - _t_rs, 2)
            except OSError as e:
                print(f"    [警告] resume.pt 写盘失败（不影响训练）: {e}")
        save_log(log_path, record)
        print(f"[4/4] 已保存 gen_{gen}.pt, 日志, 数据")

    # 最终保存最佳模型
    if best_net is not None:
        final_path = os.path.join(cfg.save_dir, "best.pt")
        torch.save({"state_dict": best_net.state_dict(),
                    "gen": summary["final_gen"],
                    "arch": best_net.arch()}, final_path)
        print(f"\n最佳模型保存至 {final_path}")
    print(f"\n训练完成: 采纳 {summary['accepted']}/{summary['accepted']+summary['rejected']} 轮")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="五子棋 AlphaZero 风格训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--games", type=int, default=40, help="每轮对局数")
    parser.add_argument("--epochs", type=int, default=4, help="每轮训练轮数")
    parser.add_argument("--n-sim", type=int, default=400,
                        help="每步 MCTS 模拟轮数 (0=按深度动态)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--max-gen", type=int, default=80, help="最大迭代轮数")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="训练小批量大小")
    # 【默认保持 16；2026-10-11 实测"batch 64 更快但更弱"，不要改大】
    # `probe_batch_scan.py`（6×128）：同进程数下 batch 16→64 让每手耗时降 37~60%
    #   （单进程 538→218 ms），8 进程聚合吞吐 11.0 → **17.5 手/秒（+59%）**。
    # 但同权重对照（`compare.py --mcts-batch-a 16 --mcts-batch-b 64`，gen_45，
    #   n_sim=400、virtual_loss=3.0、hw/pw=0.5/1.0）打到 **17:3**（A=16 胜率 0.85，
    #   20 局 p<0.001）→ **固定模拟数下 batch 64 显著更弱**。
    # 机制（待验证）：批越大，"虚拟损失"同时压住的叶子越多（64 个），
    #   选点被摊平、400 次模拟的注意力被分散 → 每手质量下降。
    # 下一步该试的是 **批量↑ + 虚拟损失↓**（或 批量↑ + n_sim↑），不是单独放大批量。
    parser.add_argument("--mcts-batch", type=int, default=16,
                        help="MCTS 批量推理大小（实测 64 提速 59%% 但棋力显著变弱，默认保持 16）")
    parser.add_argument("--max-children", type=int, default=6,
                        help="非根节点展开宽度")
    parser.add_argument("--root-max-children", type=int, default=16,
                        help="根节点展开宽度（强制点始终保留）")
    parser.add_argument("--heuristic-weight", type=float, default=6.0,
                        help="规则先验权重**起始值**（退火开启时即 anneal_h_start）")
    parser.add_argument("--policy-weight", type=float, default=0.3,
                        help="网络策略先验权重起始值")
    parser.add_argument("--no-anneal", action="store_true",
                        help="关闭规则先验权重退火（全程固定用 --heuristic-weight）")
    parser.add_argument("--anneal-start", type=int, default=20,
                        help="该代及以前保持起始权重")
    parser.add_argument("--anneal-mid", type=int, default=40,
                        help="线性插值到 2.0/1.0 的代数")
    parser.add_argument("--anneal-end", type=int, default=60,
                        help="线性插值到 0.5/1.0 的代数，之后保持不变")
    parser.add_argument("--anneal-h-end", type=float, default=0.5,
                        help="退火终点的规则先验权重")
    parser.add_argument("--blocks", type=int, default=3,
                        help="残差块数（仅在新建网络时生效；给了 --init-weights "
                             "则以权重文件的架构为准）")
    parser.add_argument("--channels", type=int, default=64,
                        help="骨干通道数（同上）")
    parser.add_argument("--value-hidden", type=int, default=64,
                        help="价值头隐层宽度（同上）")
    # ---- 【任务 6】辅助头 ----
    parser.add_argument("--aux-heads", type=str, default="",
                        help="辅助头（逗号分隔，默认空=关闭）：threat（威胁点：己方/"
                             "对方一步成五）、ownership（归属）。只影响训练损失与"
                             "网络多出来的小头，**不进入 MCTS 推理路径**，因此不拖慢"
                             "自对弈。可与 --init-weights 连用热启动：主干权重保留，"
                             "只补建小头，不必从零重训。")
    parser.add_argument("--aux-ch", type=int, default=32,
                        help="辅助头内部通道数（1×1 卷积降维，默认 32）")
    parser.add_argument("--aux-weight-threat", type=float, default=0.15,
                        help="威胁头损失权重（相对主损失，默认 0.15）")
    parser.add_argument("--aux-weight-ownership", type=float, default=0.15,
                        help="归属头损失权重（相对主损失，默认 0.15）")
    parser.add_argument("--aux-threat-pos-weight", type=float, default=20.0,
                        help="威胁头的正样本权重（默认 20）。威胁点极稀疏，"
                             "不加权会退化成整盘输出 0。")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="新 vs 旧 评估局数（每次评估的局数；黑白各半，"
                             "奇数会自动 +1）。默认 20：实测 100 局既不降门槛、"
                             "也不涨棋力，只多花 5 倍门控时间（每代 +45%% 墙钟）；"
                             "要更准请把局数花在整轮只跑一次的晋升闸 auto_eval.ps1")
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="门控评估的并行进程数（1=串行；颜色按局号分配，不影响判定）")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="采纳的点估计门槛（默认用 Config.win_rate_threshold=0.55）")
    parser.add_argument("--gate-ci-ll", type=float, default=0.0,
                        help="可选的第二道条件：Wilson 95%% 下界严格大于此值。"
                             "默认 0.0 = **关闭**（判定与历史一致，只看点估计"
                             "是否 > --gate-threshold）。⚠️ 打开后两条是『且』："
                             "n=20 时 0.50 ⇒ 需 15/20（门槛被抬到 0.75）；"
                             "n=100 时 ⇒ 需 60/100（与旧的 ≥12/20 相同）")
    parser.add_argument("--no-lr-ratchet", action="store_true",
                        help="解除 lr 棘轮：门控拒绝不再立刻降 lr，"
                             "必须连续 --lr-decay-streak 次显著退化才降一次")
    parser.add_argument("--lr-decay-streak", type=int, default=3,
                        help="配合 --no-lr-ratchet：连续几次显著退化才降 lr")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="每几代门控一次（2 = 隔代评估，省约 1/4 总时长）")
    parser.add_argument("--eval-noise-alpha", type=float, default=0.3,
                        help="评估用根噪声 alpha（必须>0，否则 20 局只有 2 个独立样本）")
    parser.add_argument("--eval-noise-eps", type=float, default=0.25,
                        help="评估用噪声混合比例")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行数据生成 worker 数（GPU 瓶颈时用 1）")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--no-resign", action="store_true", help="禁用认输机制")
    parser.add_argument("--no-noise", action="store_true", help="禁用根节点 Dirichlet 噪声")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="经验回放池容量（样本条数）")
    parser.add_argument("--buffer-gens", type=int, default=30,
                        help="经验回放池最多保留多少代")
    parser.add_argument("--fresh-ratio", type=float, default=0.7,
                        help="每个 batch 中当代新数据的比例（其余取历史）")
    parser.add_argument("--buffer-decay-half-life", type=float, default=0.0,
                        help="【任务 4】回放池按代龄做几何衰减的半衰期（单位：代）。"
                             "0=关闭（默认，与旧版逐位一致）。半衰期 H = "
                             "相隔 H 代的两代采样权重相差 2 倍；H=20 时 60 代池里"
                             "最近 20 代约占 55%% 抽样权重，用来防止旧数据淹没新策略")
    parser.add_argument("--buffer-entropy-weight", type=float, default=0.0,
                        help="【任务 4】回放池按策略目标熵加权（新颖度）的 β。"
                             "0=关闭（默认）；权重 = exp(β·H/Hmax)，β=1 时跨度 1~2.72")
    parser.add_argument("--samples-per-gen", type=float, default=2.0,
                        help="每代训练步数 = 新数据量 × 该系数 / batch_size")
    parser.add_argument("--min-steps", type=int, default=400,
                        help="每代最少训练步数")
    parser.add_argument("--no-ema", action="store_true", help="禁用 EMA 权重")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减系数")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="模型与日志保存目录（**必填**：约定写 models/<版本>/，"
                             "例如 models/V2。默认值已取消，避免忘带参数就写进 checkpoints/）")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="允许往已有训练产物的目录里写（默认拒绝，见安全闸）")
    parser.add_argument("--start-gen", type=int, default=1,
                        help="本 run 第 1 代对外的代数（接着上一轮编号，例如 61）。"
                             "退火/门控节奏仍按 run 内部代数算")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="初始化模型权重路径(默认自动加载 <save-dir>/best.pt)")
    parser.add_argument("--fresh", action="store_true",
                        help="忽略已有 best.pt，从随机权重开始训练")
    parser.add_argument("--resume", action="store_true",
                        help="从 <save-dir>/resume.pt 断点续训"
                             "（恢复优化器/EMA/回放池/代数/lr 计数）")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="指定续训状态文件（优先于 --resume）")
    parser.add_argument("--no-resume-save", action="store_true",
                        help="不写 resume.pt（默认每代写一份）")
    parser.add_argument("--resume-every", type=int, default=1,
                        help="每几代写一次 resume.pt（默认 1=每代）。写盘量与崩溃"
                             "回退粒度成反比：N=5 时平均每代只写 1/5，代价是崩溃后"
                             "最多退回 5 代自对弈数据（权重快照 gen_*.pt 仍在）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机初始化种子（A/B 对照用：两臂同种子=同初值）")
    args = parser.parse_args()

    cfg = Config()
    cfg.games_per_epoch = args.games
    cfg.epochs = args.epochs
    # n_sim=0 表示使用动态深度参数
    cfg.n_sim = None if args.n_sim <= 0 else args.n_sim
    cfg.learning_rate = args.lr
    cfg.max_epochs = args.max_gen
    cfg.batch_size = args.batch_size
    cfg.batch_size_mcts = args.mcts_batch
    cfg.n_blocks = args.blocks
    cfg.channels = args.channels
    cfg.value_hidden = args.value_hidden
    cfg.max_children = args.max_children
    cfg.root_max_children = args.root_max_children
    cfg.heuristic_weight = args.heuristic_weight
    cfg.policy_weight = args.policy_weight
    cfg.anneal_enabled = not args.no_anneal
    cfg.anneal_start = args.anneal_start
    cfg.anneal_mid = args.anneal_mid
    cfg.anneal_end = args.anneal_end
    # 退火起点跟随命令行给出的起始权重，避免"传了 --heuristic-weight 却被忽略"
    cfg.anneal_h_start = args.heuristic_weight
    cfg.anneal_p_start = args.policy_weight
    cfg.anneal_h_end = args.anneal_h_end
    cfg.eval_games = args.eval_games
    cfg.eval_workers = args.eval_workers
    # 点估计门槛：命令行给了就用命令行的，否则沿用 Config 默认（0.55）
    if args.gate_threshold is not None:
        cfg.win_rate_threshold = args.gate_threshold
    cfg.gate_ci_ll = args.gate_ci_ll
    # 【任务 6】辅助头（默认全关 ⇒ 与历史行为完全一致）
    cfg.aux_heads = args.aux_heads or ""
    cfg.aux_ch = args.aux_ch
    cfg.aux_weight_threat = args.aux_weight_threat
    cfg.aux_weight_ownership = args.aux_weight_ownership
    cfg.aux_threat_pos_weight = args.aux_threat_pos_weight
    cfg.lr_ratchet = not args.no_lr_ratchet
    cfg.lr_decay_streak = args.lr_decay_streak
    cfg.eval_every = max(1, args.eval_every)
    cfg.eval_noise_alpha = (None if args.eval_noise_alpha <= 0
                            else args.eval_noise_alpha)
    cfg.eval_noise_eps = args.eval_noise_eps
    cfg.n_workers = args.workers
    cfg.save_dir = args.save_dir
    cfg.start_gen = args.start_gen
    cfg.buffer_size = args.buffer_size
    cfg.buffer_gen_limit = args.buffer_gens
    cfg.fresh_ratio = args.fresh_ratio
    # 【任务 4】回放池采样整形（默认 0/0 ⇒ 与旧版逐位一致）
    cfg.buffer_decay_half_life = args.buffer_decay_half_life
    cfg.buffer_entropy_weight = args.buffer_entropy_weight
    cfg.samples_per_gen = args.samples_per_gen
    cfg.min_samples_per_gen = args.min_steps
    cfg.ema_enabled = not args.no_ema
    cfg.ema_decay = args.ema_decay
    if args.no_augment:
        cfg.on_the_fly_augment = False
        cfg.augment_factor = 1
    if args.no_resign:
        cfg.resign_threshold = None
    if args.no_noise:
        cfg.root_noise_alpha = None

    # 默认加载 best.pt 作为初始化（--fresh 可强制从随机权重开始）
    init_w = args.init_weights
    if init_w is None and not args.fresh:
        best_path = os.path.join(cfg.save_dir, "best.pt")
        init_w = best_path if os.path.exists(best_path) else None

    # ---- 断点续训 ----
    cfg.resume_save = not args.no_resume_save
    cfg.resume_every = max(1, int(args.resume_every))
    resume_path = args.resume_from
    if resume_path is None and args.resume:
        resume_path = os.path.join(cfg.save_dir, "resume.pt")
    if resume_path:
        print(f"断点续训: {resume_path}")
        if init_w:
            print(f"  （注意：网络权重将以 resume 里的为准，"
                  f"--init-weights/best.pt 只用于兜底）")

    # 【A/B 对照】随机初始化的种子。不指定时沿用旧的"不播种"行为；
    # 指定后两个对照臂（如 3×64 vs 6×128）从**同一个随机初值**出发，
    # 否则初值噪声会和"放大网络"的效果混在一起。
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"随机种子: {args.seed}（用于网络初始化）")

    # ---- 安全闸：不许往一个"已经有训练产物"的目录里开新训练 ----
    # 真实风险（用户 2026-10-11 提出）：`--save-dir` 默认就是 `checkpoints`，而 run3 的
    # gen_1..gen_60 / best.pt / resume.pt / train_log.jsonl 全在那儿；**不带 --save-dir
    # 再跑一次会静默覆盖它们**。其中 `data/gen_N.npz` 还是"留出集评估"
    # （probe_learning.py / plot_progress.py）的数据来源 —— 覆盖了等于把标尺弄坏。
    if resume_path is None and not args.allow_overwrite:
        existing = []
        for name in ("train_log.jsonl", "resume.pt", "best.pt"):
            if os.path.exists(os.path.join(cfg.save_dir, name)):
                existing.append(name)
        try:
            if any(f.startswith("gen_") and f.endswith(".pt")
                   for f in os.listdir(cfg.save_dir)):
                existing.append("gen_*.pt")
        except OSError:
            pass
        if existing:
            print("=" * 64)
            print(f"[拒绝启动] 目录 `{cfg.save_dir}` 里已经有训练产物："
                  f"{'、'.join(existing)}")
            print("  接着往这里跑会覆盖它们（含 data/gen_N.npz —— 留出集评估的来源）。")
            print("  请二选一：")
            print("    ① 给这次 run 一个自己的目录：  --save-dir checkpoints/<run名>")
            print("    ② 确实要覆盖：               --allow-overwrite")
            print("  （想接着上次练，请用 --resume-from <目录>/resume.pt）")
            print("=" * 64)
            raise SystemExit(3)

    print("=" * 60)
    print("训练配置摘要")
    print("=" * 60)
    for k, v in cfg.to_dict().items():
        print(f"  {k:<20} = {v}")
    print("-" * 60)
    if cfg.anneal_enabled:
        seen = set()
        for g in (1, cfg.anneal_start + 1, cfg.anneal_mid, cfg.anneal_end,
                  cfg.max_epochs):
            g = max(1, min(g, cfg.max_epochs))
            if g in seen:
                continue
            seen.add(g)
            h, p = annealed_weights(g, cfg)
            print(f"  退火 gen={g:<4} -> heuristic={h:.2f} policy={p:.2f}")
    else:
        print(f"  退火: 已关闭，全程 heuristic={cfg.heuristic_weight} "
              f"policy={cfg.policy_weight}")
    print(f"  初始化: {'随机权重' if init_w is None else init_w}")
    print("=" * 60)

    summary = train(cfg, init_weights=init_w, resume_path=resume_path)
    print(f"\n训练摘要: {summary}")
