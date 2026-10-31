"""
test_selfplay.py
================
自对弈「对局质量」验收工具（优化 阶段 0/1 的端到端检查）。

为什么需要这个脚本
------------------
搜索引擎是否修好，光看单步测试还不够，必须看**自对弈产出的对局是否合理**。
旧实现的实测结果是非常明确的病态信号：

    3 局自对弈 -> 全部 241 手、逐手完全相同的序列、全部平局
    每局出现 181 次"成五机会却不走"

也就是说训练数据是纯噪声：策略标签来自无效搜索、价值标签来自随机残局。
本脚本用同样的口径量化以下几个指标，作为"能否开始训练"的门槛：

    * 对局长度      : 应在 10~120 手内分出胜负（而不是 241 手拖到满盘）
    * 多样性        : N 局的落子序列不应完全相同（温度采样是否真的生效）
    * 漏杀次数      : 出现"一步成五"却没走的次数（必须为 0）
    * 认输机制      : 价值头持续看衰时是否提前结束（可选）

用法
----
    python test_selfplay.py                     # 默认 6 局, n_sim=400
    python test_selfplay.py --games 10 --n-sim 800
    python test_selfplay.py --model models/run3/best.pt
"""

import argparse
import io
import sys
import time

import numpy as np
import torch

from gomoku_env import Board
from network import GomokuNet
from mcts import MCTS
from train import self_play_game, Config


def _make_mcts(net, cfg, n_sim, seed):
    return MCTS(net, n_sim=n_sim, c_puct=cfg.c_puct,
                virtual_loss=cfg.virtual_loss,
                batch_size=cfg.batch_size_mcts,
                max_children=cfg.max_children,
                root_max_children=cfg.root_max_children,
                heuristic_weight=cfg.heuristic_weight,
                policy_weight=cfg.policy_weight, seed=seed)


def main():
    ap = argparse.ArgumentParser(description="自对弈对局质量验收")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-resign", action="store_true")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config()
    if args.no_resign:
        cfg.resign_threshold = None
    if args.no_noise:
        cfg.root_noise_alpha = None

    if args.model:
        # 【放大网络】按权重反推骨干规模，别写死 GomokuNet()
        from train import load_checkpoint
        net = load_checkpoint(args.model)
        print(f"已加载模型: {args.model}")
    else:
        net = GomokuNet()
        print(f"未指定模型 -> 随机初始化网络 (seed={args.seed})")
    net.eval()

    print("=" * 78)
    print(f"自对弈质量检查  n_sim={args.n_sim}  games={args.games}  "
          f"启发式权重={cfg.heuristic_weight}")
    print("旧实现基准: 3 局全部 241 手、逐手相同、全部平局、每局漏杀 181 次")
    print("=" * 78)

    mcts = _make_mcts(net, cfg, args.n_sim, args.seed)

    # ---------- 1) 用正式自对弈接口跑若干局 ----------
    lengths, results, times = [], [], []
    for g in range(args.games):
        mcts.reset_tree()
        t0 = time.time()
        game = self_play_game(
            mcts, net,
            temperature=cfg.temperature,
            temp_threshold=cfg.temp_threshold,
            noise_alpha=cfg.root_noise_alpha,
            noise_eps=cfg.root_noise_eps,
            resign_threshold=cfg.resign_threshold,
            resign_plies=cfg.resign_plies)
        dt = time.time() - t0
        n = len(game)
        lengths.append(n)
        times.append(dt)
        values = [v for _s, _p, v, u in game if u]
        results.append(values[-1] if values else 0.0)
        print(f"  第{g + 1}局: {n:3d} 手  耗时{dt:5.1f}s "
              f"({dt / max(1, n) * 1000:4.0f}ms/手)  "
              f"末步标签={results[-1]:+.0f}")

    # ---------- 2) 真实落子序列：多样性 + 漏杀统计 ----------
    seqs = []
    missed_total = 0
    for g in range(args.games):
        mcts.reset_tree()
        b = Board()
        moves = []
        while not b.is_game_over()[0]:
            player = b.current_player
            wins = {mv for mv in b.get_legal_moves()
                    if b.check_win_after(int(mv[0]), int(mv[1]), player)}
            temp = (cfg.temperature
                    if len(b.history) < cfg.temp_threshold else 0.0)
            mv, _pol = mcts.sample_move_with_policy(
                b, temp=temp, noise_alpha=cfg.root_noise_alpha,
                noise_eps=cfg.root_noise_eps)
            if mv is None:
                break
            if wins and tuple(int(v) for v in mv) not in wins:
                missed_total += 1
            moves.append((int(mv[0]), int(mv[1])))
            b.move(int(mv[0]), int(mv[1]), player)
            mcts.set_root_move(mv)
        seqs.append(tuple(moves))

    uniq = len(set(seqs))
    lens = [len(s) for s in seqs]
    n_draw = sum(1 for r in results if r == 0.0)
    n_decisive = args.games - n_draw
    med = float(np.median(lens))

    print("\n" + "=" * 78)
    print("验收指标")
    print("=" * 78)
    print(f"  对局手数      : min={min(lens)} 中位数={med:.0f} max={max(lens)}")
    print(f"  不同序列数    : {uniq}/{args.games}   （旧实现：全部相同）")
    print(f"  漏杀次数      : {missed_total}   （旧实现：每局 181 次）")
    print(f"  分出胜负局数  : {n_decisive}/{args.games}")
    print(f"  平均每手耗时  : {np.mean(times) / max(1, np.mean(lens)) * 1000:.0f} ms")

    # 判据说明（为什么不卡"最大手数"）：
    #   五子棋在双方都防守正确时**本来就会下成满盘和棋**。实测出现过 361 手满盘
    #   平局，复盘检查发现全程 0 次漏堵将五点 —— 那是正确防守而非缺陷。
    #   因此用"中位数"和"分出胜负比例"衡量整体健康度，而不是最大值。
    #   真正的硬指标是"漏杀次数 == 0"（引擎有能力抓住并完成胜利）。
    ok = True
    if missed_total != 0:
        print("  [未达标] 存在漏杀：终局价值或搜索仍有问题")
        ok = False
    if uniq < 2:
        print("  [未达标] 所有对局序列相同：温度采样/根噪声未生效")
        ok = False
    if med > 200:
        print("  [未达标] 中位手数过长：引擎普遍无法终结对局")
        ok = False
    if n_decisive * 2 < args.games:
        print(f"  [未达标] 分胜负局数不足一半（{n_decisive}/{args.games}）")
        ok = False
    print("=" * 78)
    print("[达标] 自对弈质量通过，可以开始训练" if ok else "[未达标] 请先修搜索")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
