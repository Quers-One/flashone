"""
bench.py
========
自对弈吞吐基准（用来验证"某次优化到底有没有用"）。

为什么需要它
------------
本项目的瓶颈长期在**规则计算**而不是神经网络（实测 `torch.conv2d` 只占 ~5%），
所以"感觉变快了/变慢了"毫无意义，必须用固定口径量：固定种子、固定手数、
默认 20 手，报 ms/手。历史上出现过两次"看起来很快、其实是错的"的情况
（增量棋盘定位错误导致反复评估同一局面、缓存狂命中），因此这个基准只用于
**比对相对快慢**，正确性一律以 `test_tactics.py` / `test_selfplay.py` 为准。

用法
----
    python bench.py                     # n_sim=400, 20 手
    python bench.py --n-sim 200 --moves 30
    python bench.py --profile           # 附带 cProfile 自身耗时前 15 名

参考基线（RTX 5070 Ti Laptop + Core Ultra 9 275HX）
--------------------------------------------------
    n_sim=400, 随机初始化网络: ~410~480 ms/手（填充棋盘重构前）
"""

import argparse
import cProfile
import io
import pstats
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

from gomoku_env import Board
from mcts import MCTS
from network import GomokuNet
from train import Config


def build(n_sim, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = GomokuNet()
    net.eval()
    mcts = MCTS(net, n_sim=n_sim, c_puct=cfg.c_puct,
                virtual_loss=cfg.virtual_loss,
                batch_size=cfg.batch_size_mcts,
                max_children=cfg.max_children,
                root_max_children=cfg.root_max_children,
                heuristic_weight=cfg.heuristic_weight,
                policy_weight=cfg.policy_weight, seed=seed)
    return net, mcts


def run(mcts, moves, noise=True):
    """走 moves 手（带树复用），返回 (实际手数, 每手耗时列表, 搜索痕迹)。"""
    board = Board()
    times = []
    mcts.reset_tree()
    for _ in range(moves):
        t0 = time.perf_counter()
        mv, _pol = mcts.sample_move_with_policy(
            board, temp=1.0,
            noise_alpha=(0.15 if noise else None),
            noise_eps=(0.25 if noise else 0.0))
        times.append(time.perf_counter() - t0)
        if mv is None:
            break
        board.move(int(mv[0]), int(mv[1]), board.current_player)
        mcts.set_root_move(mv)
        if board.is_game_over()[0]:
            break
    return len(times), times, mcts


def main():
    ap = argparse.ArgumentParser(description="自对弈吞吐基准")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--moves", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-noise", action="store_true",
                    help="关闭根节点 Dirichlet 噪声（与历史基线口径一致时用）")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = Config()
    net, mcts = build(args.n_sim, args.seed, cfg)
    noise = not args.no_noise

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        n, times, mcts = run(mcts, args.moves, noise=noise)
        pr.disable()
        st = pstats.Stats(pr)
        print(f"n_sim={args.n_sim}  手数={n}  总耗时={st.total_tt:.2f}s  "
              f"平均={st.total_tt / max(1, n) * 1000:.0f} ms/手")
        print("\n--- 自身耗时 (tottime) 前若干名 ---")
        st.sort_stats("tottime").print_stats(args.top)
        return

    n, times, mcts = run(mcts, args.moves, noise=noise)
    st = mcts.collect_stats(mcts._root)
    per = [t * 1000 for t in times]
    print(f"n_sim={args.n_sim}  batch_size={cfg.batch_size_mcts}  "
          f"宽度(根/其它)={cfg.root_max_children}/{cfg.max_children}")
    print(f"  手数={n}  总计={sum(times):.2f}s")
    print(f"  每手: 平均={np.mean(per):.0f}ms  中位={np.median(per):.0f}ms  "
          f"min={min(per):.0f}ms  max={max(per):.0f}ms")
    print(f"  搜索: 树节点={st['nodes']}  最大深度={st['max_depth']}  "
          f"网络前向占比={st['net_ratio']:.1%}  "
          f"棋盘重建(resync)={st['resync']}")
    print(f"  => 每手 {np.mean(per):.0f} ms；40 局 × 44 手 ≈ "
          f"{np.mean(per) / 1000 * 44 * 40 / 60:.1f} min/代")


if __name__ == "__main__":
    main()
