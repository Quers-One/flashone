"""
elo.py
======
用多个 checkpoint 之间的对局结果，拟合一个 **Elo 等级分**（Bradley-Terry 模型）。

为什么需要它
------------
门控只回答"新模型有没有比**上一个**最佳强"，而且是 20 局、标准误差 ±0.11 的
低功效检验。整个训练下来，我们真正想知道的是：**gen 60 比 gen 1 强多少？**
这不是一次两两对比能回答的，需要把多个 checkpoint 放在同一把尺子上量。

做法
----
1. 让给定的若干 checkpoint 两两对局（先后手交替，带根噪声保证每局不同）；
2. 用 Bradley-Terry 模型 `P(i 胜 j) = 1 / (1 + 10^(-(r_i - r_j)/400))`
   对全部对局结果做极大似然拟合，得到每个 checkpoint 的等级分；
3. 以第一个 checkpoint 为 0 分锚点（只定相对强弱，绝对值无意义）。

用法
----
    python elo.py --gens 1 10 20 30 40 50 60 --games 12
    python elo.py --ckpts a.pt b.pt --games 20 --n-sim 400

注意
----
* 12 局/对的噪声仍然很大（单对 ±约 0.14 胜率），所以**别把小数点当精确值看**，
  看量级即可（例如"60 代比 1 代高 150 分左右"）。
* 想更准就加 `--games`，代价是时间线性增长。
"""

import argparse
import itertools
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def expected(r_a, r_b, scale):
    """Bradley-Terry 期望胜率（含和棋算半分的约定）。"""
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / scale))


def fit_elo(pairs, scale=400.0, iters=20000, lr=8.0, prior=0.5):
    """
    极大似然拟合等级分。

    参数:
        pairs: list of (i, j, score_i, n)  —— i 对 j 打了 n 局、得分率 score_i
        prior: 伪计数先验（每对给双方各先记 prior 胜）。
               必要性：若某对是 0 胜 n 负，纯 MLE 的等级分会**发散到负无穷**
               —— 实测 2 局 0:2 时拟合给出 -264 这种毫无意义的数字。
               加半个伪计数把极端得分率向 0.5 拉一点，估计才稳定。
    返回:
        ratings: np.ndarray，第一个选手锚定为 0
    """
    names = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    idx = {n: k for k, n in enumerate(names)}
    r = np.zeros(len(names), dtype=np.float64)

    # 简单梯度上升：对数似然求导，做归一化后逐步走。
    # 步长随迭代衰减，保证收敛（这几个选手的规模不需要更复杂的方法）。
    for t in range(iters):
        g = np.zeros_like(r)
        for (a, b, s, n) in pairs:
            ia, ib = idx[a], idx[b]
            # 伪计数正则：得分率与局数都往"多打一局、各得半分"的方向修正
            s_eff = (s * n + prior) / (n + 2.0 * prior)
            n_eff = n + 2.0 * prior
            e = expected(r[ia], r[ib], scale)
            # d/dr_a [s*ln e + (1-s)*ln(1-e)] = (s - e) * ln(10)/scale
            g[ia] += (s_eff - e) * n_eff * np.log(10.0) / scale
            g[ib] -= (s_eff - e) * n_eff * np.log(10.0) / scale
        r += (lr / (1.0 + t / 2000.0)) * g
        r -= r.mean()          # 去掉整体平移的自由度
    r -= r[0]
    return names, r


def main():
    ap = argparse.ArgumentParser(description="多 checkpoint Elo 估计")
    ap.add_argument("--gens", type=int, nargs="*", default=None,
                    help="要评估的代数（读 models/run3/gen_N.pt）")
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="直接给 checkpoint 路径（与 --gens 二选一）")
    ap.add_argument("--save-dir", default="models/run3")
    ap.add_argument("--games", type=int, default=12,
                    help="每一对打多少局（先后手交替）")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--heuristic-weight", type=float, default=6.0)
    ap.add_argument("--policy-weight", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=400.0, help="Elo 刻度")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.ckpts:
        ckpts = list(args.ckpts)
    elif args.gens:
        ckpts = [os.path.join(args.save_dir, f"gen_{g}.pt") for g in args.gens]
    else:
        print("必须给 --gens 或 --ckpts")
        return 1

    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        print("以下 checkpoint 不存在，先等训练产出：")
        for m in missing:
            print("  ", m)
        return 1

    cfg = compare.Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    nets = {}
    print("加载模型：")
    for c in ckpts:
        nets[c] = compare.load_net(c)

    hw, pw = args.heuristic_weight, args.policy_weight
    na = None if args.noise_alpha <= 0 else args.noise_alpha
    n_pair = len(ckpts) * (len(ckpts) - 1) // 2
    print(f"\n两两对局: {n_pair} 对 × {args.games} 局 "
          f"(n_sim={args.n_sim}, 根噪声α={na})")
    est_total = n_pair * args.games * 5        # 实测约 1.5~5s/局（对局普遍不长）
    print(f"预计耗时 {est_total / 60:.0f} 分钟量级"
          f"（{n_pair * args.games} 局；实测波动大，看量级即可）\n")

    pairs = []
    raw = {}
    t0 = time.time()
    for a, b in itertools.combinations(ckpts, 2):
        a_wins = b_wins = draws = 0
        seqs = set()
        for g in range(args.games):
            if g % 2 == 0:
                ma = compare.make_mcts(nets[a], cfg, args.n_sim, hw, pw,
                                       seed=g, noise_alpha=na)
                mb = compare.make_mcts(nets[b], cfg, args.n_sim, hw, pw,
                                       seed=g + 5000, noise_alpha=na)
                winner, _plies, seq = compare.play(ma, mb)
                a_is_black = True
            else:
                mb = compare.make_mcts(nets[b], cfg, args.n_sim, hw, pw,
                                       seed=g, noise_alpha=na)
                ma = compare.make_mcts(nets[a], cfg, args.n_sim, hw, pw,
                                       seed=g + 5000, noise_alpha=na)
                winner, _plies, seq = compare.play(mb, ma)
                a_is_black = False
            seqs.add(seq)
            if winner == 0:
                draws += 1
            elif (winner == 1) == a_is_black:
                a_wins += 1
            else:
                b_wins += 1
        n = args.games
        score = (a_wins + 0.5 * draws) / n
        pairs.append((a, b, score, n))
        raw[(a, b)] = (a_wins, b_wins, draws, len(seqs))
        print(f"  {os.path.basename(a):>12} vs {os.path.basename(b):<12} "
              f"{a_wins:2d}胜 {b_wins:2d}负 {draws:2d}和  "
              f"得分率 {score:.3f}  独立序列 {len(seqs)}/{n}", flush=True)

    names, r = fit_elo(pairs, scale=args.scale)
    print("\n" + "=" * 66)
    print("Bradley-Terry Elo 拟合（第一个为 0 分锚点）")
    print("=" * 66)
    print(f"{'checkpoint':>14} {'Elo':>8} {'相对锚点':>10}")
    base = r[0]
    for nm, v in zip(names, r):
        print(f"{os.path.basename(nm):>14} {v:>8.1f} {v - base:>+10.1f}")
    spread = r.max() - r.min()
    print("-" * 66)
    print(f"最高与最低相差 {spread:.0f} 分（{args.scale:.0f} 刻度）")
    print("拟合残差检查（实际得分率 vs 模型预测）：")
    worst = 0.0
    for (a, b, s, n) in pairs:
        ia, ib = names.index(a), names.index(b)
        pred = expected(r[ia], r[ib], args.scale)
        worst = max(worst, abs(pred - s))
        print(f"  {os.path.basename(a):>12} vs {os.path.basename(b):<12} "
              f"实际 {s:.3f}  预测 {pred:.3f}  差 {pred - s:+.3f}")
    print(f"最大偏差 {worst:.3f}"
          + ("   <<< 偏差偏大，说明刻度/模型不太贴合，或局数太少"
             if worst > 0.25 else "   （贴合良好）"))
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
