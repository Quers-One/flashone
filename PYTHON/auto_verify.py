"""
auto_verify.py
==============
等训练跑到第 N 代，自动做一次**独立对战验证**，把结论写进文件。

为什么需要它
------------
训练里的门控胜率是"20 局、标准误差 ±0.11"的低功效指标，而且它自己就出过
两个 bug（颜色记错 + 评估完全确定性）。所以"训练到底有没有效果"这个问题
不能只听门控的，必须另做一次**独立实现**的对战检验。

这个脚本挂在后台，等 `models/run3/gen_{N}.pt` 出现并且日志里已经有第 N 代记录，
就用 compare.py 的对局代码跑 A vs B 的正面交锋，输出写到
`verify_gen{N}.txt`。

用法
----
    python auto_verify.py --a models/run3/init_gen9.pt --b-gen 20
    python auto_verify.py --a a.pt --b models/run3/gen_30.pt   # 直接指定 B
"""

import argparse
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


def wait_for(path, gen, log_path, poll=60, timeout_h=24):
    """等 gen_{gen}.pt 出现、且它在日志里已经记录完成（避免读到写一半的文件）。"""
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        done_marker = False
        if os.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        if f'"gen": {gen},' in line or f'"gen":{gen},' in line:
                            done_marker = True
            except OSError:
                pass
        if os.path.exists(path) and done_marker:
            # 再确认文件大小稳定（真正写完）
            s1 = os.path.getsize(path)
            time.sleep(3)
            if os.path.getsize(path) == s1:
                return True
        time.sleep(poll)
    return False


def main():
    ap = argparse.ArgumentParser(description="等训练到第 N 代后自动做独立验证")
    ap.add_argument("--a", required=True, help="基线 checkpoint")
    ap.add_argument("--b-gen", type=int, default=None, help="等这一代的 gen_N.pt")
    ap.add_argument("--b", default=None, help="直接指定 B（给了就不等待）")
    ap.add_argument("--save-dir", default="models/run3")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    # 【为什么需要 seed-offset 而不是只改 --seed】
    # 对局过程本身不用全局 RNG：MCTS 内部是 `np.random.default_rng(seed)`，
    # 而本脚本给第 g 局固定传 seed=g / g+5000。所以**只改 --seed 会生成
    # 逐手完全相同的对局**（实测：`--seed` 只影响网络初始化，而权重是加载的）。
    # 想拿到真正独立的第二批对局，必须给每局的 MCTS 种子加偏移。
    # 否则"合并两批 = n 翻倍"是把同一批数据数了两遍，p 值会假显著。
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="给每局 MCTS 种子加偏移（跑独立第二批用，例如 100）")
    ap.add_argument("--tag", default=None,
                    help="报告文件名里的标签（默认 gen{N}；第二批可用 gen60_b2）")
    args = ap.parse_args()

    if args.b:
        b_path = args.b
        tag = args.tag or os.path.splitext(os.path.basename(b_path))[0]
    else:
        if args.b_gen is None:
            print("必须给 --b-gen 或 --b")
            return 1
        b_path = os.path.join(args.save_dir, f"gen_{args.b_gen}.pt")
        tag = args.tag or f"gen{args.b_gen}"
        print(f"等待 {b_path} 出现（每 60s 检查一次，最多 24 小时）...",
              flush=True)
        if not wait_for(b_path, args.b_gen,
                        os.path.join(args.save_dir, "train_log.jsonl")):
            print("等待超时，放弃验证")
            return 1
        print(f"检测到 {b_path}（训练已记录第 {args.b_gen} 代），开始验证",
              flush=True)

    cfg = compare.Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net_a = compare.load_net(args.a)
    net_b = compare.load_net(b_path)

    hw, pw = 6.0, 0.3          # 与训练前 5 代的先验设置一致
    na = None if args.noise_alpha <= 0 else args.noise_alpha
    off = int(args.seed_offset)

    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("=" * 64)
    out(f"独立对战验证  A = {args.a}")
    out(f"              B = {b_path}")
    out(f"n_sim={args.n_sim}  局数={args.games}  根噪声α={na}  "
        f"启发式权重={hw}/{pw}  对局种子偏移={off}")
    out("=" * 64)

    a_bw = a_ww = b_wins = draws = 0
    lens, seqs = [], set()
    t0 = time.time()
    for g in range(args.games):
        if g % 2 == 0:
            ma = compare.make_mcts(net_a, cfg, args.n_sim, hw, pw,
                                   seed=g + off, noise_alpha=na)
            mb = compare.make_mcts(net_b, cfg, args.n_sim, hw, pw,
                                   seed=g + 5000 + off, noise_alpha=na)
            winner, plies, seq = compare.play(ma, mb)
            a_is_black = True
        else:
            mb = compare.make_mcts(net_b, cfg, args.n_sim, hw, pw,
                                   seed=g + off, noise_alpha=na)
            ma = compare.make_mcts(net_a, cfg, args.n_sim, hw, pw,
                                   seed=g + 5000 + off, noise_alpha=na)
            winner, plies, seq = compare.play(mb, ma)
            a_is_black = False
        lens.append(plies)
        seqs.add(seq)
        if winner == 0:
            draws += 1
            who = "和棋"
        elif (winner == 1) == a_is_black:
            if a_is_black:
                a_bw += 1
            else:
                a_ww += 1
            who = "A胜"
        else:
            b_wins += 1
            who = "B胜"
        out(f"  第{g + 1:2d}局: A执{'黑' if a_is_black else '白'} {who:>4} "
            f"{plies:3d} 手")

    a_wins = a_bw + a_ww
    decided = args.games - draws
    out("-" * 64)
    out(f"  A(基线) 胜 {a_wins}  (执黑 {a_bw} / 执白 {a_ww})")
    out(f"  B(新一代) 胜 {b_wins}      和棋 {draws}")
    out(f"  独立对局序列数: {len(seqs)}/{args.games}")
    if decided > 0:
        rate = a_wins / decided
        p = compare.binom_p(a_wins, decided)
        out(f"  有效对局 {decided} 局: A 胜率 {rate:.3f}  双侧二项 p={p:.3f}")
        # 【判据说明】原来这里只分"p<0.05 显著"和"否则无差异"两种，
        # 结果 16:7 这种明显一边倒、p=0.093 的结果被打印成
        # "无显著差异 —— 没有产生可测的棋力提升"，**措辞严重误导**。
        # 现在按三档说：显著 / 倾向性（0.05<=p<0.15）/ 无法区分。
        better = "B（新一代）" if rate < 0.5 else "A（基线）"
        if p < 0.05:
            out(f"  >>> {better}显著更强（p={p:.3f} < 0.05）"
                + ("—— 训练有效" if rate < 0.5 else "—— 训练在退化！"))
        elif p < 0.15:
            out(f"  >>> 倾向 {better} 更强，但**未达显著**（p={p:.3f}，"
                f"样本 {decided} 局太少）")
            out(f"      方向性证据，不能当结论用；要定论需加大 --games")
        else:
            out("  >>> 两者无法区分（p 较大且胜率接近 0.5）")
    out(f"  平均手数 {np.mean(lens):.0f}  耗时 {time.time() - t0:.0f}s")
    out("=" * 64)

    report = os.path.join(args.save_dir, f"verify_{tag}.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n结论已写入 {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
