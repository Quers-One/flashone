"""
draw_rate.py —— 逐代"和棋标签占比 / 策略目标尖锐度"汇总，用来盯**和棋振荡**。

【为什么需要】2026-10-11 发现 V1 的价值标签里"0（和棋）"占比在代之间剧烈振荡
（gen 7 = 49.9%，gen 8 = 8.7%）。和棋局意味着 361 手满盘：既拖慢每代（每局 ~180 秒），
又让价值头的学习信号趋近于 0。这个脚本把每代的这几个数拉成一张表，方便看它是
"收敛过程中的振荡"还是"结构性问题"。

用法：
    python draw_rate.py --dirs models/run3/data                    # run3（60 代）
    python draw_rate.py --dirs checkpoints/pilot/big/data          # V1
    python draw_rate.py --dirs models/run3/data checkpoints/pilot/big/data --csv out.csv
"""

import argparse
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_data import analyze_file  # noqa: E402  （复用同一套指标定义）


def gnum(path):
    m = re.search(r"gen_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description="逐代和棋率 / 目标尖锐度")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--csv", default=None, help="把逐代指标追加到这个 CSV")
    ap.add_argument("--every", type=int, default=1, help="只打印每 N 代（默认全打）")
    args = ap.parse_args()

    out_rows = []
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "gen_*.npz")), key=gnum)
        if not files:
            print(f"[跳过] {d} 里没有 gen_*.npz")
            continue
        print(f"\n{'=' * 78}\n{d}  共 {len(files)} 代\n{'=' * 78}")
        print(f"{'gen':>4} {'样本':>6} {'和棋0':>7} {'+1':>7} {'-1':>7} "
              f"{'one-hot':>8} {'有效候选':>9}")
        stats = []
        for i, p in enumerate(files):
            r = analyze_file(p, verbose=False)
            g = gnum(p)
            stats.append((g, r))
            if i % max(1, args.every) == 0 or i == len(files) - 1:
                print(f"{g:>4} {r['n']:>6} {r['zero']:>7.1%} {r['pos']:>7.1%} "
                      f"{r['neg']:>7.1%} {r['onehot']:>8.1%} {r['supp']:>9.1f}")
            out_rows.append((d, g, r))

        zeros = sorted(r["zero"] for _, r in stats)
        supps = sorted(r["supp"] for _, r in stats)
        mid = len(zeros) // 2
        print(f"{'-' * 78}")
        print(f"和棋标签占比: 最小 {zeros[0]:.1%}  中位 {zeros[mid]:.1%}  最大 {zeros[-1]:.1%}"
              f"   （跨度 {(zeros[-1] - zeros[0]) * 100:.1f} 个百分点）")
        print(f"有效候选点  : 最小 {supps[0]:.1f}  中位 {supps[mid]:.1f}  最大 {supps[-1]:.1f}")
        hi = sum(1 for z in zeros if z > 0.35)
        print(f"和棋占比 >35% 的代: {hi}/{len(zeros)}")

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write("dir,gen,samples,draw_frac,pos_frac,neg_frac,"
                        "onehot,near_onehot,entropy,effective_candidates,dup_rate,conflict\n")
            for d, g, r in out_rows:
                f.write(f"{d},{g},{r['n']},{r['zero']:.4f},{r['pos']:.4f},{r['neg']:.4f},"
                        f"{r['onehot']:.4f},{r['near']:.4f},{r['han']:.4f},"
                        f"{r['supp']:.3f},{r['dup']:.4f},{r['conflict']}\n")
        print(f"\n逐代指标已追加: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
