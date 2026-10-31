"""
analyze_data.py
===============
分析自对弈产出的训练数据本身是否健康（不需要 GPU，纯 CPU）。

为什么需要它
------------
`loss_p` / `loss_v` 只能说明"网络能不能拟合目标"，不能说明**目标本身好不好**。
如果自对弈数据是退化的（例如策略目标几乎全是规则强制的一手点、或者胜负标签
严重偏向某一方），那网络再努力也只是在学一个平凡函数 —— 这会表现为
"loss 看着还行，但棋力原地踏步"。

本工具回答四个具体问题：
  1) 胜负是否严重偏向一方？        -> 偏向会让价值头靠"猜当前执子方"就拿到低 loss
  2) 策略目标有多尖锐？多少是 one-hot？
     one-hot 比例高 = 规则强制覆盖在主导标签 -> 策略头只是在背规则
  3) 局面重复度如何？（同一局面出现多次、标签是否一致）
  4) 对局长度分布（是否长期拖到 100+ 手，即双方都不会终结）

用法
----
    python analyze_data.py                      # 分析 models/run3/data/gen_*.npz
    python analyze_data.py --dir <数据目录>
    python analyze_data.py --per-gen            # 逐代列出，看趋势
"""

import argparse
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np


def ent(p):
    """一行概率分布的熵（nat）。"""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def analyze_file(path, verbose=True):
    d = np.load(path)
    st, pol, val = d["states"], d["policies"], d["values"]
    n = len(val)

    # ---- 1) 胜负标签分布（从行棋方视角）----
    n_pos = int((val > 0.5).sum())
    n_neg = int((val < -0.5).sum())
    n_zero = n - n_pos - n_neg

    # ---- 2) 策略目标尖锐度 ----
    mx = pol.max(axis=1)
    onehot = float((mx > 0.99).mean())          # 完全 one-hot
    near_onehot = float((mx > 0.9).mean())      # 近似 one-hot
    ents = np.array([ent(p) for p in pol])
    support = np.exp(ents)                      # 有效候选点数

    # ---- 3) 局面重复度 ----
    # 用 states 的字节串做指纹（19*19*3 = 1083 字节，先去重再算重复率）
    flat = np.ascontiguousarray(st, dtype=np.uint8).reshape(n, -1)
    uniq, inv, counts = np.unique(flat, axis=0, return_inverse=True,
                                  return_counts=True)
    dup_rate = float(1.0 - len(uniq) / n)
    # 同一局面是否出现过互相矛盾的标签（同样的盘面 + 同样的行棋方，值不同）
    conflict = 0
    if len(uniq) < n:
        for k in np.flatnonzero(counts > 1)[:2000]:
            vs = val[inv == k]
            if vs.size > 1 and (vs.max() - vs.min()) > 0.5:
                conflict += 1

    if verbose:
        print(f"\n--- {os.path.basename(path)}  样本 {n} ---")
        print(f"  胜负标签: +1 {n_pos/n:5.1%}   -1 {n_neg/n:5.1%}   "
              f"0 {n_zero/n:5.1%}   (失衡度 {abs(n_pos-n_neg)/n:5.1%})")
        print(f"  策略目标: one-hot {onehot:5.1%}   近 one-hot {near_onehot:5.1%}"
              f"   平均熵 {ents.mean():.2f} nat   有效候选 {support.mean():5.1f} 点")
        print(f"  局面重复率 {dup_rate:5.1%}   标签冲突局面 {conflict}")
    return dict(n=n, pos=n_pos / n, neg=n_neg / n, zero=n_zero / n,
                onehot=onehot, near=near_onehot, han=float(ents.mean()),
                supp=float(support.mean()), dup=dup_rate, conflict=conflict)


def main():
    ap = argparse.ArgumentParser(description="自对弈数据健康度分析")
    ap.add_argument("--dir", default=os.path.join("models", "run3", "data"))
    ap.add_argument("--per-gen", action="store_true", help="逐代列出")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"目录不存在: {args.dir}")
        return 1
    fs = glob.glob(os.path.join(args.dir, "gen_*.npz"))

    def gnum(p):
        m = re.search(r"gen_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    fs.sort(key=gnum)
    if not fs:
        print(f"{args.dir} 里没有 gen_*.npz")
        return 1

    rows = []
    if args.per_gen:
        for p in fs:
            rows.append((gnum(p), analyze_file(p)))
    else:
        # 只汇总：把全部数据合并后一次性统计（避免按代重复打印）
        allst, allpol, allval = [], [], []
        for p in fs:
            d = np.load(p)
            allst.append(d["states"])
            allpol.append(d["policies"])
            allval.append(d["values"])
        st = np.concatenate(allst)
        pol = np.concatenate(allpol)
        val = np.concatenate(allval)
        tmp = os.path.join(args.dir, "_merged_for_analysis.npz")
        np.savez(tmp, states=st, policies=pol, values=val)
        try:
            stat = analyze_file(tmp)
        finally:
            os.remove(tmp)
        rows.append((0, stat))
        print(f"\n（合并了 {len(fs)} 个文件: gen_"
              f"{gnum(fs[0])} ~ gen_{gnum(fs[-1])}）")

    print("\n" + "=" * 72)
    print("结论提示")
    print("=" * 72)
    st0 = rows[-1][1]
    if st0["onehot"] > 0.25 or st0["near"] > 0.60:
        print("  [!] 策略目标过于尖锐：过半标签集中在极少数点上。")
        print("      说明规则强制覆盖（成五/必挡）在主导标签，策略头主要在背规则。")
        print("      这不是 bug，但要意识到：退火把 heuristic_weight 降下来之后，")
        print("      标签会自然变平，届时 loss_p 的基准也会跟着变。")
    if abs(st0["pos"] - st0["neg"]) > 0.15:
        print(f"  [!] 胜负标签失衡 {st0['pos']:.1%} vs {st0['neg']:.1%}：")
        print("      价值头可能靠'猜当前执子方'就拿到偏低的 loss_v。")
        print("      解读 loss_v 时要考虑这一点，别把 0.10 当成'判断力很强'。")
    if st0["dup"] > 0.05:
        print(f"  [!] 局面重复率 {st0['dup']:.1%}：回放池里同一局面出现多次。")
        print("      若标签互相矛盾（上面那个计数），会增加价值目标的噪声。")
    if st0["supp"] < 3:
        print(f"  [!] 策略目标平均只有 {st0['supp']:.1f} 个有效候选点：")
        print("      搜索的访问分布高度集中，可能宽度限制过紧或规则覆盖过强。")
    if (st0["onehot"] <= 0.25 and st0["near"] <= 0.60
            and abs(st0["pos"] - st0["neg"]) <= 0.15 and st0["dup"] <= 0.05):
        print("  数据看起来健康：标签平衡、目标不过分尖锐、重复率低。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
