"""
val_value.py
============
在**固定的验证集**上评估各代 checkpoint 的价值头质量。

为什么需要它（这是对 loss 曲线那个坑的正面解决）
------------------------------------------------
训练日志里的 `loss_v` **不能跨代比较**：每代的标签都来自"那一代网络自己的
搜索"，目标是移动的。实测 18 代下来 `loss_v` 毫无趋势（相关系数 +0.087），
而同期独立对战证明棋力涨了约 162 分 —— 也就是说训练日志里的 loss 曲线
什么也说明不了。

正确做法：**把某一代的数据冻结成验证集**，之后所有 checkpoint 都在**同一批
(局面, 胜负标签)** 上评估。这样得到的 loss 才是可比的，能真正回答
"价值头有没有一代代变强"。

用法
----
    # 用 gen_10 的数据当冻结验证集，评估 gen_2/10/18
    python val_value.py --frozen models/run3/data/gen_10.npz --gens 2 10 18

    # 也可以直接给 checkpoint 路径
    python val_value.py --frozen <npz> --ckpts a.pt b.pt

注意
----
* 验证局的标签是"当时那局的实际胜负"，对后来更强的网络来说，这些局面里
  有一部分是**当时下错的**（标签本身不完美）。所以这个指标衡量的是
  "对同一批历史结果的拟合能力"，不是绝对棋力。
* 但它**跨代可比**，这正是我们要的：如果新一代在这个集合上反而更准，
  说明价值判断确实在进步。
"""

import argparse
import glob
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from network import GomokuNet


def load_net(path):
    """按权重反推架构加载（放大网络后必须，见 train.load_checkpoint_ex）。"""
    from train import load_checkpoint
    net = load_checkpoint(path)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser(description="冻结验证集上的价值头评估")
    ap.add_argument("--frozen", required=True,
                    help="用作冻结验证集的 npz（例如 models/run3/data/gen_10.npz）")
    ap.add_argument("--gens", type=int, nargs="*", default=None)
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--save-dir", default="models/run3")
    ap.add_argument("--limit", type=int, default=4000,
                    help="最多用多少条样本（默认 4000，够稳定且很快）")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    d = np.load(args.frozen)
    st, val = d["states"], d["values"]
    if args.limit and len(st) > args.limit:
        # 均匀抽样，而不是取前 N 条（前 N 条往往集中在少数几局的开局）
        idx = np.linspace(0, len(st) - 1, args.limit).astype(int)
        st, val = st[idx], val[idx]
    print(f"冻结验证集: {args.frozen}")
    print(f"  样本 {len(st)} 条   标签分布: +1 {np.mean(val > 0.5):.1%} / "
          f"-1 {np.mean(val < -0.5):.1%} / 0 {np.mean(np.abs(val) <= 0.5):.1%}")
    base = float(np.mean(val ** 2))
    print(f"  基准 MSE（一律预测 0）= {base:.4f}   基准符号准确率 = "
          f"{max(np.mean(val > 0.5), np.mean(val < -0.5)):.1%}")

    if args.ckpts:
        paths = list(args.ckpts)
    elif args.gens:
        paths = [os.path.join(args.save_dir, f"gen_{g}.pt") for g in args.gens]
    else:
        print("必须给 --gens 或 --ckpts")
        return 1
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("没有可用的 checkpoint")
        return 1
    paths.sort(key=lambda p: int(re.search(r"gen_(\d+)",
                                            os.path.basename(p)).group(1))
               if re.search(r"gen_(\d+)", os.path.basename(p)) else 0)

    print(f"\n{'checkpoint':>14} {'MSE':>9} {'比基准好':>9} {'符号准确率':>10}")
    print("-" * 48)
    rows = []
    for p in paths:
        net = load_net(p)
        preds = []
        with torch.no_grad():
            for i in range(0, len(st), args.batch):
                t = torch.from_numpy(
                    st[i:i + args.batch].astype(np.float32)).to(net.device)
                _pol, v = net.forward(t)
                preds.append(v.cpu().numpy())
        pr = np.concatenate(preds)
        mse = float(np.mean((pr - val) ** 2))
        decided = np.abs(val) > 0.5
        acc = float(np.mean(np.sign(pr[decided]) == np.sign(val[decided])))
        rows.append((p, mse, acc))
        print(f"{os.path.basename(p):>14} {mse:>9.4f} "
              f"{1 - mse / base:>8.1%} {acc:>10.1%}")

    print("-" * 48)
    # 判据必须看"是否优于基准"。实测踩过两个坑：
    #  坑1（记忆效应）：拿 gen_k.npz 当验证集去评 gen_k.pt，那个 checkpoint
    #       正是**训练过这批样本**的，MSE 会好得离谱（实测 0.2156 vs 别人的 1.2），
    #       而 gen_14/gen_18 因为回放池保留 30 代，也见过这批数据。
    #       所以冻结集必须选"所有被评 checkpoint 都没训过"的那一代 ——
    #       唯一的干净选择是**最新的那一代**（只有比它早的 checkpoint 没训过）。
    #  坑2（分布偏移）：自对弈每一代的数据分布都在动。实测在干净的时间留出集上，
    #       **所有**价值头的 MSE 都**差于"一律预测 0"的基准**（-25% ~ -120%），
    #       说明这些局面整体落在老网络的分布之外，指标被分布偏移主导，
    #       此时"最晚的比最早的略好"完全不足以支撑"价值头变强"的结论。
    below_base = all(r[1] > base for r in rows)
    if below_base:
        print(f"注意：所有 checkpoint 的 MSE 都**差于基准**（{base:.4f}），")
        print("      说明分布偏移主导了本指标 —— 这批局面整体在老网络的分布之外。")
        print("      此时**不能**据此判断价值头强弱，请改用独立对战（compare.py/elo.py）。")
    if len(rows) >= 2:
        dmse = rows[-1][1] - rows[0][1]
        dacc = rows[-1][2] - rows[0][2]
        print(f"最早 -> 最晚: MSE {rows[0][1]:.4f} -> {rows[-1][1]:.4f} "
              f"({dmse:+.4f})   符号准确率 {rows[0][2]:.1%} -> {rows[-1][2]:.1%} "
              f"({dacc:+.1%})")
        if below_base:
            print(">>> 结论：本验证集无法回答该问题（见上面的分布偏移说明）")
        elif dmse < 0 and dacc > 0:
            print(">>> 价值头在当前验证集上确实变准了（跨代可比，且优于基准）")
        elif dmse > 0 and dacc < 0:
            print(">>> 价值头变差了 —— 值得查训练")
        else:
            print(">>> 变化不明确（MSE 与符号准确率方向不一致或幅度太小）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
