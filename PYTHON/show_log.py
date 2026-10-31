"""
show_log.py
===========
把 `models/run3/train_log.jsonl` 汇总成一张表（训练期间随时看进度用）。

用法
----
    python show_log.py                          # 默认读 models/run3/train_log.jsonl
    python show_log.py --log <路径>             # 指定日志
    python show_log.py --tail 10                # 只看最后 10 代
"""

import argparse
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def fmt(v, spec="{:.3f}"):
    return spec.format(v) if v is not None else "   -   "


def main():
    ap = argparse.ArgumentParser(description="训练日志汇总")
    ap.add_argument("--log", default=os.path.join("models", "run3", "train_log.jsonl"))
    ap.add_argument("--tail", type=int, default=0, help="只看最后 N 代（0=全部）")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"日志不存在: {args.log}")
        return

    rows = []
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[警告] 跳过损坏的一行: {line[:80]}")
    if not rows:
        print("日志为空")
        return
    if args.tail > 0:
        rows = rows[-args.tail:]

    print(f"{'gen':>4} {'lr':>9} {'raw':>6} {'ema':>6} {'采用':>7} "
          f"{'来源':>8} {'样本':>6} {'loss_p':>7} {'loss_v':>7} "
          f"{'退火h/p':>11} {'时间':>8}")
    print("-" * 96)
    accepted = rejected = skipped = 0
    for r in rows:
        acc = r.get("accepted")
        if acc is None:
            tag = "不门控"
            skipped += 1
        elif acc:
            tag = "[采纳]"
            accepted += 1
        else:
            tag = "[拒绝]"
            rejected += 1
        ts = str(r.get("timestamp", ""))[-8:]
        lp = r.get("loss_policy")
        lv = r.get("loss_value")
        print(f"{r['gen']:>4} {r['lr']:>9.6f} {fmt(r.get('win_rate_raw')):>6} "
              f"{fmt(r.get('win_rate_ema')):>6} {tag:>7} "
              f"{str(r.get('chosen')):>8} {r.get('data_samples', 0):>6} "
              f"{fmt(lp, '{:.4f}') if lp is not None else '   -   ':>7} "
              f"{fmt(lv, '{:.4f}') if lv is not None else '   -   ':>7} "
              f"{str(r.get('heuristic_weight')) + '/' + str(r.get('policy_weight')):>11} "
              f"{ts:>8}")
    print("-" * 96)
    gates = [r for r in rows if r.get("win_rate") is not None
             and r.get("accepted") is not None and r["gen"] != 1]
    if gates:
        wrs = [r["win_rate"] for r in gates]
        print(f"门控 {len(gates)} 次: 采纳 {accepted} / 拒绝 {rejected} / "
              f"不门控 {skipped}  |  胜率 min={min(wrs):.3f} "
              f"max={max(wrs):.3f} 末次={wrs[-1]:.3f}")
    # 损失趋势比胜率信噪比高：loss_p 的"什么都没学"基准是 ln(361)=5.89
    lps = [r["loss_policy"] for r in rows if r.get("loss_policy") is not None]
    lvs = [r["loss_value"] for r in rows if r.get("loss_value") is not None]
    if len(lps) >= 2:
        print(f"损失趋势 (loss_p 基准 ln361=5.89): "
              f"loss_p {lps[0]:.4f} -> {lps[-1]:.4f}   "
              f"loss_v {lvs[0]:.4f} -> {lvs[-1]:.4f}")
    print(f"代数: {rows[0]['gen']} ~ {rows[-1]['gen']}   "
          f"回放池={rows[-1].get('buffer_total', 0)} 条 / "
          f"{rows[-1].get('buffer_gens', 0)} 代")
    if len(rows) >= 2:
        t0 = rows[0].get("timestamp")
        t1 = rows[-1].get("timestamp")
        print(f"记录时间: {t0} ~ {t1}")


if __name__ == "__main__":
    main()
