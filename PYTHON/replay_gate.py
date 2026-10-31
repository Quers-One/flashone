#!/usr/bin/env python
"""replay_gate.py —— 用本项目**真实的**历史门控决策，回放新旧两套门控协议。

任务验收标准里写着：

    回放 100 个历史模型对局，误拒率从 74.8% 降到 <10%

这个脚本就是拿来回答那句话的。三个模式：

  analytic（默认）
      把 models/*/train_log.jsonl 里每一条"有门控"的记录当作一次真实对局。
      新协议在同一批对局上会怎么判？用**后验预测**算，不拿观测胜率当真值：
          观测到 20 局赢 k 局 ⇒ p ~ Beta(k+1, 21-k)（均匀先验）
          新协议再抽 X ~ Binomial(100, p)，判 X ≥ 60
      这个复合分布是 Beta-二项分布，尾部概率有精确闭式（见 betabinom_tail_ge），
      所以结果是确定的、不需要蒙特卡洛。

  power
      与历史无关的纯功效表：P(拒绝 | 真实胜率 p) 在新旧协议下各是多少。
      验收里那个 74.8% 就是这条曲线在 p=0.50 的取值。

  real
      真打局。挑若干历史对局，同一对模型跑 20 局与 100 局各一次，比较判决。
      慢（约 2 秒/局，8 进程），用来抽样验证 analytic 的结论。
      注意：当时若走的是 EMA 分支（train_log 的 chosen == "ema"），影子权重没有
      落盘、无法复现，这类记录会被跳过。

用法：
    python replay_gate.py                          # 后验预测回放（全部 run）
    python replay_gate.py --mode power             # 功效表
    python replay_gate.py --mode real --pairs 6 --workers 8
    python replay_gate.py --json out.json          # 存一份机器可读的结果
"""

import argparse
import json
import math
import os
import random
import sys
import time

# ⚠️ 下面这几个常量是**任务书里的对照方案**（"如果把门控改成 100 局 + CI 下界 0.50
# 会怎样"），**不是现役训练配置**。现役是 train.Config 的 `eval_games=20` +
# `gate_ci_ll=0.0`（= 关闭 CI 条件，判定与历史一致）—— 决定与理由见 NOTES.md §18.6，
# 由 verify_gate_protocol.py 的 E 组盯着。改这几个常量只影响本脚本的回放结论。
GATE_Z = 1.96
GATE_GAMES_NEW = 100
GATE_THRESHOLD = 0.55
GATE_CI_LL = 0.50
OLD_GAMES = 20
OLD_K = 12          # 旧协议：阈值 0.55 + 20 局 ⇒ 实际要 ≥12/20（= 现役协议）


def wilson_lo(k, n, z=GATE_Z):
    if n <= 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def elo_from_score(s):
    s = min(max(s, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / s - 1.0)


def binom_pmf(k, n, p):
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    lc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(lc + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_tail_ge(k, n, p):
    """P(X >= k), X ~ Binomial(n, p)。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(binom_pmf(i, n, p) for i in range(k, n + 1)))


def reject_prob(n, k_min, p):
    """P(拒绝 | 真实胜率 p)：赢局数 < k_min 就拒。

    注意是 P(X ≤ k_min-1) = 1 - P(X ≥ k_min)；写成 binom_tail_ge(0, …) 就恒等于 1。
    """
    if k_min <= 0:
        return 0.0
    return 1.0 - binom_tail_ge(k_min, n, p)


def betabinom_tail_ge(k, n, a, b):
    """P(X >= k)，其中 X ~ Binomial(n, p) 且 p ~ Beta(a, b)。

    精确式（Beta-二项分布）：
        P(X=x) = C(n,x) · B(a+x, b+n-x) / B(a,b)
    用它就等于"对一个只知道 20 局观测结果的对局，按后验预测重放 100 局"。
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    tot = 0.0
    for x in range(k, n + 1):
        lc = math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
        lnum = (math.lgamma(a + x) + math.lgamma(b + n - x)
                - math.lgamma(a + b + n))
        tot += math.exp(lc + lnum - lB)
    return min(1.0, tot)


# --------------------------------------------------------------- 读历史
def read_gate_rows(runs, models_dir="models"):
    """读 models/<run>/train_log.jsonl 里所有"有门控"的记录。"""
    rows = []
    for run in runs:
        path = os.path.join(models_dir, run, "train_log.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                wr = r.get("win_rate")
                if wr is None:
                    continue                      # 这一代没门控
                n = int(r.get("n_games") or OLD_GAMES)
                if n <= 0:
                    continue
                rows.append({
                    "run": run,
                    "gen": r.get("gen"),
                    "local_gen": r.get("local_gen"),
                    "n": n,
                    "wins": int(round(float(wr) * n)),
                    "win_rate": float(wr),
                    "accepted": bool(r.get("accepted")),
                    "chosen": r.get("chosen"),
                    "win_rate_ema": r.get("win_rate_ema"),
                    "timestamp": r.get("timestamp"),
                })
    return rows


# --------------------------------------------------------------- 模式
def mode_analytic(rows, args):
    """后验预测回放：同一批历史对局，新协议会怎么判。"""
    k_new = None
    for k in range(GATE_GAMES_NEW + 1):
        if k / GATE_GAMES_NEW > GATE_THRESHOLD and wilson_lo(k, GATE_GAMES_NEW) > GATE_CI_LL:
            k_new = k
            break
    if k_new is None:
        print("❌ 新协议有效门槛无解")
        return 1

    print("=" * 78)
    print("[1] 历史门控总体（models/*/train_log.jsonl）")
    print(f"    {'run':<10} {'记录数':>7} {'历史采纳':>9} {'历史拒绝':>9} "
          f"{'平均胜率':>9} {'中位胜率':>9}")
    per_run = {}
    for run in sorted({r['run'] for r in rows}):
        rs = [r for r in rows if r["run"] == run]
        acc = sum(1 for r in rs if r["accepted"])
        wrs = sorted(r["win_rate"] for r in rs)
        med = wrs[len(wrs) // 2] if wrs else float("nan")
        per_run[run] = rs
        print(f"    {run:<10} {len(rs):>7} {acc:>9} {len(rs)-acc:>9} "
              f"{sum(wrs)/len(wrs):>9.3f} {med:>9.3f}")
    all_acc = sum(1 for r in rows if r["accepted"])
    print(f"    {'合计':<10} {len(rows):>7} {all_acc:>9} {len(rows)-all_acc:>9} "
          f"{sum(r['win_rate'] for r in rows)/len(rows):>9.3f}")

    print("\n[2] 后验预测回放：新协议（%d 局、须 ≥%d）在同一批对局上的判决"
          % (GATE_GAMES_NEW, k_new))
    print("    观测 k/20 ⇒ p~Beta(k+1,21-k) ⇒ 再抽 X~Bin(100,p)，判 X≥%d" % k_new)
    print(f"    {'run':<10} {'历史采纳':>8} {'新协议仍采纳':>12} {'保真率':>8} "
          f"{'历史拒绝':>8} {'改判采纳':>9} {'翻案率':>8}")
    tot_keep = tot_keep_n = tot_flip = tot_flip_n = 0
    new_acc_expect = 0.0
    detail = []
    for run in sorted(per_run):
        rs = per_run[run]
        keep = keep_n = flip = flip_n = 0
        for r in rs:
            a_post = r["wins"] + 1
            b_post = r["n"] - r["wins"] + 1
            p_acc = betabinom_tail_ge(k_new, GATE_GAMES_NEW, a_post, b_post)
            new_acc_expect += p_acc
            detail.append({"run": run, "gen": r["gen"], "wins20": r["wins"],
                           "accepted_old": r["accepted"], "p_accept_new": p_acc})
            if r["accepted"]:
                keep_n += 1
                keep += p_acc
            else:
                flip_n += 1
                flip += p_acc
        tot_keep += keep; tot_keep_n += keep_n
        tot_flip += flip; tot_flip_n += flip_n
        print(f"    {run:<10} {keep_n:>8} {keep:>12.1f} "
              f"{(keep/keep_n if keep_n else float('nan')):>7.1%} "
              f"{flip_n:>8} {flip:>9.1f} "
              f"{(flip/flip_n if flip_n else float('nan')):>7.1%}")
    print(f"    {'合计':<10} {tot_keep_n:>8} {tot_keep:>12.1f} "
          f"{(tot_keep/tot_keep_n if tot_keep_n else float('nan')):>7.1%} "
          f"{tot_flip_n:>8} {tot_flip:>9.1f} "
          f"{(tot_flip/tot_flip_n if tot_flip_n else float('nan')):>7.1%}")
    print(f"\n    新协议在这批对局上的期望采纳率 = "
          f"{new_acc_expect/len(rows):.1%}（历史实际采纳率 "
          f"{all_acc/len(rows):.1%}）")

    print("\n[3] 误拒率（固定真值 p，与历史无关）—— 验收标准里的那个数")
    print(f"    {'真实胜率':>8} {'≈Elo':>7} {'旧协议(20局≥12)':>16} "
          f"{'新协议(100局≥%d)' % k_new:>18}")
    ptab = []
    for p in (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70):
        ro = reject_prob(OLD_GAMES, OLD_K, p)
        rn = reject_prob(GATE_GAMES_NEW, k_new, p)
        ptab.append({"p": p, "elo": elo_from_score(p), "reject_old": ro,
                     "reject_new": rn})
        flag = "  ← 反而更差" if rn > ro else ""
        print(f"    {p:>8.2f} {elo_from_score(p):>+7.1f} {ro:>15.1%} "
              f"{rn:>17.1%}{flag}")

    # 找 <10% 的临界真值
    lo, hi = 0.5, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if reject_prob(GATE_GAMES_NEW, k_new, mid) > 0.10:
            lo = mid
        else:
            hi = mid
    pcross = (lo + hi) / 2

    print(f"\n    ⚠ 新协议把误拒率压到 <10% 需要真实胜率 ≈{pcross:.3f}"
          f"（≈{elo_from_score(pcross):+.0f} Elo）")
    print(f"    ⚠ 而本项目门控层面的真实效应是 +12~+35 Elo（p≈0.52~0.55），"
          f"那一档新协议误拒 {reject_prob(GATE_GAMES_NEW, k_new, 0.55):.1%}")

    print("\n[4] 结论")
    print(f"    * 验收标准「误拒率从 74.8% 降到 <10%」在 n=100 + 阈值>0.55 + "
          f"CI下界>0.50 下**不可能成立**：")
    print(f"      有效门槛 = {k_new}/100 = {k_new/100:.3f}，与旧协议的有效门槛 "
          f"{OLD_K}/{OLD_GAMES} = {OLD_K/OLD_GAMES:.3f} 相同 ⇒ 同一档真值下"
          f"拒绝得更多。")
    print(f"    * 在历史对局上，历史上被采纳的 {tot_keep_n} 条里，新协议只还会"
          f"采纳 {tot_keep:.1f} 条（{tot_keep/tot_keep_n:.1%}）⇒ 训练会因为门控"
          f"过不去而大面积停摆。")
    print(f"    * 要让「误拒<10%」与「采纳者真实胜率>0.55」同时成立，需要 "
          f"~658 局；只区分 0.50 与 0.60 需要 ~168 局。")
    return 0, {"k_new": k_new, "rows": len(rows), "keep_rate": tot_keep / max(1, tot_keep_n),
               "flip_rate": tot_flip / max(1, tot_flip_n),
               "new_accept_expected": new_acc_expect / len(rows),
               "old_accept": all_acc / len(rows), "pcross": pcross,
               "power": ptab, "detail": detail}


def mode_power(rows, args):
    k_new = None
    for k in range(GATE_GAMES_NEW + 1):
        if k / GATE_GAMES_NEW > GATE_THRESHOLD and wilson_lo(k, GATE_GAMES_NEW) > GATE_CI_LL:
            k_new = k
            break
    print("=" * 78)
    print(f"功效表：P(拒绝 | 真实胜率 p)")
    print(f"  旧协议 = {OLD_GAMES} 局、胜率>{GATE_THRESHOLD}（无 CI 条件）"
          f" ⇒ 有效门槛 {OLD_K}/{OLD_GAMES}")
    print(f"  新协议 = {GATE_GAMES_NEW} 局、胜率>{GATE_THRESHOLD} 且 Wilson 下界>"
          f"{GATE_CI_LL} ⇒ 有效门槛 {k_new}/{GATE_GAMES_NEW}")
    print(f"  {'真实胜率':>8} {'≈Elo':>7} {'旧':>9} {'新':>9} {'谁更好':>8}")
    out = []
    for p in (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70):
        ro = reject_prob(OLD_GAMES, OLD_K, p)
        rn = reject_prob(GATE_GAMES_NEW, k_new, p)
        who = "旧" if ro < rn else ("新" if rn < ro else "平")
        out.append({"p": p, "elo": elo_from_score(p), "old": ro, "new": rn})
        print(f"  {p:>8.2f} {elo_from_score(p):>+7.1f} {ro:>8.1%} {rn:>8.1%} {who:>8}")
    print("\n  【规则菜单】n=100 换门槛：")
    print(f"  {'需胜局':>6} {'门槛':>6} {'误受@0.50':>10} {'误拒@0.55':>10} "
          f"{'误拒@0.60':>10} {'误拒@0.65':>10}")
    for k in (44, 50, 55, 56, 60):
        print(f"  {k:>6} {k/100:>6.2f} {1-reject_prob(100,k,0.50):>9.1%} "
              f"{reject_prob(100,k,0.55):>10.1%} {reject_prob(100,k,0.60):>10.1%} "
              f"{reject_prob(100,k,0.65):>10.1%}")
    return 0, {"k_new": k_new, "power": out}


def mode_real(rows, args):
    """真打局：同一对模型，20 局与 100 局各跑一次，比判决。"""
    import subprocess
    py = sys.executable
    # 重建"当时的对手"：门控比的是 current gen vs 上一次被采纳的 gen
    last_acc = {}
    pairs = []
    for r in sorted(rows, key=lambda x: (x["run"], x["gen"] or 0)):
        run = r["run"]
        if r["chosen"] == "ema":
            continue                      # EMA 影子权重没落盘，无法复现
        prev = last_acc.get(run)
        if prev is not None and r["gen"] is not None:
            pairs.append({"run": run, "cand": r["gen"], "anchor": prev,
                          "wins20": r["wins"], "accepted": r["accepted"]})
        if r["accepted"]:
            last_acc[run] = r["gen"]
    random.seed(1234)
    random.shuffle(pairs)
    pairs = pairs[:args.pairs]
    print(f"真打局回放：{len(pairs)} 对（20 局 + 100 局各一次，{args.workers} 进程）")
    results = []
    for i, pr in enumerate(pairs, 1):
        cpath = f"models/{pr['run']}/gen_{pr['cand']}.pt"
        apath = f"models/{pr['run']}/gen_{pr['anchor']}.pt"
        if not (os.path.isfile(cpath) and os.path.isfile(apath)):
            print(f"  [{i}/{len(pairs)}] 跳过（权重缺失）{cpath} / {apath}")
            continue
        row = dict(pr)
        for n in (OLD_GAMES, GATE_GAMES_NEW):
            j = f"models/{pr['run']}/eval/replay_{pr['cand']}v{pr['anchor']}_{n}.json"
            os.makedirs(os.path.dirname(j), exist_ok=True)
            t0 = time.time()
            cmd = [py, "-u", "ladder.py", "--candidate", cpath, "--anchor", apath,
                   "--games", str(n), "--workers", str(args.workers), "--json", j]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0 or not os.path.isfile(j):
                print(f"  [{i}/{len(pairs)}] {n} 局失败 rc={p.returncode}")
                row[f"score{n}"] = None
                continue
            with open(j, "r", encoding="utf-8") as f:
                d = json.load(f)
            row[f"score{n}"] = d.get("score")
            row[f"elo{n}"] = d.get("elo")
            row[f"sec{n}"] = round(time.time() - t0, 1)
        k20 = None if row.get("score20") is None else round(row["score20"] * OLD_GAMES)
        k100 = None if row.get("score100") is None else round(row["score100"] * GATE_GAMES_NEW)
        row["accept_old"] = None if k20 is None else (k20 >= OLD_K)
        row["accept_new"] = None if k100 is None else (
            k100 / GATE_GAMES_NEW > GATE_THRESHOLD
            and wilson_lo(k100, GATE_GAMES_NEW) > GATE_CI_LL)
        row["k20"], row["k100"] = k20, k100
        results.append(row)
        print(f"  [{i}/{len(pairs)}] {pr['run']} gen{pr['cand']} vs gen{pr['anchor']}: "
              f"20局 {k20}/20 判{row['accept_old']} | 100局 {k100}/100 判{row['accept_new']} "
              f"({row.get('sec100','?')}s)")
    agree = sum(1 for r in results if r["accept_old"] == r["accept_new"])
    print(f"\n  一致 {agree}/{len(results)}；"
          f"20 局判采纳 {sum(1 for r in results if r['accept_old'])} 条，"
          f"100 局判采纳 {sum(1 for r in results if r['accept_new'])} 条")
    return 0, {"pairs": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="analytic",
                    choices=["analytic", "power", "real"])
    ap.add_argument("--runs", default="V1,V2,V3,run3")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--pairs", type=int, default=6, help="real 模式跑几对")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = read_gate_rows([s for s in args.runs.split(",") if s], args.models_dir)
    if args.mode != "power" and not rows:
        print("❌ 没读到任何有门控的历史记录")
        return 1
    print(f"读到 {len(rows)} 条历史门控记录"
          f"（run={args.runs}）\n")

    if args.mode == "analytic":
        rc, payload = mode_analytic(rows, args)
    elif args.mode == "power":
        rc, payload = mode_power(rows, args)
    else:
        rc, payload = mode_real(rows, args)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "runs": args.runs, **payload},
                      f, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
