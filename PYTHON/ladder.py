"""
ladder.py
=========
**单锚点强度标尺**：以冻结的 `gen_60` 为参照，把任意模型换算成"相对它的 Elo 差值"。

为什么用"单锚点"而不是"全阶梯"
------------------------------
* 全循环赛不可行：63 个 checkpoint = 1953 对 × 12 局 ≈ 2.3 万局 ≈ 27 小时。
* 单锚点只要 **一个固定参照物 + 若干局**（默认 48 局，约 3~5 分钟），
  就能给出"新模型比 gen_60 强/弱多少分 ± 误差"——跨实验、跨时间都可直接比。
* 锚点本身不参与拟合，所以**尺度不会漂**：Elo(gen_60) 恒为 0。

数学
----
设候选对锚点的得分率（和棋算半分）为 s：
        ΔElo = 400 · log10( s / (1 - s) )
s 由有限局数估出，用 **Wilson 区间** 给出 95% 置信区间，再映射成 Elo 区间
（s=0 或 1 时用 1/(2n) 与 1-1/(2n) 兜底，避免 ±∞）。

用法
----
    # ① 冻结锚点（把 gen_60 复制进 checkpoints/anchors/ 并写下测量条件）
    python ladder.py --init-anchor models/run3/gen_60.pt

    # ② 测一个候选模型（默认 48 局，先后手交替）
    python ladder.py --candidate checkpoints/phase2/gen_20.pt --games 48
    python ladder.py --candidate models/run3/gen_30.pt --games 48 --quiet

    # ③ 自检：拿锚点自己跟自己打，应得 s≈0.5、ΔElo≈0
    python ladder.py --candidate checkpoints/anchors/gen_60.pt --games 16

【重要】锚点强度依赖"测量条件"：引擎版本、先验权重、n_sim、根噪声。
这些都会随锚点一起写进 `checkpoints/anchors/<name>.json`；换了引擎或权重，
整把尺子都要重测（否则数字不可比）。
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                     # noqa: E402
import torch                           # noqa: E402

import compare                         # noqa: E402
from mcts import MCTS                  # noqa: E402

ANCHOR_DIR = os.path.join("checkpoints", "anchors")
# 「当前标尺」指针：里面写锚点名（可带或不带 .pt）。切换标尺只改这一个文件，
# 所有 `ladder.py` 调用（不显式给 --anchor 时）自动跟着走。
ACTIVE_PTR = os.path.join(ANCHOR_DIR, "ACTIVE.txt")

# 【引擎指纹口径 · 2026-10-11 扩容】原来只有 mcts.py 决定指纹（`engine_fp()` 的 sha1），
# 而 `engine_fingerprint()` 记的 4 个文件里**漏了 heuristics.py 与 gomoku_env.py**
# —— 搜索选点直接依赖这两个（规则先验、合法手、成五判定），
# 所以"改了 heuristics.py 而指纹不变、数字静默作废"是可能的。现在分开两档：
#   * ENGINE_FILES = 会改变搜索行为的文件 → 决定"引擎是否可比"（硬警告）
#   * TOOL_FILES   = 只作留痕（改工具不该像换了引擎）
ENGINE_FILES = ("mcts.py", "heuristics.py", "gomoku_env.py")
TOOL_FILES = ("train.py", "compare.py", "network.py")
FP_FILES = ENGINE_FILES + TOOL_FILES


# ------------------------------------------------------------------ 统计

def wilson(k, n, z=1.96):
    """二项比例的 Wilson 95% 区间（比正态近似在小样本下稳）。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - half) / d), min(1.0, (c + half) / d)


def elo_from_score(s):
    """得分率 -> Elo 差（400 刻度）。0/1 无法换算，返回 None。"""
    if s <= 0.0 or s >= 1.0:
        return None
    return 400.0 * math.log10(s / (1.0 - s))


def elo_ci(k, n):
    """由得分率与其 Wilson 区间给出 Elo 差与区间。"""
    lo, hi = wilson(k, n)
    eps = 1.0 / (2 * n) if n else 0.5
    lo_c = min(max(lo, eps), 1 - eps)
    hi_c = min(max(hi, eps), 1 - eps)
    return (elo_from_score(k / n) if 0 < k < n else None,
            elo_from_score(lo_c), elo_from_score(hi_c))


def fmt_elo(v):
    return "   n/a" if v is None else f"{v:+7.1f}"


# ------------------------------------------------------------------ 锚点

def file_sha1(path, n=8):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def engine_fingerprint():
    fp = {}
    for f in FP_FILES:
        p = os.path.join(HERE, f)
        fp[f] = file_sha1(p) if os.path.exists(p) else "?"
    return fp


def engine_all_short(fp=None):
    """**引擎行为**的组合短哈希（只覆盖 ENGINE_FILES）。

    为什么要有它：`engine_fingerprint()` 是逐文件的字典，人读没问题，
    但"引擎整体变没变"需要一眼可判。更重要的是**它只覆盖会改变搜索行为的文件**
    —— `train.py`/`compare.py` 是工具，改它们不该动这个哈希（否则每次改工具
    都像换了引擎）。
    """
    fp = fp or engine_fingerprint()
    h = hashlib.sha1()
    for k in ENGINE_FILES:
        h.update(f"{k}:{fp.get(k, '?')}|".encode())
    return h.hexdigest()[:8]


def init_anchor(src, name=None, hw=6.0, pw=0.3, n_sim=400, noise=0.3,
                note=""):
    """把一个 checkpoint 冻结成锚点，并写下"测量条件"。"""
    import shutil
    os.makedirs(ANCHOR_DIR, exist_ok=True)
    name = name or os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(ANCHOR_DIR, f"{name}.pt")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    meta = {
        "name": name,
        "path": dst.replace("\\", "/"),
        "sha1_8": file_sha1(dst),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine_fingerprint": engine_fingerprint(),
        "measure_conditions": {
            "heuristic_weight": hw, "policy_weight": pw,
            "n_sim": n_sim, "root_noise_alpha": noise,
            "note": "双方都用这套；换引擎/换权重必须整把尺子重测",
        },
        "elo_reference": 0.0,
        "note": note,
    }
    mp = os.path.join(ANCHOR_DIR, f"{name}.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"锚点已冻结: {dst}")
    print(f"  sha1(8)={meta['sha1_8']}  引擎指纹 mcts.py={meta['engine_fingerprint']['mcts.py']}")
    print(f"  测量条件: hw/pw={hw}/{pw}  n_sim={n_sim}  根噪声α={noise}")
    print(f"  元数据: {mp}")
    return meta


def load_anchor_meta(path):
    mp = os.path.splitext(path)[0] + ".json"
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    return None


def active_anchor():
    """
    当前标尺（不显式给 `--anchor` 时的默认锚点）。

    读 `checkpoints/anchors/ACTIVE.txt`；没有或指向不存在的文件就退回 gen_60.pt。
    重定标尺的做法见 `new_ruler.ps1` / NOTES §12.4。
    """
    try:
        with open(ACTIVE_PTR, encoding="utf-8") as f:
            name = f.read().strip()
        if name:
            p = os.path.join(ANCHOR_DIR,
                             name if name.endswith(".pt") else name + ".pt")
            if os.path.exists(p):
                return p
    except OSError:
        pass
    return os.path.join(ANCHOR_DIR, "gen_60.pt")


def set_active(name):
    """把某个锚点设为当前标尺。"""
    os.makedirs(ANCHOR_DIR, exist_ok=True)
    with open(ACTIVE_PTR, "w", encoding="utf-8") as f:
        f.write(name if name.endswith(".pt") else name + ".pt")
    print(f"当前标尺已切换 -> {ACTIVE_PTR} = "
          f"{name if name.endswith('.pt') else name + '.pt'}")


# ------------------------------------------------------------------ 对局

def _duel_worker(payload):
    """子进程里跑一批局（按局号切分）。

    【为什么必须逐局一致】每局的种子与先后手完全由局号决定（见 `duel`）：
        g%2==0 -> 候选执黑 seed=g，锚点 seed=g+5000
        g%2==1 -> 锚点执黑 seed=g，候选 seed=g+5000
    所以"按局号取模切分"不会改变任何一局的内容 —— 并行与串行的结果必须**逐局相同**。
    这条性质由 `ladder_parallel_check`（见 NOTES §15.8）验收。
    """
    (cand_path, anchor_path, game_ids, kw, max_ply, hw, pw, n_sim, wi_lock) = payload
    wi, total_w = wi_lock
    import compare
    from train import load_checkpoint_ex

    cfg = compare.Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    net_c, _ = load_checkpoint_ex(cand_path)
    net_a, _ = load_checkpoint_ex(anchor_path)
    net_c.eval()
    net_a.eval()

    out = []
    for g in game_ids:
        cand_black = (g % 2 == 0)
        # 【同 serial】谁执黑由 g%2 决定，且执黑方先传给 play（原实现漏了这一点）
        if cand_black:
            ma = compare.make_mcts(net_c, cfg, n_sim, hw, pw,
                                   seed=g, noise_alpha=0.3)
            mb = compare.make_mcts(net_a, cfg, n_sim, hw, pw,
                                   seed=g + 5000, noise_alpha=0.3)
            winner, plies, seq = compare.play(ma, mb, max_ply=max_ply)
        else:
            mb = compare.make_mcts(net_a, cfg, n_sim, hw, pw,
                                   seed=g, noise_alpha=0.3)
            ma = compare.make_mcts(net_c, cfg, n_sim, hw, pw,
                                   seed=g + 5000, noise_alpha=0.3)
            winner, plies, seq = compare.play(mb, ma, max_ply=max_ply)
        out.append((g, winner, plies, seq))
        # 【实时可见】逐局打一行（带 worker 编号）。
        # 不这么做的话，父进程要等 pool.map 全部返回才开始打印 → 日志在整个测量
        # 期间是空的，看不出进度（第一版就是这样，被自己的实测发现）。
        print(f"    [w{wi}/{total_w}] 第{g + 1:3d}局 完成（{len(out)}/{len(game_ids)}，"
              f"{plies} 手）", flush=True)
    return out


def duel(net_cand, net_anchor, cfg, games, n_sim, hw, pw, seed=0,
         quiet=False, max_ply=361, n_workers=1,
         cand_path=None, anchor_path=None):
    """
    候选 vs 锚点，先后手交替，双方各自独立的搜索树（**每局重建**）。

    每局重建是硬要求：MCTS 跨手复用树并维护增量棋盘历史，跨局复用会让
    新棋盘和上一局历史打架、搜索塌陷（实测把 42:6 变成 5:3 的假结论）。

    `n_workers > 1` 时按局号切分到多个进程（**测量不再被训练抢机器拖成 1/2.4 速度**）。
    因为每局内容只由局号决定，并行结果与串行**逐局相同** —— 这一点有验收测试。
    """
    kw = {"c_puct": cfg.c_puct, "virtual_loss": cfg.virtual_loss,
          "batch_size_mcts": cfg.batch_size_mcts,
          "max_children": cfg.max_children,
          "root_max_children": cfg.root_max_children}
    t0 = time.time()
    parallel = (n_workers and n_workers > 1 and cand_path and anchor_path
                and games >= n_workers)
    if parallel:
        import multiprocessing as mp
        jobs = []
        for wi in range(n_workers):
            ids = [g for g in range(games) if g % n_workers == wi]
            if ids:
                jobs.append((cand_path, anchor_path, ids, kw, max_ply, hw, pw, n_sim,
                             (wi + 1, min(n_workers, games))))
        print(f"  并行测量：{len(jobs)} 进程 × "
              f"{'/'.join(str(len(j[2])) for j in jobs)} 局", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(jobs)) as pool:
            res = pool.map(_duel_worker, jobs)
        per_game = sorted([r for sub in res for r in sub], key=lambda t: t[0])
    else:
        per_game = []
        for g in range(games):
            cand_black = (g % 2 == 0)
            # 【2026-10-11 修 P0 bug】原实现在两个分支都写 `compare.play(ma, mb)`，
            # 而 `ma` 在两个分支里都是 net_cand —— 于是**候选方永远执黑**，
            # 而 cand_black 却按 g%2 交替 ⇒ 半数对局的胜负**记反**。
            # 后果（数学上必然）：报出的得分 =(p_even + 1 − p_odd)/2，
            # **任何"对称的强弱差"都会被压成 ≈0.5** —— 也就是说这把尺子
            # 对"两边同时变强/变弱"完全不敏感（今天所有 indistinguishable 都受此影响）。
            # 正确写法与 compare.py 一致：谁执黑由 g%2 决定，且**执黑方先传给 play**。
            if cand_black:
                ma = compare.make_mcts(net_cand, cfg, n_sim, hw, pw,
                                       seed=g, noise_alpha=0.3)
                mb = compare.make_mcts(net_anchor, cfg, n_sim, hw, pw,
                                       seed=g + 5000, noise_alpha=0.3)
                winner, plies, seq = compare.play(ma, mb)      # 候选执黑
            else:
                mb = compare.make_mcts(net_anchor, cfg, n_sim, hw, pw,
                                       seed=g, noise_alpha=0.3)
                ma = compare.make_mcts(net_cand, cfg, n_sim, hw, pw,
                                       seed=g + 5000, noise_alpha=0.3)
                winner, plies, seq = compare.play(mb, ma)      # 锚点执黑（候选执白）
            per_game.append((g, winner, plies, seq))

    c_wins = a_wins = draws = 0
    c_black = c_white = 0
    d_black = d_white = 0
    seqs, lens = set(), []
    for g, winner, plies, seq in per_game:
        cand_black = (g % 2 == 0)
        seqs.add(seq)
        lens.append(plies)
        if winner == 0:
            draws += 1
            if cand_black:
                d_black += 1
            else:
                d_white += 1
            who = "和"
        elif (winner == 1) == cand_black:
            c_wins += 1
            if cand_black:
                c_black += 1
            else:
                c_white += 1
            who = "候选胜"
        else:
            a_wins += 1
            who = "锚点胜"
        if not quiet:
            print(f"  第{g + 1:2d}局: 候选执{'黑' if cand_black else '白'} "
                  f"{who:>3}  {plies:3d} 手", flush=True)

    score = (c_wins + 0.5 * draws) / games if games else 0.0
    # 把"和棋半分"折算成 2n 个半分局（胜=2、和=1、负=0），这样可以直接套
    # Wilson/二项工具，且与得分率 s 完全自洽（k/2n == s）。
    # 注意：同一局内的两个半分局并不独立，所以区间是**近似**，仅用于给量级。
    k_units = int(round(2 * c_wins + draws))
    elo, lo, hi = elo_ci(k_units, 2 * games)
    return {
        "games": games, "c_wins": c_wins, "a_wins": a_wins, "draws": draws,
        "c_black": c_black, "c_white": c_white,
        "d_black": d_black, "d_white": d_white,
        "score": score, "elo": elo, "elo_lo": lo, "elo_hi": hi,
        "p_binom": compare.binom_p(k_units, 2 * games) if games else 1.0,
        "seqs": len(seqs), "mean_ply": float(np.mean(lens)),
        "secs": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description="单锚点强度标尺")
    ap.add_argument("--init-anchor", default=None,
                    help="把该 checkpoint 冻结成锚点（复制进 checkpoints/anchors/）")
    ap.add_argument("--anchor", default=None,
                    help="锚点 checkpoint（默认读 checkpoints/anchors/ACTIVE.txt；"
                         "没有则用 gen_60.pt）")
    ap.add_argument("--name", default=None,
                    help="冻结锚点时用的名字（默认取文件名；best.pt 这类名字请显式给）")
    ap.add_argument("--set-active", action="store_true",
                    help="冻结后把该锚点设为当前标尺（写 ACTIVE.txt）")
    ap.add_argument("--note", default="",
                    help="冻结锚点时写进 json 的备注（例如口径选择理由）")
    ap.add_argument("--json", default=None,
                    help="把结果写成机器可读 JSON（供自动化流程用，避免解析中文）")
    ap.add_argument("--candidate", default=None, help="要测的模型")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--heuristic-weight", type=float, default=6.0)
    ap.add_argument("--policy-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="测量并行进程数（按局号切分，结果与串行逐局相同）。"
                         "训练同时跑时单进程尺子会掉到 ~2 局/分钟，"
                         "给 8 可让 200 局重新回到 ~40 分钟量级。")
    args = ap.parse_args()
    args.anchor = args.anchor or active_anchor()

    if args.init_anchor:
        meta = init_anchor(args.init_anchor, name=args.name,
                           hw=args.heuristic_weight,
                           pw=args.policy_weight, n_sim=args.n_sim,
                           note=args.note)
        if args.set_active:
            set_active(meta["name"])
        if not args.candidate:
            return 0

    if not args.candidate:
        print("需要 --candidate（或用 --init-anchor 只冻结锚点）")
        return 1
    if not os.path.exists(args.anchor):
        print(f"锚点不存在: {args.anchor}　先跑 --init-anchor")
        return 1

    meta = load_anchor_meta(args.anchor)
    if meta:
        cond = meta.get("measure_conditions", {})
        if (abs(cond.get("heuristic_weight", 6.0) - args.heuristic_weight) > 1e-9
                or abs(cond.get("policy_weight", 0.3) - args.policy_weight) > 1e-9):
            print(f"[警告] 当前权重 {args.heuristic_weight}/{args.policy_weight} "
                  f"与锚点记录的 {cond.get('heuristic_weight')}/"
                  f"{cond.get('policy_weight')} 不一致 —— Elo 不可比！")
        cur = engine_fingerprint()
        meta_fp = meta.get("engine_fingerprint", {}) or {}
        # 逐文件比：老锚点（扩容前冻结的）没有 heuristics.py / gomoku_env.py 这两项，
        # 缺项一律跳过（不能因为"记录里没有"就当成"变了"）；只对 ENGINE_FILES 报硬警告。
        for fname in ENGINE_FILES:
            old_v = meta_fp.get(fname)
            if old_v is None:
                continue
            if old_v != cur.get(fname):
                print(f"[警告] 引擎与冻结锚点时不符（{fname} "
                      f"{old_v} -> {cur.get(fname)}）—— 整把尺子需要重测！")
        if meta_fp and not all(meta_fp.get(f) for f in ENGINE_FILES):
            print("[提示] 该锚点是旧口径冻结的（未记录 heuristics.py / gomoku_env.py）"
                  "；本次测量仍会把它们写进新锚点/结果里。")

    cfg = compare.Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("=" * 70)
    print(f"锚点 = {args.anchor}（Elo 定义为 0）")
    print(f"候选 = {args.candidate}　局数 = {args.games}　n_sim = {args.n_sim}"
          f"　权重 = {args.heuristic_weight}/{args.policy_weight}")
    print(f"引擎指纹: mcts.py {engine_fingerprint()['mcts.py']}　引擎全 {engine_all_short()}"
          f"　（换引擎=整把尺子重测，见 checkpoints/ENGINE_CHANGE_NOTICE.md）")
    print("=" * 70)

    net_c = compare.load_net(args.candidate)
    net_a = compare.load_net(args.anchor)
    r = duel(net_c, net_a, cfg, args.games, args.n_sim,
             args.heuristic_weight, args.policy_weight,
             seed=args.seed, quiet=args.quiet, n_workers=args.workers,
             cand_path=args.candidate, anchor_path=args.anchor)

    print("-" * 70)
    print(f"  候选胜 {r['c_wins']}（执黑 {r['c_black']} / 执白 {r['c_white']}）   "
          f"锚点胜 {r['a_wins']}   和 {r['draws']}")
    print(f"  独立序列 {r['seqs']}/{r['games']}　平均 {r['mean_ply']:.0f} 手　"
          f"耗时 {r['secs']:.0f}s")
    print(f"  得分率 s = {r['score']:.3f}")
    # 颜色均衡：候选执黑/执白的局数与各自得分率。两者局数必须各半，
    # 否则总分被颜色优势污染；两者得分率之差就是这次测量的颜色偏差。
    n_black = (r["games"] + 1) // 2
    n_white = r["games"] // 2
    s_black = (r["c_black"] + 0.5 * r["d_black"]) / n_black if n_black else 0.0
    s_white = (r["c_white"] + 0.5 * r["d_white"]) / n_white if n_white else 0.0
    color_bias = elo_from_score(s_black) - elo_from_score(s_white)
    print(f"  颜色均衡：执黑 {n_black} 局得分 {s_black:.3f}（胜 {r['c_black']}）"
          f"　执白 {n_white} 局得分 {s_white:.3f}（胜 {r['c_white']}）"
          f"　颜色偏差 = {fmt_elo(color_bias)}")
    print(f"  ΔElo（相对锚点）= {fmt_elo(r['elo'])}   "
          f"95% 区间 [{fmt_elo(r['elo_lo'])}, {fmt_elo(r['elo_hi'])}]")
    if r["elo"] is not None and r["elo_lo"] is not None:
        if r["elo_lo"] > 0:
            print("  >>> 显著强于锚点")
        elif r["elo_hi"] < 0:
            print("  >>> 显著弱于锚点")
        else:
            print("  >>> 区间跨 0：与锚点无法区分（要更窄就加 --games）")
    print("-" * 70)

    # ---- 机器可读结果（自动化流程靠它，避免解析中文日志）----
    if args.json:
        verdict = "unknown"
        if r["elo"] is not None and r["elo_lo"] is not None:
            if r["elo_lo"] > 0:
                verdict = "stronger"
            elif r["elo_hi"] < 0:
                verdict = "weaker"
            else:
                verdict = "indistinguishable"
        out = {
            "candidate": args.candidate,
            "anchor": args.anchor,
            "anchor_name": (meta or {}).get("name"),
            "games": r["games"], "wins": r["c_wins"], "losses": r["a_wins"],
            "draws": r["draws"], "score": r["score"],
            "cand_black_wins": r["c_black"], "cand_white_wins": r["c_white"],
            "black_games": n_black, "white_games": n_white,
            "draws_black": r["d_black"], "draws_white": r["d_white"],
            "score_as_black": s_black, "score_as_white": s_white,
            "color_bias_elo": color_bias,
            "elo": r["elo"], "elo_lo": r["elo_lo"], "elo_hi": r["elo_hi"],
            "verdict": verdict,
            "n_sim": args.n_sim,
            "heuristic_weight": args.heuristic_weight,
            "policy_weight": args.policy_weight,
            "independent_sequences": r["seqs"],
            "mean_ply": r["mean_ply"], "secs": r["secs"],
            "engine_fingerprint_mcts": engine_fingerprint()["mcts.py"],
            "engine_full": engine_all_short(),
            "engine_fingerprint": engine_fingerprint(),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  机器可读结果: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
