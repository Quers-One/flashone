"""
compare.py
==========
两个 checkpoint 的**独立**直接对弈验证（不复用训练里的门控代码）。

为什么需要它
------------
训练里的门控曾经有一个致命 bug：`evaluate_models` 两个分支都写
`new_won = (winner == 1)`，而 `play_match(mcts_a, mcts_b)` 里 a 永远执黑，
所以"新模型执白"的那半数对局实际上记的是**旧模型**的胜负。
后果是门控拿噪声做决定（实测出现过 raw=0.000，含义其实是"20 局全是白胜"），
还顺手把学习率降了 4 次。

这个工具用**独立实现**的对弈循环 + 显式的分颜色记账，用来回答一个
训练日志回答不了的问题：**gen_N 到底比 gen_M 强还是弱？**

用法
----
    python compare.py --a models/run3/gen_1.pt --b models/run3/gen_9.pt
    python compare.py --a a.pt --b b.pt --games 40 --n-sim 400

输出会分颜色报告（A 执黑多少胜、执白多少胜），并用二项检验给出
"胜率显著偏离 0.5" 的判断，避免把 10 局 6 胜当成真的变强。
"""

import argparse
import io
import json
import os
import sys
import time

# 【不要用 io.TextIOWrapper(sys.stdout.buffer) 这种写法】
# 它会新建一个包装器；将来这个包装器被 GC 时 __del__ 会**关掉底层 buffer**，
# 于是真正还在用的 sys.stdout 报 "I/O operation on closed file"。
# 更致命的是：本模块会被 auto_verify.py import，import 时的副作用会污染对方进程
# （实测就是这样炸的）。`reconfigure` 直接改现有流的编码，不新建对象，安全。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from gomoku_env import Board
from mcts import MCTS
from network import GomokuNet
from train import Config, load_checkpoint_ex

# 决定"引擎是否可比"的文件（见 engine_fp_all 的说明）
ENGINE_FILES = ("mcts.py", "heuristics.py", "gomoku_env.py")


def load_net(path):
    """按权重反推架构加载（放大网络后必须，见 train.load_checkpoint_ex）。

    这是 compare / arena / ladder 三个工具**共用**的加载器，改这里就够。
    """
    net, ckpt = load_checkpoint_ex(path)
    net.eval()
    gen = ckpt.get("gen") if isinstance(ckpt, dict) else None
    print(f"  已加载 {path}  (gen={gen})")
    return net


def make_mcts(net, cfg, n_sim, hw, pw, seed=0, noise_alpha=None,
              noise_eps=0.25, batch=None, vl=None):
    """batch/vl/n_sim 都允许"每侧不同"，用于**同权重对照实验**：

    * `batch` —— MCTS 批量（`probe_batch_scan.py` 量到 16→64 自对弈 +59%，但同权重
      对照显示 batch 64 棋力崩到 0.15 胜率，见 NOTES §15.7）
    * `vl`    —— 虚拟损失。批量越大，同时被虚拟损失压住的叶子越多、选点越被摊平，
      所以"放大批量"必须和"调小 vl"一起试。
    * `n_sim` —— 每手模拟数。也可以把吞吐收益换成"更多模拟"。
    """
    return MCTS(net, n_sim=n_sim, c_puct=cfg.c_puct,
                virtual_loss=float(vl) if vl else cfg.virtual_loss,
                batch_size=int(batch) if batch else cfg.batch_size_mcts,
                max_children=cfg.max_children,
                root_max_children=cfg.root_max_children,
                heuristic_weight=hw, policy_weight=pw,
                root_noise_alpha=noise_alpha, root_noise_eps=noise_eps,
                seed=seed)


def play(mcts_black, mcts_white, max_ply=361):
    """黑方 mcts_black vs 白方 mcts_white，返回 (winner, plies, 落子序列)。"""
    board = Board()
    mcts_black.reset_tree()
    mcts_white.reset_tree()
    plies = 0
    seq = []
    while plies < max_ply:
        over, winner = board.is_game_over()
        if over:
            return winner, plies, tuple(seq)
        me = board.current_player
        m = mcts_black if me == 1 else mcts_white
        best = m.get_best_action(board)
        if best is None:
            return 0, plies, tuple(seq)
        x, y = int(best[0]), int(best[1])
        if not board.move(x, y, me):
            return 0, plies, tuple(seq)
        m.set_root_move((x, y))
        seq.append((x, y))
        plies += 1
    return 0, plies, tuple(seq)


def _worker_games(payload):
    """子进程：按局号跑一批对局，返回 [(g, winner, plies, seq)]。

    每局的内容只由局号决定（先后手在 g%2，种子在 g / g+5000），
    所以"按局号取模切分"不改变任何一局 —— 并行与串行必须**逐局相同**。
    这条性质与 `ladder.py` 的 `_duel_worker` 完全同源，验收方式也一样。
    """
    import compare
    from train import load_checkpoint_ex

    cfg = compare.Config()
    for k, v in payload["cfg_kw"].items():
        setattr(cfg, k, v)
    net_a, _ = load_checkpoint_ex(payload["a"])
    net_b, _ = load_checkpoint_ex(payload["b"])
    net_a.eval()
    net_b.eval()

    out = []
    for g in payload["ids"]:
        if g % 2 == 0:
            ma = compare.make_mcts(net_a, cfg, payload["nsa"], payload["hwa"],
                                   payload["pwa"], seed=g, noise_alpha=payload["na"],
                                   noise_eps=payload["neps"], batch=payload["ba"],
                                   vl=payload["vla"])
            mb = compare.make_mcts(net_b, cfg, payload["nsb"], payload["hwb"],
                                   payload["pwb"], seed=g + 5000, noise_alpha=payload["na"],
                                   noise_eps=payload["neps"], batch=payload["bb"],
                                   vl=payload["vlb"])
            winner, plies, seq = compare.play(ma, mb)          # A 执黑
        else:
            mb = compare.make_mcts(net_b, cfg, payload["nsb"], payload["hwb"],
                                   payload["pwb"], seed=g, noise_alpha=payload["na"],
                                   noise_eps=payload["neps"], batch=payload["bb"],
                                   vl=payload["vlb"])
            ma = compare.make_mcts(net_a, cfg, payload["nsa"], payload["hwa"],
                                   payload["pwa"], seed=g + 5000, noise_alpha=payload["na"],
                                   noise_eps=payload["neps"], batch=payload["ba"],
                                   vl=payload["vla"])
            # 【必须 play(mb, ma)】第一个参数永远是执黑方。原 worker 写成 play(ma, ma)
            # 会让 A 在奇数局也执黑，于是与串行不一致（8 局串行 3:5 vs 并行 5:3 抓到的就是这个）。
            winner, plies, seq = compare.play(mb, ma)          # A 执白
        out.append((g, winner, plies, seq))
        print(f"    [w{payload['wi']}/{payload['nw']}] 第{g + 1:3d}局 完成"
              f"（{len(out)}/{len(payload['ids'])}，{plies} 手）", flush=True)
    return out


def binom_p(k, n, p=0.5):
    """双侧二项检验的粗略 p 值（n<=80 时用精确求和）。"""
    from math import comb
    if n == 0:
        return 1.0
    pmf = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(v for v in pmf if v <= obs + 1e-12))


def engine_fp_all():
    """引擎**行为**的组合短哈希：`mcts.py` + `heuristics.py` + `gomoku_env.py`。

    为什么除了 `engine_fp()`（只看 mcts.py）还要有这个：搜索选点不只依赖 `mcts.py`，
    规则先验/合法手/成五判定在 `heuristics.py` 与 `gomoku_env.py` 里 ——
    只改这两个文件时，旧指纹**不会变**，但棋已经不一样了，数字会静默作废。
    这个组合哈希与 `ladder.engine_all_short()` 逐字节一致（同样的文件名顺序与拼接格式），
    所以各工具报出来的值可以直接对照。

    `train.py` / `compare.py` / `network.py` **故意不算进来**：它们是工具，
    改工具不该看起来像"换了引擎"。
    """
    import hashlib

    def sha1_8(path):
        hh = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hh.update(chunk)
        return hh.hexdigest()[:8]

    try:
        h = hashlib.sha1()
        for k in ENGINE_FILES:
            h.update(f"{k}:{sha1_8(k)}|".encode())
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def engine_fp():
    """
    当前引擎指纹（`mcts.py` 前 8 位）。

    【为什么要打印它】2026-10-10 22:16:54 改了一次 `mcts.py`（补对称终局判定），
    同一对模型、同样 48 局的得分率从 0.875 变成 0.438。所以任何强度数字都必须
    知道自己是在哪把尺子上量的 —— 以后所有对局工具的产物都自带这行指纹。
    详见 `checkpoints/ENGINE_CHANGE_NOTICE.md`。
    """
    import hashlib
    try:
        h = hashlib.sha1()
        with open("mcts.py", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser(description="两个 checkpoint 直接对弈")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--heuristic-weight", type=float, default=6.0)
    ap.add_argument("--policy-weight", type=float, default=0.3)
    ap.add_argument("--noise-alpha", type=float, default=0.3,
                    help="根节点 Dirichlet 噪声 alpha；<=0 关闭")
    ap.add_argument("--noise-eps", type=float, default=0.25,
                    help="噪声混合比例")
    ap.add_argument("--seed", type=int, default=0)
    # 【同权重对照用】两侧的 MCTS 批量可以不同：默认都用 Config 的值；
    # --mcts-batch 同时设两侧，--mcts-batch-a / -b 分别设。
    ap.add_argument("--mcts-batch", type=int, default=0,
                    help="同时设 A/B 两侧的 MCTS 批量（0 = 用 Config 默认）")
    ap.add_argument("--mcts-batch-a", type=int, default=0)
    ap.add_argument("--mcts-batch-b", type=int, default=0)
    # 每侧 n_sim / 虚拟损失（0 = 用 --n-sim / Config）
    ap.add_argument("--n-sim-a", type=int, default=0)
    ap.add_argument("--n-sim-b", type=int, default=0)
    ap.add_argument("--vl", type=float, default=0.0,
                    help="同时设 A/B 的虚拟损失（0 = Config）")
    ap.add_argument("--vl-a", type=float, default=0.0)
    ap.add_argument("--vl-b", type=float, default=0.0)
    # 【每侧不同先验权重】用于"纯网络先验 vs 融合先验"这类**配置**对照：
    # 0.0 是合法取值，所以默认必须是 None（不能用 `or` 兜底）。
    ap.add_argument("--heuristic-weight-a", type=float, default=None)
    ap.add_argument("--policy-weight-a", type=float, default=None)
    ap.add_argument("--heuristic-weight-b", type=float, default=None)
    ap.add_argument("--policy-weight-b", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="并行对局进程数（按局号切分，结果与串行逐局相同）")
    ap.add_argument("--json", default=None,
                    help="把结果写成 JSON（供脚本/自动化判读；中文字段会乱码，机器读用这个）")
    args = ap.parse_args()

    cfg = Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net_a = load_net(args.a)
    net_b = load_net(args.b)

    hw, pw = args.heuristic_weight, args.policy_weight
    # 每侧覆盖（None = 用公共值）；hw 可以合法地为 0.0，所以必须显式判 None
    hwa = hw if args.heuristic_weight_a is None else args.heuristic_weight_a
    pwa = pw if args.policy_weight_a is None else args.policy_weight_a
    hwb = hw if args.heuristic_weight_b is None else args.heuristic_weight_b
    pwb = pw if args.policy_weight_b is None else args.policy_weight_b
    na = None if args.noise_alpha <= 0 else args.noise_alpha
    ba = args.mcts_batch_a or args.mcts_batch or 0
    bb = args.mcts_batch_b or args.mcts_batch or 0
    nsa = args.n_sim_a or args.n_sim
    nsb = args.n_sim_b or args.n_sim
    vla = args.vl_a or args.vl or 0.0
    vlb = args.vl_b or args.vl or 0.0
    print(f"\n启发式权重={hw}/{pw}  局数={args.games}"
          + (f"　【每侧覆盖】A={hwa}/{pwa}  B={hwb}/{pwb}"
             if (hwa, pwa, hwb, pwb) != (hw, pw, hw, pw) else ""))
    print(f"A: n_sim={nsa} 批量={ba or cfg.batch_size_mcts} "
          f"虚拟损失={vla or cfg.virtual_loss}")
    print(f"B: n_sim={nsb} 批量={bb or cfg.batch_size_mcts} "
          f"虚拟损失={vlb or cfg.virtual_loss}")
    print(f"（Config 默认：批量 {cfg.batch_size_mcts}、虚拟损失 {cfg.virtual_loss}）")
    if na is None:
        print("[警告] 未开根噪声：greedy 落子是完全确定性的，"
              "同一先后手会走出**完全相同**的对局 —— 20 局其实只有 2 局独立样本！")

    # ---- 对局：并行（按局号切分）或串行，两者结果必须逐局相同 ----
    kw = {"c_puct": cfg.c_puct, "virtual_loss": cfg.virtual_loss,
          "batch_size_mcts": cfg.batch_size_mcts,
          "max_children": cfg.max_children,
          "root_max_children": cfg.root_max_children}
    t0 = time.time()
    if args.workers > 1 and args.games >= args.workers:
        import multiprocessing as mp
        jobs = []
        for wi in range(args.workers):
            ids = [g for g in range(args.games) if g % args.workers == wi]
            if not ids:
                continue
            jobs.append({"a": args.a, "b": args.b, "ids": ids,
                         "hwa": hwa, "pwa": pwa, "hwb": hwb, "pwb": pwb,
                         "nsa": nsa, "nsb": nsb, "ba": ba, "bb": bb,
                         "vla": vla, "vlb": vlb, "na": na,
                         "neps": args.noise_eps, "cfg_kw": kw,
                         "wi": wi + 1, "nw": args.workers})
        print(f"并行对局：{len(jobs)} 进程 × "
              f"{'/'.join(str(len(j['ids'])) for j in jobs)} 局", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(jobs)) as pool:
            res = pool.map(_worker_games, jobs)
        per_game = sorted([r for sub in res for r in sub], key=lambda t: t[0])
    else:
        per_game = []
        for g in range(args.games):
            # 每两局交换一次先后手
            if g % 2 == 0:
                ma = make_mcts(net_a, cfg, nsa, hwa, pwa, seed=g,
                               noise_alpha=na, noise_eps=args.noise_eps, batch=ba, vl=vla)
                mb = make_mcts(net_b, cfg, nsb, hwb, pwb, seed=g + 5000,
                               noise_alpha=na, noise_eps=args.noise_eps, batch=bb, vl=vlb)
                winner, plies, seq = play(ma, mb)          # A 执黑
            else:
                mb = make_mcts(net_b, cfg, nsb, hwb, pwb, seed=g,
                               noise_alpha=na, noise_eps=args.noise_eps, batch=bb, vl=vlb)
                ma = make_mcts(net_a, cfg, nsa, hwa, pwa, seed=g + 5000,
                               noise_alpha=na, noise_eps=args.noise_eps, batch=ba, vl=vla)
                winner, plies, seq = play(mb, ma)          # A 执白
            per_game.append((g, winner, plies, seq))

    a_black_wins = a_white_wins = b_wins = draws = 0
    lens = []
    seqs = set()
    for g, winner, plies, seq in per_game:
        a_is_black = (g % 2 == 0)
        seqs.add(seq)
        lens.append(plies)
        if winner == 0:
            draws += 1
            who = "和棋"
        elif (winner == 1) == a_is_black:
            if a_is_black:
                a_black_wins += 1
            else:
                a_white_wins += 1
            who = "A胜"
        else:
            b_wins += 1
            who = "B胜"
        print(f"  第{g + 1:2d}局: A执{'黑' if a_is_black else '白'}  "
              f"{who:>4}  {plies:3d} 手", flush=True)

    a_wins = a_black_wins + a_white_wins
    decided = args.games - draws
    print("\n" + "=" * 60)
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print("-" * 60)
    print(f"  A 胜 {a_wins}  (执黑 {a_black_wins} / 执白 {a_white_wins})")
    print(f"  B 胜 {b_wins}")
    print(f"  和棋 {draws}")
    print(f"  独立对局序列数: {len(seqs)}/{args.games}"
          f"{'   <<< 重复太多，统计无意义！' if len(seqs) < args.games // 2 else ''}")
    print(f"  A 的分色胜率: 执黑 {a_black_wins}/{args.games - args.games // 2}  "
          f"执白 {a_white_wins}/{args.games // 2}")
    rate = (a_wins / decided) if decided > 0 else None
    p = binom_p(a_wins, decided) if decided > 0 else None
    if decided > 0:
        better = "B" if rate < 0.5 else "A"
        if p < 0.05:
            verdict = (f"{better} 显著更强"
                       + ("（A 强于 B）" if rate > 0.5 else "（B 强于 A）"))
        elif p < 0.15:
            verdict = (f"倾向 {better} 更强，但未达显著（p={p:.3f}，"
                       f"{decided} 局样本不足）—— 方向性证据，不能当结论")
        else:
            verdict = "两者无法区分"
        print(f"  有效对局(非和棋) {decided} 局: A 胜率 {rate:.3f}  "
              f"双侧二项 p={p:.3f}")
        print(f"  >>> {verdict}")
        # 分颜色记账：确认优势不是"某一方执黑必胜"造成的假象
        print(f"  分颜色: A执黑 {a_black_wins}胜 / A执白 {a_white_wins}胜 "
              f"（各 {args.games - args.games // 2} 与 {args.games // 2} 局）")
    print(f"  平均手数 {np.mean(lens):.0f}  耗时 {time.time() - t0:.0f}s")
    print(f"  引擎指纹: mcts.py {engine_fp()}　引擎全 {engine_fp_all()}"
          f"　（换引擎=整把尺子重测，见 checkpoints/ENGINE_CHANGE_NOTICE.md）")
    print(f"  MCTS 批量: A={ba or cfg.batch_size_mcts}  B={bb or cfg.batch_size_mcts}")
    print(f"  搜索配置: A n_sim={nsa}/vl={vla or cfg.virtual_loss}"
          f"　B n_sim={nsb}/vl={vlb or cfg.virtual_loss}")
    if args.json:
        out = {
            "a": args.a, "b": args.b, "games": args.games,
            "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "a_black_wins": a_black_wins, "a_white_wins": a_white_wins,
            "decided": decided, "a_rate": rate, "p_value": p,
            "n_sim_a": nsa, "n_sim_b": nsb,
            "mcts_batch_a": ba or cfg.batch_size_mcts,
            "mcts_batch_b": bb or cfg.batch_size_mcts,
            "vl_a": vla or cfg.virtual_loss, "vl_b": vlb or cfg.virtual_loss,
            "heuristic_weight": hw, "policy_weight": pw,
            # 上面两个是"默认权重"，只在一侧没单独指定时才有意义。
            # 用 --heuristic-weight-a/-b 时真正的生效值在下面这 4 个字段里 ——
            # 只记默认值会让人把"纯网络 vs 融合"误读成"两边都是 6.0/0.3"。
            "heuristic_weight_a": hwa, "policy_weight_a": pwa,
            "heuristic_weight_b": hwb, "policy_weight_b": pwb,
            "unique_sequences": len(seqs),
            "mean_plies": float(np.mean(lens)),
            "seconds": round(time.time() - t0, 1),
            "engine_mcts": engine_fp(),
            "engine_full": engine_fp_all(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  JSON: {args.json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
