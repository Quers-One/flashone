"""
arena.py
========
**固定对手阶梯**：给模型一个"绝对强度"坐标（这是本项目此前唯一缺的东西）。

为什么需要它
------------
`elo.py` / `compare.py` 只能回答"这代比那代强多少"——都是自己跟自己比。
一旦要回答"它到底有多强"，就需要**不随训练变化的参照物**：
随机、纯规则贪心、α-β minimax。有了固定阶梯，跨代数、跨实验的数字才可比。

阶梯（由弱到强）
----------------
    random      均匀随机合法点
    greedy      纯规则贪心：一步成五 -> 必挡 -> 最高棋型分（无搜索）
    minimax_d2  α-β 搜索，深度 2，每层取前 12 个候选（并列随机打破）
    minimax_d3  同上，深度 3（约 0.3 秒/手，慢但明显更强）

用法
----
    python arena.py --a models/run3/best.pt --b greedy --games 40
    python arena.py --a models/run3/best.pt --ladder --games 24     # 对全阶梯各打一轮
    python arena.py --a models/run3/gen_30.pt --b models/run3/gen_60.pt --games 40

输出：分颜色记账 + 双侧二项检验（判据与 compare.py 完全一致，便于互相对照）。
"""

import argparse
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
import heuristics                      # noqa: E402
from gomoku_env import Board           # noqa: E402

LADDER = ["random", "greedy", "minimax_d2", "minimax_d3"]


# ------------------------------------------------------------------ 对手

class RandomOpp:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return None
        return legal[int(self.rng.integers(len(legal)))]


class GreedyOpp:
    name = "greedy"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return None
        raw, wins, blocks = heuristics.move_scores(board, legal)
        for forced in (wins, blocks):           # 先成五，再必挡
            if forced:
                idx = sorted(forced)
                return legal[idx[int(self.rng.integers(len(idx)))]]
        best = float(np.max(raw))
        idx = [i for i in range(len(raw)) if raw[i] >= best - 1e-9]   # 并列随机
        return legal[idx[int(self.rng.integers(len(idx)))]]


class MinimaxOpp:
    """α-β 搜索 + 手工评估（评估 = 双方最强威胁分之差）。"""

    def __init__(self, depth=2, width=12, seed=0):
        self.depth = int(depth)
        self.width = int(width)
        self.name = f"minimax_d{self.depth}"
        self.rng = np.random.default_rng(seed)
        self.root_player = 1

    # ---- 静态评估：站在 root_player 视角 ----
    def _static(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return 0.0
        me = self.root_player
        opp = 3 - me
        raw_me, wins_me, _ = heuristics.move_scores(board, legal, me)
        if wins_me:
            return 1e6
        raw_op, wins_op, _ = heuristics.move_scores(board, legal, opp)
        if wins_op:
            return -1e6
        return float(np.max(raw_me) - np.max(raw_op))

    def _order(self, board):
        legal = board.get_legal_moves()
        if not legal:
            return []
        raw, wins, blocks = heuristics.move_scores(board, legal)
        pri = {int(i): 2 for i in (list(wins) + list(blocks))}
        order = sorted(range(len(legal)),
                       key=lambda i: (-pri.get(i, 0), -raw[i]))
        return [legal[i] for i in order[:self.width]]

    def _search(self, board, depth, alpha, beta):
        over, winner = board.is_game_over()
        if over:
            if winner == 0:
                return 0.0
            return 1e6 if winner == self.root_player else -1e6
        if depth == 0:
            return self._static(board)
        moves = self._order(board)
        if not moves:
            return 0.0
        maximizing = board.current_player == self.root_player
        best = -float("inf") if maximizing else float("inf")
        for mv in moves:
            nb = board.copy()
            nb.move(mv[0], mv[1], board.current_player)
            val = self._search(nb, depth - 1, alpha, beta)
            if maximizing:
                best = max(best, val)
                alpha = max(alpha, best)
            else:
                best = min(best, val)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def choose(self, board):
        moves = self._order(board)
        if not moves:
            return None
        self.root_player = board.current_player
        best_val, best_moves = -float("inf"), []
        alpha, beta = -float("inf"), float("inf")
        for mv in moves:
            nb = board.copy()
            nb.move(mv[0], mv[1], board.current_player)
            val = self._search(nb, self.depth - 1, alpha, beta)
            if val > best_val + 1e-9:
                best_val, best_moves = val, [mv]
            elif abs(val - best_val) <= 1e-9:
                best_moves.append(mv)
            alpha = max(alpha, best_val)
        return best_moves[int(self.rng.integers(len(best_moves)))]


class AZOpp:
    """
    用**另一个 checkpoint** 当对手（同引擎、同先验权重，先后手交替）。

    【为什么要这层适配器】MCTS 的接口是 `get_best_action(board)`，而本文件里
    对手的统一接口是 `choose(board)` —— 直接把 MCTS 当对手会报
    `'MCTS' object has no attribute 'choose'`（实测踩过）。
    """

    def __init__(self, net, cfg, n_sim, hw, pw, seed, name):
        self.net, self.cfg = net, cfg
        self.n_sim, self.hw, self.pw = n_sim, hw, pw
        self.seed = seed
        self.name = name
        self.new_game(seed)

    def new_game(self, seed):
        """
        每局开始时**重建**搜索树。

        【为什么必须重建】MCTS 会跨手复用搜索树，并且维护一份增量棋盘历史
        （`_board_history`）。若把同一个 MCTS 实例跨局复用，新棋盘的走向会和
        上一局的历史打架 → `_seek` 反复重建、搜索质量塌陷。实测症状：
        `gen_60` 对 `gen_1` 只赢 5:3（而 48 局独立验证是 42:6）—— 假结论。
        """
        self.mcts = compare.make_mcts(self.net, self.cfg, self.n_sim,
                                      self.hw, self.pw, seed=seed,
                                      noise_alpha=0.3)

    def choose(self, board):
        mv = self.mcts.get_best_action(board)
        if mv is not None:
            self.mcts.set_root_move((int(mv[0]), int(mv[1])))
        return mv


def make_opponent(spec, seed=0):
    if spec == "random":
        return RandomOpp(seed)
    if spec == "greedy":
        return GreedyOpp(seed)
    if spec.startswith("minimax"):
        d = 2
        if "_d" in spec:
            d = int(spec.split("_d")[1])
        return MinimaxOpp(depth=d, width=12, seed=seed)
    return None


# ------------------------------------------------------------------ 对局

def play_one(net, opponent, cfg, n_sim, hw, pw, az_black, seed, max_ply=361):
    """模型 vs 对手，返回 (winner, plies, seq)。winner: 1黑/2白/0平。"""
    board = Board()
    mcts = compare.make_mcts(net, cfg, n_sim, hw, pw, seed=seed,
                             noise_alpha=0.3) if net is not None else None
    seq = []
    for _ in range(max_ply):
        over, winner = board.is_game_over()
        if over:
            return winner, len(seq), tuple(seq)
        me = board.current_player
        az_turn = (me == 1) == az_black
        if az_turn:
            mv = mcts.get_best_action(board)
            if mv is not None:
                mcts.set_root_move((int(mv[0]), int(mv[1])))
        else:
            mv = opponent.choose(board)
        if mv is None:
            return 0, len(seq), tuple(seq)
        x, y = int(mv[0]), int(mv[1])
        if not board.move(x, y, me):
            return 0, len(seq), tuple(seq)
        seq.append((x, y))
    return 0, len(seq), tuple(seq)


def run_duel(net, opp, cfg, games, n_sim, hw, pw, seed=0, quiet=False,
             label="模型"):
    """模型执黑/白交替，与对手打 games 局。返回统计字典。"""
    az_wins = opp_wins = draws = 0
    az_black_wins = az_white_wins = 0
    seqs, lens = set(), []
    t0 = time.time()
    for g in range(games):
        az_black = (g % 2 == 0)
        seed_g = seed * 1000 + g
        if hasattr(opp, "new_game"):        # 有状态的对手（另一个模型）每局重建
            opp.new_game(seed_g + 7)
        winner, plies, seq = play_one(net, opp, cfg, n_sim, hw, pw,
                                      az_black, seed_g)
        seqs.add(seq)
        lens.append(plies)
        if winner == 0:
            draws += 1
            who = "和"
        elif (winner == 1) == az_black:
            az_wins += 1
            if az_black:
                az_black_wins += 1
            else:
                az_white_wins += 1
            who = "模型胜"
        else:
            opp_wins += 1
            who = "对手胜"
        if not quiet:
            print(f"  第{g + 1:2d}局: 模型执{'黑' if az_black else '白'} "
                  f"{who:>3}  {plies:3d} 手", flush=True)
    decided = games - draws
    rate = az_wins / decided if decided else 0.0
    p = compare.binom_p(az_wins, decided) if decided else 1.0
    return {"games": games, "decided": decided, "az_wins": az_wins,
            "opp_wins": opp_wins, "draws": draws,
            "az_black_wins": az_black_wins, "az_white_wins": az_white_wins,
            "rate": rate, "p": p, "seqs": len(seqs),
            "mean_ply": float(np.mean(lens)), "secs": time.time() - t0,
            "opp": getattr(opp, "name", str(opp)), "label": label}


def print_result(r, vs_name):
    print("-" * 66)
    print(f"  {r['label']} 胜 {r['az_wins']}（执黑 {r['az_black_wins']} / "
          f"执白 {r['az_white_wins']}）    对手({vs_name}) 胜 {r['opp_wins']}    "
          f"和 {r['draws']}")
    print(f"  独立对局序列 {r['seqs']}/{r['games']}   平均 {r['mean_ply']:.0f} 手   "
          f"耗时 {r['secs']:.0f}s")
    if r["decided"]:
        print(f"  有效 {r['decided']} 局：模型胜率 {r['rate']:.3f}   "
              f"双侧二项 p={r['p']:.4f}")
        if r["p"] < 0.05:
            who = "模型显著更强" if r["rate"] > 0.5 else "对手显著更强"
            print(f"  >>> {who}")
        else:
            print("  >>> 样本内无法区分")
    print("-" * 66)


def main():
    ap = argparse.ArgumentParser(description="固定对手阶梯（绝对强度锚点）")
    ap.add_argument("--a", required=True, help="模型 checkpoint（.pt）")
    ap.add_argument("--b", default=None, help="单个对手：random/greedy/minimax_d2/minimax_d3 或另一个 .pt")
    ap.add_argument("--ladder", action="store_true", help="对整条阶梯各打一轮")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--heuristic-weight", type=float, default=6.0)
    ap.add_argument("--policy-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = compare.Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = LADDER if args.ladder else ([args.b] if args.b else [])
    if not targets:
        print("需要 --b <对手> 或 --ladder")
        return 1

    t_all = time.time()
    print("=" * 66)
    print(f"模型 A = {args.a}　局数 = {args.games}/对手　n_sim={args.n_sim}")
    print("=" * 66)
    rows = []
    for i, spec in enumerate(targets):
        opp = make_opponent(spec, seed=args.seed + i)
        if opp is None:                       # 当作另一个 checkpoint
            if not os.path.exists(spec):
                print(f"跳过未知对手 {spec}（既不是阶梯名也不是存在的 .pt）")
                continue
            net_b = compare.load_net(spec)
            opp = AZOpp(net_b, cfg, args.n_sim, args.heuristic_weight,
                        args.policy_weight, seed=args.seed + i,
                        name=os.path.basename(spec))
        net_a = compare.load_net(args.a)
        print(f"\n--- 对手：{getattr(opp, 'name', spec)} ---")
        r = run_duel(net_a, opp, cfg, args.games, args.n_sim,
                     args.heuristic_weight, args.policy_weight,
                     seed=args.seed + i, quiet=args.quiet)
        print_result(r, getattr(opp, "name", spec))
        rows.append(r)

    if len(rows) > 1:
        print("\n" + "=" * 66)
        print(f"{'对手':<14}{'模型胜率':>10}{'战绩':>14}{'p':>10}{'平均手数':>10}")
        print("-" * 66)
        for r in rows:
            print(f"{r['opp']:<14}{r['rate']:>10.3f}"
                  f"{str(r['az_wins']) + ':' + str(r['opp_wins']):>14}"
                  f"{r['p']:>10.4f}{r['mean_ply']:>10.0f}")
        print("=" * 66)
    print(f"总计耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
