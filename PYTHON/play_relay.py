"""
play_relay.py
=============
**单步**人机对战：把"一局棋"拆成多次调用，状态存在 JSON 里。

为什么需要它
------------
`play.py` 是交互式 stdin 循环，需要一个真正的终端。如果你是通过聊天/手机
下棋（没有终端可敲），就需要一个"我替你落子、把棋盘的 ASCII 图发给你"的通道。
本脚本每次只做一手：接收你的一手 -> 打印棋盘 -> AI 应一手 -> 打印棋盘。

坐标约定（与 play.py 一致）
--------------------------
    * `x y`：两数写法，**1~19**，x 是列（左→右），y 是行（下→上）
      例：`10 10` = 天元
    * `J10`：字母+数字写法，A~S 表示列 1~19，数字表示行 1~19
      例：`J10` 与 `10 10` 等价

用法
----
    python play_relay.py --new                 # 开新局（你执黑先手）
    python play_relay.py --new --human-white   # 开新局（你执白，AI 先手）
    python play_relay.py --move "10 10"        # 你落子，AI 应一手
    python play_relay.py --move "J10"          # 同上，字母写法
    python play_relay.py --show                 # 只看当前棋盘，不下子
    python play_relay.py --model models/run3/gen_60.pt --n-sim 1600

状态文件默认 `checkpoints/relay_game.json`（`--state` 可改）。

注意：每一步都是一棵**新**搜索树（不跨手复用），所以每手都是干净的 400 次模拟；
这和 `play.py` 的"树复用"在强度上同级（复用只是省算力，不多给信息）。
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gomoku_env import Board            # noqa: E402
from mcts import MCTS                   # noqa: E402
from play import load_model             # noqa: E402

DEFAULT_STATE = os.path.join("checkpoints", "relay_game.json")
LETTERS = "ABCDEFGHIJKLMNOPQRS"        # 19 列


def parse_move(text):
    """'10 10' 或 'J10' -> 内部 0-based (x, y)；非法返回 None。"""
    text = text.strip().upper().replace(",", " ")
    parts = text.split()
    try:
        if len(parts) == 1 and len(parts[0]) >= 2:
            head = parts[0][0]
            if head.isalpha():
                if head not in LETTERS:
                    return None
                x = LETTERS.index(head) + 1
                y = int(parts[0][1:])
            else:
                return None
        elif len(parts) == 2:
            x, y = int(parts[0]), int(parts[1])
        else:
            return None
    except ValueError:
        return None
    if not (1 <= x <= 19 and 1 <= y <= 19):
        return None
    return x - 1, y - 1


def to_label(mv):
    """内部 (x, y) -> 'J10' 这种人类标签。"""
    x, y = mv
    return f"{LETTERS[x]}{y + 1}"


def build_board(moves):
    b = Board()
    for (x, y, pl) in moves:
        b.move(int(x), int(y), int(pl))
    return b


def show(board, human, net, moves, last_ai=None, note=""):
    print("\n" + "=" * 46)
    print("     " + " ".join(f"{LETTERS[x]}" for x in range(19)))
    for y in range(18, -1, -1):
        # 【P0 已修·2026-10-12】同 play.py：Board.board 下标是 [行=y, 列=x]，
        # [:, y] 取的是第 y **列** ⇒ 棋盘被转置显示。详见 play.py 里的长注释。
        row = board.board[y, :]
        cells = " ".join(" ●" if v == 1 else (" ○" if v == 2 else " .")
                         for v in row)
        print(f"{y + 1:2d}   {cells}")
    print("=" * 46)
    print(f"你 = {'黑 ●' if human == 1 else '白 ○'}    "
          f"手数 = {len(moves)}    "
          f"轮到 {'你' if board.current_player == human else 'AI'}")
    if last_ai is not None:
        print(f"AI 上一手: {to_label(last_ai)}")
    # 网络价值头的局面估计（与 play.py 同口径）
    _, value = net.policy_value(board.get_state_tensor())
    ai_win = (value + 1) / 2 if board.current_player != human else 1 - (value + 1) / 2
    filled = int(round(ai_win * 20))
    print(f"局面胜率(网络估值): AI {ai_win * 100:5.1f}%  |{'#' * filled}{'-' * (20 - filled)}|")
    if note:
        print(note)
    print("=" * 46)


def main():
    ap = argparse.ArgumentParser(description="单步人机对战（聊天/命令行皆可）")
    ap.add_argument("--new", action="store_true", help="开新局")
    ap.add_argument("--move", type=str, default=None, help="你这一手，如 '10 10' 或 'J10'")
    ap.add_argument("--show", action="store_true", help="只看棋盘，不下子")
    ap.add_argument("--human-white", action="store_true", help="你执白（默认执黑）")
    ap.add_argument("--model", type=str, default="models/run3/best.pt")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--state", type=str, default=DEFAULT_STATE)
    ap.add_argument("--show-probs", action="store_true")
    args = ap.parse_args()

    human = 2 if args.human_white else 1
    ai = 3 - human

    if args.new:
        state = {"human": human, "model": args.model, "n_sim": args.n_sim,
                 "moves": [], "over": False, "winner": None}
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        print(f"[新局] 你执{'白 ○' if human == 2 else '黑 ●'}，AI 执"
              f"{'黑 ●' if human == 2 else '白 ○'}；模型 {args.model}，n_sim={args.n_sim}")
    else:
        if not os.path.exists(args.state):
            print(f"没有棋局状态 {args.state}，先运行 --new")
            return 1
        with open(args.state, encoding="utf-8") as f:
            state = json.load(f)
        human = int(state.get("human", human))
        ai = 3 - human
        args.model = args.model if args.model != "models/run3/best.pt" else state.get("model", args.model)
        args.n_sim = state.get("n_sim", args.n_sim)

    moves = [tuple(m) for m in state["moves"]]
    net = load_model(args.model)
    board = build_board(moves)

    over, winner = board.is_game_over()
    if over or state.get("over"):
        show(board, human, net, moves, note="【本局已结束】")
        return 0

    last_ai = None
    if args.move is not None and not args.show:
        mv = parse_move(args.move)
        if mv is None:
            print(f"看不懂这一手: {args.move!r}（用 '10 10' 或 'J10'，1~19）")
            return 1
        if board.current_player != human:
            print("现在不是你的回合（AI 该走）—— 直接再跑一次让它应手即可")
            return 1
        if not board.move(mv[0], mv[1], human):
            occ = [(x, y) for (x, y, p) in moves]
            print(f"{to_label(mv)} 不能下（已占或非法）。已占点 {len(occ)} 个。")
            return 1
        moves.append((mv[0], mv[1], human))
        board = build_board(moves)
        print(f"你落子: {to_label(mv)}")

    # 轮到 AI 就让它应一手
    over, winner = board.is_game_over()
    if not over and board.current_player == ai:
        mcts = MCTS(net, n_sim=args.n_sim, c_puct=1.0, virtual_loss=3.0,
                    batch_size=16, max_children=6, root_max_children=16,
                    heuristic_weight=6.0, policy_weight=0.3)
        best = mcts.get_best_action(board)
        if best is None:
            print("AI 无合法落子（棋盘满了）")
        else:
            last_ai = (int(best[0]), int(best[1]))
            board.move(last_ai[0], last_ai[1], ai)
            moves.append((last_ai[0], last_ai[1], ai))

    state["moves"] = [list(m) for m in moves]
    over, winner = board.is_game_over()
    state["over"], state["winner"] = bool(over), (int(winner) if over else None)
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)

    note = ""
    if over:
        note = ("【平局】棋盘下满。" if winner == 0 else
                f"【{'黑 ●' if winner == 1 else '白 ○'} 获胜】"
                f"{'你' if winner == human else 'AI'}赢了这局！")
    show(board, human, net, moves, last_ai=last_ai, note=note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
