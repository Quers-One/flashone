"""临时脚本：逐手复盘一局记录（跑完即删）。

输入：play.py --record 产出的 JSONL
输出：每一手前后，双方是否各有"一手成五"（杀），AI 是否漏杀/漏防；
      以及树复用的证据（root_total vs n_sim 请求值）。
"""

import copy
import json
import sys

from gomoku_env import Board, BLACK, WHITE


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def fives(board, player):
    """返回该方"落一手即成五"的点集合（只枚举已有棋子半径 2 内的空点）。"""
    out = []
    b = board.board
    cand = set()
    ys, xs = (b != 0).nonzero()
    for y, x in zip(ys, xs):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = int(y) + dy, int(x) + dx
                if 0 <= ny < 19 and 0 <= nx < 19 and b[ny, nx] == 0:
                    cand.add((nx + 1, ny + 1))       # 对外坐标 1~19
    for (x, y) in cand:
        t = copy.deepcopy(board)
        if t.current_player != player:
            t.current_player = player
        if t.move(x - 1, y - 1, player):
            over, w = t.is_game_over()
            if over and w == player:
                out.append((x, y))
    return out


def main(path):
    rows = load(path)
    hdr = rows[0]
    ai = 3 - int(hdr["human_player"])
    human = int(hdr["human_player"])
    moves = [r for r in rows if r["type"] == "move"]
    end = [r for r in rows if r["type"] == "end"]
    print(f"记录 {path}：n_sim={hdr['n_sim']} hw/pw={hdr['hw']}/{hdr['pw']} "
          f"人执{'黑' if human == 1 else '白'} 引擎={hdr['engine_mcts']} "
          f"手数={len(moves)}")
    if end:
        print(f"结束记录: {json.dumps({k: v for k, v in end[-1].items() if k != 'board'}, ensure_ascii=False)}")

    board = Board()
    print(f"\n{'ply':>3} {'走子':>9} {'谁的杀':>7}  {'AI有杀':>7} {'你有杀':>7}  "
          f"{'复用/请求':>10} {'首选占比':>8}")
    flags = []
    for i, r in enumerate(moves):
        who = "AI" if r["who"] == "ai" else "人"
        x, y = r["mv"]
        pre_ai = fives(board, ai) if board.current_player == ai else None
        pre_hu = fives(board, human) if board.current_player == human else None
        # 统一在"轮到谁"的语义下检查双方（战术点与轮次无关）
        ai_f = pre_ai if pre_ai is not None else fives(board, ai)
        hu_f = pre_hu if pre_hu is not None else fives(board, human)
        reuse = ""
        top = ""
        if r["who"] == "ai":
            tot, req = r.get("root_total_visits") or 0, r.get("n_sim_requested") or 0
            reuse = f"{tot}/{req}"
            if r.get("top"):
                share = r["top"][0][2] / max(tot, 1)
                top = f"{share:6.1%}"
        ok = board.move(x - 1, y - 1, board.current_player)
        mark = ""
        if not ok:
            mark = "  <<< 记录里的这手在重建时非法！"
        note = []
        if r["who"] == "ai":
            if ai_f and (x, y) not in ai_f:
                note.append(f"漏杀{ai_f}")
            if hu_f and (x, y) not in hu_f and (x, y) not in ai_f:
                # 对手有杀、自己也没杀 → 必须挡在对手的杀点上
                note.append(f"可能漏防{hu_f}")
        else:
            if hu_f and (x, y) not in hu_f:
                note.append(f"（你漏杀{hu_f}）")
        if note:
            flags.append((i + 1, who, (x, y), note))
        print(f"{i + 1:>3} {who:>3}({x:>2},{y:>2}) {'':>7}  "
              f"{(','.join(f'{a},{b}' for a, b in ai_f) or '-'):>7} "
              f"{(','.join(f'{a},{b}' for a, b in hu_f) or '-'):>7}  {reuse:>10} {top:>8}"
              f"{'  ' + ' '.join(note) if note else ''}{mark}")

    print("\n=== 值得看的位置 ===")
    for ply, who, mv, note in flags[:12]:
        print(f"  ply {ply}（{who}）走 {mv}：{'；'.join(note)}")
    if not flags:
        print("  没有发现漏杀/漏防（双方都没出现过一手杀）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "games/v1gen60_1.jsonl")
