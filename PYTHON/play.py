"""
play.py
=======
人机对战入口：加载训练好的模型，控制台输入坐标对战。

用法:
    python play.py                          # 默认加载 models/run3/best.pt
    python play.py --model path.pt          # 指定模型路径
    python play.py --human-white            # 人类执白(默认执黑)
    python play.py --n-sim 800              # 每步 MCTS 模拟轮数
    python play.py --show-probs             # 显示 AI 候选概率
    python play.py --model models/V1/gen_60.pt --n-sim 1600 --live --stats
                                            # KataGo 式：实时看它怎么算 + 落子后看算力分布

棋盘坐标: x 列(0~18) 横轴, y 行(0~18) 纵轴。
输入如 "10 10" 表示在 (x=10, y=10) 落子。

【--live / --stats 是怎么做到的】**不改 mcts.py**（改它 = 换引擎指纹 = 所有历史数字作废）。
搜索跑的时候，`MCTS._root.children` 里已经实时带着每个候选的
    `.n` 访问次数、`.w` 累计价值（Q = w/n）、`.prior` 网络先验 ——
所以外部起一个**只读**轮询线程就能一边算一边显示（GIL 下读 int/float 对显示足够安全，
即使快照偶尔不一致也不影响搜索本身）。KataGo 的实时面板本质也是这个（搜索在跑 + 定时汇报），
区别只是它是多线程 C++、我们是单线程 Python。

对照旧版的关键差异：
    * 旧版 mcts.py 的搜索只做到 2 层深度、丢弃 93% 算力、连一步成五都找不到，
      因此这里"AI 很弱"并不是模型的问题而是搜索的问题。现已修复：
      同样的 n_sim 下深度提升到 6~11 层，成五/必挡/做活四 100% 正确。
    * Tree Reuse 现在真正有效（旧版每手都会因为根只展开 top-20 而丢弃整棵树，
      实测每手 175ms 全是重算）。
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time

from gomoku_env import Board
from network import GomokuNet
from mcts import MCTS


def engine_fp() -> str:
    """当前引擎指纹（`mcts.py` 前 8 位）——复盘记录里必须带上，否则数字不知道出自哪把尺子。"""
    try:
        h = hashlib.sha1()
        with open("mcts.py", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def engine_fp_all() -> str:
    """引擎**行为**组合短哈希（mcts.py + heuristics.py + gomoku_env.py）。

    与 `compare.engine_fp_all()` / `ladder.engine_all_short()` 同口径：只改
    `heuristics.py` 或 `gomoku_env.py` 时旧指纹不变，但棋已经不一样了 —— 复盘记录
    里两个都写，才不会被静默误导。
    """
    try:
        h = hashlib.sha1()
        for name in ("mcts.py", "heuristics.py", "gomoku_env.py"):
            hh = hashlib.sha1()
            with open(name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    hh.update(chunk)
            h.update(f"{name}:{hh.hexdigest()[:8]}|".encode())
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def board_lines(board) -> list:
    """棋盘快照（行 19→1、列 1→19；'.' 空 / '1' 黑 / '2' 白），用于复盘精确重建。"""
    return ["".join("." if v == 0 else str(int(v)) for v in row)
            for row in board.board[::-1]]


def load_model(path: str) -> GomokuNet:
    """
    加载模型权重。

    兼容说明：network.py 的策略/价值头已从"展平+全连接"升级为
    "全卷积策略头 + 全局池化价值头"，旧 checkpoint 的这两层无法加载。

    【放大网络】走 `train.load_checkpoint`（按权重反推骨干规模），不要在这里
    自己 `GomokuNet()` —— 那样 6×128 的权重会被整包丢掉、只剩随机初始化，
    对战结果毫无意义（网页版 `web_play.py` 也走这个函数）。
    """
    from train import load_checkpoint
    net = load_checkpoint(path)
    net.eval()
    return net


def display_board(board: Board, human_player: int, mcts: MCTS,
                  show_probs: bool = False):
    """
    在控制台显示棋盘。
    行号从上到下(18~0)，列号从左到右(0~18)。
    """
    print("\n" + "=" * 40)
    # 列标题（显示 1~19，内部 0~18）
    print("    " + " ".join(f"{x:2d}" for x in range(1, 20)))
    for y in range(18, -1, -1):
        # 【P0 已修·2026-10-12】原来写的是 board.board[:, y]，注释还写着"第 y 行"。
        # 但 Board.board 是 numpy 数组、下标是 **[行=y, 列=x]**（gomoku_env.py:167
        # `self.board[y, x]`），所以 [:, y] 取的是**第 y 列**。后果：整张棋盘的行列
        # 被交换显示 —— 你输入 "3 10"，子画在第 3 行第 10 列（真实落点是第 10 行
        # 第 3 列）。引擎/胜负判定/复盘记录全都没错，**只有显示是转置的**，
        # 所以它看起来像"落子不准"，而且天元、对角线上的子看着完全正常。
        row = board.board[y, :]  # 第 y 行（所有列）
        chars = []
        for x in range(19):
            v = row[x]
            if v == 1:
                chars.append("●")
            elif v == 2:
                chars.append("○")
            else:
                chars.append(".")
        # 每个格子 2 字符宽 + 1 空格分隔，与列标题对齐
        cells = " ".join(f" {c}" for c in chars)
        print(f"{y+1:2d}  {cells}")
    print("=" * 40)
    print(f"执子方: {'黑(●)' if board.current_player == 1 else '白(○)'}  "
          f"{'[你]' if board.current_player == human_player else '[AI]'}")
    if show_probs:
        # 只取前5个候选，避免打印过多
        probs = mcts.get_action_probs(board, temp=1.0, n_sim=200)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
        print("AI 候选(前5): " + ", ".join(f"({x+1},{y+1})={p:.3f}"
                                            for (x, y), p in top))
    # 显示局面胜率（网络价值头评估）
    _, value = mcts.net.policy_value(board.get_state_tensor())
    # value 在 [-1, 1]，当前执子方胜率 = (value + 1) / 2
    current_win = (value + 1.0) / 2.0
    ai_player = 3 - human_player
    if board.current_player == ai_player:
        ai_win = current_win
    else:
        ai_win = 1.0 - current_win
    bar_len = 20
    filled = int(round(ai_win * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"局面胜率(网络估值): AI {ai_win*100:5.1f}%  |{bar}|")
    print("=" * 40)


def parse_input() -> tuple:
    """解析用户输入坐标(1~19)，返回内部 0~18 的 (x, y) 或 None。"""
    while True:
        try:
            line = input("输入落子坐标 (x y)，如 '10 10'，或 'q' 退出: ").strip()
            if line.lower() in ("q", "quit", "exit"):
                return None
            parts = line.split()
            if len(parts) != 2:
                print("格式错误，请输入 'x y' 两个数字")
                continue
            x, y = int(parts[0]), int(parts[1])
            if not (1 <= x <= 19 and 1 <= y <= 19):
                print("坐标超出范围(1~19)")
                continue
            return (x - 1, y - 1)
        except (ValueError, EOFError):
            print("输入无效")
            continue


def root_snapshot(mcts, top: int = 10):
    """读根节点的子节点：返回 (已算模拟数, [(x, y, n, q, p), ...] 按访问次数降序)。

    **只读**，不改引擎：数据本来就在 `MCTS._root.children` 里
    （`.n` 访问次数、`.w` 累计价值、`.prior` 网络先验）。
    """
    root = getattr(mcts, "_root", None)
    if root is None:
        return 0, []
    rows, total = [], 0
    for mv, ch in list(root.children.items()):
        n = int(getattr(ch, "n", 0))
        total += n
        if n <= 0:
            continue
        w = float(getattr(ch, "w", 0.0))
        rows.append((int(mv[0]), int(mv[1]), n, w / n,
                     float(getattr(ch, "prior", 0.0))))
    rows.sort(key=lambda r: -r[2])
    return total, rows[:top]


class LiveMonitor(threading.Thread):
    """搜索进行中，每 interval 秒把根节点快照覆盖打印成一行（daemon，不阻塞退出）。"""

    def __init__(self, mcts, target: int, interval: float = 0.2):
        super().__init__(daemon=True)
        self.mcts = mcts
        self.target = int(target or 0)
        self.interval = max(0.05, float(interval))
        self._stop = False
        self._last_len = 0

    def run(self):
        while not self._stop:
            total, rows = root_snapshot(self.mcts, top=3)
            if rows:
                b = rows[0]
                line = (f"  思考中 {total}"
                        f"{'/' + str(self.target) if self.target else ''} 次模拟"
                        f"　当前最佳 ({b[0] + 1},{b[1] + 1})"
                        f" (n={b[2]} Q你={b[3]:+.3f} P={b[4]:.3f})")
                if len(rows) > 1:
                    s = rows[1]
                    line += (f"　次优 ({s[0] + 1},{s[1] + 1})"
                             f" (n={s[2]} Q你={s[3]:+.3f})")
                sys.stdout.write("\r" + line + " " * max(0, self._last_len - len(line)))
                sys.stdout.flush()
                self._last_len = len(line)
            time.sleep(self.interval)

    def stop(self):
        self._stop = True
        self.join(timeout=1.0)
        sys.stdout.write("\r" + " " * self._last_len + "\r")
        sys.stdout.flush()


def print_move_stats(snap, chosen, top: int = 10):
    """打印这一手的算力分布（访问次数 / Q 搜索估值 / P 网络先验）。

    坐标用**双数字 (x,y)**，与棋盘显示和输入格式一致（早期版本用了 H11 那种
    字母+数字写法，跟棋盘对不上——2026-10-11 改回统一）。
    """
    total, rows = snap
    if not rows:
        return
    print(f"  算力分布（这手共 {total} 次模拟，前 {len(rows)} 个候选）：")
    print("  注：Q 是该手之后**从你（人类）视角**的搜索估值 —— MCTS 子节点轮到你先走；"
          "所以 Q 越接近 +1 表示你越优。")
    print(f"    {'#':>2} {'落子(x,y)':>9} {'访问':>6} {'占比':>7} {'Q(你视角)':>10} {'P(先验)':>8}")
    for i, (x, y, n, q, p) in enumerate(rows, 1):
        mark = "  ← AI 选它" if chosen is not None and (x, y) == tuple(chosen) else ""
        print(f"    {i:>2} {f'({x + 1},{y + 1})':>9} {n:>6} {n / total:>6.1%} "
              f"{q:>+10.3f} {p:>8.3f}{mark}")


def play_game(model: GomokuNet, human_player: int, n_sim: int,
              show_probs: bool = False, live: bool = False,
              live_every: float = 0.2, stats: bool = False,
              stats_top: int = 10, hw: float = 6.0, pw: float = 0.3,
              record: str = None, no_reuse: bool = False) -> int:
    """
    人机对战一局，返回胜者(1黑/2白/0平)。

    参数:
        model:       GomokuNet 实例
        human_player: 人类执子方 (1=黑, 2=白)
        n_sim:       MCTS 模拟轮数 (0=按深度动态)
        show_probs:  是否显示 AI 候选概率
        live:        搜索中实时刷新"算力面板"（KataGo 式；建议配 n_sim>=1600）
        stats:       落子后打印这一手的访问分布（n / Q / P）
        hw, pw:      规则先验 / 网络策略权重。默认 6.0/0.3（标尺口径）；
                     想按"训练时的配方"下就用 --hw 0.5 --pw 1.0（退火终点）
        record:      JSONL 落盘路径：整局序列 + 每手 AI 的根节点分布（复盘用）
        no_reuse:    【--no-reuse】落子后 `reset_tree()` 而不是 `set_root_move()`，
                     彻底关掉树复用（连 value/policy 缓存一起清）。
                     人机对战里复用是**真实存在**的：实测 root_total/n_sim 平均 1.6×、
                     最高 2.64×（人类落子也调 set_root_move，所以整棵树跟着整局走）。
                     注意：compare.py / 门控 play_match 是**双方各一棵树**，
                     每手哈希不匹配就重建 → 那里本来就没有复用，这个开关只在
                     play.py 和 train.self_play_game（共享树）里才有意义。
    """
    board = Board()
    # 与 train.py 使用同一套搜索参数
    if n_sim <= 0:
        n_sim = None
    mcts = MCTS(model, n_sim=n_sim, c_puct=1.0, virtual_loss=3.0,
                batch_size=16, max_children=6, root_max_children=16,
                heuristic_weight=hw, policy_weight=pw)

    rec = None
    if record:
        d = os.path.dirname(os.path.abspath(record))
        if d:
            os.makedirs(d, exist_ok=True)
        rec = open(record, "a", encoding="utf-8")

        def w(obj):
            rec.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rec.flush()

        w({"type": "header", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "human_player": human_player, "n_sim": n_sim or 0,
           "hw": hw, "pw": pw, "mcts_batch": 16,
           "engine_mcts": engine_fp(),
           "engine_full": engine_fp_all(),
           "note": "坐标是 1~19 的 (x,y)；board 是 19 行（上→下）字符串，. 空 1 黑 2 白"})
    else:
        def w(_obj):
            return
    ai_player = 3 - human_player

    mcts.reset_tree()
    display_board(board, human_player, mcts, show_probs)

    while True:
        over, winner = board.is_game_over()
        if over:
            if winner == 0:
                print("\n【平局】棋盘下满，无人获胜。")
            else:
                name = "黑(●)" if winner == 1 else "白(○)"
                who = "你" if winner == human_player else "AI"
                print(f"\n【{name} 获胜】{who} 赢了这局！")
            w({"type": "end", "winner": winner, "ply": len(board.history),
               "board": board_lines(board)})
            if rec:
                rec.close()
            return winner

        if board.current_player == human_player:
            # 人类落子
            while True:
                move = parse_input()
                if move is None:
                    w({"type": "end", "winner": None, "note": "人类中途退出"})
                    if rec:
                        rec.close()
                    return 0
                x, y = move
                if board.move(x, y, human_player):
                    if no_reuse:
                        mcts.reset_tree()
                    else:
                        mcts.set_root_move((x, y))
                    w({"type": "move", "who": "human", "ply": len(board.history),
                       "mv": [x + 1, y + 1], "board": board_lines(board)})
                    break
                else:
                    print("该位置已占或非法，请重新输入")
        else:
            # AI 落子
            print("\nAI 思考中...")
            mon = None
            if live:
                mon = LiveMonitor(mcts, mcts.n_sim or 0, live_every)
                mon.start()
            t0 = time.perf_counter()
            best = mcts.get_best_action(board)
            dt = time.perf_counter() - t0
            if mon:
                mon.stop()
            # 【顺序要紧】快照必须在 set_root_move 之前取：它会把根推进到选中的子节点
            # （树复用），推进之后 children 就变成"下一层"了，这手的分布就没了。
            snap = root_snapshot(mcts, stats_top)
            if best is None:
                print("AI 无合法落子")
                return 0
            x, y = int(best[0]), int(best[1])
            board.move(x, y, ai_player)
            if no_reuse:
                # 【--no-reuse】整棵树作废：下一手从空树重新搜 n_sim 次，
                # 不再继承任何旧访问统计（value/policy 缓存也一起清）。
                mcts.reset_tree()
            else:
                mcts.set_root_move((x, y))
            msg = f"AI 落子: ({x+1}, {y+1})"
            if dt > 0 and mcts.n_sim:
                msg += f"　用时 {dt:.2f}s（{mcts.n_sim / dt:.0f} 次模拟/秒）"
            print(msg)
            if stats:
                print_move_stats(snap, (x, y), stats_top)
            # 复盘记录：**这手的根分布 + 复用后的总访问数**（判断树复用是否把它带偏的关键）
            w({"type": "move", "who": "ai", "ply": len(board.history),
               "mv": [x + 1, y + 1], "seconds": round(dt, 3),
               "n_sim_requested": mcts.n_sim or 0,
               "root_total_visits": snap[0],
               "top": [[int(a) + 1, int(b) + 1, int(n), round(float(q), 4),
                        round(float(p), 4)] for a, b, n, q, p in snap[1]],
               "board": board_lines(board)})

        display_board(board, human_player, mcts, show_probs)


def main():
    parser = argparse.ArgumentParser(description="五子棋人机对战")
    parser.add_argument("--model", type=str, default="models/run3/best.pt",
                        help="模型权重路径")
    parser.add_argument("--human-white", action="store_true",
                        help="人类执白(默认执黑)")
    parser.add_argument("--n-sim", type=int, default=400,
                        help="每步 MCTS 模拟轮数 (0=按深度动态)")
    parser.add_argument("--show-probs", action="store_true",
                        help="显示 AI 候选概率")
    parser.add_argument("--live", action="store_true",
                        help="搜索中实时刷新算力面板（KataGo 式；建议配 --n-sim 1600+）")
    parser.add_argument("--live-every", type=float, default=0.2,
                        help="实时面板刷新间隔秒数（默认 0.2）")
    parser.add_argument("--stats", action="store_true",
                        help="AI 落子后打印这一手的访问分布（访问次数/Q/P）")
    parser.add_argument("--stats-top", type=int, default=10,
                        help="--stats 打印多少个候选（默认 10）")
    parser.add_argument("--hw", type=float, default=6.0,
                        help="规则先验权重（默认 6.0＝标尺口径；"
                             "按训练配方下用 --hw 0.5）")
    parser.add_argument("--pw", type=float, default=0.3,
                        help="网络策略权重（默认 0.3＝标尺口径；按训练配方用 --pw 1.0）")
    parser.add_argument("--record", type=str, default=None,
                        help="把整局落子序列 + 每手 AI 的搜索分布写成 JSONL（复盘用，"
                             "例如 --record games/v1gen60_1.jsonl）")
    parser.add_argument("--no-reuse", action="store_true",
                        help="关掉树复用（每手从空树重搜）。用于对照"
                             "\"复用是不是把它带偏了\"：复用下 root_total/n_sim 平均 1.6×、最高 2.64×")
    args = parser.parse_args()

    human_player = 2 if args.human_white else 1
    print("=" * 40)
    print("五子棋 人机对战")
    print("=" * 40)
    print(f"人类执: {'白(○)' if human_player == 2 else '黑(●)'}")
    print(f"AI 执:   {'黑(●)' if human_player == 2 else '白(○)'}")
    print(f"模型:    {args.model}")
    print(f"MCTS:    c_puct=1.0, virtual_loss=3.0, "
          f"n_sim={args.n_sim or '动态'}, hw/pw={args.hw}/{args.pw}")
    print("输入 'q' 退出")
    if args.live:
        print(f"实时面板: 开（每 {args.live_every}s 刷新；日志会打印到 stdout，"
              f"重定向到文件时看不到）")
    if args.stats:
        print(f"算力分布: 开（每手打印前 {args.stats_top} 个候选）")
    if args.record:
        print(f"复盘记录: {args.record}（整局序列 + 每手根分布，JSONL 追加）")
    if args.no_reuse:
        print("树复用:  关（--no-reuse：每手从空树重搜，会明显变慢）")

    try:
        model = load_model(args.model)
    except FileNotFoundError:
        print(f"未找到模型 {args.model}")
        print("请先运行 train.py 生成模型，或使用 --model 指定路径")
        return
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("开始对战...\n")
    play_game(model, human_player, args.n_sim, args.show_probs,
              live=args.live, live_every=args.live_every,
              stats=args.stats, stats_top=args.stats_top,
              hw=args.hw, pw=args.pw, record=args.record,
              no_reuse=args.no_reuse)
    print("\n感谢游玩！")


if __name__ == "__main__":
    main()
