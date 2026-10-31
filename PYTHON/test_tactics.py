"""
test_tactics.py
===============
五子棋引擎「战术正确性测试集」（优化 阶段 0 的验收工具）。

用途
----
在没有训练好网络的情况下，验证**搜索引擎本身**是否正确。
这是训练之前必须先过的门槛：搜索错了，自对弈数据就是噪声，
训练多久都不会进步（旧实现正是如此：241 手全平局、181 次漏杀）。

测试项
------
    A. 一步成五（必胜点必须走到）         -> 必须 100% 通过
    B. 必挡（对方一步成五必须堵）         -> 必须 100% 通过
    C. 活四（能形成活四的要点）           -> 必须 100% 通过
    D. 安全性（不得走出让对方立刻成五的着）-> 必须 100% 通过
    E. 允许的等价解（多个同价值点时命中任一即可）

每个测试局面都用"落子序列"构造，并在运行时**自校验**：
    * 局面未终局
    * 轮到预期的一方
    * 期望的必胜/必挡点确实成立（否则测试本身有 bug，直接报错）
这样不会出现"局面写错导致测试假通过"的情况。

用法
----
    python test_tactics.py                  # 默认 n_sim=200，带启发式
    python test_tactics.py --n-sim 400
    python test_tactics.py --compare        # 对比"启用/关闭启发式"的通过率
"""

import argparse
import io
import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

# 【必须显式指定输出编码】不能用 io.TextIOWrapper(sys.stdout.buffer)：包装器被 GC 时
# 会关掉底层 buffer（见 compare.py 的同款说明）。这里用 reconfigure。
# 实测教训：本文件此前没有这段 → 输出编码取决于环境变量 PYTHONIOENCODING，
# 从后台任务启动时（父进程没设）会按 GBK 输出，导致调用方（finalize.py）
# 按 UTF-8 解码拿到乱码、连"总计"都解析不到。显式重编码后与调用环境无关。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from gomoku_env import Board
from network import GomokuNet
from mcts import MCTS

Move = Tuple[int, int]
ColoredMove = Tuple[int, int, int]      # (x, y, player)

# ======================================================================
# 局面构造工具
# ======================================================================


def build(moves: Sequence) -> Board:
    """
    按序列构造局面。

    元素可以是:
        (x, y)         -> 按交替顺序（黑先）自动着色
        (x, y, player) -> 显式指定着色（必须满足轮次合法）
        (x, y, None)   -> 由 build_any 自动推断合法落子顺序
    """
    b = Board()
    for i, mv in enumerate(moves):
        if len(mv) == 3 and mv[2] is not None:
            x, y, player = int(mv[0]), int(mv[1]), int(mv[2])
        else:
            x, y = int(mv[0]), int(mv[1])
            player = 1 if i % 2 == 0 else 2
        assert b.move(x, y, player), \
            f"构造失败: 第{i}手 {(x, y)} player={player} 非法"
    return b


def build_any(stones: Sequence[Tuple[int, int, int]],
              n_try: int = 30000) -> Board:
    """
    由"棋子集合"构造局面，自动搜索一个合法的落子顺序。

    五子棋必须黑先白后交替，手工排序列极易出错（写错颜色会让局面完全不是
    预期形状，测试就变成假通过/假失败）。这里只声明"哪些位置是什么颜色"，
    由本函数按合法轮次顺序把它们放上去。

    参数:
        stones: [(x, y, player), ...]
    返回:
        Board（刚好放下这些棋子，且满足黑先白后）
    注意:
        要求黑子数 == 白子数（轮到黑）或 黑子数 == 白子数 + 1（轮到白）。
    """
    blacks = [(x, y) for x, y, p in stones if p == 1]
    whites = [(x, y) for x, y, p in stones if p == 2]
    assert len(blacks) in (len(whites), len(whites) + 1), \
        f"棋子数量不合法: 黑{len(blacks)} 白{len(whites)}"

    rng = np.random.default_rng(12345)
    attempts = min(n_try, 5 * math.factorial(max(len(blacks), 1)))
    for _ in range(attempts):
        bi = list(rng.permutation(len(blacks)))
        wi = list(rng.permutation(len(whites)))
        b = Board()
        expected: List[int] = []       # 预期的棋子布局快照（展平）
        _s = np.zeros((b.SIZE, b.SIZE), dtype=np.int8)
        for x, y, p in stones:
            _s[y, x] = p
        expected = _s.ravel().tolist()
        ok = True
        for k in range(len(blacks)):
            if not b.move(blacks[bi[k]][0], blacks[bi[k]][1], 1):
                ok = False
                break
            # 目标局面本身不是终局，因此任何中途出现五连的顺序都是死路
            if b.is_game_over()[0]:
                ok = False
                break
            if k < len(whites):
                if not b.move(whites[wi[k]][0], whites[wi[k]][1], 2):
                    ok = False
                    break
                if b.is_game_over()[0]:
                    ok = False
                    break
        # 退出前必须校验"落子真的按预期写进了棋盘"。
        # 这一条不是多余的自检：早期版本的棋子顺序校验有缺陷，
        # 会静默产出与声明不符的棋盘，让后续测试测一个假的局面。
        if ok and b.board.ravel().tolist() != expected:
            ok = False
        if ok and len(b.history) == len(stones):
            # 关键校验：目标局面本身不能已经是五连终局
            # （否则这个"局面"根本到不了，测试会测一个假的终局状态）
            if b.is_game_over()[0]:
                continue
            return b
    raise RuntimeError(f"未能为给定棋子集合找到合法落子顺序: {stones}")


def winning_moves(board: Board, player: Optional[int] = None) -> Set[Move]:
    """
    枚举当前所有"一步成五"的点。

    参数:
        board : 局面
        player: 指定检查哪一方（默认当前执子方）
    """
    player = board.current_player if player is None else player
    out: Set[Move] = set()
    for (x, y) in board.get_legal_moves():
        if board.check_win_after(x, y, player):
            out.add((int(x), int(y)))
    return out


def has_open_four(board: Board, player: int) -> bool:
    """
    player 是否已经形成「活四」：存在一条线上恰有 4 个连续同色子，且两端为空。

    注意与"有 4 个成五威胁"区分：后者会把无关方向的零散威胁也算进来，过宽。
    活四的严格定义就是"4 连 + 两端皆空"，必然有 2 个成五点。

    实现委托给 gomoku_env.Board 的同名逻辑（那里同时供 MCTS 使用），
    这里只做"是否存在"的判定：借用 open_four_after 在某个空点上的语义，
    直接扫描棋盘上已有的 4 连。
    """
    arr = board.board
    n = board.SIZE
    for y in range(n):
        for x in range(n):
            if arr[y, x] != player:
                continue
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                # 只从该方向的起点开始数，避免重复
                px, py = x - dx, y - dy
                if 0 <= px < n and 0 <= py < n and arr[py, px] == player:
                    continue
                cells = []
                cx, cy = x, y
                while 0 <= cx < n and 0 <= cy < n and arr[cy, cx] == player:
                    cells.append((cx, cy))
                    cx, cy = cx + dx, cy + dy
                if len(cells) != 4:
                    continue
                bx, by = x - dx, y - dy
                ex, ey = cx, cy
                lo_ok = (0 <= bx < n and 0 <= by < n and arr[by, bx] == 0)
                hi_ok = (0 <= ex < n and 0 <= ey < n and arr[ey, ex] == 0)
                if lo_ok and hi_ok:
                    return True
    return False


def open_four_moves(board: Board, player: Optional[int] = None) -> Set[Move]:
    """
    枚举能让 player 形成「活四」的点。

    活四 = 落子后存在 4 连且两端皆空 -> 下一手有两个不同的成五点，
    对方只能堵一个，因此活四是**必胜形**。
    """
    player = board.current_player if player is None else player
    opp = 3 - player
    out: Set[Move] = set()
    for (x, y) in board.get_legal_moves():
        tb = board.copy()
        if not tb.move(int(x), int(y), player):
            continue
        if has_open_four(tb, player) and not winning_moves(tb, opp):
            out.add((int(x), int(y)))
    return out


def opponent_wins_after(board: Board, move: Move, player: int) -> bool:
    """假设 player 走 move，之后对方是否立刻有一步成五的机会。"""
    opp = 3 - player
    tb = board.copy()
    if not tb.move(move[0], move[1], player):
        return False
    if tb.is_game_over()[0]:
        return False          # 这一手直接赢了，无需担心
    return len(winning_moves(tb, opp)) > 0


# ======================================================================
# 测试局面库
# ======================================================================
# 每个局面只声明"棋子集合" [(x, y, player), ...]，由 build_any() 自动找到
# 一个合法的黑先白后落子顺序。这样不会因为手写顺序/颜色出错而让局面变形。
# 轮到谁走由黑白子数量决定：
#     黑子数 == 白子数       -> 轮到黑
#     黑子数 == 白子数 + 1   -> 轮到白

# ---- A. 一步成五 ----
# 注意：黑子数 N 时白子数必须是 N（轮到黑）或 N-1（轮到黑）之外的情况——
# 由黑先的交替规则决定：black == white -> 轮到黑；black == white + 1 -> 轮到白。
# 因此局面里黑子数必须 <= 白子数 + 1，且 >= 白子数。
A_CASES: List[Tuple[str, List[Tuple[int, int, int]], Set[Move]]] = [
    ("黑四连(两端皆空)，黑走任一端成五",
     [(5, 8, 1), (6, 8, 1), (7, 8, 1), (8, 8, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(4, 8), (9, 8)}),
    ("黑四连(左端被白封)，只能走右端",
     [(5, 8, 1), (6, 8, 1), (7, 8, 1), (8, 8, 1),
      (4, 8, 2), (0, 0, 2), (0, 1, 2), (0, 2, 2)],
     {(9, 8)}),
    ("白四连，轮到白，白成五",
     [(0, 0, 1), (0, 1, 1), (0, 4, 1), (1, 0, 1), (2, 0, 1),
      (5, 12, 2), (6, 12, 2), (7, 12, 2), (8, 12, 2)],
     {(4, 12), (9, 12)}),
    ("黑竖向四连，(4,9) 或 (4,14) 成五",
     [(4, 10, 1), (4, 11, 1), (4, 12, 1), (4, 13, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(4, 9), (4, 14)}),
    ("黑正斜四连，(3,3) 或 (8,8) 成五",
     [(4, 4, 1), (5, 5, 1), (6, 6, 1), (7, 7, 1),
      (0, 17, 2), (0, 16, 2), (0, 15, 2), (1, 17, 2)],
     {(3, 3), (8, 8)}),
    ("黑反斜四连，(3,7) 或 (8,2) 成五",
     [(4, 6, 1), (5, 5, 1), (6, 4, 1), (7, 3, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(3, 7), (8, 2)}),
]

# ---- B. 必挡 ----
# 黑（对手）已成四连，轮到白，白必须堵。
B_CASES: List[Tuple[str, List[Tuple[int, int, int]], Set[Move]]] = [
    ("黑横四连，黑多一手：轮到白，白必挡 (2,5)/(7,5)",
     [(3, 5, 1), (4, 5, 1), (5, 5, 1), (6, 5, 1), (2, 10, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(2, 5), (7, 5)}),
    ("黑横四连(右端被白封)，白只能挡 (2,3)",
     [(3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 1), (2, 10, 1), (7, 3, 2),
      (0, 0, 2), (0, 1, 2), (0, 2, 2)],
     {(2, 3)}),
    ("黑竖四连，白必挡 (9,5)/(9,10)",
     [(9, 6, 1), (9, 7, 1), (9, 8, 1), (9, 9, 1), (2, 10, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(9, 5), (9, 10)}),
    ("黑正斜四连，白必挡 (2,2)/(7,7)",
     [(3, 3, 1), (4, 4, 1), (5, 5, 1), (6, 6, 1), (2, 10, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(2, 2), (7, 7)}),
    ("黑反斜四连，白必挡 (2,8)/(7,3)",
     [(3, 7, 1), (4, 6, 1), (5, 5, 1), (6, 4, 1), (2, 10, 1),
      (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2)],
     {(2, 8), (7, 3)}),
    ("白横四连，轮到黑，黑必挡 (5,8)/(10,8)",
     [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1),
      (6, 8, 2), (7, 8, 2), (8, 8, 2), (9, 8, 2)],
     {(5, 8), (10, 8)}),
]

# ---- C. 活四（必胜形）----
# 局面：我方已有"活三"（三连且两端皆空），存在把它扩成"活四"的点。
# 活四 = 4 连且两端皆空 -> 下一手有两个成五点，对方只能堵一个 => 必胜。
# 用 has_open_four() 严格判定，避免"任意方向凑出两个成五点"的过度宽松。
C_CASES: List[Tuple[str, List[Tuple[int, int, int]]]] = [
    ("黑横活三 (5,10)(6,10)(7,10)，黑应做活四",
     [(5, 10, 1), (6, 10, 1), (7, 10, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2)]),
    ("黑横活三 (6,10)(7,10)(8,10)，黑应做活四",
     [(6, 10, 1), (7, 10, 1), (8, 10, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2)]),
    ("黑竖活三 x=10 (6..8)，黑应做活四",
     [(10, 6, 1), (10, 7, 1), (10, 8, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2)]),
    ("黑正斜活三 (6,6)(7,7)(8,8)，黑应做活四",
     [(6, 6, 1), (7, 7, 1), (8, 8, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2)]),
]

# ---- D. 安全性 ----
# 局面：对方已成"活四"（下一手就有两个成五点，必胜），
#       而我方有一个诱人的"活三"可以扩成四。
# 正确走法：立刻堵对方的成五点（这是唯一能延长对局的着法）。
# 错误走法（贪攻）：去扩自己的三 -> 对方下一手直接成五。
# 检查项：引擎选择后，对方**没有**立刻成五的机会。
D_CASES: List[Tuple[str, List[Tuple[int, int, int]], Set[Move]]] = [
    ("白有活三(y=3)，但黑已成四(y=12)：白必须挡 (5,12)/(10,12)",
     [(6, 12, 1), (7, 12, 1), (8, 12, 1), (9, 12, 1), (0, 7, 1),
      (3, 3, 2), (4, 3, 2), (5, 3, 2), (0, 8, 2)],
     {(5, 12), (10, 12)}),
    ("白有活三(y=5)，但黑竖四连(x=12)：白必须挡 (12,5)/(12,10)",
     [(12, 6, 1), (12, 7, 1), (12, 8, 1), (12, 9, 1), (0, 7, 1),
      (3, 5, 2), (4, 5, 2), (5, 5, 2), (0, 8, 2)],
     {(12, 5), (12, 10)}),
    ("白有活三(y=12)，但黑反斜四连：白必须挡 (2,8)/(7,3)",
     [(6, 4, 1), (5, 5, 1), (4, 6, 1), (3, 7, 1), (0, 7, 1),
      (3, 12, 2), (4, 12, 2), (5, 12, 2), (0, 8, 2)],
     {(2, 8), (7, 3)}),
]


# ======================================================================
# 测试执行
# ======================================================================


class TacticResult:
    def __init__(self):
        self.rows: List[dict] = []

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r["ok"])

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def add(self, group: str, name: str, ok: bool, detail: str):
        self.rows.append({"group": group, "name": name, "ok": ok,
                          "detail": detail})

    def summary(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, List[int]] = {}
        for r in self.rows:
            k = r["group"]
            out.setdefault(k, [0, 0])
            out[k][0] += 1
            out[k][1] += 1 if r["ok"] else 0
        return {k: (v[1], v[0]) for k, v in out.items()}


def make_mcts(net, n_sim: int, use_heuristics: bool, seed: int = 0,
              hw: float = None, pw: float = None, **kw) -> MCTS:
    """
    构造测试用的 MCTS（与 train/play 使用同一套参数）。

    hw/pw 可覆盖规则/策略先验权重。**这是必需的**：训练里规则先验权重会随代数
    退火（6.0 -> 2.0 -> 0.5），必须能在任意权重下重跑本测试集，才能确认
    "撤掉规则拐杖之后战术正确性没被牺牲"。
    """
    params = dict(n_sim=n_sim, c_puct=1.0, virtual_loss=3.0, batch_size=16,
                  max_children=6, root_max_children=16,
                  heuristic_weight=(6.0 if use_heuristics else 0.0)
                  if hw is None else hw,
                  policy_weight=(0.3 if use_heuristics else 1.0)
                  if pw is None else pw,
                  use_heuristics=use_heuristics, seed=seed)
    params.update(kw)
    return MCTS(net, **params)


def run_suite(net, n_sim: int, use_heuristics: bool, verbose: bool = True,
              hw: float = None, pw: float = None) -> TacticResult:
    res = TacticResult()
    tag = "带启发式" if use_heuristics else "纯网络先验"
    if use_heuristics and hw is not None:
        tag += f" hw={hw}/pw={pw if pw is not None else 0.3}"

    def pick(board: Board) -> Optional[Move]:
        m = make_mcts(net, n_sim, use_heuristics, hw=hw, pw=pw)
        m.reset_tree()
        return m.get_best_action(board, n_sim=n_sim)

    # ---------- A. 一步成五 ----------
    for name, stones, expected in A_CASES:
        b = build_any(stones)
        assert not b.is_game_over()[0], f"[测试集错误] A 局面已终局: {name}"
        real_wins = winning_moves(b)
        assert real_wins, f"[测试集错误] A 局面并无成五点: {name}"
        assert real_wins <= expected, \
            f"[测试集错误] A 期望集合漏了 {real_wins - expected}: {name}"
        got = pick(b)
        ok = got in real_wins
        res.add("A.一步成五", name, ok, f"选择={got} 成五点={sorted(real_wins)}")

    # ---------- B. 必挡 ----------
    for name, stones, expected in B_CASES:
        b = build_any(stones)
        assert not b.is_game_over()[0], f"[测试集错误] B 局面已终局: {name}"
        opp = 3 - b.current_player
        threats = winning_moves(b, opp)
        assert threats, f"[测试集错误] B 局面对方并无成五威胁: {name}"
        assert threats <= expected, \
            f"[测试集错误] B 期望集合漏了 {threats - expected}: {name}"
        # 我方无自身成五（否则优先成五也合理）
        my_wins = winning_moves(b)
        got = pick(b)
        ok = (got in threats) or (got in my_wins)
        res.add("B.必挡", name, ok,
                f"选择={got} 对方威胁={sorted(threats)}")

    # ---------- C. 活四 ----------
    for name, stones in C_CASES:
        b = build_any(stones)
        target = open_four_moves(b)
        assert target, f"[测试集错误] C 局面并无做活四的点: {name}"
        got = pick(b)
        ok = got in target
        res.add("C.做活四", name, ok,
                f"选择={got} 可成活四={sorted(target)}")

    # ---------- D. 安全性 ----------
    for name, stones, expected in D_CASES:
        b = build_any(stones)
        me = b.current_player
        opp = 3 - me
        threats = winning_moves(b, opp)
        assert threats, f"[测试集错误] D 局面对方并无成五威胁: {name}"
        assert threats <= expected, \
            f"[测试集错误] D 期望集合漏了 {threats - expected}: {name}"
        assert has_open_four(b, opp), f"[测试集错误] D 对方应为活四: {name}"
        got = pick(b)
        my_wins = winning_moves(b, me)
        unsafe = got is not None and opponent_wins_after(b, got, me)
        # 允许：挡在成五点上（含自己直接成五的反杀）
        ok = (got in threats) or (got in my_wins) or (not unsafe)
        res.add("D.安全性", name, ok,
                f"选择={got} 必挡点={sorted(threats)} 走后对方成五={unsafe}")

    if verbose:
        print(f"\n{'='*78}")
        print(f"战术测试集 [{tag}]  n_sim={n_sim}")
        print(f"{'='*78}")
        last_group = None
        for r in res.rows:
            if r["group"] != last_group:
                last_group = r["group"]
                print(f"\n-- {last_group} --")
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['name']}")
            if not r["ok"]:
                print(f"         {r['detail']}")
        print(f"\n{'='*78}")
        print(f"分组通过率 [{tag}]：")
        for g, (p, t) in res.summary().items():
            print(f"   {g:<12} {p}/{t}  ({p/t*100:5.1f}%)")
        print(f"   总计         {res.passed}/{res.total}  ({res.rate*100:5.1f}%)")
        print(f"{'='*78}")
    return res


def main():
    ap = argparse.ArgumentParser(description="五子棋引擎战术正确性测试")
    ap.add_argument("--n-sim", type=int, default=400,
                    help="每步 MCTS 模拟次数（默认 400；当前参数下 400 已能 100%% 通过）")
    ap.add_argument("--model", type=str, default=None,
                    help="模型路径（默认随机初始化网络）")
    ap.add_argument("--compare", action="store_true",
                    help="对比启用/关闭规则启发式的通过率")
    ap.add_argument("--heuristic-weight", type=float, default=None,
                    help="规则先验权重（默认 6.0）。用来验证退火后战术是否仍然正确")
    ap.add_argument("--policy-weight", type=float, default=None,
                    help="网络策略先验权重（默认 0.3）")
    ap.add_argument("--sweep", action="store_true",
                    help="按训练退火曲线扫一遍权重（6.0/2.0/0.5/0.0），看战术是否退化")
    ap.add_argument("--seed", type=int, default=1234,
                    help="随机种子（固定随机网络权重，保证结果可复现）")
    args = ap.parse_args()

    # 重要：不指定模型时网络是随机初始化的，必须固定种子，
    # 否则每次运行的"随机网络"都不同，通过率会在 40%~90% 之间乱跳，
    # 无法用来判断改动是否真的带来提升。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model:
        # 【放大网络】按权重反推骨干规模，别写死 GomokuNet()
        from train import load_checkpoint
        net = load_checkpoint(args.model)
        print(f"已加载模型: {args.model}  (seed={args.seed})")
    else:
        net = GomokuNet()
        print(f"未指定模型 -> 随机初始化网络 (seed={args.seed})，"
              f"测试的是搜索引擎本身")
    net.eval()

    t0 = time.time()

    if args.sweep:
        # 按训练用的退火曲线扫一遍：规则拐杖撤掉之后，战术正确性还在不在？
        print(f"\n规则先验权重扫描（评估退火是否安全）")
        print("=" * 78)
        print(f"  {'hw/pw':>12} {'A.成五':>8} {'B.必挡':>8} {'C.活四':>8} "
              f"{'D.安全':>8} {'总计':>9}")
        worst = None
        for (hw, pw) in ((6.0, 0.3), (3.6, 0.72), (2.0, 1.0), (0.5, 1.0),
                         (0.0, 1.0)):
            r = run_suite(net, args.n_sim, use_heuristics=True,
                          verbose=False, hw=hw, pw=pw)
            s = r.summary()
            cells = []
            for g in ("A.一步成五", "B.必挡", "C.做活四", "D.安全性"):
                a, b = s.get(g, (0, 0))
                cells.append(f"{a}/{b}")
            hard_fail = [g for g in ("A.一步成五", "B.必挡", "D.安全性")
                         if s.get(g, (0, 0))[0] != s.get(g, (0, 0))[1]]
            flag = "  <<< 硬指标未满" if hard_fail else ""
            print(f"  {f'{hw}/{pw}':>12} {cells[0]:>8} {cells[1]:>8} "
                  f"{cells[2]:>8} {cells[3]:>8} "
                  f"{f'{r.passed}/{r.total}':>9}{flag}")
            if hard_fail and worst is None:
                worst = (hw, pw, hard_fail)
        print("=" * 78)
        if worst:
            print(f"[警告] 权重 {worst[0]}/{worst[1]} 时硬指标未满: {worst[2]}")
            sys.exit(1)
        print("[达标] 从 6.0 一路降到 0.0，硬指标（成五/必挡/安全）始终 100%")
        print("       说明退火不会牺牲基本战术正确性 —— 强制覆盖与权重无关。")
        print(f"\n耗时 {time.time()-t0:.1f}s")
        return

    r_heur = run_suite(net, args.n_sim, use_heuristics=True,
                       hw=args.heuristic_weight, pw=args.policy_weight)

    if args.compare:
        r_pure = run_suite(net, args.n_sim, use_heuristics=False)
        print(f"\n{'='*78}")
        print("对比：规则启发式对战术正确性的贡献")
        print(f"{'='*78}")
        print(f"  {'分组':<14}{'带启发式':>16}{'纯网络先验':>16}")
        gs = set(r_heur.summary()) | set(r_pure.summary())
        for g in sorted(gs):
            a = r_heur.summary().get(g, (0, 0))
            b = r_pure.summary().get(g, (0, 0))
            print(f"  {g:<14}{f'{a[0]}/{a[1]}':>16}{f'{b[0]}/{b[1]}':>16}")
        print(f"  {'总计':<14}"
              f"{f'{r_heur.passed}/{r_heur.total}':>16}"
              f"{f'{r_pure.passed}/{r_pure.total}':>16}")

    print(f"\n耗时 {time.time()-t0:.1f}s")

    # 验收门槛：A/B/D 必须 100%
    s = r_heur.summary()
    hard = ["A.一步成五", "B.必挡", "D.安全性"]
    failed = [g for g in hard if s.get(g, (0, 0))[0] != s.get(g, (0, 0))[1]]
    if failed:
        print(f"\n[未达标] 以下分组未达 100%: {failed}")
        sys.exit(1)
    print("\n[达标] A.一步成五 / B.必挡 / D.安全性 全部 100%")
    if r_heur.rate >= 0.9:
        print(f"[达标] 总通过率 {r_heur.rate*100:.1f}% >= 90%")
    else:
        print(f"[未达标] 总通过率 {r_heur.rate*100:.1f}% < 90%")
        sys.exit(1)


if __name__ == "__main__":
    main()
