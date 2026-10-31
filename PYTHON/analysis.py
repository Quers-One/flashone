"""
analysis.py
===========
**分析模式**（KataGo Analysis 风格）的后端：把一个局面的每个候选点摊开成
"访问次数 / 胜率 / 先验"，以及一条整盘胜率曲线。

只读层
------
本模块**只读**棋盘、只**新建** MCTS 实例，绝不修改
`mcts.py` / `heuristics.py` / `gomoku_env.py`（改了 = 换引擎 = 历史 Elo 全废，
引擎指纹 `engine_mcts = sha1(mcts.py)[:8] = aabfbd47` 必须保持不变）。
需要的三个数据源都在现成接口里：

* `MCTS._root.children[move]` → `.n` 访问次数、`.w`/`.q` 价值、`.prior` 融合先验；
* `net.policy_value(state)`    → 网络原始策略（"网络直觉"）与价值头；
* `Board.check_win_after`      → 一眼成五 / 必挡标注。

⚠️ 两个极易搞错的**视角**问题（本模块用 `mover_win_from_child_q` 统一处理）
--------------------------------------------------------------------------------
1. `child.q` 是**落子之后轮到的那一方**（= 对手）视角的价值，不是落子方视角。
   想要"我下这一手的胜率"必须取反：`(1 - child.q) / 2`。
   （根节点 `root.q` 相反，它就是从"当前执子方"视角，直接 `(q+1)/2`。）
2. 网络价值头 `value` 是"**当前执子方**胜率"，同一局面换手就换视角。
   逐手曲线要跨手比较，必须统一折算到**黑方胜率**再画。

坐标约定（与 `web_play.py` 的 P0 修复保持一致）
-----------------------------------------------
`Board.board` 是 numpy 数组，下标是 **[行=y, 列=x]**；本模块对外一律用
`(x=列, y=行)`，密集下标 `y * 19 + x`。前端表格第 0 行是 y=18。

自检
----
    python analysis.py            # 6 项，含"把转置 bug 塞回去必须 FAIL"
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:                                        # 与 web_play.py 同口径：控制台 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np

from gomoku_env import Board
from mcts import MCTS

SIZE = 19
N_CELLS = SIZE * SIZE                      # 361
# 分析模式下让**根节点展开全部合法点**（默认搜索只展开 16 个，会让很多点
# 拿不到访问次数）。这是 `mcts.py` 的构造参数，不是改引擎。
ANALYZE_MAX_CHILDREN = N_CELLS
BAD_MOVE_LOSS = 0.12                       # 胜率单步跌幅超过它 = 疑似坏棋


# ----------------------------------------------------------------- 视角折算

def win_from_value(v: float) -> float:
    """价值头（**当前执子方**视角，[-1,1]）→ 该方胜率 [0,1]。"""
    return (float(v) + 1.0) / 2.0


def mover_win_from_child_q(q: float) -> float:
    """子节点 q（**落子后轮到的那一方**视角）→ **落子方**胜率。

    这是本模块最容易写反的一处：写成 `(q+1)/2` 会把"必胜的一手"显示成 0%
    （自检 ③ 专门盯它）。
    """
    return (1.0 - float(q)) / 2.0


def win_for_player(black_win: float, player: int) -> float:
    """把"黑方胜率"折算成指定一方的胜率。"""
    return float(black_win) if int(player) == 1 else 1.0 - float(black_win)


def black_win_of(board, value: float) -> float:
    """价值头 → **黑方胜率**（跨手可比的口径，见模块头注释第 2 条）。"""
    return win_from_value(value) if board.current_player == 1 else 1.0 - win_from_value(value)


# ----------------------------------------------------------------- 单局面分析

def analyze_board(board, net, n_sim: int = 800,
                  heuristic_weight: float = 0.5, policy_weight: float = 1.0,
                  full_root: bool = True,
                  root_max_children: Optional[int] = None,
                  top_k: int = 10,
                  mcts=None, net_pol=None, net_val=None,
                  elapsed_override: Optional[float] = None) -> Dict:
    """对一个局面做完整分析，**不修改 board**。

    参数:
        board   : `gomoku_env.Board`（只读）
        net     : `network.GomokuNet`（只需 policy_value / evaluate_batch）
        n_sim   : 模拟次数
        full_root: True 时根节点展开全部合法点（分析模式要的）；False = 对局口径
        mcts    : **已有搜索树**（持续搜索用）。传了就不新建、不清空 ——
                  于是访问次数会在同一棵树上**只增不减**，这正是 KataGo 分析模式的手感。
        net_pol / net_val : 网络先验与价值头（持续搜索时整局只算一次，传进来复用）
        elapsed_override  : 覆盖 `elapsed`（持续搜索要报"从开始到现在"的总时长）

    返回:
        dict —— `points` 是**密集 361 格**的记录列表（下标 i = y*19 + x），
        一次请求拿全，前端不用逐点再问。
    """
    t0 = time.time()
    mover = int(board.current_player)
    over, winner = board.is_game_over()
    legal = [(int(x), int(y)) for (x, y) in board.get_legal_moves()]
    legal_set = set(legal)

    # ---- 网络直觉：整盘策略 + 价值头（一次前向；持续搜索由调用方复用）----
    if net_pol is None or net_val is None:
        net_pol, net_val = net.policy_value(board.get_state_tensor())
    net_pol = np.asarray(net_pol, dtype=np.float64).reshape(-1)

    # ---- 搜索结论：新建一个 MCTS 实例，绝不碰对局那棵树 ----
    children: Dict[Tuple[int, int], object] = {}
    root_win: Optional[float] = None
    root_n = 0
    root_children = 0
    stats: Dict[str, float] = {}
    root = None
    if not over and legal and int(n_sim) > 0:
        m = mcts
        if m is None:                              # 一次性：新建一棵搜索树
            width = (root_max_children if root_max_children is not None
                     else (ANALYZE_MAX_CHILDREN if full_root else 16))
            m = MCTS(net, n_sim=int(n_sim), c_puct=1.0, virtual_loss=3.0,
                     batch_size=16, max_children=6, root_max_children=int(width),
                     heuristic_weight=float(heuristic_weight),
                     policy_weight=float(policy_weight))
            fresh = True
        else:
            fresh = False                          # 复用：继续在同一棵树上加模拟
        m.get_action_probs(board, temp=0, n_sim=int(n_sim))
        root = getattr(m, "_root", None)
        if root is not None:
            root_win = win_from_value(root.q)          # 根 q = 当前执子方视角
            root_n = int(root.n)
            root_children = len(root.children)
            for mv, ch in root.children.items():
                children[(int(mv[0]), int(mv[1]))] = ch
            if fresh:
                # collect_stats 要遍历**整棵树**；持续搜索里树会长到几万节点，
                # 每批都算一遍纯属浪费（而且这个信息只对一次性分析有意义）。
                try:
                    stats = m.collect_stats(root)
                except Exception:                      # noqa: BLE001
                    stats = {}

    # ---- 逐格组装（密集 361，含非法/已占点）----
    points: List[Dict] = []
    for y in range(SIZE):
        for x in range(SIZE):
            occ = int(board.board[y, x])               # [行=y, 列=x]
            rec = {"x": x, "y": y, "occ": occ, "legal": (x, y) in legal_set,
                   "n": 0, "win": None, "prior": None,
                   "netp": round(float(net_pol[y * SIZE + x]), 5),
                   "w5": False, "b5": False, "terminal": False}
            if rec["legal"]:
                mover_now = (not over) and mover == board.current_player
                if mover_now:
                    rec["w5"] = bool(board.check_win_after(x, y, mover))
                    if not rec["w5"]:
                        rec["b5"] = bool(board.check_win_after(x, y, 3 - mover))
                ch = children.get((x, y))
                if ch is not None:
                    rec["prior"] = round(float(ch.prior), 5)
                    rec["terminal"] = bool(ch.is_terminal)
                    if ch.n > 0:
                        rec["n"] = int(ch.n)
                        rec["win"] = round(mover_win_from_child_q(ch.w / ch.n), 4)
            points.append(rec)

    # ---- 候选排行（按访问次数，与 get_best_action 同口径）----
    ranked = [p for p in points if p["legal"] and p["n"] > 0]
    ranked.sort(key=lambda r: (-r["n"], -(r["win"] or 0.0)))
    cands = []
    for p in ranked[:max(1, int(top_k))]:
        cands.append({"x": p["x"], "y": p["y"], "n": p["n"], "win": p["win"],
                      "prior": p["prior"], "netp": p["netp"],
                      "w5": p["w5"], "b5": p["b5"]})
    best = None
    if ranked:
        b = ranked[0]
        best = {"x": b["x"], "y": b["y"], "n": b["n"], "win": b["win"]}
        for c in cands:                                # Δ = 相对最佳点的胜率差
            c["dwin"] = None if (c["win"] is None or b["win"] is None) else round(c["win"] - b["win"], 4)
    by_win = max((p for p in ranked if p["win"] is not None),
                 key=lambda r: r["win"], default=None)

    return {
        "ply": len(board.history),
        "mover": mover,
        "over": bool(over),
        "winner": int(winner) if over else None,
        "legal_count": len(legal),
        "n_sim": int(n_sim),
        "full_root": bool(full_root),
        "root_n": root_n,
        "root_children": root_children,
        "root_win": None if root_win is None else round(root_win, 4),
        "net_win": round(win_from_value(net_val), 4),
        "best": best,
        "best_by_win": None if by_win is None else {
            "x": by_win["x"], "y": by_win["y"], "n": by_win["n"], "win": by_win["win"]},
        "cands": cands,
        "points": points,
        "tree": {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))},
        "elapsed": (round(float(elapsed_override), 3) if elapsed_override is not None
                    else round(time.time() - t0, 3)),
    }


# ----------------------------------------------------------------- 持续搜索
# KataGo 分析模式的手感：**搜索一直在跑**，访问次数只增不减，胜率随之收敛，
# 界面上数字一直在动。做法只有一句话 —— 在**同一棵搜索树**上不停追加模拟。

class ContinuousAnalyzer:
    """后台线程持续加深同一棵搜索树，前端轮询最新快照。

    【为什么不是"每批新建一棵树"】那样访问次数会每批清零，看起来在动、其实没有收敛。
    自检⑦专门盯这一点：同一局面的 `total` 必须**单调不减**且整体在涨。

    【线程模型】只有本对象的后台线程碰它自己的 MCTS；外界通过 `snapshot()` 拿
    加锁后的只读副本。搜索树永远是"当前局面"的（局面变了就 `start()` 新的）。
    `stop()` 在一个 batch 之内返回（batch=400 实测 ≈20 ms），所以点"停止"是即时的。

    【为什么要有 pause】本机分析极快（800 模拟 0.04 s ≈ 2 万次/秒）。
    全力跑会把 CPU/GPU 吃满、页面数字快到看不清；默认 0.12 s 一批，
    约 3 千次/秒、每秒刷新约 8 次 —— 既有"一直在搜"的手感，又留得住机器。
    """

    def __init__(self, net=None, hw: float = 0.5, pw: float = 1.0):
        self.net = net
        self.hw = float(hw)
        self.pw = float(pw)
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snap: Optional[Dict] = None
        self._meta: Dict = {"running": False, "total": 0, "batch": 0,
                            "tps": 0.0, "started": 0.0, "max_sims": 0,
                            "full_root": True, "ply": 0}

    # ---- 控制 ----
    def start(self, board, batch: int = 400, full_root: bool = True,
              max_sims: int = 0, pause: float = 0.12, top_k: int = 10) -> Dict:
        """在 `board`（会深拷贝，之后不读外界棋盘）上开始持续搜索。"""
        self.stop()
        b = board.copy()
        with self._lock:
            self._snap = None
            self._meta = {"running": True, "total": 0, "batch": int(batch),
                          "tps": 0.0, "started": time.time(),
                          "max_sims": int(max_sims), "full_root": bool(full_root),
                          "ply": len(b.history)}
        self._stop.clear()
        self._th = threading.Thread(
            target=self._loop,
            args=(b, int(batch), bool(full_root), int(max_sims), float(pause), int(top_k)),
            name="continuous-analysis", daemon=True)
        self._th.start()
        # 等第一份结果再返回：否则调用方拿到的是"空浮层"，前端第一屏会闪一下空的。
        # 正常几十毫秒就好；最多等 2 秒（网络/模型异常时也不至于把接口卡死）。
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._lock:
                if self._snap is not None:
                    break
            time.sleep(0.01)
        return self.snapshot()

    def stop(self, timeout: float = 5.0) -> bool:
        th, self._th = self._th, None
        self._stop.set()
        if th is not None and th.is_alive():
            th.join(timeout=timeout)
        with self._lock:
            self._meta["running"] = False
        return True

    def running(self) -> bool:
        with self._lock:
            return bool(self._meta.get("running"))

    def snapshot(self) -> Dict:
        """最新结果（含 running/total/tps）。还没出第一份结果时返回空点集。"""
        with self._lock:
            meta = dict(self._meta)
            snap = dict(self._snap) if self._snap else None
        if snap is None:
            return {"running": bool(meta.get("running")), "total": int(meta.get("total", 0)),
                    "batch": int(meta.get("batch", 0)), "tps": float(meta.get("tps", 0.0)),
                    "ply": int(meta.get("ply", 0)), "points": [], "cands": [],
                    "elapsed": 0.0, "root_n": 0, "legal_count": 0}
        snap.update({"running": bool(meta.get("running")), "total": int(meta.get("total", 0)),
                     "batch": int(meta.get("batch", 0)), "tps": float(meta.get("tps", 0.0)),
                     "max_sims": int(meta.get("max_sims", 0)),
                     "full_root": bool(meta.get("full_root", True))})
        return snap

    # ---- 后台循环（每批 = 在同一棵树上追加 batch 次模拟）----
    def _loop(self, board, batch, full_root, max_sims, pause, top_k) -> None:
        try:
            net_pol, net_val = self.net.policy_value(board.get_state_tensor())
            net_pol = np.asarray(net_pol, dtype=np.float64).reshape(-1)
            width = ANALYZE_MAX_CHILDREN if full_root else 16
            m = MCTS(self.net, n_sim=batch, c_puct=1.0, virtual_loss=3.0,
                     batch_size=16, max_children=6, root_max_children=int(width),
                     heuristic_weight=self.hw, policy_weight=self.pw)
            t0 = time.time()
            while not self._stop.is_set():
                snap = analyze_board(board, self.net, n_sim=batch,
                                     heuristic_weight=self.hw, policy_weight=self.pw,
                                     full_root=full_root, top_k=top_k,
                                     mcts=m, net_pol=net_pol, net_val=net_val,
                                     elapsed_override=time.time() - t0)
                total = int(snap.get("root_n") or 0)
                with self._lock:
                    self._snap = snap
                    self._meta["total"] = total
                    self._meta["tps"] = round(total / max(1e-6, time.time() - t0), 1)
                if max_sims and total >= max_sims:
                    break
                if pause > 0:
                    self._stop.wait(pause)
        except Exception as exc:                       # noqa: BLE001
            with self._lock:
                self._meta["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._meta["running"] = False


# ----------------------------------------------------------------- 整盘胜率曲线

def winrate_curve(moves: Sequence[Sequence[int]], net, n_sim: int = 0,
                  heuristic_weight: float = 0.5, policy_weight: float = 1.0) -> Dict:
    """按逐手记录重放，给出**每一手落子前**的胜率与这一手造成的跌幅。

    参数:
        moves : [[x, y, player], ...]（`web_play.Game.moves` 的格式）
        n_sim : 0 = 只用网络价值头（快，整盘 1~2 秒）；>0 = 每个局面跑一次浅搜索
                （慢，局数 × n_sim 次模拟；根只展开 16 个点 = 对局口径）

    返回:
        dict：
            points : 每个局面一个点，含 black_win（**统一黑方视角**）、
                     mover / mover_win（该手落子方的胜率）、loss（这一手丢了多少）
            worst  : loss 排序的疑似坏棋（loss > BAD_MOVE_LOSS）
            mode   : "net" 或 "mcts(n)"
    """
    board = Board()
    pts: List[Dict] = []
    use_mcts = int(n_sim) > 0
    engine = MCTS(net, n_sim=int(n_sim), c_puct=1.0, virtual_loss=3.0,
                  batch_size=16, max_children=6, root_max_children=16,
                  heuristic_weight=float(heuristic_weight),
                  policy_weight=float(policy_weight)) if use_mcts else None

    def eval_pos(b) -> float:
        """返回该局面的**黑方胜率**。"""
        if engine is None:
            _, v = net.policy_value(b.get_state_tensor())
            return black_win_of(b, v)
        engine.reset_tree()
        engine.get_action_probs(b, temp=0, n_sim=int(n_sim))
        root = getattr(engine, "_root", None)
        if root is None or root.n <= 0:
            _, v = net.policy_value(b.get_state_tensor())
            return black_win_of(b, v)
        cur = win_from_value(root.q)                  # 当前执子方视角
        return cur if b.current_player == 1 else 1.0 - cur

    # 位置 0..L：位置 i = 走完前 i 手
    wins: List[float] = [eval_pos(board)]
    for step, (x, y, p) in enumerate(moves):
        if not board.move(int(x), int(y), int(p)):
            raise ValueError(f"第 {step + 1} 手 ({x},{y}) 无法重放（记录与规则不符）")
        wins.append(eval_pos(board))

    for i, (x, y, p) in enumerate(moves):
        before, after = wins[i], wins[i + 1]
        mover_win = win_for_player(before, p)
        mover_after = win_for_player(after, p)
        pts.append({"ply": i, "x": int(x), "y": int(y), "player": int(p),
                    "black_win": round(before, 4),
                    "mover_win": round(mover_win, 4),
                    "mover_win_after": round(mover_after, 4),
                    "loss": round(mover_win - mover_after, 4),
                    "black_win_after": round(after, 4)})

    worst = sorted([q for q in pts if q["loss"] > BAD_MOVE_LOSS],
                   key=lambda r: -r["loss"])
    return {"points": pts, "worst": worst, "final_black_win": round(wins[-1], 4),
            "mode": f"mcts({int(n_sim)})" if use_mcts else "net",
            "threshold": BAD_MOVE_LOSS}


# ----------------------------------------------------------------- 自检

class _StubNet:
    """零依赖假网络：均匀策略 + 常数价值。

    用它跑分析，搜索的行为就**完全由规则（成五/必挡）决定**，
    于是自检的期望值是确定的，不受"模型强不强"的影响。
    """

    def __init__(self, value: float = 0.0):
        self.value_const = float(value)
        self.calls: List[float] = []

    def policy_value(self, state):                     # noqa: ARG002
        self.calls.append(self.value_const)
        return np.full(N_CELLS, 1.0 / N_CELLS, dtype=np.float64), self.value_const

    def evaluate_batch(self, states):
        b = int(np.asarray(states).shape[0])
        self.calls.append(self.value_const)
        return (np.full((b, N_CELLS), 1.0 / N_CELLS, dtype=np.float64),
                np.full(b, self.value_const, dtype=np.float64))


def _mk_board(stones: Sequence[Tuple[int, int, int]]) -> Board:
    """手摆一个局面（允许不按黑先白后，直接指定落子方）。"""
    b = Board()
    for (x, y, p) in stones:
        if b.current_player != p:
            b.current_player = p            # 手摆局面：直接指定执子方
        if not b.move(x, y, p):
            raise RuntimeError(f"自检夹具落子失败 ({x},{y},{p})")
    return b


def _probe_a_board() -> Board:
    """黑 (0..3, 5) 四连、白 (0..2, 10) 三连 ⇒ **黑走 (4,5) 立刻成五**。"""
    return _mk_board([(0, 5, 1), (1, 5, 1), (2, 5, 1), (3, 5, 1),
                      (0, 10, 2), (1, 10, 2), (2, 10, 2)])


def _probe_b_board() -> Board:
    """白 (5..8, 8) 四连、两端皆空 ⇒ 黑**怎么挡都输**（活的四）。"""
    b = _mk_board([(5, 8, 2), (6, 8, 2), (7, 8, 2), (8, 8, 2),
                   (0, 10, 1), (1, 10, 1), (2, 10, 1)])
    if b.current_player != 1:
        b.current_player = 1                 # 要"黑方走"（必败方）
    return b


def _check_probe_a(res: Dict) -> List[str]:
    """① 探针 A 的断言集合。反向验证复用同一函数（喂转置结果给它）。"""
    fails: List[str] = []
    P = {(p["x"], p["y"]): p for p in res["points"]}
    win_cell = P.get((4, 5))
    tran_cell = P.get((5, 4))
    if win_cell is None or tran_cell is None:
        return ["分析结果缺 (4,5) / (5,4) —— 返回结构不对"]
    if not win_cell["w5"]:
        fails.append("(4,5) 应被标为『一步成五』(w5) 却没标")
    if tran_cell["w5"]:
        fails.append("转置点 (5,4) 被误标为成五点（朝向反了）")
    if res["best"] is None or (res["best"]["x"], res["best"]["y"]) != (4, 5):
        fails.append(f"最佳点应为 (4,5)，实际 {res['best']}")
    elif not (win_cell["win"] is not None and win_cell["win"] > 0.95):
        fails.append(f"(4,5) 是必胜手，胜率却只有 {win_cell['win']}（视角可能取反了）")
    if res["root_win"] is None or res["root_win"] < 0.95:
        fails.append(f"必胜局面的根胜率应接近 1，实际 {res['root_win']}")
    if res["legal_count"] <= 0 or len(res["points"]) != N_CELLS:
        fails.append(f"points 应有 {N_CELLS} 格，实际 {len(res['points'])}")
    return fails


def _check_probe_b(res: Dict) -> List[str]:
    """② 必败局面：所有被搜索到的点胜率都必须很低，且根胜率 ≈ 0。"""
    fails: List[str] = []
    P = {(p["x"], p["y"]): p for p in res["points"]}
    if not P[(4, 8)]["b5"] or not P[(9, 8)]["b5"]:
        fails.append("(4,8)/(9,8) 应被标为『必挡』(b5)")
    if P[(4, 8)]["w5"]:
        fails.append("(4,8) 是白方的成五点，不该标成黑方 w5")
    if P[(8, 4)]["b5"]:
        fails.append("转置点 (8,4) 被误标为必挡点（朝向反了）")
    if res["root_win"] is None or res["root_win"] > 0.05:
        fails.append(f"必败局面的根胜率应≈0，实际 {res['root_win']}")
    hot = [p for p in res["points"] if p["legal"] and p["n"] > 0]
    if not hot:
        fails.append("没有任何点拿到访问次数（搜索没跑起来）")
    for p in hot:
        if p["win"] is not None and p["win"] > 0.2:
            fails.append(f"必败局面里 ({p['x']},{p['y']}) 的胜率 {p['win']} 偏高")
    return fails


def _transpose_result(res: Dict) -> Dict:
    """把分析结果里的 (x,y) 全部对调 —— 复现"沿主对角线镜像"的旧 bug。"""
    out = dict(res)
    out["points"] = [dict(p, x=p["y"], y=p["x"]) for p in res["points"]]
    if res.get("best"):
        out["best"] = dict(res["best"], x=res["best"]["y"], y=res["best"]["x"])
    return out


def _gate(name: str, fails: List[str], extra: str = "") -> bool:
    ok = not fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('　' + extra) if extra else ''}")
    for f in fails:
        print(f"        · {f}")
    return ok


def _continuous_series_problems(totals: List[int]) -> List[str]:
    """判定一串"持续搜索访问次数"快照是否真的是**同一棵树在长**。

    抽成函数是为了能反向验证：喂进"每批重建树"（恒定）或"回退"的假序列必须报错，
    否则门禁⑦就是假的绿灯。
    """
    if len(totals) < 2:
        return [f"快照太少（{len(totals)} 份）"]
    probs = []
    if totals != sorted(totals):
        probs.append(f"访问次数出现回退 ⇒ 树被重建：{totals[:8]}")
    if totals[-1] <= totals[0]:
        probs.append(f"访问次数没增长（{totals[0]} → {totals[-1]}）⇒ 每批都在新建树")
    return probs


def self_test(verbose: bool = True) -> int:
    """6 项自检。**把 bug 塞回去必须 FAIL**（③④ 就是干这个的）。"""
    print("=" * 76)
    print("分析模式自检（analysis.py）")
    print("=" * 76)
    n_fail = 0

    # ---- ① 朝向 + 必胜手 + 视角 ----
    ra = analyze_board(_probe_a_board(), _StubNet(0.0), n_sim=400)
    f1 = _check_probe_a(ra)
    n_fail += 0 if _gate("① 全盘评估：必胜手 (4,5) 被找到且标为 w5（朝向正确）", f1,
                         f"最佳={ra['best']}　用时 {ra['elapsed']}s") else 1

    # ---- ② 必败局面 + 必挡标注 ----
    rb = analyze_board(_probe_b_board(), _StubNet(0.0), n_sim=400)
    f2 = _check_probe_b(rb)
    n_fail += 0 if _gate("② 必败局面：根胜率≈0，(4,8)/(9,8) 标为必挡", f2,
                         f"根胜率={rb['root_win']}") else 1

    # ---- ③ 视角：胜率的正负号必须站在"落子方" ----
    f3: List[str] = []
    P = {(p["x"], p["y"]): p for p in ra["points"]}
    if P[(4, 5)]["win"] is None or abs(P[(4, 5)]["win"] - 1.0) > 1e-6:
        f3.append(f"必胜手的胜率应为 1.0，实际 {P[(4, 5)]['win']}")
    if abs(mover_win_from_child_q(-1.0) - 1.0) > 1e-9:
        f3.append("mover_win_from_child_q(-1) 应为 1.0（子节点 q 是对手视角）")
    if abs(mover_win_from_child_q(1.0) - 0.0) > 1e-9:
        f3.append("mover_win_from_child_q(+1) 应为 0.0")
    if abs(win_for_player(0.7, 2) - 0.3) > 1e-9:
        f3.append("win_for_player 的白方折算错了")
    n_fail += 0 if _gate("③ 视角折算：child.q 是对手视角，落子方胜率必须取反", f3) else 1

    # ---- ④ 反向验证：朝向左移一格 → 上面 ①② 必须 FAIL ----
    inv_ok = True
    for tag, res, chk in (("①", ra, _check_probe_a), ("②", rb, _check_probe_b)):
        caught = bool(chk(_transpose_result(res)))
        if verbose:
            print(f"        {'[OK] 转置被抓到' if caught else '[X] 转置没被抓到'}：{tag}")
        inv_ok = inv_ok and caught
    n_fail += 0 if _gate("④ 反向验证：把 (x,y) 对调后 ①② 必须 FAIL", [] if inv_ok else
                         ["转置后的结果仍然通过了断言 —— 门禁是假的"]) else 1

    # ---- ⑤ 曲线：长度 / 黑方视角折算 / 跌幅方向 ----
    f5: List[str] = []
    moves = [[9, 9, 1], [10, 10, 2], [9, 10, 1]]
    cv = winrate_curve(moves, _StubNet(1.0), n_sim=0)      # 常数价值 1.0
    if len(cv["points"]) != len(moves):
        f5.append(f"曲线应有 {len(moves)} 个点，实际 {len(cv['points'])}")
    else:
        # 价值恒为 +1 ⇒ "轮到谁谁必胜"：黑走时黑方胜率 1，白走时黑方胜率 0
        want_bw = [1.0, 0.0, 1.0]
        got_bw = [p["black_win"] for p in cv["points"]]
        if got_bw != want_bw:
            f5.append(f"黑方视角折算错了：期望 {want_bw}，实际 {got_bw}")
        if any(p["loss"] <= 0.5 for p in cv["points"]):
            f5.append("跌幅方向反了（把必胜手交出去应为正跌幅）")
        if any(p["mover_win"] != 1.0 for p in cv["points"]):
            f5.append("落子方胜率应为 1.0")
    cv2 = winrate_curve(moves, _StubNet(0.0), n_sim=60)     # 真跑一遍搜索
    if cv2["mode"] != "mcts(60)" or len(cv2["points"]) != len(moves):
        f5.append(f"MCTS 模式曲线不对：{cv2['mode']} / {len(cv2['points'])} 点")
    elif any(not (0.0 <= p["black_win"] <= 1.0) for p in cv2["points"]):
        f5.append("MCTS 模式胜率越界")
    n_fail += 0 if _gate("⑤ 胜率曲线：逐手重放、统一黑方视角、跌幅归因正确", f5,
                         f"模式 {cv2['mode']}") else 1

    # ---- ⑥ 密集结构与合法性一致 ----
    f6: List[str] = []
    for res, tag in ((ra, "A"), (rb, "B")):
        idx_ok = all(p["x"] == i % SIZE and p["y"] == i // SIZE
                     for i, p in enumerate(res["points"]))
        if not idx_ok:
            f6.append(f"[{tag}] points 的下标不是 y*19+x（前端浮层会对不上格）")
        occ = sum(1 for p in res["points"] if p["occ"])
        legal_legal = sum(1 for p in res["points"] if p["legal"])
        if occ + legal_legal > N_CELLS:
            f6.append(f"[{tag}] 已占点与合法点数量矛盾（{occ}+{legal_legal} > {N_CELLS}）")
        if any(p["legal"] and p["occ"] for p in res["points"]):
            f6.append(f"[{tag}] 已占点被标成了合法点")
        if res["root_n"] <= 0:
            f6.append(f"[{tag}] root_n={res['root_n']}（搜索没记账）")
    n_fail += 0 if _gate("⑥ 密集 361 格：下标=y*19+x，合法/已占不矛盾", f6) else 1

    # ---- ⑦ 持续搜索：同一棵树里访问次数只增不减；stop 之后真的停 ----
    f7: List[str] = []
    ana = ContinuousAnalyzer(_StubNet(0.0), hw=0.5, pw=1.0)
    ana.start(_probe_a_board(), batch=100, pause=0.01)
    totals: List[int] = []
    t0 = time.time()
    while time.time() - t0 < 6.0 and len(totals) < 25:
        s = ana.snapshot()
        if s.get("points"):
            totals.append(int(s.get("total") or 0))
        time.sleep(0.04)
    stopped_snap = ana.snapshot()
    ana.stop()
    if len(totals) < 2:
        f7.append(f"持续搜索没产出足够快照（只拿到 {len(totals)} 份）")
    else:
        f7.extend(_continuous_series_problems(totals))
    # 反向验证：喂"每批重建树"（恒定）与"回退"的假序列，判定必须报错
    fake_flat = [100, 100, 100, 100]
    fake_back = [500, 300, 100, 50]
    for tag, series in (("每批重建树", fake_flat), ("访问回退", fake_back)):
        if not _continuous_series_problems(series):
            f7.append(f"反向验证失败：假的『{tag}』序列 {series} 被判成正常 ⇒ 判定是假的")
        elif verbose:
            print(f"        [OK] 被抓到：{tag} {series}")
    after = int(ana.snapshot().get("total") or 0)
    time.sleep(0.25)
    if int(ana.snapshot().get("total") or 0) != after:
        f7.append("stop() 之后访问次数还在涨 ⇒ 后台线程没停住")
    if ana.running():
        f7.append("stop() 之后 running 仍为 True")
    if not stopped_snap.get("running"):
        f7.append("快照里的 running 字段没反映'正在搜索'")
    if len(stopped_snap.get("points") or []) != N_CELLS:
        f7.append("持续模式的快照不是 361 点")
    if (stopped_snap.get("tps") or 0) <= 0:
        f7.append("快照没报增速 tps")
    n_fail += 0 if _gate("⑦ 持续搜索：同一棵树上访问只增不减，stop 即刻生效", f7,
                         f"{totals[0] if totals else 0} → {totals[-1] if totals else 0} 次访问，"
                         f"{stopped_snap.get('tps')} 次/秒") else 1

    print("-" * 76)
    if n_fail:
        print(f">>> 【自检未过】{n_fail} 项")
        return n_fail
    print(">>> 【自检通过】分析层朝向、视角、曲线归因正确，且转置 bug 能被抓到")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="分析模式后端自检")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return self_test(verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
