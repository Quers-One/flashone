"""
heuristics.py
=============
五子棋「棋型规则先验」模块（优化 阶段 1）。

为什么需要它
------------
纯随机初始网络 + 弱搜索在 19×19 五子棋上学不到任何东西：策略先验是噪声，
价值头也是噪声，自对弈只能产出「241 手全平局」这种无信息对局。
本模块把五子棋的确定性短程战术（成五、必挡、冲四、活三）以规则形式
注入先验，让引擎在**不训练**的情况下就具备基本战术能力，使后续 RL
有意义的对局（30~60 手、有胜负）作为起点。

设计要点
--------
* **绝不被网络覆盖**：出现「一步成五」或「对方一步成五」时，只保留这些
  强制点，网络先验被完全忽略（forced win / forced block）。
* **其余情况温和融合**：强制点之外的棋型分与网络概率按权重混合，
  保留网络的学习空间，不把网络变成纯查表。
* 纯 numpy 向量化取值，单次评估约几十微秒，相对网络前向可忽略。

对外接口
--------
    move_scores(board, legal=None, me=None) -> (raw, wins_idx, blocks_idx)
    compute_prior(board, legal, net_probs=None, ...) -> (priors, forced_idx)

性能说明（第二版 —— 实测后的重构）
---------------------------------
第一版的 `_line_stats` 用 numpy 逐格索引（`arr[yy, xx]`，约 150ns/次）+
逐格越界判断，并靠"临时落子 -> 扫描 -> 还原"来试探棋型。cProfile 显示
`heuristics` 占了搜索 73% 的时间，其中 `_line_stats` 是最大的一块。

第二版改成读 `Board._pad`（带 4 格围墙的 27×27 纯 Python list 镜像）：
    * 取值变成 list 下标（约 30ns），越界判断完全消失
    * 不再需要"临时落子/还原"（填充棋盘只读就能问"若我落这里会怎样"）
    * 一次方向扫描**同时**算出我方与对方的棋型（原来要扫两遍）
于是"成五判定"也被合并进来了：`count >= 5` 就是成五点，
不必再单独调 `check_win_after`（原实现每个候选点要额外调 2 次）。
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from gomoku_env import SIZE

# 方向：横、竖、正斜、反斜
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))

# ----------------------------------------------------------------------
# 关于"连子查询表"的一段历史（保留说明，避免后人重复踩坑）
# ----------------------------------------------------------------------
# 为了给棋型扫描提速，曾经在这里预计算过一张
#     LINE_INDEX[方向, y, x, 9]  —— 每条线的 9 个格子索引（越界指向哨兵槽）
#     LINE_OUT[方向, y, x, 9]    —— 对应的"越界"掩码
# 的索引表，用来把逐格循环换成 numpy 花式索引。
# 结论：**这是负优化，已删除**。原因：9 个元素的数组上 numpy 的每次调用开销
# 远大于 9 次 Python 比较，实测反而慢 3 倍。
# 真正有效的做法是 gomoku_env 里的**填充棋盘**（`Board._pad`）：
# 用"围墙值 -1"替代越界判断，同时把逐格取值从 numpy 换成 list 下标。


# ======================================================================
# 威胁分级基准分（数值必须**量级分离**，这是关键设计）
# ======================================================================
# 为什么不能用连续棋型分（如 off_me + 0.9*off_op）：
#   实测过一个局面：白方必须堵黑方的四连，但白方自己也有一个"活三+好形"的点。
#   连续打分给出的分数是 60007 vs 54008 —— 既然只差 10%，
#   搜索就会把访问次数摊到那个"看起来不错"的点上，最后漏挡。
#   把"能否形成四连（冲四/活四）"与"能否形成活三"做成 **10 倍量级** 差距后，
#   搜索会优先把模拟预算花在真正的威胁点上。
TACTICAL_OPEN_FOUR = 1_000_000   # 我能做活四（严格必胜）
TACTICAL_FOUR = 100_000          # 我能做冲四（单点成五威胁）
TACTICAL_BLOCK_FOUR = 10_000     # 我能阻止对方形成四连（含堵活三的端点）
TACTICAL_OPEN_THREE = 1_000      # 我能做活三
TACTICAL_BLOCK_THREE = 500       # 我能阻止对方做活三
TACTICAL_TWO = 100               # 二连
TACTICAL_ONE = 4                 # 单子

# 强制点（必胜 / 必挡）的象征性分数，保证一定排在最高位
FORCED_WIN_SCORE = 1e9
FORCED_BLOCK_SCORE = 1e8

# 融合权重：网络概率的等效权重（加在归一化后的启发式分布上）
DEFAULT_HEURISTIC_WEIGHT = 1.0
DEFAULT_POLICY_WEIGHT = 1.0


def _pattern_score(count: int, open_ends: bool) -> float:
    """
    把"连子数 + 是否两端开放"映射为威胁分。

    量级分离（见文件顶部说明，这是避免"堵冲四的点"打分低于"做活三的点"的关键）：
        4 连两端开放 -> 活四（严格必胜）  1,000,000
        4 连单端     -> 冲四              100,000
        3 连两端开放 -> 活三                1,000
        3 连单端     -> 眠三                  500
        2 连                               100~200
        1 连                                    4
    """
    if count >= 5:
        return FORCED_WIN_SCORE
    if count == 4:
        return TACTICAL_OPEN_FOUR if open_ends else TACTICAL_FOUR
    if count == 3:
        return TACTICAL_OPEN_THREE if open_ends else TACTICAL_BLOCK_THREE
    if count == 2:
        return TACTICAL_TWO * (2 if open_ends else 1)
    return TACTICAL_ONE


def move_scores(board, legal: Optional[Sequence[Tuple[int, int]]] = None,
                me: Optional[int] = None):
    """
    计算所有合法点的启发式分数。

    打分语义（按优先级）：
        1) 我落这里直接成五             -> FORCED_WIN_SCORE（必胜）
        2) 对方落这里会成五（必须堵）    -> FORCED_BLOCK_SCORE（必挡）
        3) 我落这里能做活四             -> TACTICAL_OPEN_FOUR（严格必胜）
        4) 我落这里能做冲四             -> TACTICAL_FOUR
        5) 对方落这里能做活四/冲四       -> TACTICAL_BLOCK_FOUR（必须限制）
        6) 我做活三 / 堵对方活三         -> TACTICAL_OPEN_THREE / BLOCK_THREE
        7) 其余按一般棋型分

    参数:
        board: gomoku_env.Board 实例
        legal: 合法点序列；None 时用 board.get_legal_moves()
        me   : 打分视角（默认 board.current_player）

    返回:
        raw      : np.ndarray(len(legal),) 威胁综合分，与 legal 一一对应
        wins_idx : set[int]  当前执子方"一步成五"点在 legal 中的下标
        blocks_idx: set[int] 对方"一步成五"点（必须堵）的下标

    实现要点（性能）：
        * 每个候选点只调**一次** `board.line_after_both`，一次方向扫描同时得到
          我方与对方的 (连子数, 是否开放)；原实现要扫两遍 + 2 次 check_win_after。
        * 邻域预筛改成读填充棋盘的 8 个 list 单元（无 numpy 切片）。
        * 返回下标集合而不是 (x,y) 元组集合，调用方可以直接做数组赋值。
    """
    if legal is None:
        legal = board.get_legal_moves()
    if me is None:
        me = board.current_player

    n = len(legal)
    raw = np.empty(n, dtype=np.float64)
    wins: Set[int] = set()
    blocks: Set[int] = set()

    line_after_both = board.line_after_both
    has_neighbor = board.has_neighbor

    for k in range(n):
        x, y = legal[k]
        x = int(x)
        y = int(y)

        # ---- 0) 廉价预筛：8 邻域没有任何棋子 -> 最多形成"孤二"，给最低分
        if not has_neighbor(x, y):
            raw[k] = TACTICAL_ONE
            continue

        c_me, o_me, _b_me, c_op, o_op, _b_op = line_after_both(x, y, me)
        if c_me == 0:
            # 越界 / 已被占用（合法点里理论上不会出现）：保守给 0
            raw[k] = 0.0
            continue

        # ---- 1) 我落这里立刻成五 ----
        if c_me >= 5:
            raw[k] = FORCED_WIN_SCORE
            wins.add(k)
            continue

        # ---- 2) 对方落这里立刻成五（我必须堵）----
        if c_op >= 5:
            raw[k] = FORCED_BLOCK_SCORE
            blocks.add(k)
            continue

        # ---- 3) 常规：我方威胁 + 对方威胁的防守价值 ----
        # off_me: 我在此落子的威胁等级
        # def_op: 对方在此落子的威胁等级 == 我占住它的防守价值
        # 取两者较大者，并给次要项一点附加分（兼顾"攻守兼备"的点）。
        off_me = _pattern_score(c_me, o_me)
        def_op = _pattern_score(c_op, o_op)
        raw[k] = max(off_me, def_op * 0.8) + 0.1 * min(off_me, def_op)

    return raw, wins, blocks


def _has_neighbor_removed_note():
    """
    （占位说明）旧版 `_has_neighbor(arr, x, y)` 已删除。

    它用 numpy 的 3×3 切片做"邻域有没有子"的预筛；现在这个判断搬到了
    `gomoku_env.Board.has_neighbor(x, y)`，读填充棋盘的 8 个 list 单元，
    比 numpy 切片快得多（切片每次约 1~2µs，8 次 list 读约 0.2µs）。
    """
    raise NotImplementedError


def compute_prior(board, legal: Sequence[Tuple[int, int]],
                  net_probs: Optional[np.ndarray] = None,
                  heuristic_weight: float = DEFAULT_HEURISTIC_WEIGHT,
                  policy_weight: float = DEFAULT_POLICY_WEIGHT,
                  use_heuristics: bool = True):
    """
    计算合法点的先验分布，用于 MCTS 扩展。

    参数:
        board          : Board 实例
        legal          : 合法点列表，顺序与返回值一一对应
        net_probs      : 网络策略在 legal 上的概率（已归一化）；None 表示无网络
        heuristic_weight: 启发式分布权重
        policy_weight  : 网络分布权重
        use_heuristics : False 时退化为纯网络先验（用于对照实验 / 纯 RL 训练）

    返回:
        (priors, forced_idx)
            priors     : np.ndarray (len(legal),) 归一化先验；
                         出现强制点（成五/必挡）时只保留强制点
            forced_idx : set[int] 强制点在 legal 中的下标（= 成五点 ∪ 必挡点）。
                         调用方（MCTS 的宽度控制）需要它来保证这些点**永不被裁掉**。
                         顺带把这份信息返回，就省掉了调用方再算一遍成五
                         （原实现 `_forced_moves` 会对每个候选点再调 2 次
                          `check_win_after`，实测 20 手就要 143 万次）。

    规则:
        * 出现「一步成五」-> 只保留这些点，均匀分配（等价于必胜）
        * 否则出现「对方一步成五」-> 只保留这些堵点，均匀分配
        * 否则：启发式分 -> softmax 风格归一化 -> 与网络概率按权重线性融合
    """
    n = len(legal)
    if n == 0:
        return np.zeros(0, dtype=np.float64), set()

    if not use_heuristics:
        if net_probs is None:
            return np.full(n, 1.0 / n, dtype=np.float64), set()
        p = np.asarray(net_probs, dtype=np.float64).copy()
        s = p.sum()
        return (p / s if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)), set()

    raw, wins, blocks = move_scores(board, legal)
    # 宽度控制要保护的强制点是"成五点 ∪ 必挡点"（与旧 _forced_moves 语义一致）
    forced_union: Set[int] = wins | blocks

    # ---- 强制点：完全忽略网络 ----
    forced = wins if wins else blocks
    if forced:
        out = np.zeros(n, dtype=np.float64)
        out[list(forced)] = 1.0
        s = out.sum()
        if s > 0:
            return out / s, forced_union
        # 强制点为空（理论不应发生）-> 落到常规路径

    # ---- 常规：棋型分归一化 ----
    # 威胁分已按量级分离（1e6 / 1e5 / 1e4 / 1e3 ...），直接平方根压缩再归一化，
    # 既保留量级差异，又避免活四彻底压死活三导致搜索宽度归零。
    comp = np.sqrt(np.maximum(raw, 0.0))
    tot = comp.sum()
    heur = comp / tot if tot > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    if net_probs is None or policy_weight <= 0:
        return heur, forced_union

    pol = np.asarray(net_probs, dtype=np.float64).copy()
    pol = np.maximum(pol, 0.0)
    s = pol.sum()
    pol = pol / s if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    mixed = heuristic_weight * heur + policy_weight * pol
    s = mixed.sum()
    return (mixed / s if s > 0 else np.full(n, 1.0 / n, dtype=np.float64)), forced_union


def describe_move(board, x: int, y: int) -> str:
    """调试用：给出某点的棋型描述。"""
    me = board.current_player
    opp = 3 - me
    if board.check_win_after(x, y, me):
        return "成五(胜)"
    if board.check_win_after(x, y, opp):
        return "对方成五点(必挡)"
    c, o, _b = board.line_after(x, y, me)
    c2, o2, _b2 = board.line_after(x, y, opp)
    names = {5: "五", 4: "四", 3: "三", 2: "二", 1: "一"}
    return (f"我{names.get(c, c)}{'活' if o else '眠'} / "
            f"敌{names.get(c2, c2)}{'活' if o2 else '眠'}")
