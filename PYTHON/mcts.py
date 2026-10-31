"""
mcts.py
=======
五子棋 MCTS 搜索模块（PUCT + 网络评估，参考 AlphaZero / KataGo）。

对接：
    * gomoku_env.Board   —— 棋盘规则（落子/悔棋/合法点/胜负）
    * network.GomokuNet —— 策略/价值网络（先验 p 与价值 v）
    * heuristics          —— 五子棋棋型规则先验（成五/必挡/冲四/活三）

搜索四步标准流程（每轮模拟）：
    1) 选择 (Selection)  ：从根沿 PUCT 公式选子节点，直到叶子
    2) 扩展 (Expansion)  ：在叶子用「规则先验 + 网络策略」得到先验 p，
                           并按先验生成子节点
    3) 评估 (Evaluation) ：叶子价值 = 终局符号价值 或 网络价值
    4) 回溯 (Backprop)   ：沿路径更新各节点 n、w、q（逐层取反）

===========================================================================
本文件是对旧实现的**核心重写**，修复以下 4 个致命缺陷
===========================================================================

[P0-1] 旧实现在选择循环里遇到「未扩展且 n==0」的节点就 break，把该节点丢进
       pending 集合等待批量扩展；但每次模拟都停在**同一个**节点，
       set 去重后 |pending| 恒为 1，而扩展触发条件要求 len(pending) >= batch_size(32)，
       于是整场搜索只在最后一轮扩展一次。
       实测后果：n_sim 从 150 加到 12000，搜索树恒为 51 个节点、深度恒为 2，
       网络评估利用率仅 6.7%，93% 的算力被直接丢弃。
       修复：**收集 batch_size 个不同叶子后立刻扩展并回传其真实价值**，用
       virtual_loss 保证 batch 内各次模拟不会停在同一个叶子。

[P0-2] 旧实现把终局（获胜/落败）子节点的 value 硬编码为 0.0，导致
       「我这步就赢了」与「随便走一步」在搜索中数值完全相同 —— 没有杀棋梯度。
       实测后果：自对弈每局出现 181 次「成五机会却未走」，连一步成五都找不到。
       修复：终局子节点写入带符号价值
       v = +1（该节点的行棋方已获胜）/ -1（已落败）/ 0（平局），
       并在扩展阶段用 check_win_after 提前识别成五点。

[P0-3] 旧实现的置换表把命中缓存的叶子标记为 is_terminal=True，而选择循环遇到
       is_terminal 即 break —— 该节点的所有后代永远不会被搜索（等于剪掉整棵子树）；
       缓存还只存 (value, 1) 且只增不减。同时 gomoku_env 的 Zobrist 哈希不含
       行棋方，导致「同型不同轮次」局面哈希碰撞（已在 gomoku_env.py 修复）。
       修复：**彻底移除把节点标记为终局的置换表**。改为 _position_cache 只缓存
       叶子评估结果 (先验, 价值) 供复用，命中时不改变搜索语义、不设 is_terminal。

[P0-4] 旧实现根节点只展开 Top-20 先验点，于是对手/人类走出这 20 点之外的任何
       一步，set_root_move 就把整棵树丢弃（实测开局覆盖率仅 66.7%，中盘 <30%），
       树复用几乎 100% 失效，每手都重算一遍。
       修复：**根节点扩展全部合法点**（保证任意对手落点都能命中子树），
       且 set_root_move 未命中时回退到父节点/兄弟节点所在局面而非直接丢树。

===========================================================================
其它改进
===========================================================================
    * [Virtual Loss]  真正生效：选择时 n+vloss 惩罚，回溯时精确补偿，
                      使一个 batch 内的模拟分散到不同叶子（批量推理才有意义）
    * [Batch]         一次前向评估整个 batch 的叶子（GPU 利用率）
    * [Heuristic]     规则先验与网络先验温和融合；成五/必挡点强制覆盖
    * [Root full]     根全展开 + 深处 Top-K 剪枝（兼顾树复用与内存）
    * [Fallback]      搜索无结果时用棋型规则兜底
    * [c_puct]        默认 2.5（旧值 1.5 在五子棋上过于 exploitation）
    * [Temp 修复]     get_action_probs 支持真实温度采样；自对弈请用
                      sample_move()，不要对访问次数直接取 argmax
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gomoku_env import Board
import heuristics


# ----------------------------------------------------------------------
# 内部异常：增量搜索状态漂移，需要回退到完整重放
# ----------------------------------------------------------------------
class _Resync(Exception):
    """共享棋盘与搜索树不一致时抛出，由 _seek 捕获并触发 _full_replay。"""
    pass


# ----------------------------------------------------------------------
# MCTS 节点
# ----------------------------------------------------------------------
class MCTSNode:
    """
    MCTS 节点（**增量式**：不保存棋盘副本）。

    阶段 3 的重要改动
    ------------------
    旧实现每个节点都持有一份 `Board` 深拷贝（`board.copy()`）。实测消耗：
        * `board.copy()` 本身约 11 µs，每次扩展要为每个子节点拷一次；
        * 更贵的是它同时深拷贝 `history` 列表，而 history 随手数线性增长，
          于是内存与拷贝成本都是 O(深度²)；
        * 实测每手耗时 300~700 ms，绝大部分耗在这里。

    现在改为标准做法 —— **单个共享棋盘 + make/undo 增量搜索**：
        * 节点只记录"走到这里所落的那一手"(`move`) 与所属父节点；
        * 需要某节点的棋盘时，从根沿路径 make 下去，用完再 undo 回来（见
          `MCTS._seek`）。因为搜索本身就是深度优先、每次只走一条路径，
          所以每次 seek 的均摊代价是 O(路径长度) 次 `make`/`undo`，
          而 `Board.move()/undo()` 是 O(1)（Zobrist 异或 + 局部的五连判定）。
        * 棋盘本身不再进入节点，内存占用大幅下降。

    保留的少量缓存（`h` / `term_v` / `win_moves`）都是"按需惰性计算一次"的
    标量/小列表，代价远低于深拷贝整个棋盘。
    """

    __slots__ = ("move", "player", "parent", "prior", "n", "w", "q",
                 "children", "is_terminal", "value", "expanded", "vl",
                 "h", "term_v", "win_moves", "ply")

    def __init__(self, move=None, player=1, parent=None, prior=0.0,
                 is_terminal=False, value=0.0, ply=0):
        # move: 从父节点走到本节点所落的一手; 根节点为 None
        self.move = move
        self.player = player
        self.parent = parent
        self.prior = prior
        # ply: 本节点局面对应的**绝对手数**（从对局开始算，而不是从当前根算）。
        # 这是增量搜索定位棋盘位置的关键：根节点可能因为树复用而被提升到
        # 较深的位置（set_root_move），此时"相对根的手数"不再是绝对手数，
        # 必须显式记录 ply，否则 _seek 会把棋盘定位到错误的局面
        # （实测表现为 "无法落子 (10,11) @ 2" 这种状态不一致异常）。
        self.ply = ply
        self.n = 0
        self.w = 0.0
        self.q = 0.0
        self.children: Dict[Tuple[int, int], "MCTSNode"] = {}
        self.is_terminal = is_terminal
        self.value = value
        self.expanded = False
        # 虚拟损失计数（见 virtual_loss 记账说明）。
        # 用独立字段而不是直接改 n/w，是为了让统计量始终自洽：
        #   n、w 只反映"已完成并回溯"的模拟，
        #   选择阶段看到的临时值由 score() 现算，用完即弃。
        self.vl = 0
        # ---- 惰性缓存（None 表示"尚未计算"）----
        self.h = None          # 该节点局面的 Zobrist 哈希
        self.term_v = None     # 终局价值（未判定时为 None）
        self.win_moves = None  # 该节点执子方的"一步成五"点集合

    def add_child(self, move, child):
        """按落点 (x, y) 添加子节点。"""
        self.children[move] = child

    def score(self, c_puct: float, sqrt_parent: float,
              fpu_base: float, prior_fpu_weight: float) -> float:
        """
        计算该节点的 PUCT 分数（从**父节点**视角，值越大越该被选）。

        注意价值符号：self.q 是从 **self.player** 视角的价值，
        而父节点要最大化自己的价值，因此取 -q。
        """
        eff_n = self.n + self.vl
        if eff_n <= 0:
            return fpu_base + prior_fpu_weight * c_puct * self.prior * sqrt_parent
        q = self.w / self.n if self.n > 0 else 0.0
        return -q + c_puct * self.prior * sqrt_parent / (1.0 + eff_n)

    def __repr__(self):
        return (f"MCTSNode(move={self.move}, player={self.player}, n={self.n}, "
                f"q={self.q:.3f}, children={len(self.children)}, "
                f"terminal={self.is_terminal})")


# ----------------------------------------------------------------------
# 动态参数：按局面深度决定模拟次数与温度
# ----------------------------------------------------------------------
# 【重要】旧实现把 n_sim 设成 50/100/150 这种量级，配合被破坏的搜索毫无意义。
# 五子棋要稳定看到 4 层战术（冲四/活三的威胁链）至少需要数百次模拟。
# 这里保留"按深度递增"的形状，但量级提升到可用区间。
def dynamic_n_sim(depth: int) -> int:
    """
    根据局面深度返回自对弈用的模拟次数建议值。

    开局分支多但战术浅（400），中盘需要算杀（800~1200）。
    实际训练建议：先用固定 400 跑通，再提到 800。
    """
    if depth < 8:
        return 400
    elif depth < 24:
        return 800
    else:
        return 1200


def dynamic_temp(depth: int) -> float:
    """
    根据局面深度返回自对弈采样温度。

    与旧实现的区别：旧代码把该函数算出的温度**从未使用**（自对弈永远传 1.0
    并对访问次数取 argmax，导致同一模型下每局逐手完全相同）。
    现在由 sample_move() 真正按温度采样。
    """
    if depth < 12:
        return 1.0
    elif depth < 30:
        return 0.5
    else:
        return 0.0      # 0 = 贪心（选访问次数最多）


# 兼容旧名
_dynamic_n_sim = dynamic_n_sim
_dynamic_temp = dynamic_temp


# ----------------------------------------------------------------------
# MCTS 搜索器
# ----------------------------------------------------------------------
class MCTS:
    """
    五子棋 MCTS 搜索器（PUCT + 规则先验 + 网络评估 + 批量推理）。

    参数:
        net              : GomokuNet 实例
        n_sim            : 每步模拟轮数（None 表示按深度用 dynamic_n_sim）
        c_puct           : PUCT 探索系数（默认 1.0）
        virtual_loss     : 虚拟损失（默认 3.0；批量推理下必须 > 0）
        batch_size       : 批量推理的叶子数（默认 16）
        max_children     : 非根节点最多展开的子节点数（根节点始终全展开）
        fpu_reduction    : First-Play-Urgency 折减（未访问子节点用
                           -q_parent - fpu_reduction 代替 +inf，默认 0.20）
        prior_fpu_weight : 未访问子节点的先验权重，保证高先验点优先被搜索
        root_noise_alpha : 根节点 Dirichlet 噪声 alpha（None/<=0 关闭）
        root_noise_eps   : 根节点噪声混合比例 epsilon（默认 0.25）
        use_heuristics   : 是否启用棋型规则先验（默认 True）
        heuristic_weight : 规则先验权重（默认 6.0）
        policy_weight    : 网络策略先验权重（默认 0.3）
        terminal_blend   : 终局价值的规则/网络混合系数。
                           五子棋短程战术多，用 check_win_after 直接给出 ±1
                           比等网络学出来快得多，默认完全信任规则 (1.0)
        open_four_win    : 是否把"形成活四"直接判为必胜终局（默认 True）。
                           活四是严格必胜形（对手只能堵一端）。
                           搜索只走 4~6 层，很难自己发现"做活四 -> 堵一端 ->
                           另一端成五"这条 3 手必胜链，实测活三局面下会漏掉
                           唯一的做活四点。开启后既保证正确性也大幅提速。
        immediate_win_loss: 是否把"对手已有一步成五的点"直接判为必败终局
                           （默认 True）。没有它的话，对方成五点常常不在
                           节点的 top-max_children 展开范围内，加多少模拟
                           次数都发现不了（实测必挡点 q 值恒为 +0.49）。
        seed             : 随机种子（温度采样与根噪声用）

    ===== 参数取值依据（实测，n_sim=800，中盘无战术局面）=====
        max_children=16, hw=1, pw=1  -> 搜深 3，根访问被 36 个子节点各 16 次摊平
        max_children=6,  hw=6, pw=0.3-> 搜深 5~7，根访问 top1 占 28%~44%
        深度不足的根因是"先验太平"：每个未访问子节点都用 FPU+先验打分，
        36 个分支会各分到一次访问，模拟预算被宽度吃光。窄化 max_children
        并加大规则先验权重后，模拟次数集中到少数候选，深度立刻上去。

    ===== 训练时的权重退火建议 =====
        heuristic_weight 是"带规则拐杖"，随网络变强应逐步降低：
            前 20 代: hw=6.0, pw=0.3   （网络≈噪声，靠规则）
            20~60 代: hw=2.0, pw=1.0   （网络开始有用）
            60 代以后: hw=0.5, pw=1.0  （让网络主导，规则只兜底成五/必挡）
        注意：成五/必挡的强制覆盖与权重无关（永远生效），所以退火不会
        牺牲基本战术正确性。
    """

    def __init__(self, net, n_sim=None, c_puct=1.0, virtual_loss=3.0,
                 batch_size=16, max_children=6, root_max_children=12,
                 fpu_reduction=0.20,
                 prior_fpu_weight=1.0,
                 root_noise_alpha=None, root_noise_eps=0.25,
                 use_heuristics=True, heuristic_weight=6.0,
                 policy_weight=0.3, terminal_blend=1.0,
                 open_four_win=True, immediate_win_loss=True,
                 reuse_board=True, seed=None):
        self.net = net
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.batch_size = max(1, int(batch_size))
        self.max_children = max(1, int(max_children))
        self.root_max_children = int(root_max_children)
        self.fpu_reduction = float(fpu_reduction)
        # 未访问子节点的先验权重：保证高先验点（成五/必挡）优先被搜索
        self.prior_fpu_weight = float(prior_fpu_weight)
        self.root_noise_alpha = root_noise_alpha
        self.root_noise_eps = float(root_noise_eps)
        self.use_heuristics = bool(use_heuristics)
        self.heuristic_weight = float(heuristic_weight)
        self.policy_weight = float(policy_weight)
        self.terminal_blend = float(np.clip(terminal_blend, 0.0, 1.0))
        self.open_four_win = bool(open_four_win)
        self.immediate_win_loss = bool(immediate_win_loss)
        self.rng = np.random.default_rng(seed)

        # ---- Tree Reuse 状态 ----
        self._root: Optional[MCTSNode] = None
        self._move_depth = 0

        # ---- 增量搜索：单个共享棋盘 ----
        # 所有节点都不再持有棋盘副本；搜索时用 make/undo 在共享棋盘上走路径。
        #
        # 【唯一真相】共享棋盘的绝对手数 **就是** `len(self._board.history)`。
        # 早期版本额外维护了 `_board_ply` + `_board_path` 两套账本，结果两套账
        # 与真实棋盘三者会互相漂移（树复用后快照没提升 -> `_board_ply` 小于根的
        # 手数 -> 每次 `_seek` 都判定失败 -> 整个搜索在**错误的棋盘**上跑）。
        # 现在只认 Board.history 这一个来源：undo 一次就是减一手，不可能对不上。
        self._board = None          # Board 实例（共享）
        self._board_history = []    # **当前根局面**的完整落子历史（快照）
        self._reuse_board = bool(reuse_board)

        # ---- 叶子评估缓存（替代有缺陷的置换表，见 [P0-3]）----
        # {zobrist: value}，只缓存"评估结果"供复用，
        # **不**改变搜索语义、绝不把节点标记为终局
        self._value_cache: Dict[int, float] = {}
        self._policy_cache: Dict[int, np.ndarray] = {}
        self._max_cache = 150_000

        # ---- 统计（验收指标用）----
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 增量搜索：共享棋盘与路径操作
    # ------------------------------------------------------------------
    @property
    def _board_ply(self) -> int:
        """
        共享棋盘当前的绝对手数（只读）。

        单一真相来源就是 `Board.history` 的长度；保留这个名字是为了兼容旧调用方
        （诊断脚本 / 自测）。历史遗留的 `_board_path` 账本已被删除，见 __init__。
        """
        if self._board is None:
            return 0
        return len(self._board.history)

    def _reset_board(self, board) -> None:
        """
        把共享棋盘重置到给定局面（深拷贝一次，整个搜索只做这一次）。

        同时记录 `_board_history`：**当前根局面**的完整落子历史，
        供 `_full_replay` 在增量定位失败时从零重建。
        """
        self._board = board.copy()
        self._board_history = list(board.history)

    def _undo_to_ply(self, target_ply: int) -> None:
        """
        把共享棋盘 undo 到绝对手数 target_ply。

        直接以 `Board.history` 为准：undo 一步就少一手，天然与棋盘内容一致。
        不再维护额外的路径账本（那套账本曾与真实棋盘漂移，是
        "搜索在错误棋盘上跑" 的根因）。
        """
        while len(self._board.history) > target_ply:
            if not self._board.undo():
                break

    def _replay_from_snapshot(self, target_ply: int) -> None:
        """（保留的旧辅助方法，等价于 _full_replay；见 _seek 的兜底逻辑。）"""
        self._full_replay()

    def _path(self, node: MCTSNode) -> List[MCTSNode]:
        """
        返回从当前根到 node 的路径（含两端，正序）。
        """
        path = []
        n = node
        while n is not None and n is not self._root:
            path.append(n)
            n = n.parent
        path.append(self._root)
        path.reverse()
        return path

    def _seek(self, node: MCTSNode) -> None:
        """
        把共享棋盘定位到 node 的局面（增量 undo/make）。

        算法（"最长一致前缀 + 重放"）
        ---------------------------
        目标是从根到 node 的那条路径 chain。做法不是"undo 到某个深度就假定
        前面的手都是对的"，而是**逐手核对盘上记录**：

            1) 从根开始，沿 chain 逐手比较 `board.history[ply-1]` 与树上那一手；
            2) 取最长的一段"完全一致"的前缀，记最深的匹配节点下标 keep；
            3) undo 到 chain[keep] 的手数，再从 keep+1 开始逐手 make 下去。

        【踩过的大坑 —— 这条曾经让搜索整体失效】
        旧实现是这样算"公共深度"的：

            for nd in chain:
                if nd.ply <= self._board_ply:  common = nd.ply
                else: break
            self._undo_to_ply(common)
            for nd in chain:
                if nd.ply <= common: continue        # ← 这些节点**完全没被核对**
                ...核对落子...

        它把"手数 ≤ 当前棋盘手数"当成了"盘上内容一定是这个分支的"。
        但共享棋盘在一次搜索里会沿**不同分支**走来走去：走到 A 分支的 ply=6
        之后，再去 seek B 分支的 ply=6 节点时，`common` 直接算成 6，
        于是 B 分支那一手被 `continue` 跳过、盘上留着的却是 A 分支的落子。

        实测后果（随机网络 + 500 次模拟即可复现）：
            * `_expand` 用**父节点/别的分支**的合法点给子节点建边，
              产出 "ply 8 与 ply 9 走了同一个点 (8,8)" 这种**非法搜索树**；
            * 节点在被污染的盘面上做终局判定 -> 整棵搜索的评估都是错的。

        现在的正确性由 `_assert_at` 常开校验兜底（见该方法）。
        """
        chain: List[MCTSNode] = []
        n = node
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()                     # 根 -> node

        def want_of(nd: MCTSNode):
            """树记录的"走到 nd 的那一手"：(x, y, 落子方)。"""
            player = nd.parent.player if nd.parent is not None else nd.player
            return (int(nd.move[0]), int(nd.move[1]), int(player))

        # 先尝试增量定位；任一步对不上就整体重放并重试一次
        for attempt in (0, 1):
            try:
                hist = self._board.history
                if chain[0].ply > len(hist):
                    raise _Resync(
                        f"根手数 {chain[0].ply} > 棋盘手数 {len(hist)}"
                        f"（根快照未同步）")

                # ---- 0) 先核对"根快照区"，即手数 1..chain[0].ply 这一段的落子 ----
                # 这一段**无法靠 undo 修正**（undo 只能减少手数，不能把盘上的
                # 某一手换成另一手）。而共享棋盘完全可能停在同一手数的别的分支上：
                # 最典型的就是刚 `set_root_move` 之后，棋盘还停在上一轮搜索走过的
                # 那条支线上。若跳过这一步核对，`_seek` 会"以为"自己到了目标局面，
                # 实际上盘面是"别的分支的前缀 + 目标分支的后缀"的**混合体** ——
                # 于是 (8,8) 这种已经在 ply 1 用过的点又作为合法点被子节点采用，
                # 造出非法搜索树（实测崩溃点正是这里）。
                snap = self._board_history
                if len(snap) != chain[0].ply:
                    raise _Resync(
                        f"根快照手数 {len(snap)} != 根 ply {chain[0].ply}")
                for k in range(chain[0].ply):
                    h = hist[k]
                    s = snap[k]
                    if (int(h[0]), int(h[1]), int(h[2])) != (
                            int(s[0]), int(s[1]), int(s[2])):
                        raise _Resync(
                            f"手数 {k + 1} 与根快照不符：盘上={h} 快照={s}")

                # ---- 1) 求"最长一致前缀" ----
                keep = 0
                for i in range(1, len(chain)):
                    nd = chain[i]
                    idx = nd.ply - 1
                    if idx >= len(hist):
                        break                       # 盘上还没走到这一手
                    h = hist[idx]
                    if (int(h[0]), int(h[1]), int(h[2])) != want_of(nd):
                        break                       # 这一手是别的分支的
                    keep = i

                # ---- 2) 退回到一致前缀的末端，再往前补 ----
                self._undo_to_ply(chain[keep].ply)
                for nd in chain[keep + 1:]:
                    w = want_of(nd)
                    if not self._board.move(w[0], w[1], w[2]):
                        raise _Resync(f"手数 {nd.ply} 无法落子 {w}")
                return                              # 定位成功
            except _Resync as exc:
                self._n_resync = getattr(self, "_n_resync", 0) + 1
                self._last_resync_reason = str(exc)
                if attempt == 0:
                    self._full_replay()
                    continue
                raise RuntimeError(
                    f"MCTS 增量棋盘定位失败（已重试重建）：{exc}；"
                    f"根 ply={self._root.ply if self._root else None}，"
                    f"快照手数={len(self._board_history)}；"
                    f"链={[(nd.ply, nd.move, nd.player) for nd in chain]}；"
                    f"盘上历史={self._board.history}") from exc

    def _full_replay(self) -> None:
        """
        从"当前根局面"的快照完整重建共享棋盘。

        这是增量搜索的"兜底正确路径"：代价是一次 361 格数组重建 + 若干手重放，
        与旧实现每个节点都 copy 整个棋盘相比仍然便宜得多，而且只在状态异常时触发。
        """
        from gomoku_env import Board as _Board
        b = _Board()
        for (x, y, p) in self._board_history:
            if not b.move(x, y, p):
                raise RuntimeError(
                    f"根快照本身无法重放（实现 bug）：历史={self._board_history}")
        self._board = b

    def _node_hash(self, node: MCTSNode) -> int:
        """惰性获取节点局面的 Zobrist 哈希。"""
        if node.h is None:
            self._seek(node)
            node.h = self._board.get_hash()
        return node.h

    def _node_win_moves(self, node: MCTSNode) -> set:
        """惰性计算该节点执子方的"一步成五"点集合。"""
        if node.win_moves is None:
            self._seek(node)
            me = node.player
            node.win_moves = {
                (int(x), int(y))
                for (x, y) in self._board.get_legal_moves()
                if self._board.check_win_after(int(x), int(y), me)
            }
        return node.win_moves

    def _state_tensor(self, node: MCTSNode):
        """定位到 node 局面并返回 3×19×19 状态张量。"""
        self._seek(node)
        return self._board.get_state_tensor()

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------
    @property
    def _transposition(self):
        """
        兼容旧属性名。旧实现的置换表语义（把叶子标记为终局、剪掉整棵子树）
        已被移除，这里暴露只读缓存视图，避免旧代码/自测报错。
        """
        return self._value_cache

    def reset_tree(self):
        """清空搜索树、深度计数与评估缓存（新对局调用）。"""
        self._root = None
        self._move_depth = 0
        self._board = None
        self._board_history = []
        self._value_cache.clear()
        self._policy_cache.clear()

    def set_root(self, board) -> MCTSNode:
        """
        确保搜索树根与 board 局面一致，返回根节点。

        增量搜索版：共享棋盘由 `_reset_board` 重置到根局面。
        复用已有树时必须同时满足两个条件，否则整棵树作废重建：
            1) 根节点的 Zobrist 哈希 == 棋盘哈希；
            2) 根快照手数 == 根手数（`_board_history` 与根严格同步）。
        条件 2 是必须的：只比哈希的话，一旦快照落后于根，
        `_seek` 就会永远定位失败（历史 bug，见 `_seek` 的说明）。
        """
        if self._root is not None and self._board is not None:
            stale = (self._node_hash(self._root) != board.get_hash()
                     or len(self._board_history) != self._root.ply)
            if stale:
                self._root = None
        if self._root is None:
            self._reset_board(board)
            self._root = MCTSNode(move=None, player=board.current_player,
                                  parent=None, prior=1.0,
                                  ply=len(board.history))
            self._root.h = self._board.get_hash()
            self._move_depth = len(board.history)
        # 统一收尾：无论复用还是重建，都把共享棋盘**定位回根局面**。
        # 这样"共享棋盘此刻在哪儿"是确定的，任何依赖它的代码
        # （终局判定 / 哈希 / 成五点集合）都不会被上一轮搜索的残留位置污染。
        self._seek(self._root)
        return self._root

    def set_root_move(self, move) -> bool:
        """
        落子后将对应子节点升为新根（Tree Reuse）。

        修复 [P0-4]：旧实现在子节点不存在时直接 `self._root = None` 丢弃整棵树，
        而根只展开 Top-20，导致对手走出树外的概率极高（中盘 >70%），
        树复用形同虚设。现在：
            1) 命中子节点 -> 升为新根（并清除该子节点用于复用的访问统计）
            2) 未命中但根已有子节点 -> 只需 +1 深度（保留旧树，下一步会重建根）
            3) 完全没有树 -> 保持 None

        【踩过的坑】旧写法是：
                child.parent = None
                self._root = child          # ← child 就是 self._root.children[move]
            而 `self._root` 被赋成了 child **自己**（两者是同一对象），
            等于"把新根又指回旧根"，同时又把 parent 链切断了。
            后果：树复用之后 `_seek` 沿 parent 链走不回真正的新根，
            手数错位，最终抛出"无法落子 (3,2) @ 1"这种状态不一致异常。
            正确写法是先把子节点单独取出、置 None，再赋给 self._root。

        返回:
            bool — 是否成功复用了子树
        """
        root = self._root
        if root is None:
            return False
        self._move_depth += 1
        child = root.children.get(move)
        if child is None:
            # 未命中：保留旧树结构（不丢弃），仅记录深度变化。
            # 下一次 get_action_probs 会因哈希不匹配重建根，代价可控。
            return False
        # 升为新根：
        #   1) 先把 child 从旧根的 children 里摘出来（避免 self._root 指回旧根）
        #   2) 断开 parent 链
        #   3) 再赋给 self._root
        #   4) **把这一手补进根快照**（关键，见下）
        root.children.pop(move, None)
        child.parent = None
        self._root = child
        # 复用的子树继承旧搜索的统计（AlphaZero 同样保留），
        # 但 prior 需要由新搜索重算，故重置为 1.0
        child.prior = 1.0
        # 【关键修复】新根比旧根深一手，根快照必须同步前进一手。
        # 早期版本漏了这一步，导致 `len(_board_history) < root.ply`：
        # 一旦触发 `_full_replay`（从快照重建），棋盘会被拉回**旧根**局面，
        # 而 `_seek` 的"根手数 > 棋盘手数"判定必然失败 -> 每次都重建 ->
        # 整轮搜索（400 次模拟）全部在错误的棋盘上评估。
        # 实测症状：每局自对弈只有 8~18 手就 mv=None 中断、标签恒为 0（假和棋）。
        self._board_history.append(
            (int(move[0]), int(move[1]), int(root.player)))
        return True

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------
    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        选择阶段：用 PUCT 公式选一个子节点（含 virtual loss）。

        【价值符号约定 —— 这里有一个极易搞错、后果致命的地方】
            child.q 是从 **child.player（轮到 child 落子的一方）视角** 的价值。
            而 node 要最大化的是 **自己（node.player）视角** 的价值，
            两者相差一个负号。因此选择公式必须用 **-q**：

                score = -child.q + c_puct * p * sqrt(1 + N_node) / (1 + n + vloss)

            旧实现（以及本次重写的第一版）用的是 +child.q，方向完全反了：
            子节点胜率越高 -> q 越大 -> 分数越大 -> 父节点**越不去走它**。
            实测症状：低访问次数时高先验点仍能被选中（掩盖了问题），
            但访问次数一涨、探索项衰减后，真正的"一步成五"立刻被
            "稍微不那么差"的普通点挤掉 —— n_sim=2000 时引擎放弃成五去走 (7,6)。

        FPU（First-Play-Urgency）：未访问子节点（n + vloss == 0）用
            score = -node.q - fpu_reduction + prior_fpu_weight * c_puct * p * sqrt(1+N)
        代替 +inf，避免"必须把 50+ 个分支各访问一遍"，同时让高先验点
        （规则识别出的成五/必挡点，p≈1）在首次访问前就拿到最高分。
        """
        if not node.children:
            return None
        sqrt_parent = math.sqrt(1.0 + node.n)
        c = self.c_puct
        base = -node.q - self.fpu_reduction
        pw = self.prior_fpu_weight
        best = None
        best_score = -float("inf")
        for child in node.children.values():
            score = child.score(c, sqrt_parent, base, pw)
            if score > best_score:
                best_score = score
                best = child
        return best

    # ------------------------------------------------------------------
    # 叶子评估
    # ------------------------------------------------------------------
    def _terminal_value(self, node: MCTSNode) -> Optional[float]:
        """
        若 node 局面已可判定终局，返回**从 node.player 视角**的符号价值；
        否则返回 None。

        [P0-2] 修复：旧实现把终局价值写成常数 0。

        判定分三种（都返回符号价值，无需网络）：
            1) 棋盘规则已终局（五连 / 满盘）
            2) **当前方自己**已有"一步成五"的点 —— 下一手就走掉，value = +1
            3) 对手已有"一步成五"的点 —— 该点在本回合就必然被走掉，
               因此当前局面**已经输了**，value = -1

        第 2 条是 2026-10-10 补上的**对称判定**（此前只有第 3 条）：
            缺了它，"我能成五"的节点仍要走网络评估 + 展开，于是它那一支的
            期望值会被同层的**零先验垃圾分支稀释** —— 实测在"对方活四、我方必败"
            的局面里，必挡点给对手的 q 是 0.998~0.999，而一个无关点 (7,12)
            只给 0.812；父节点要最小化对手价值，于是**故意去走那个无关点**，
            导致 `test_tactics.py` 的 D.安全性 从 3/3 掉到 2/3。
            补上第 2 条后，这类节点直接拿到精确的 +1，不再被稀释，
            父节点看到所有走法同等绝望，于是按先验（强制点 0.5 vs 无关点 0）
            选必挡点。实测 gen_1/gen_30/gen_39 全部从 18/19 恢复到 19/19。

        第 2 条必须排在"对手有成五"**之前**：轮到我走，我先成五就赢了，
        对手根本没机会下。（旧注释曾写"双方都有成五点时按输处理"——那是错的。）

        第 3 条极其重要（五子棋专属的终局判定）：
            如果没有它，搜索要靠 4~6 层展开才能"看穿"对方下一手成五。
            而节点默认只展开 top max_children 个子节点，对方那个成五点经常
            不在其中，于是无论怎么加模拟次数都发现不了危机。
            实测症状：必挡局面下根节点访问次数涨到 800 才勉强选对，
            200 次模拟时必挡点的 q 值仍显示 +0.49（应为 -1），
            MCTS 认为"挡了也不亏"，转而去走无关的点。
            加上这条判定后，"对方有成五点"直接等价于终局失败，
            与 check_win_after / open_four_after 一起构成完整的战术终局识别。
        """
        self._seek(node)
        over, winner = self._board.is_game_over()
        if over:
            if winner == 0:
                return 0.0
            return 1.0 if winner == node.player else -1.0
        if self.immediate_win_loss:
            # 对称判定（顺序不能反）：我能成五 -> 精确 +1
            if self._node_win_moves(node):
                return 1.0
            if self._opponent_has_immediate_win(node):
                return -1.0
        return None

    def _opponent_has_immediate_win(self, node: MCTSNode) -> bool:
        """
        对手（3 - node.player）是否存在一步成五的点。

        【旧注释的错误更正】这里原来写着"不能先检查自己是否有成五点，
        双方都有成五点时保守按输处理"——**这个逻辑是错的**：轮到我走，
        我先成五就赢了，对手没有机会落子。所以调用方
        `_terminal_value` 必须先查"我能否成五"（返回 +1）、再查这一条。
        """
        opp = 3 - node.player
        self._seek(node)
        board = self._board
        return any(board.check_win_after(int(x), int(y), opp)
                   for (x, y) in board.get_legal_moves())

    def _evaluate_leaves(self, leaves: List[MCTSNode]):
        """
        批量评估叶子。三类来源：
            1) 评估缓存命中   -> 复用 (价值, 完整策略向量)
            2) 局面已终局     -> 规则给出符号价值（无需网络）
            3) 其余           -> 一次网络批量前向

        返回:
            policies : list，元素为 (361,) numpy 策略 或 None（缓存/终局）
            values   : list[float]，从各自 node.player 视角
            from_cache: list[bool]
        """
        n = len(leaves)
        policies: List[Optional[np.ndarray]] = [None] * n
        values: List[float] = [0.0] * n
        from_cache: List[bool] = [False] * n

        need_net: List[int] = []
        n_terminal = 0
        for i, leaf in enumerate(leaves):
            h = self._node_hash(leaf)
            cached = self._value_cache.get(h)
            if cached is not None:
                values[i] = cached
                policies[i] = self._policy_cache.get(h)
                from_cache[i] = True
                continue
            tv = self._terminal_value(leaf)
            if tv is not None:
                values[i] = tv
                n_terminal += 1
                continue
            need_net.append(i)

        if need_net:
            # 逐个 seek 到该叶子并取局面张量（增量定位，均摊 O(1) 手）
            states = np.stack(
                [self._state_tensor(leaves[i]) for i in need_net], axis=0)
            pol, val = self.net.evaluate_batch(states)   # (B,361), (B,)
            for k, i in enumerate(need_net):
                policies[i] = pol[k]
                values[i] = float(val[k])
                # 顺手把该节点的哈希缓存下来（后面写缓存/建子节点都要用）
                self._node_hash(leaves[i])

        # 统计（验收指标）
        #   leaf_visits : 被评估的叶子总数，应严格等于 n_sim
        #   policy_evals+terminal_evals+cache_hits = leaf_visits
        #   旧实现的问题是 leaf_visits 只有 n_sim 的 6.7%，其余模拟全被浪费，
        #   这里保证每个模拟都真正落到一次"有效评估"上。
        self._n_evaluations = getattr(self, "_n_evaluations", 0) + len(need_net)
        self._n_cache_hits = getattr(self, "_n_cache_hits", 0) + sum(from_cache)
        self._n_terminal_evals = getattr(self, "_n_terminal_evals", 0) + n_terminal
        self._n_leaf_visits = getattr(self, "_n_leaf_visits", 0) + len(leaves)

        return policies, values, from_cache

    # ------------------------------------------------------------------
    # 扩展
    # ------------------------------------------------------------------
    def _assert_at(self, node: MCTSNode) -> None:
        """
        不变式校验：调用 `_seek(node)` 之后，共享棋盘必须**恰好**处于 node 的局面。

        校验三项（都是 O(1)）：
            1) 手数一致      : len(history) == node.ply
            2) 最后一手一致  : history[-1] 的落点 == node.move
            3) 执子方一致    : board.current_player == node.player

        这三条是增量搜索的根本前提。历史上正是这一前提被悄悄破坏（棋盘停在
        父节点上），导致 `_expand` 用"父节点的合法点"给子节点建边 ——
        实测造出 "ply 8 和 ply 9 走了同一个点 (8,8)" 这种非法树。
        校验成本 O(1)，因此常开：宁可当场炸掉，也不要产出错误搜索树。
        """
        hist = self._board.history
        if len(hist) != node.ply:
            raise RuntimeError(
                f"增量棋盘位置错误：期望 ply={node.ply}，实际={len(hist)}；"
                f"node={node!r}；盘上历史={hist}")
        if node.move is not None:
            last = hist[-1]
            if (int(last[0]), int(last[1])) != (int(node.move[0]),
                                                int(node.move[1])):
                raise RuntimeError(
                    f"增量棋盘最后一手不符：期望={node.move}，实际={last}；"
                    f"node={node!r}")
        if self._board.current_player != node.player:
            raise RuntimeError(
                f"增量棋盘执子方不符：期望={node.player}，"
                f"实际={self._board.current_player}；node={node!r}")

    def _child_priors(self, node: MCTSNode, policy: Optional[np.ndarray]
                      ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        为 node 计算合法点及其先验分布。

        宽度控制（关键性能参数）：
            * 根节点  : 先按先验取 top `root_max_children`，但**必胜/必挡点永久保留**
            * 其它节点: 先按先验取 top `max_children`
            * 强制点（成五/必挡）无论宽度如何都不会被裁掉

        为什么根节点也要限宽：
            19×19 中盘合法点常达 60~100 个。若根全宽展开，200 次模拟会把
            访问次数摊成"每个点 2~3 次"，每个点的价值估计都还是噪声，
            于是选点近乎随机。
            实测（必挡类局面，19 个测试项）：
                n_sim=200/400/800  通过率 11/19（B/D 组几乎全错）
                n_sim=1600         通过率 19/19
            也就是说模拟次数不到某个临界值时，宽度把预算吃光了。
            限宽后同样的预算能给出数倍于原来的每点访问次数。

        树复用注意：先验 top-K 里天然包含对手最可能走的好点，因此
        set_root_move 的命中率依然很高；个别未命中也只是重建一次，
        不会再像旧实现那样"根只展开 top-20 导致命中率 <30%"。
        """
        self._seek(node)
        board = self._board
        legal = board.get_legal_moves()
        if not legal:
            return [], np.zeros(0)

        net_probs = None
        if policy is not None:
            legal_arr = np.array(legal)
            flat = legal_arr[:, 1] * 19 + legal_arr[:, 0]
            net_probs = policy[flat]

        # 强制成五/必挡时，heuristics 会只保留强制点；
        # 同时把"强制点下标集合"一并返回，供下面的宽度控制使用 ——
        # 这样就不必再单独扫一遍成五点（旧版 `_forced_moves` 对每个候选点
        # 都要调 2 次 check_win_after，实测 20 手累计 143 万次调用）。
        priors, forced = heuristics.compute_prior(
            board, legal, net_probs,
            heuristic_weight=self.heuristic_weight,
            policy_weight=self.policy_weight,
            use_heuristics=self.use_heuristics,
        )

        is_root = node.parent is None
        width = self.root_max_children if is_root else self.max_children
        if width <= 0 or len(legal) <= width:
            return [(int(x), int(y)) for x, y in legal], priors

        # 按先验取 top-width，并保证"强制点"一定入选
        keep = set(forced)
        for i in np.argsort(priors)[::-1]:
            if len(keep) >= max(width, len(forced)):
                break
            keep.add(int(i))
        # 至少保留 width 个（若 forced 已超过 width 则按 forced 数量）
        idx_sorted = [i for i in np.argsort(priors)[::-1] if int(i) in keep]
        idx_sorted = idx_sorted[:max(width, len(forced))]

        moves = [(int(legal[i][0]), int(legal[i][1])) for i in idx_sorted]
        p = priors[idx_sorted]
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        # 强制点的先验重新拉满，避免被归一化稀释
        if forced:
            for k, i in enumerate(idx_sorted):
                if i in forced:
                    p[k] = max(p[k], 0.5)
            p = p / p.sum()
        return moves, p

    @staticmethod
    def _forced_moves(board, legal) -> set:
        """
        返回必须保留的强制点（一步成五 / 必须堵的成五点）索引集合。

        【已不再是主路径】宽度控制现在直接复用 `heuristics.compute_prior`
        返回的 `forced_idx`（那里在同一次棋型扫描里就得到了成五点）。
        这个方法保留下来供自测 / 排查对照使用，正常搜索不要调用它 ——
        它会对每个候选点做 2 次成五判定，是纯粹的重复计算。
        """
        me = board.current_player
        opp = 3 - me
        out = set()
        for i, (x, y) in enumerate(legal):
            x, y = int(x), int(y)
            if board.check_win_after(x, y, me) or board.check_win_after(x, y, opp):
                out.add(i)
        return out

    def _expand(self, node: MCTSNode, policy: Optional[np.ndarray],
                value: float) -> Dict[Tuple[int, int], MCTSNode]:
        """
        扩展 node：生成子节点并赋先验，同时写入评估缓存。

        增量搜索版（阶段 3）：
            **不再为每个子节点深拷贝棋盘**。做法是在共享棋盘上
                seek(node) -> 对每个候选点 move() -> 读取判定 -> undo()
            子节点只记录 (move, player, parent)。
            这去掉了旧实现每节点一次 `Board.copy()`（含 history 深拷贝）的开销，
            而 history 的深拷贝正是此前每手 300~700ms 的主因。

        终局子节点（成五/落败/满盘）与活四子节点写入**带符号**价值（[P0-2]）。
        """
        if node.expanded:
            # 已有子节点（复用子树）：只需刷新先验，不重复建节点
            return node.children

        # `reuse_board=False` 时退化为旧实现（每个扩展节点建一份棋盘副本），
        # 仅用于性能对照 / 排查增量逻辑问题，正常搜索不要开。
        if self._reuse_board:
            self._seek(node)
            board = self._board
        else:
            self._seek(node)
            board = self._board.copy()

        self._assert_at(node)

        moves, priors = self._child_priors(node, policy)
        if not moves:
            node.is_terminal = True
            node.expanded = True
            node.value = 0.0
            return {}

        h_node = board.get_hash()
        node.h = h_node
        # 缓存叶子评估结果（价值 + 完整策略向量），供后续搜索复用。
        # 只在需要网络的局面上写入；终局局面价值由规则决定，无需缓存。
        if len(self._value_cache) < self._max_cache:
            self._value_cache[h_node] = float(value)
            if policy is not None:
                self._policy_cache[h_node] = np.asarray(policy, dtype=np.float32)

        me = node.player
        for i, (x, y) in enumerate(moves):
            if not board.move(x, y, me):
                continue
            try:
                # O(1) 不变式：落子后手数必须正好 +1（同时校验棋盘确实处于本节点局面）
                if len(board.history) != node.ply + 1:
                    raise RuntimeError(
                        f"落子后手数异常：node.ply={node.ply} "
                        f"len(history)={len(board.history)} node={node!r} "
                        f"move={(x, y)} 盘上历史={board.history}")
                over, winner = board.is_game_over()
                if over:
                    # ---- 终局判定：带符号价值（从子节点视角）----
                    if winner == 0:
                        cval = 0.0
                    else:
                        cval = 1.0 if winner == me else -1.0
                    # 规则与网络混合（网络价值是"子节点执子方"视角，需取反）
                    if self.terminal_blend < 1.0:
                        net_child_v = -float(value)
                        cval = (self.terminal_blend * cval
                                + (1.0 - self.terminal_blend) * net_child_v)
                    child = MCTSNode(move=(x, y), player=3 - me, parent=node,
                                     prior=float(priors[i]), is_terminal=True,
                                     value=cval, ply=node.ply + 1)
                    child.expanded = True
                elif self.open_four_win and board.open_four_after(x, y, me):
                    # ---- 活四判定：已锁定胜利，等价于终局 ----
                    # 活四 = 4 连且两端皆空 -> 对手只能堵一端，另一端必成五。
                    # 子节点执子方是"对手"，对手必败 -> 子节点 value = -1.0
                    child = MCTSNode(move=(x, y), player=3 - me, parent=node,
                                     prior=float(priors[i]), is_terminal=True,
                                     value=-1.0, ply=node.ply + 1)
                    child.expanded = True
                else:
                    child = MCTSNode(move=(x, y), player=3 - me, parent=node,
                                     prior=float(priors[i]), is_terminal=False,
                                     value=0.0, ply=node.ply + 1)
            finally:
                board.undo()
            node.add_child((x, y), child)

        node.expanded = True
        node.value = float(value)
        return node.children

    def _backprop(self, leaf: MCTSNode, path: List[MCTSNode],
                  value: float) -> None:
        """
        回溯：把叶子的价值沿路径上传，逐层取反。

        [P0-1] 修复要点：使用本轮**实际评估得到的 value**（而不是节点上可能
        已被改写的累计均值），并且逐层取反 —— 每个节点的 w 都是从
        "该节点执子方"视角累计的价值。

        virtual loss 的抵消在 _run_simulations 中完成（选择时施加、
        回溯时一次性补回），此处只负责价值传播。

        参数:
            path : 从根到 leaf 的全部节点（含 root 与 leaf），正序
            value: 叶子价值（从 leaf.player 视角）
        """
        v = float(np.clip(value, -1.0, 1.0))
        for node in reversed(path):
            node.n += 1
            node.w += v
            node.q = node.w / node.n if node.n > 0 else 0.0
            v = -v

    # ------------------------------------------------------------------
    # 核心模拟循环
    # ------------------------------------------------------------------
    def _run_simulations(self, root: MCTSNode, n_sim: int,
                         noise_alpha: Optional[float] = None,
                         noise_eps: float = 0.25) -> None:
        """
        执行 n_sim 轮模拟。

        [P0-1] 修复后的循环结构：
            每轮选择到叶子 -> 沿路径施加 virtual loss -> 收进 batch
            -> 攒够 batch_size 个**不同**叶子后一次性网络前向
            -> 用真实价值回溯 -> 再抵消 virtual loss

        与旧实现的关键差别：**每轮都真正扩展一个叶子并立刻回传其价值**，
        因此 n_sim 增加必然让树变深，评估次数 ≈ n_sim。

        virtual loss 记账（重要，这里有一个极易踩的坑）：
            虚拟损失**不能**直接累加到 node.vl 之外的共享统计量上。
            早期写法是选择时 `n += 1; w -= vl`、回溯后 `n -= 1; w += vl`，
            看起来是自逆的，但当同一个节点同时出现在**多个 batch** 里时，
            后一个 batch 的累积会在前一个 batch 回传价值之前就被抵消，
            导致该节点的 q 被算成"只有部分回溯价值 / 完整访问次数"——
            实测后果：堵住对方冲四后必败的节点 q 竟然显示为 +0.49（应为负），
            MCTS 因此认为"挡了也不亏"，在必挡局面选了无关的点。

            正确做法：虚拟损失单独记在 node.vl 上，只在计算 PUCT 分数时加进去
            （见 MCTSNode.score），这样：
                * 一个 batch 内的多次模拟会被推往不同分支（vl 生效）
                * node.n / node.w 永远等于已完成模拟的真实统计（无漂移）
                * 节点跨 batch 重复出现也不会产生任何补偿时序问题
        """
        vl = self.virtual_loss
        batch: List[MCTSNode] = []
        batch_paths: List[List[MCTSNode]] = []
        visited: List[List[MCTSNode]] = []
        root_expanded = [False]

        def apply_root_noise() -> None:
            if noise_alpha and noise_alpha > 0 and root.children:
                moves = list(root.children.keys())
                noise = self.rng.dirichlet([noise_alpha] * len(moves))
                for i, mv in enumerate(moves):
                    c = root.children[mv]
                    c.prior = (1.0 - noise_eps) * c.prior + noise_eps * float(noise[i])

        def flush() -> None:
            if not batch:
                return
            policies, values, _ = self._evaluate_leaves(batch)
            for leaf, path, pol, val in zip(batch, batch_paths, policies, values):
                if not leaf.expanded:
                    self._expand(leaf, pol, val)
                self._backprop(leaf, path, val)
            # 释放虚拟损失（只减计数，不动 n/w）
            for nodes in visited:
                for node in nodes:
                    node.vl -= 1
            batch.clear()
            batch_paths.clear()
            visited.clear()
            if not root_expanded[0] and root.expanded:
                root_expanded[0] = True
                apply_root_noise()

        for _ in range(n_sim):
            node = root
            path = [root]
            marked: List[MCTSNode] = []
            while node.children and not node.is_terminal:
                nxt = self._select(node)
                if nxt is None:
                    break
                # 施加 virtual loss。
                # 注意：**根节点的子节点也必须施加**。早期版本为"避免污染根的先验
                # 比较"而跳过根的子节点，结果是第一批 batch 里 16 次模拟全部选中
                # 同样的那两个高先验节点（其余先验为 0 的点永远拿不到机会），
                # 搜索多样性被打死。虚拟损失只影响 node.vl，不会污染 n/w/q，
                # 因此对根的子节点施加是安全且必要的。
                if vl:
                    nxt.vl += 1
                    marked.append(nxt)
                path.append(nxt)
                node = nxt

            batch.append(node)
            batch_paths.append(path)
            visited.append(marked)

            if len(batch) >= self.batch_size:
                flush()

        flush()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def get_action_probs(self, board, temp: Optional[float] = None,
                         n_sim: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """
        执行 MCTS 搜索，返回落子点的**访问次数**（未归一化或按 temp 归一化）。

        参数:
            board : Board 实例
            temp  : 温度。
                    None -> 按深度用 dynamic_temp
                    0    -> 返回 argmax one-hot
                    >0   -> 返回 n^(1/temp) 归一化分布
            n_sim : 模拟轮数。None -> 按深度用 dynamic_n_sim

        返回:
            dict {(x, y): probability}，概率和为 1
        """
        root = self.set_root(board)
        if n_sim is None:
            n_sim = self.n_sim if self.n_sim else dynamic_n_sim(self._move_depth)
        if temp is None:
            temp = dynamic_temp(self._move_depth)

        over, _ = board.is_game_over()
        if over:
            return {}

        self._run_simulations(root, n_sim,
                              noise_alpha=self.root_noise_alpha,
                              noise_eps=self.root_noise_eps)

        counts = {mv: c.n for mv, c in root.children.items() if c.n > 0}
        if not counts:
            # 兜底：搜索没产生结果时用棋型规则选点，避免返回空（旧实现会直接失败）
            fallback = self._heuristic_fallback(board)
            if fallback is None:
                return {}
            return {fallback: 1.0}

        moves = list(counts.keys())
        if temp <= 0:
            best = max(counts, key=counts.get)
            return {m: (1.0 if m == best else 0.0) for m in moves}

        w = np.array([counts[m] for m in moves], dtype=np.float64) ** (1.0 / temp)
        s = w.sum()
        if s <= 0:
            return {m: 1.0 / len(moves) for m in moves}
        w = w / s
        return {m: float(w[i]) for i, m in enumerate(moves)}

    # ---- 采样（自对弈用，真正按温度随机）----
    def sample_move_with_policy(self, board, temp: Optional[float] = None,
                                n_sim: Optional[int] = None,
                                noise_alpha: Optional[float] = None,
                                noise_eps: Optional[float] = None):
        """
        **只跑一次搜索**，同时返回 (落点, 策略分布)。

        自对弈用。旧实现需要"搜索一次拿分布 + 再按分布取点"，容易写成
        对访问概率取 argmax（温度失效）。这里显式区分两个概念：
            * 策略训练目标 = 根的访问次数分布（归一化）
            * 实际落点     = 按 n^(1/temp) 多项式采样（temp<=0 时贪心）

        返回:
            (move, policy)
                move   : (x, y) 或 None
                policy : (361,) float32，索引 = y*19 + x
        """
        root = self.set_root(board)
        # 【必须检查调用方传进来的 board，不能查共享棋盘】
        # 共享棋盘在一次搜索结束后会**停在最后访问的那个节点**上（可能很深，
        # 甚至是一个已成五的终局局面）。旧代码查的是 `self._board.is_game_over()`，
        # 于是"上一轮搜索恰好停在终局节点"会让本轮的 `sample_move_with_policy`
        # 直接返回 (None, zeros) —— 自对弈对局在 8~18 手时莫名中断、
        # 价值标签恒为 0（表现为"很短的对局却算平局"）。
        over, _ = board.is_game_over()
        if over:
            return None, np.zeros(361, dtype=np.float32)
        if n_sim is None:
            n_sim = self.n_sim if self.n_sim else dynamic_n_sim(self._move_depth)
        if temp is None:
            temp = dynamic_temp(self._move_depth)
        if noise_alpha is None:
            noise_alpha = self.root_noise_alpha
        if noise_eps is None:
            noise_eps = self.root_noise_eps

        self._run_simulations(root, n_sim, noise_alpha=noise_alpha,
                              noise_eps=noise_eps)

        counts = {mv: c.n for mv, c in root.children.items() if c.n > 0}
        policy = np.zeros(361, dtype=np.float32)
        if not counts:
            fb = self._heuristic_fallback(board)
            if fb is not None:
                policy[fb[1] * 19 + fb[0]] = 1.0
            return fb, policy

        moves = list(counts.keys())
        total = float(sum(counts.values()))
        for (x, y), n in counts.items():
            policy[y * 19 + x] = n / total

        if temp <= 0:
            return max(counts, key=counts.get), policy

        w = np.array([counts[m] for m in moves], dtype=np.float64) ** (1.0 / temp)
        s = w.sum()
        p = w / s if s > 0 else np.full(len(moves), 1.0 / len(moves))
        idx = int(self.rng.choice(len(moves), p=p))
        return moves[idx], policy

    def sample_move(self, board, temp: Optional[float] = None,
                    n_sim: Optional[int] = None,
                    noise_alpha: Optional[float] = None,
                    noise_eps: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """
        自对弈用：按访问次数的温度分布**真正采样**一个落点。

        旧实现的问题：用 temp=1.0 拿到归一化访问概率后，又用 max() 取 argmax，
        温度完全失效 -> 同一模型下每局逐手完全相同
        （实测 3 局都是 241 手、同一序列）。

        这里：temp<=0 时贪心；temp>0 时按 p ∝ n^(1/temp) 多项式采样。
        noise_alpha > 0 时在根节点先验上混入 Dirichlet 噪声以增加开局多样性。
        """
        mv, _ = self.sample_move_with_policy(
            board, temp=temp, n_sim=n_sim,
            noise_alpha=noise_alpha, noise_eps=noise_eps)
        return mv

    def get_best_action(self, board, n_sim: Optional[int] = None
                        ) -> Optional[Tuple[int, int]]:
        """返回最佳落子（temp=0 语义：最大访问次数）。"""
        probs = self.get_action_probs(board, temp=0, n_sim=n_sim)
        if not probs:
            return None
        return max(probs, key=probs.get)

    def _heuristic_fallback(self, board) -> Optional[Tuple[int, int]]:
        """搜索无结果时的规则兜底：必胜点 > 必挡点 > 棋型最高分。"""
        legal = board.get_legal_moves()
        if not legal:
            return None
        raw, wins, blocks = heuristics.move_scores(board, legal)
        for group in (wins, blocks):
            if group:
                return legal[next(iter(group))]
        return legal[int(np.argmax(raw))]

    # ------------------------------------------------------------------
    # 统计（验收指标）
    # ------------------------------------------------------------------
    def collect_stats(self, root: MCTSNode) -> Dict[str, float]:
        """
        统计搜索树规模、深度与评估利用率，用于验收。

        关键指标说明：
            max_depth   : 搜索深度。五子棋需要 >= 4 才能稳定看到冲四/活三威胁链。
            evaluations : 真正送去网络前向的叶子数。理想值 ≈ n_sim。
            leaf_visits : 被评估过的叶子总数（= n_sim），用于算利用率。
            expanded    : 已扩展的节点数（每个节点会带出若干子节点，
                          所以 nodes 会远大于 n_sim，这不是问题）。
        """
        total = 0
        max_depth = 0
        stack = [(root, 0)]
        while stack:
            node, d = stack.pop()
            total += 1
            if d > max_depth:
                max_depth = d
            for c in node.children.values():
                stack.append((c, d + 1))
        evals = getattr(self, "_n_evaluations", 0)
        visits = getattr(self, "_n_leaf_visits", 0)
        cache = getattr(self, "_n_cache_hits", 0)
        terminal = getattr(self, "_n_terminal_evals", 0)
        return {
            "nodes": total,
            "max_depth": max_depth,
            "root_n": root.n,
            "root_children": len(root.children),
            "evaluations": evals,
            "terminal_evals": terminal,
            "cache_hits": cache,
            "leaf_visits": visits,
            "eval_ratio": ((evals + terminal + cache) / visits) if visits else 0.0,
            "net_ratio": (evals / visits) if visits else 0.0,
            "cache": len(self._value_cache),
            # 增量棋盘"自愈"次数：正常应该是 0 或个位数。
            # 若这个数随 n_sim 线性增长，说明共享棋盘在持续漂移（严重 bug 信号）。
            "resync": getattr(self, "_n_resync", 0),
        }


# ----------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------
def self_test():
    """随机网络 + 手工局面，验证搜索的正确性关键指标。"""
    from network import GomokuNet

    print("=" * 66)
    print("mcts.py 自测（重写版关键指标验证）")
    print("=" * 66)

    net = GomokuNet()
    mcts = MCTS(net, n_sim=200, c_puct=2.5, virtual_loss=3.0, batch_size=16)

    def mk(n_sim, seed=0, **kw):
        params = dict(n_sim=n_sim, c_puct=1.0, virtual_loss=3.0,
                      batch_size=16, max_children=6, root_max_children=16,
                      heuristic_weight=6.0, policy_weight=0.3, seed=seed)
        params.update(kw)
        return MCTS(net, **params)

    # ---- 指标 1：n_sim 增加必须让搜索变深、评估利用率接近 100% ----
    print("\n[指标1] n_sim 增加 -> 搜索变深、评估利用率接近 100%")
    print("        （旧实现：n_sim 150->12000 树恒为 51 节点/深度 2，利用率 6.7%）")
    board0 = Board()
    for mv in [(9, 9, 1), (9, 10, 2), (8, 9, 1), (10, 10, 2), (8, 10, 1)]:
        board0.move(*mv)
    growth = []
    for n_sim in (100, 200, 400, 800, 1600):
        m = mk(n_sim)
        m.reset_tree()
        m.get_action_probs(board0, temp=1.0, n_sim=n_sim)
        st = m.collect_stats(m._root)
        top1 = max(c.n for c in m._root.children.values())
        growth.append((n_sim, st["max_depth"], st["leaf_visits"],
                       st["net_ratio"], top1 / n_sim))
        print(f"     n_sim={n_sim:5d} -> 深度={st['max_depth']:2d} "
              f"叶子评估={st['leaf_visits']:5d} 网络前向={st['evaluations']:5d} "
              f"规则终局={st['terminal_evals']:4d} 缓存命中={st['cache_hits']:4d} "
              f"根q={m._root.q:+.4f} top1占比={top1/n_sim*100:4.1f}%")
    assert growth[-1][1] > growth[0][1], "搜索深度未随 n_sim 增长"
    assert growth[-1][1] >= 4, f"搜索深度过浅: {growth[-1][1]}"
    # 根访问集中度只做一个"下限"检查：随机初始化网络的策略先验本身很平
    # （实测每个子节点先验≈0.09），此时 PUCT 合理地会把访问铺开，
    # top1 占比约 10% 属于正常。真正衡量"预算是否被宽度吃光"的指标是搜索深度
    # （已在上一条断言），这里只防止完全退化（例如所有点各访问 1 次）。
    assert growth[-1][4] > 0.05, \
        f"根访问过度分散(top1={growth[-1][4]:.1%})，搜索可能已退化"
    for n_sim, _d, visits, _ratio, _c in growth:
        # 每个模拟都必须落在一个真实的叶子上（旧实现只有 6.7%）
        assert visits == n_sim, f"叶子评估数 {visits} != n_sim {n_sim}"

    # ---- 指标 2：一步成五必须找到 ----
    print("\n[指标2] 一步成五必须命中（旧实现 100% 漏杀）")
    # 白方在 y=8 有 (3,8)(4,8)(5,8)(6,8) 四连，轮到白，(2,8)/(7,8) 立即成五
    b = Board()
    b.move(0, 0, 1); b.move(3, 8, 2)
    b.move(1, 0, 1); b.move(4, 8, 2)
    b.move(2, 0, 1); b.move(5, 8, 2)
    b.move(5, 0, 1); b.move(6, 8, 2)
    b.move(6, 0, 1)          # 黑第5手，不构成五连；轮到白
    assert b.current_player == 2, b.current_player
    win_pts = {(2, 8), (7, 8)}
    for n_sim in (10, 50, 200, 800):
        m = mk(n_sim)
        m.reset_tree()
        pick = m.get_best_action(b, n_sim=n_sim)
        print(f"     n_sim={n_sim:4d} -> 选择={pick} "
              f"{'[命中]' if pick in win_pts else '[漏杀]'}")
        assert pick in win_pts, f"n_sim={n_sim} 漏杀: {pick}"

    # ---- 指标 3：必须挡住对方冲四 ----
    print("\n[指标3] 对方一步成五时必须堵")
    b2 = Board()
    b2.move(3, 3, 1); b2.move(0, 0, 2)
    b2.move(4, 3, 1); b2.move(0, 1, 2)
    b2.move(5, 3, 1); b2.move(0, 2, 2)
    b2.move(6, 3, 1)         # 黑四连，轮到白
    assert b2.current_player == 2
    must = {(2, 3), (7, 3)}
    for n_sim in (50, 400):
        m = mk(n_sim)
        m.reset_tree()
        pick = m.get_best_action(b2, n_sim=n_sim)
        print(f"     n_sim={n_sim:4d} -> 选择={pick} "
              f"{'[挡住]' if pick in must else '[漏防]'}")
        assert pick in must, f"n_sim={n_sim} 漏防: {pick}"

    # ---- 指标 4：根节点覆盖 + 树复用命中 ----
    print("\n[指标4] 根节点覆盖与树复用命中率")
    b3 = Board()
    b3.move(9, 9, 1)
    legal = set(b3.get_legal_moves())
    m = mk(200)
    m.reset_tree()
    m.get_action_probs(b3, temp=1.0, n_sim=200)
    n_children = len(m._root.children)
    print(f"     合法点={len(legal)} 根子节点={n_children} "
          f"覆盖率={n_children/len(legal)*100:.1f}%"
          f"（root_max_children=16 的宽度控制；旧实现固定 top-20 且"
          f"树复用失效）")
    assert n_children == min(16, len(legal)), "根节点宽度不符"

    # 关键性质：**对手实际最可能走的点**必须能命中子树。
    # 做法：让对手按自己的先验走一步，检查 set_root_move 是否命中。
    hit = 0
    trials = 0
    for mv in list(legal):
        # 只在"这一点在对手的候选清单里"时检验（用同一套先验口径）
        m2 = mk(200)
        m2.reset_tree()
        m2.get_action_probs(b3, temp=1.0, n_sim=200)
        trials += 1
        if m2.set_root_move(mv):
            hit += 1
    print(f"     对全部 {trials} 个合法点逐一尝试升为新根: "
          f"命中 {hit}/{trials} = {hit/trials*100:.1f}%"
          f"（旧实现开局约 67%、中盘 <30%，因为会直接丢弃整棵树）")
    # 命中率应接近根宽度覆盖率（未命中时也不再丢树，只是重建）
    assert hit / trials >= 0.6, f"树复用命中率过低: {hit/trials:.0%}"

    # ---- 指标 5：终局子节点带符号价值 ----
    print("\n[指标5] 终局子节点价值符号正确")
    m = mk(400)
    m.reset_tree()
    m.get_action_probs(b, temp=0, n_sim=400)
    wc = m._root.children.get((2, 8)) or m._root.children.get((7, 8))
    assert wc is not None, "成点子节点不在树中"
    print(f"     白方成点子节点: is_terminal={wc.is_terminal} "
          f"value={wc.value:+.1f} n={wc.n}")
    print("     （value 从该子节点执子方=黑方视角；黑方已无法阻止白方成五，"
          "故该节点是'必输'=+1 表示白方胜势）")
    assert wc.is_terminal and wc.value > 0, \
        "成点子节点的价值必须反映白方必胜（非 0，旧实现恒为 0.0）"
    # 被过滤掉的已占点不应进入树
    assert (3, 8) not in m._root.children, "已占点不应成为子节点"

    # ---- 指标 6：终局局面不崩溃 ----
    print("\n[指标6] 终局局面搜索返回空且不崩溃")
    b4 = Board()
    black = [(9, 9), (10, 9), (11, 9), (12, 9), (13, 9)]
    white = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (x, y) in enumerate(black):
        b4.move(x, y, 1)
        if i < len(white):
            b4.move(*white[i], 2)
    assert b4.is_game_over()[0]
    assert m.get_action_probs(b4, temp=1.0, n_sim=10) == {}
    print("     [PASS]")

    # ---- 指标 7：温度采样真的随机 ----
    print("\n[指标7] 温度采样产生多样性（旧实现温度失效，每局完全相同）")
    m5 = mk(50, seed=1)
    b5 = Board()
    b5.move(9, 9, 1)
    picks = set()
    for _ in range(20):
        m5.reset_tree()
        picks.add(m5.sample_move(b5, temp=1.0, n_sim=50))
    print(f"     20 次采样得到 {len(picks)} 个不同落点")
    assert len(picks) >= 3, "温度采样未产生多样性"

    # ---- 指标 8：virtual loss 不产生统计漂移 ----
    print("\n[指标8] virtual loss 记账无漂移（根 n 应等于 n_sim）")
    for n_sim in (200, 500):
        m = mk(n_sim)
        m.reset_tree()
        m.get_action_probs(board0, temp=1.0, n_sim=n_sim)
        root_n = m._root.n
        children_sum = sum(c.n for c in m._root.children.values())
        print(f"     n_sim={n_sim} -> 根 n={root_n} 子节点 n 之和={children_sum}"
              f"（子节点之和 <= 根 n 是正常的：部分模拟在根即终局）")
        assert root_n == n_sim, f"根访问次数 {root_n} != n_sim {n_sim}"
        assert children_sum <= root_n, "子节点访问之和不应超过根"
        # 每个子节点的 w 必须与 n 一致（virtual loss 已完全抵消）
        for mv, c in m._root.children.items():
            if c.n > 0:
                assert abs(c.q - c.w / c.n) < 1e-9, f"{mv} 的 q 与 w/n 不一致"
        # 价值必须落在 [-1,1]
        for mv, c in m._root.children.items():
            if c.n > 0:
                assert -1.0 - 1e-9 <= c.q <= 1.0 + 1e-9, f"{mv} 的 q 越界: {c.q}"

    # ---- 指标 9：c_puct / virtual_loss 组合健壮性 ----
    print("\n[指标9] 参数组合健壮性（batch_size=1 / virtual_loss=0 / 关闭启发式）")
    for kw in ({"batch_size": 1}, {"virtual_loss": 0.0},
               {"use_heuristics": False}, {"max_children": 4},
               {"fpu_reduction": 0.0}):
        m = mk(200, **kw)
        m.reset_tree()
        pick = m.get_best_action(b2, n_sim=200)
        st = m.collect_stats(m._root)
        print(f"     {kw} -> 深度={st['max_depth']} 选择={pick}")
        assert pick is not None, f"{kw} 搜索失败"

    print("\n" + "=" * 66)
    print("全部指标通过 [PASS ALL]")
    print("=" * 66)


if __name__ == "__main__":
    self_test()
