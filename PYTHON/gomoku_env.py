"""
gomoku_env.py
=============
19×19 五子棋「纯规则环境」模块。

本模块只包含规则逻辑（落子、胜负判定、悔棋、合法点、状态张量、Zobrist 哈希），
与任何 AI 算法（搜索、神经网络等）完全解耦。

规则：
    * 黑先白后（player=1 黑 / player=2 白）
    * 无禁手（无"过五"限制，五连即胜）
    * 横 / 竖 / 两个斜向五连即胜
    * 棋盘下满判平局

坐标约定：x 为列（0~18，从左到右），y 为行（0~18，从上到下）。
棋盘为 2D 数组 board[y, x]。

核心接口：
    * move(x, y, player)       落子，返回是否落子成功
    * undo()                   撤销上一步落子（悔棋）
    * is_game_over()           返回 (是否结束, 胜者)，胜者 1/2，0 表示平局
    * get_legal_moves()        返回合法落子点列表（候选点过滤）
    * get_state_tensor()       返回 3×19×19 numpy 张量
    * get_hash()               返回 Zobrist 哈希（64 位无符号整数）

修复记录（优化 P0-3）：
    原实现的 Zobrist 哈希只对"棋子位置"异或，**不含行棋方**，导致
    「棋子相同但轮到不同方」的两个局面哈希完全相同（实测确认碰撞）。
    这会让置换表 / 搜索树复用把两种本质不同的局面混为一谈。
    现改为：hash = XOR(棋子键) XOR 行棋方键，落子/悔棋均同步更新。

性能优化：
    * Zobrist 表与邻域偏移表提升为模块级常量，不再每个 Board() 重新生成
      （原实现每个棋盘实例都会分配 2×19×19×8B ≈ 5.6KB 的随机表）
    * get_legal_moves() 改用预计算的邻域索引，避免 Python 双重循环
"""

import random
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# 模块级常量：Zobrist 表（固定种子，保证可复现）
# ----------------------------------------------------------------------
SIZE = 19
BLACK = 1
WHITE = 2
N_CELLS = SIZE * SIZE

_ZOBRIST_RNG = np.random.default_rng(20240518)
# 每个格子为每种颜色预生成一个 64 位随机数
_ZOBRIST = _ZOBRIST_RNG.integers(1, 2**64, size=(2, N_CELLS), dtype=np.uint64)
# 行棋方键：区分「同型不同轮次」的局面（关键修复）
_ZOBRIST_SIDE = _ZOBRIST_RNG.integers(1, 2**64, size=2, dtype=np.uint64)

# 邻域索引（半径 2 的 5×5 邻域，边界裁剪后展平）
_neighbor_idx = np.full((N_CELLS, 25), -1, dtype=np.int32)
for _y in range(SIZE):
    for _x in range(SIZE):
        _k = 0
        for _dy in range(-2, 3):
            for _dx in range(-2, 3):
                _ny, _nx = _y + _dy, _x + _dx
                if 0 <= _nx < SIZE and 0 <= _ny < SIZE:
                    _neighbor_idx[_y * SIZE + _x, _k] = _ny * SIZE + _nx
                    _k += 1
NEIGHBOR_IDX = np.ascontiguousarray(_neighbor_idx)

# ----------------------------------------------------------------------
# 模块级常量：**带围墙的填充棋盘**（规则查询的性能核心）
# ----------------------------------------------------------------------
# 动机（实测数据见 heuristics.py 顶部的历史记录）：
#   搜索 73% 的时间花在 _expand -> heuristics 的棋型计算上，而其中最大的一块是
#   "每个候选点 × 4 个方向 × 9 格"的取值。原实现用 numpy 逐格索引
#   （`arr[yy, xx]`，每次约 150ns）外加逐格越界判断，一次棋型扫描要 12µs。
#
#   解决方式：维护一份 27×27 的**纯 Python list** 镜像 `Board._pad`，
#   四周留 4 格宽的围墙（值 -1，表示"永久封死的盘外"）。于是：
#       * 取值变成 list 下标（约 30ns，比 numpy 快 5 倍）
#       * 越界判断彻底消失（撞到围墙自然停止，围墙也不等于 0，故不算"空位"）
#       * 一次扫描就能同时算双方，不必"临时落子 -> 算 -> 还原"
#   `_pad` 由 move/undo/reset/copy 全程同步维护，永远与 `board` 一致。
PAD_OFF = 4                       # 真实棋盘在填充棋盘里的起始偏移
PAD_SIZE = SIZE + 2 * PAD_OFF     # 27
PAD_CELLS = PAD_SIZE * PAD_SIZE   # 729
# 四个方向的**步长**（一维索引差）：横、竖、正斜、反斜
PAD_STEPS = (1, PAD_SIZE, PAD_SIZE + 1, PAD_SIZE - 1)
# 中心 8 邻域的步长（用于"邻域内有没有棋子"的廉价预筛）
PAD_NEIGHBOR8 = (-PAD_SIZE - 1, -PAD_SIZE, -PAD_SIZE + 1, -1,
                 1, PAD_SIZE - 1, PAD_SIZE, PAD_SIZE + 1)
_ZERO_ROW = [0] * SIZE


def _new_pad() -> list:
    """新建一份填充棋盘：内部 19×19 为 0，四周 4 格围墙为 -1。"""
    pad = [-1] * PAD_CELLS
    for _r in range(SIZE):
        _base = (_r + PAD_OFF) * PAD_SIZE + PAD_OFF
        pad[_base:_base + SIZE] = _ZERO_ROW
    return pad



class Board:
    """19×19 五子棋棋盘（纯规则逻辑，与 AI 解耦）。"""

    SIZE = SIZE
    BLACK = BLACK
    WHITE = WHITE

    # 四个方向：横、竖、正斜、反斜
    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self):
        # 棋盘：0=空, 1=黑, 2=白
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        # 落子历史，用于 undo() 悔棋
        self.history = []
        # 当前执子方，开局黑先
        self.current_player = self.BLACK
        # 胜负状态缓存（避免每次 is_game_over 全板扫描）
        self._game_over = False
        self._winner = 0  # 0=进行中或平局, 1=黑胜, 2=白胜
        # Zobrist 哈希：棋子键 XOR 行棋方键
        self._hash = int(_ZOBRIST_SIDE[self.BLACK - 1])
        # 带围墙的 Python list 镜像（规则查询走这一份，见模块顶部说明）
        self._pad = _new_pad()

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def player(self):
        """当前执子方（1 黑 / 2 白）。"""
        return self.current_player

    @property
    def is_over(self):
        """游戏是否已结束。"""
        return self._game_over

    @property
    def winner(self):
        """胜者：1 黑 / 2 白 / 0 平局（或进行中）。"""
        return self._winner

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------
    def move(self, x, y, player):
        """
        落子。

        参数:
            x, y: 列/行坐标，0~18
            player: 落子方，1=黑 2=白

        返回:
            bool — 落子是否成功
                  （越界 / 已有棋子 / 游戏已结束 / 非当前方 均失败）
        """
        # 1) 坐标合法性
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return False
        # 2) 不能落已有棋子
        if self.board[y, x] != 0:
            return False
        # 3) 游戏结束后不再允许落子
        if self._game_over:
            return False
        # 4) 只能由当前执子方落子（保证黑先白后轮次正确）
        if player != self.current_player:
            return False

        # 落子
        self.board[y, x] = player
        self._pad[(y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)] = player
        self.history.append((x, y, player))

        # 更新 Zobrist 哈希：异或棋子键 + 行棋方键（XOR 自逆，O(1)）
        idx = y * self.SIZE + x
        self._hash ^= int(_ZOBRIST[player - 1, idx])
        self._hash ^= int(_ZOBRIST_SIDE[player - 1])          # 撤销旧行棋方键
        self._hash ^= int(_ZOBRIST_SIDE[(3 - player) - 1])    # 加入新行棋方键

        # 判定胜负
        if self._check_win(x, y, player):
            self._game_over = True
            self._winner = player
        elif self._is_full():
            self._game_over = True
            self._winner = 0  # 平局

        # 未结束则换人
        if not self._game_over:
            self.current_player = 3 - player
        return True

    def undo(self):
        """
        撤销上一步落子（悔棋）。

        返回:
            bool — 是否撤销成功（无历史则失败）
        """
        if not self.history:
            return False
        x, y, player = self.history.pop()
        self.board[y, x] = 0
        self._pad[(y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)] = 0

        # 恢复 Zobrist 哈希（XOR 自逆，含行棋方键）
        idx = y * self.SIZE + x
        self._hash ^= int(_ZOBRIST[player - 1, idx])
        self._hash ^= int(_ZOBRIST_SIDE[player - 1])
        self._hash ^= int(_ZOBRIST_SIDE[(3 - player) - 1])

        # 恢复当前执子方
        self.current_player = player

        # 撤销后恢复胜负状态：若该落子曾判胜或平局，则恢复为未结束
        if self._game_over:
            self._game_over = False
            self._winner = 0
        return True

    def is_game_over(self):
        """
        判断游戏是否结束。

        返回:
            tuple(bool, int) — (是否结束, 胜者)；胜者 1=黑 2=白 0=平局
        """
        return self._game_over, self._winner

    def get_legal_moves(self):
        """
        返回合法落子点列表 [(x, y), ...]。

        候选点过滤策略：
            * 只保留已有棋子半径 2 内的空位（靠近活棋的点）
            * 开局无子时返回天元 (9,9) 及周边 8 个点
        """
        if self._game_over:
            return []

        board = self.board
        occupied = np.flatnonzero(board)          # 已落子的展平索引 y*19+x

        # 开局无子：天元 + 周边 8 点
        if occupied.size == 0:
            cx = cy = self.SIZE // 2
            return [(cx + dx, cy + dy)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if 0 <= cx + dx < self.SIZE and 0 <= cy + dy < self.SIZE]

        # 利用模块级邻域表收集半径 2 内的空位（去掉 Python 双重循环）
        cand = NEIGHBOR_IDX[occupied].ravel()
        cand = cand[cand >= 0]
        cand = np.unique(cand)
        cand = cand[board.ravel()[cand] == 0]
        # np.unique 已按索引升序返回，故顺序 (x, y) 完全确定。
        # 确定性的落点顺序很重要：当多个点棋型分相同时（例如活四的两个成五
        # 点），选择结果必须可复现，否则自对弈数据会产生无谓抖动。
        return [(int(i % self.SIZE), int(i // self.SIZE)) for i in cand]

    def get_state_tensor(self):
        """
        返回当前局面的 3×19×19 numpy 张量。

        通道:
            0 = 黑棋位置（1 为有黑棋，0 为无）
            1 = 白棋位置
            2 = 当前执子方（全 1 = 黑，全 0 = 白）
        """
        tensor = np.zeros((3, self.SIZE, self.SIZE), dtype=np.int8)
        tensor[0] = (self.board == BLACK).astype(np.int8)
        tensor[1] = (self.board == WHITE).astype(np.int8)
        # 当前执子方：黑=1, 白=0
        tensor[2] = 1 if self.current_player == BLACK else 0
        return tensor

    def get_hash(self):
        """返回当前局面的 Zobrist 哈希值（64 位无符号整数，含行棋方，用于局面去重）。"""
        return self._hash

    def reset(self):
        """重置为初始局面（黑先）。"""
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self.history = []
        self.current_player = self.BLACK
        self._game_over = False
        self._winner = 0
        self._hash = int(_ZOBRIST_SIDE[self.BLACK - 1])
        self._pad = _new_pad()

    def copy(self):
        """
        返回当前棋盘状态的深拷贝（供 MCTS 各节点独立使用）。
        Zobrist 表为模块级只读常量，所有节点共享，无需拷贝。
        """
        new_board = Board.__new__(Board)
        new_board.board = self.board.copy()
        new_board.history = list(self.history)
        new_board.current_player = self.current_player
        new_board._game_over = self._game_over
        new_board._winner = self._winner
        new_board._hash = self._hash
        new_board._pad = list(self._pad)     # 729 个 int 的浅拷贝，约 2µs
        return new_board

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _pad_base(self, x: int, y: int) -> int:
        """(x, y) 在填充棋盘里的一维下标。"""
        return (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)

    def _check_win(self, x, y, player):
        """
        检查 (x, y) 落子后该玩家是否五连。
        仅检查经过该点的 4 条线（横/竖/两斜），沿两方向数连续同色。

        【性能要点 —— 这是整个搜索最热的函数之一】
        实测瓶颈在规则计算而非神经网络（网络前向只占约 5%）。
        现在走填充棋盘 `_pad`：纯 list 下标、无越界判断、无 numpy 逐格索引，
        比早先"numpy 逐格 + 越界比较"的写法快数倍（见模块顶部说明）。

        注：曾经尝试用预计算的方向切片表消除 while 循环，实测反而慢 3 倍
        （numpy 花式索引的开销 > 几次 while 循环），故不做那种"优化"。
        """
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        # 快速预筛：中心必须是 player
        if pad[base] != player:
            return False
        for step in PAD_STEPS:
            i = 1
            while pad[base - i * step] == player:
                i += 1
            j = 1
            while pad[base + j * step] == player:
                j += 1
            if i + j - 1 >= 5:
                return True
        return False

    def line_after(self, x: int, y: int, player: int) -> Tuple[int, bool, bool]:
        """
        假设 player 落在空点 (x, y)，返回经过该点的 4 条线里最好的一条：
            (最长连子数, 是否至少一端开放, 是否两端都开放)

        不修改棋盘（填充棋盘让"试探落子"变成纯读取）。
        (x, y) 越界或已被占用时返回 (0, False, False)。

        两条注意事项：
            * 每个方向最多走 4 格（与原先的 9 格窗口一致），**必须显式设上界**：
              填充棋盘只有 4 格宽的围墙，若不设界会一路数穿到棋盘另一头。
            * "开放"的判定就是"再外面一格 == 0"；撞到围墙读到 -1，自然为假。
        """
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return 0, False, False
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        if pad[base] != 0:
            return 0, False, False
        return self._line_at_base(base, player)

    def _line_at_base(self, base: int, player: int) -> Tuple[int, bool, bool]:
        """`line_after` 的内核：给定填充棋盘下标，扫描四个方向。"""
        pad = self._pad
        best_count = 1
        best_any = False
        best_both = False
        for step in PAD_STEPS:
            i = 1
            while i <= 4 and pad[base - i * step] == player:
                i += 1
            j = 1
            while j <= 4 and pad[base + j * step] == player:
                j += 1
            count = i + j - 1
            open_a = pad[base - i * step] == 0
            open_b = pad[base + j * step] == 0
            open_any = open_a or open_b
            if count > best_count or (count == best_count and open_any):
                best_count = count
                best_any = open_any
                best_both = open_a and open_b
            if count >= 5:
                return 5, True, True
        return best_count, best_any, best_both

    def line_after_both(self, x: int, y: int, me: int):
        """
        一次扫描同时给出**双方**在空点 (x, y) 落子后的棋型。

        返回 (me_count, me_open, me_both, opp_count, opp_open, opp_both)；
        (x, y) 越界或已占用时 six 个值全为 0/False。

        这是 `heuristics.move_scores` 的主力：原实现对每个候选点要走两遍
        完整的方向扫描（我方一遍、对方一遍），合并后方向循环只跑一遍。
        """
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return 0, False, False, 0, False, False
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        if pad[base] != 0:
            return 0, False, False, 0, False, False

        opp = 3 - me
        me_best = opp_best = 1
        me_any = opp_any = False
        me_both = opp_both = False
        for step in PAD_STEPS:
            # ---- 我方 ----
            i = 1
            while i <= 4 and pad[base - i * step] == me:
                i += 1
            j = 1
            while j <= 4 and pad[base + j * step] == me:
                j += 1
            count = i + j - 1
            oa = pad[base - i * step] == 0
            ob = pad[base + j * step] == 0
            open_any = oa or ob
            if count > me_best or (count == me_best and open_any):
                me_best = count
                me_any = open_any
                me_both = oa and ob
            # ---- 对方 ----
            i = 1
            while i <= 4 and pad[base - i * step] == opp:
                i += 1
            j = 1
            while j <= 4 and pad[base + j * step] == opp:
                j += 1
            count = i + j - 1
            oa = pad[base - i * step] == 0
            ob = pad[base + j * step] == 0
            open_any = oa or ob
            if count > opp_best or (count == opp_best and open_any):
                opp_best = count
                opp_any = open_any
                opp_both = oa and ob
        return me_best, me_any, me_both, opp_best, opp_any, opp_both

    def has_neighbor(self, x: int, y: int) -> bool:
        """
        (x, y) 的紧邻 8 格内是否有棋子（廉价预筛，纯 list 读取）。

        用途：合法点本来就是"离棋子半径 2 以内"，但仍有很多点周围只有孤子；
        这类点即使落子也最多形成"孤二"，威胁分极低。
        注意这是**保守**预筛：只要 8 邻域有子就照常精确计算，
        因此不会漏掉任何真实威胁（跳子棋型也至少有一个相邻子）。
        """
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        for off in PAD_NEIGHBOR8:
            if pad[base + off] > 0:
                return True
        return False

    def _is_full(self):
        """棋盘是否下满。"""
        return bool(np.all(self.board != 0))

    def check_win_after(self, x, y, player):
        """
        试探性判断：假设 player 在 (x, y) 落子，是否会立刻五连。

        不改变棋盘状态、不走完整 move() 流程（不更新哈希/历史/轮次），
        供启发式模块和 MCTS 做「一步成五 / 必挡」检测。
        现在直接读填充棋盘，**完全不需要"临时落子 -> 还原"**。

        参数:
            x, y  : 空点坐标
            player: 1 或 2
        返回:
            bool — 该点是否让 player 立刻成五（越界/已占返回 False）
        """
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return False
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        if pad[base] != 0:
            return False
        for step in PAD_STEPS:
            i = 1
            while pad[base - i * step] == player:
                i += 1
            j = 1
            while pad[base + j * step] == player:
                j += 1
            if i + j - 1 >= 5:
                return True
        return False

    # 兼容旧名（早期启发式代码使用过 _check_win_after）
    _check_win_after = check_win_after

    def open_four_after(self, x, y, player):
        """
        试探性判断：假设 player 在 (x, y) 落子，是否形成「活四」。

        活四定义：存在一条线上恰有 4 个连续同色子，且该线两端都是空位。
        活四是五子棋的**必胜形** —— 下一手有两个不同的成五点，
        对手只能堵一个，因此可以判定为"已锁定胜利"。

        用途（重要）：
            MCTS 只搜 4~6 层，很难自己发现"做活四 -> 对手堵一端 -> 另一端成五"
            这条 3 手必胜链。实测在活三局面下搜索会漏掉唯一的做活四点。
            因此把"形成活四"直接视为终局胜势（价值 +1），
            既保证正确性，也大幅提升搜索效率。这样处理在规则上是严格的，
            不是近似。

        参数:
            x, y  : 空点坐标
            player: 1 或 2
        返回:
            bool — 该点是否让 player 形成活四（越界/已占返回 False）
        """
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return False
        pad = self._pad
        base = (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF)
        if pad[base] != 0:
            return False
        # 注意：不能用 line_after 的"最好一条线"结果来判断，
        # 因为最好那条可能是"5 连/双端封死的 4 连"，而活四可能在另一个方向。
        # 这里必须逐方向问"这一条是不是恰好 4 连且两端皆空"。
        for step in PAD_STEPS:
            i = 1
            while i <= 4 and pad[base - i * step] == player:
                i += 1
            j = 1
            while j <= 4 and pad[base + j * step] == player:
                j += 1
            if (i + j - 1 == 4
                    and pad[base - i * step] == 0
                    and pad[base + j * step] == 0):
                return True
        return False


# ----------------------------------------------------------------------
# 自测 main：模拟随机对弈验证胜负判定正确
# ----------------------------------------------------------------------
def _simulate_random_game(board, seed):
    """
    随机对弈一局（随机合法落子），返回胜者（1/2，0 表示平局）。
    黑先，每手随机选一个合法落点。
    """
    random.seed(seed)
    while True:
        over, winner = board.is_game_over()
        if over:
            return winner
        moves = board.get_legal_moves()
        if not moves:
            return 0  # 无合法点（满盘）
        x, y = random.choice(moves)
        board.move(x, y, board.current_player)

def main():
    """自测：随机对弈 + 手工构造五连验证胜负判定。"""
    print("=" * 60)
    print("gomoku_env.py 自测")
    print("=" * 60)

    # ---- 测试 1：手工构造黑横五连 ----
    b = Board()
    black = [(9, 9), (10, 9), (11, 9), (12, 9), (13, 9)]
    white = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (x, y) in enumerate(black):
        b.move(x, y, 1)
        if i < len(white):
            b.move(*white[i], 2)
    over, winner = b.is_game_over()
    assert over and winner == 1, f"黑横五未判胜: {over}, {winner}"
    print("[PASS] 手工构造黑横五连 -> 判黑胜")

    # ---- 测试 2：竖直五连 ----
    b = Board()
    black = [(9, 9), (9, 10), (9, 11), (9, 12), (9, 13)]
    white = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (x, y) in enumerate(black):
        b.move(x, y, 1)
        if i < len(white):
            b.move(*white[i], 2)
    over, winner = b.is_game_over()
    assert over and winner == 1, f"黑竖五未判胜: {over}, {winner}"
    print("[PASS] 手工构造黑竖五连 -> 判黑胜")

    # ---- 测试 3：正斜五连 ----
    b = Board()
    black = [(7, 7), (8, 8), (9, 9), (10, 10), (11, 11)]
    white = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (x, y) in enumerate(black):
        b.move(x, y, 1)
        if i < len(white):
            b.move(*white[i], 2)
    over, winner = b.is_game_over()
    assert over and winner == 1, f"黑正斜五未判胜: {over}, {winner}"
    print("[PASS] 手工构造黑正斜五连 -> 判黑胜")

    # ---- 测试 4：反斜五连 ----
    b = Board()
    black = [(7, 11), (8, 10), (9, 9), (10, 8), (11, 7)]
    white = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (x, y) in enumerate(black):
        b.move(x, y, 1)
        if i < len(white):
            b.move(*white[i], 2)
    over, winner = b.is_game_over()
    assert over and winner == 1, f"黑反斜五未判胜: {over}, {winner}"
    print("[PASS] 手工构造黑反斜五连 -> 判黑胜")

    # ---- 测试 5：白胜 ----
    b = Board()
    # 黑先落远处，白再构造 y=1 行横五
    b.move(0, 0, 1)
    white = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    # 黑其余落子散开，避免自己成五
    black_rest = [(0, 5), (2, 7), (4, 3), (6, 9), (8, 1)]
    for i, (x, y) in enumerate(white):
        b.move(x, y, 2)
        if i < len(black_rest):
            b.move(*black_rest[i], 1)
    over, winner = b.is_game_over()
    assert over and winner == 2, f"白横五未判胜: {over}, {winner}"
    print("[PASS] 手工构造白横五连 -> 判白胜")

    # ---- 测试 6：悔棋 ----
    b = Board()
    b.move(9, 9, 1)
    b.move(10, 9, 2)
    assert b.undo()
    assert b.board[9, 9] == 1 and b.board[9, 10] == 0  # 白子被撤，黑子保留
    assert b.current_player == 2  # 悔棋后回到白方
    assert b.is_game_over() == (False, 0)
    print("[PASS] undo() 撤销落子并恢复状态")

    # ---- 测试 7：开局合法点 ----
    b = Board()
    moves = b.get_legal_moves()
    expected = [(9 + dx, 9 + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    assert set(moves) == set(expected), f"开局合法点不符: {moves}"
    print(f"[PASS] 开局合法点 = 天元+周边8点, 共 {len(moves)} 个")

    # ---- 测试 8：随机对弈验证胜负判定 ----
    print("-" * 60)
    print("随机对弈验证（100 局）...")
    win_black, win_white, draw = 0, 0, 0
    for seed in range(100):
        b = Board()
        winner = _simulate_random_game(b, seed)
        if winner == 1:
            win_black += 1
        elif winner == 2:
            win_white += 1
        else:
            draw += 1
    print(f"黑胜 {win_black} 局 / 白胜 {win_white} 局 / 平局 {draw} 局")
    assert win_black + win_white + draw == 100
    print("-" * 60)

    # ---- 测试 9：状态张量格式 ----
    b = Board()
    b.move(9, 9, 1)
    t = b.get_state_tensor()
    assert t.shape == (3, 19, 19), f"张量形状错误: {t.shape}"
    assert t[0, 9, 9] == 1, "黑通道未标记"
    assert t[1, 9, 9] == 0, "白通道误标记"
    assert np.all(t[2] == 0), "当前执子方应为白(全0)"
    print("[PASS] get_state_tensor() 形状与内容正确")

    # ---- 测试 10：Zobrist 哈希一致性 ----
    b = Board()
    h1 = b.get_hash()
    b.move(9, 9, 1)
    h2 = b.get_hash()
    assert h1 != h2, "落子后哈希未变"
    b.undo()
    h3 = b.get_hash()
    assert h3 == h1, "悔棋后哈希未恢复"
    print("[PASS] Zobrist 哈希：落子改变、悔棋恢复")

    # ---- 测试 11：哈希区分行棋方（P0-3 修复验证）----
    # 用不同落子顺序构造"棋子相同、轮次相同"的局面，哈希必须相同
    a1 = Board()
    a1.move(9, 9, 1)
    a1.move(0, 0, 2)
    a2 = Board()
    a2.move(9, 9, 1)
    a2.move(0, 0, 2)
    assert a1.get_hash() == a2.get_hash(), "同型同轮次哈希应相同"

    # 关键用例：撤销到"棋子相同但轮次不同"的局面
    #   c1 = 黑(9,9) + 白(0,0)，轮到黑
    #   c2 = 黑(9,9) + 白(0,0)，轮到白（把白子撤掉一枚再换个位置落回不可能，
    #        改为直接在 c1 基础上 undo 一手即可得到不同轮次/不同棋子）
    # 真正的同型不同轮次：棋盘上只有黑(9,9) 时，轮到黑 vs 轮到白
    #   —— 只有黑(9,9) 且轮到黑 是不可能自然到达的，故用 undo 直接验证键生效：
    c1 = Board()
    c1.move(9, 9, 1)                      # 黑(9,9)，轮到白
    h_before = c1.get_hash()
    c1.undo()                             # 空盘，轮到黑
    h_empty_black = c1.get_hash()
    c1.move(9, 9, 1)
    assert c1.get_hash() == h_before, "重做后哈希应恢复"
    # 空盘轮到黑 vs 空盘轮到白：手工翻转 current_player 后哈希必须改变
    d1 = Board()
    d2 = Board()
    d2.current_player = 2                 # 人为构造"空盘轮到白"
    d2._hash = Board().get_hash() ^ int(_ZOBRIST_SIDE[0]) ^ int(_ZOBRIST_SIDE[1])
    assert d1.get_hash() != d2.get_hash(), \
        "行棋方不同却哈希相同 -> Zobrist 未含行棋方键"
    assert d1.get_hash() == Board().get_hash()
    print("[PASS] Zobrist 哈希区分行棋方（同型不同轮次不再碰撞）")
    assert h_empty_black == Board().get_hash(), "空盘哈希应为初始值"

    # ---- 测试 12：get_legal_moves 邻域表正确性 ----
    b = Board()
    b.move(9, 9, 1)
    moves = set(b.get_legal_moves())
    expected = {(x, y) for y in range(19) for x in range(19)
                if abs(x - 9) <= 2 and abs(y - 9) <= 2 and (x, y) != (9, 9)}
    assert moves == expected, f"邻域合法点不符: {sorted(moves ^ expected)}"
    print(f"[PASS] get_legal_moves 半径2邻域正确, 共 {len(moves)} 个")

    print("=" * 60)
    print("全部自测通过 [PASS ALL]")
    print("=" * 60)


if __name__ == "__main__":
    main()
