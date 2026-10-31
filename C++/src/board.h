// board.h — 19x19 五子棋规则核心（性能关键层）
//
// 设计直接对标 PYTHON/gomoku_env.py，但把「带围墙填充棋盘 + 半径2邻域计数」
// 做成了可以整块 memcpy 的 POD，这样 MCTS 每个节点拷贝一个局面只要 ~1KB memcpy，
// 而不是 Python 里 list(pad) + numpy 数组 + history 的三份拷贝。
//
// 与 Python 版的语义对齐点（改这里必须同步改那边的测试）：
//   * 黑先白后，无禁手，五连即胜（长连也算胜）
//   * 平局：361 格下满
//   * 合法点 = 已有棋子半径 2（切比雪夫距离）以内的空点；空盘时只给天元 3x3
//   * 合法点顺序 = 扁平下标 y*SIZE+x 升序（Python 用 np.unique 就是这个顺序）
//   * line_after_* 系列只在「假设落子」时读取填充棋盘，不修改任何状态
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gk {

inline constexpr int SIZE = 19;
inline constexpr int N_CELLS = SIZE * SIZE;              // 361
inline constexpr int PAD_OFF = 4;
inline constexpr int PAD_SIZE = SIZE + 2 * PAD_OFF;      // 27
inline constexpr int PAD_CELLS = PAD_SIZE * PAD_SIZE;    // 729

inline constexpr int BLACK = 1;
inline constexpr int WHITE = 2;
inline constexpr int8_t WALL = -1;

// 四个方向的一维步长：横、竖、正斜(\)、反斜(/)
inline constexpr int PAD_STEPS[4] = {1, PAD_SIZE, PAD_SIZE + 1, PAD_SIZE - 1};
// 中心 8 邻域步长
inline constexpr int PAD_NEIGHBOR8[8] = {-PAD_SIZE - 1, -PAD_SIZE, -PAD_SIZE + 1, -1,
                                         1, PAD_SIZE - 1, PAD_SIZE, PAD_SIZE + 1};

inline constexpr int pad_index(int x, int y) { return (y + PAD_OFF) * PAD_SIZE + (x + PAD_OFF); }
inline constexpr int xy_of(int idx) { return idx; }  // 文档用：idx = y*SIZE+x

struct Pos {
    int8_t p[PAD_CELLS];      // 填充棋盘：-1 围墙 / 0 空 / 1 黑 / 2 白
    uint8_t near2[N_CELLS];   // 每格半径 2 内的棋子数（>0 即为合法候选点）
    uint64_t hash;            // Zobrist（含行棋方）
    int16_t stones;           // 已落子数
    int8_t player;            // 当前行棋方

    Pos() { reset(); }

    void reset();
    void clear_pad();

    // 落子：调用方保证 (x,y) 在盘内且为空。返回「是否形成五连」。
    bool place(int x, int y, int who);
    // 撤子：调用方保证 (x,y) 上是 who。恢复行棋方为 who。
    void unplace(int x, int y, int who);

    inline int at(int x, int y) const { return p[pad_index(x, y)]; }
    inline bool is_empty(int x, int y) const { return at(x, y) == 0; }

    // ---- 规则查询 ----
    // 合法点（扁平下标升序）。空盘时返回天元 3x3（也是升序）。
    void legal_moves(std::vector<int>& out) const;
    int legal_count() const;
    bool has_legal() const;

    // 该点落子后 who 是否立刻五连（不修改状态）
    inline bool check_win_after(int idx, int who) const;
    // 该点落子后 who 是否形成「活四」（4 连且两端皆空）
    inline bool open_four_after(int idx, int who) const;
    // 该点 8 邻域内是否有棋子
    inline bool has_neighbor(int idx) const;
    // 一次性给出双方在 idx 落子后的棋型（count / 任意一端开放 / 两端皆开放）
    inline void line_after_both(int idx, int me, int& c_me, bool& o_me, int& c_op, bool& o_op) const;
    // 单向版（调试 / 描述用）
    inline void line_after(int idx, int who, int& count, bool& open_any, bool& open_both) const;

    // 3x19x19 局面张量：ch0 黑、ch1 白、ch2 行棋方（黑=1 白=0）
    void to_tensor(float* dst) const;

    inline friend void swap(Pos& a, Pos& b) noexcept {
        std::swap(a.hash, b.hash);
        std::swap(a.stones, b.stones);
        std::swap(a.player, b.player);
        // 大盘面整体交换（编译器会展开成 SIMD 拷贝）
        for (int i = 0; i < PAD_CELLS; ++i) std::swap(a.p[i], b.p[i]);
        for (int i = 0; i < N_CELLS; ++i) std::swap(a.near2[i], b.near2[i]);
    }
};

// ---------------------------------------------------------------------------
// Zobrist 表（模块级常量，进程生命周期内只初始化一次）
// ---------------------------------------------------------------------------
struct Zobrist {
    uint64_t key[2][N_CELLS];
    uint64_t side[2];
};
const Zobrist& zobrist();

// ---------------------------------------------------------------------------
// 内联实现
// ---------------------------------------------------------------------------
inline bool Pos::check_win_after(int idx, int who) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    if (p[base] != 0) return false;
    const int8_t w = static_cast<int8_t>(who);
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        int i = 1, j = 1;
        while (p[base - i * step] == w) ++i;
        while (p[base + j * step] == w) ++j;
        if (i + j - 1 >= 5) return true;
    }
    return false;
}

inline bool Pos::open_four_after(int idx, int who) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    if (p[base] != 0) return false;
    const int8_t w = static_cast<int8_t>(who);
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        int i = 1, j = 1;
        while (i <= 4 && p[base - i * step] == w) ++i;
        while (j <= 4 && p[base + j * step] == w) ++j;
        if (i + j - 1 == 4 && p[base - i * step] == 0 && p[base + j * step] == 0) return true;
    }
    return false;
}

inline bool Pos::has_neighbor(int idx) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    for (int k = 0; k < 8; ++k)
        if (p[base + PAD_NEIGHBOR8[k]] > 0) return true;
    return false;
}

inline void Pos::line_after_both(int idx, int me, int& c_me, bool& o_me, int& c_op, bool& o_op) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    if (p[base] != 0) { c_me = 0; o_me = false; c_op = 0; o_op = false; return; }
    const int8_t a = static_cast<int8_t>(me);
    const int8_t b = static_cast<int8_t>(3 - me);
    int best_a = 1, best_b = 1;
    bool any_a = false, any_b = false;
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        // ---- 我方 ----
        int i = 1, j = 1;
        while (i <= 4 && p[base - i * step] == a) ++i;
        while (j <= 4 && p[base + j * step] == a) ++j;
        int cnt = i + j - 1;
        bool oa = p[base - i * step] == 0;
        bool ob = p[base + j * step] == 0;
        bool any = oa || ob;
        if (cnt > best_a || (cnt == best_a && any)) { best_a = cnt; any_a = any; }
        // ---- 对方 ----
        i = 1; j = 1;
        while (i <= 4 && p[base - i * step] == b) ++i;
        while (j <= 4 && p[base + j * step] == b) ++j;
        cnt = i + j - 1;
        oa = p[base - i * step] == 0;
        ob = p[base + j * step] == 0;
        any = oa || ob;
        if (cnt > best_b || (cnt == best_b && any)) { best_b = cnt; any_b = any; }
    }
    c_me = best_a; o_me = any_a; c_op = best_b; o_op = any_b;
}

inline void Pos::line_after(int idx, int who, int& count, bool& open_any, bool& open_both) const {
    const int base = pad_index(idx % SIZE, idx / SIZE);
    if (p[base] != 0) { count = 0; open_any = false; open_both = false; return; }
    const int8_t w = static_cast<int8_t>(who);
    int best = 1;
    bool any = false, both = false;
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        int i = 1, j = 1;
        while (i <= 4 && p[base - i * step] == w) ++i;
        while (j <= 4 && p[base + j * step] == w) ++j;
        const int cnt = i + j - 1;
        if (cnt >= 5) { count = 5; open_any = true; open_both = true; return; }
        const bool oa = p[base - i * step] == 0;
        const bool ob = p[base + j * step] == 0;
        const bool a2 = oa || ob;
        if (cnt > best || (cnt == best && a2)) { best = cnt; any = a2; both = oa && ob; }
    }
    count = best; open_any = any; open_both = both;
}

}  // namespace gk
