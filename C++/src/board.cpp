// board.cpp — 规则核心的非内联部分
#include "board.h"

#include <algorithm>

namespace gk {

// ---------------------------------------------------------------------------
// Zobrist：splitmix64 生成的固定表。
// 注意：与 Python 版（numpy default_rng(20240518)）**不共享随机数序列**，
// 因此两边的 hash 数值不同 —— 只用于本引擎内部的置换表/复用判断，
// 不参与任何跨语言对比。需要跨语言对齐时用局面张量而不是 hash。
// ---------------------------------------------------------------------------
static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const Zobrist& zobrist() {
    static Zobrist z = [] {
        Zobrist t{};
        uint64_t s = 0xC0FFEE1234567ULL;
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < N_CELLS; ++i) t.key[c][i] = splitmix64(s);
        t.side[0] = splitmix64(s);
        t.side[1] = splitmix64(s);
        return t;
    }();
    return z;
}

void Pos::clear_pad() {
    std::memset(p, WALL, sizeof(p));
    for (int y = 0; y < SIZE; ++y) {
        int8_t* row = p + (y + PAD_OFF) * PAD_SIZE + PAD_OFF;
        std::memset(row, 0, SIZE);
    }
}

void Pos::reset() {
    clear_pad();
    std::memset(near2, 0, sizeof(near2));
    stones = 0;
    player = BLACK;
    hash = zobrist().side[BLACK - 1];
}

static inline bool win_at_base(const int8_t* p, int base, int who) {
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

bool Pos::place(int x, int y, int who) {
    const int base = pad_index(x, y);
    p[base] = static_cast<int8_t>(who);

    // 半径 2 邻域计数（-2..2 的 5x5，棋盘外的不算）
    const int y0 = std::max(0, y - 2), y1 = std::min(SIZE - 1, y + 2);
    const int x0 = std::max(0, x - 2), x1 = std::min(SIZE - 1, x + 2);
    for (int yy = y0; yy <= y1; ++yy) {
        uint8_t* row = near2 + yy * SIZE;
        for (int xx = x0; xx <= x1; ++xx) ++row[xx];
    }

    const Zobrist& z = zobrist();
    hash ^= z.key[who - 1][y * SIZE + x];
    hash ^= z.side[who - 1];
    hash ^= z.side[(3 - who) - 1];

    ++stones;
    const bool win = win_at_base(p, base, who);
    player = static_cast<int8_t>(3 - who);
    return win;
}

void Pos::unplace(int x, int y, int who) {
    const int base = pad_index(x, y);
    p[base] = 0;

    const int y0 = std::max(0, y - 2), y1 = std::min(SIZE - 1, y + 2);
    const int x0 = std::max(0, x - 2), x1 = std::min(SIZE - 1, x + 2);
    for (int yy = y0; yy <= y1; ++yy) {
        uint8_t* row = near2 + yy * SIZE;
        for (int xx = x0; xx <= x1; ++xx) --row[xx];
    }

    const Zobrist& z = zobrist();
    hash ^= z.key[who - 1][y * SIZE + x];
    hash ^= z.side[who - 1];
    hash ^= z.side[(3 - who) - 1];

    --stones;
    player = static_cast<int8_t>(who);
}

void Pos::legal_moves(std::vector<int>& out) const {
    out.clear();
    if (stones == 0) {
        const int c = SIZE / 2;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) out.push_back((c + dy) * SIZE + (c + dx));
        return;
    }
    for (int i = 0; i < N_CELLS; ++i) {
        if (near2[i] == 0) continue;
        if (p[pad_index(i % SIZE, i / SIZE)] != 0) continue;
        out.push_back(i);
    }
}

int Pos::legal_count() const {
    if (stones == 0) return 9;
    int n = 0;
    for (int i = 0; i < N_CELLS; ++i)
        if (near2[i] && p[pad_index(i % SIZE, i / SIZE)] == 0) ++n;
    return n;
}

bool Pos::has_legal() const {
    if (stones == 0) return true;
    for (int i = 0; i < N_CELLS; ++i)
        if (near2[i] && p[pad_index(i % SIZE, i / SIZE)] == 0) return true;
    return false;
}

void Pos::to_tensor(float* dst) const {
    float* c0 = dst;
    float* c1 = dst + N_CELLS;
    float* c2 = dst + 2 * N_CELLS;
    const float side = (player == BLACK) ? 1.0f : 0.0f;
    for (int i = 0; i < N_CELLS; ++i) {
        const int v = p[pad_index(i % SIZE, i / SIZE)];
        c0[i] = (v == BLACK) ? 1.0f : 0.0f;
        c1[i] = (v == WHITE) ? 1.0f : 0.0f;
        c2[i] = side;
    }
}

}  // namespace gk
