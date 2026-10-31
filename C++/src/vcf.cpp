// vcf.cpp — 正向 VCF（连续冲四）静态搜索的实现
//
// 搜索骨架与 Rapfi 的 SimpleVCF 同构，但 Rapfi 有增量棋型表，我们只有填充棋盘，
// 所以用「5 格连窗口整盘扫描」一次拿到三类信息：
//
//   一个 5 连窗口里，记 ca = 攻方子数、cd = 守方子数、空点 E：
//     * ca == 4 且 |E| == 1 → 该空点是攻方的「一步成五点」
//     * cd == 4 且 |E| == 1 → 该空点是守方的「一步成五点」
//     * ca == 3 且 |E| == 2 → 这两个空点都是攻方的「冲四点」
//       （在任一空点落子后，窗口变成 4 攻子 + 1 空，那个空点即成五）
//
// 递归规则（攻方 = 进入搜索时的行棋方，整条链由我们攻击、对手只能挡）：
//   1. 我方有成五点 → 直接成五，胜。
//   2. 对手有成五点：
//        ≥2 个 → 只能堵一个，必败，未证明。
//        ==1 个 → 这一手被迫只能堵它，且堵完必须成四，否则未证明。
//   3. 否则逐个试冲四点 p：落 p 后数我方成五点 nf
//        nf >= 2 → 双威胁，胜
//        nf == 1 → 对手唯一挡点 q，落 q 后递归
//        nf == 0 → p 其实没成四，跳过
//   4. 全部失败 → 未证明（返回 0，**不代表必败**）。
//
// 关键细节（都是踩过的坑）：
//   * 落子后必须检查「是否直接成五」——对手被迫堵的那个点可能正好也让对手成五。
//   * 候选点的「冲四」判定不能用「近似两个空点」猜，必须走窗口统计。
#include "vcf.h"

#include <cstring>

namespace gk {
namespace {

// 单次搜索的临时缓冲。下标一律是**填充棋盘下标**（pad index）。
struct Scratch {
    int att_five[16];
    int n_att_five;
    int def_five[16];
    int n_def_five;
    int rush[N_CELLS];
    int n_rush;
    uint8_t seen[N_CELLS];
};

inline int pad_to_cell(int pad) {
    return (pad / PAD_SIZE - PAD_OFF) * SIZE + (pad % PAD_SIZE - PAD_OFF);
}

inline bool push_unique(int* arr, int& n, int cap, int pad) {
    for (int i = 0; i < n; ++i)
        if (arr[i] == pad) return true;
    if (n >= cap) return false;
    arr[n++] = pad;
    return true;
}

inline void consider_window(const Pos& b, int base, int step, int att, int def, Scratch& s) {
    int ca = 0, cd = 0;
    int e0 = -1, e1 = -1;
    for (int k = 0; k < 5; ++k) {
        const int q = base + k * step;
        const int v = b.p[q];
        if (v == att) {
            ++ca;
        } else if (v == def) {
            ++cd;
        } else if (v == 0) {
            if (e0 < 0) e0 = q;
            else if (e1 < 0) e1 = q;
            else return;  // 3 个以上空点，本窗口无信息
        } else {
            return;  // 墙
        }
    }
    if (ca == 4 && e0 >= 0 && e1 < 0) {
        push_unique(s.att_five, s.n_att_five, 16, e0);
    } else if (cd == 4 && e0 >= 0 && e1 < 0) {
        push_unique(s.def_five, s.n_def_five, 16, e0);
    } else if (ca == 3 && e0 >= 0 && e1 >= 0) {
        // 两个空点都是冲四点（见文件头说明）
        const int c0 = pad_to_cell(e0);
        if (!s.seen[c0] && s.n_rush < N_CELLS) {
            s.seen[c0] = 1;
            s.rush[s.n_rush++] = e0;
        }
        const int c1 = pad_to_cell(e1);
        if (!s.seen[c1] && s.n_rush < N_CELLS) {
            s.seen[c1] = 1;
            s.rush[s.n_rush++] = e1;
        }
    }
}

void scan_board(const Pos& b, int att, int def, Scratch& s) {
    s.n_att_five = 0;
    s.n_def_five = 0;
    s.n_rush = 0;
    std::memset(s.seen, 0, sizeof(s.seen));

    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x + 4 < SIZE; ++x) consider_window(b, pad_index(x, y), 1, att, def, s);
    for (int x = 0; x < SIZE; ++x)
        for (int y = 0; y + 4 < SIZE; ++y)
            consider_window(b, pad_index(x, y), PAD_SIZE, att, def, s);
    for (int y = 0; y + 4 < SIZE; ++y)
        for (int x = 0; x + 4 < SIZE; ++x)
            consider_window(b, pad_index(x, y), PAD_SIZE + 1, att, def, s);
    for (int y = 0; y + 4 < SIZE; ++y)
        for (int x = 4; x < SIZE; ++x)
            consider_window(b, pad_index(x, y), PAD_SIZE - 1, att, def, s);
}

// 刚落子于 pad 的 who 是否已构成五连（只查经过 pad 的四条线）。
inline bool five_at(const Pos& b, int pad, int who) {
    const int8_t w = static_cast<int8_t>(who);
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        int i = 1, j = 1;
        while (b.p[pad - i * step] == w) ++i;
        while (b.p[pad + j * step] == w) ++j;
        if (i + j - 1 >= 5) return true;
    }
    return false;
}

// 落子 pad 之后，攻方**新产生**的成五点。必然都经过 pad，所以只扫含 pad 的 20 个窗口。
int five_points_at(const Pos& b, int pad, int att, int* out) {
    int n = 0;
    for (int d = 0; d < 4; ++d) {
        const int step = PAD_STEPS[d];
        for (int off = 0; off < 5; ++off) {
            const int base = pad - off * step;
            int empty = -1;
            bool ok = true;
            for (int k = 0; k < 5; ++k) {
                const int q = base + k * step;
                const int x = q % PAD_SIZE, y = q / PAD_SIZE;
                if (x < PAD_OFF || x >= PAD_OFF + SIZE || y < PAD_OFF || y >= PAD_OFF + SIZE) {
                    ok = false;
                    break;
                }
                const int v = b.p[q];
                if (v == 0) {
                    if (empty >= 0) {
                        ok = false;
                        break;
                    }
                    empty = q;
                } else if (v != att) {
                    ok = false;
                    break;
                }
            }
            if (!ok || empty < 0) continue;
            bool dup = false;
            for (int i = 0; i < n; ++i)
                if (out[i] == empty) dup = true;
            if (!dup && n < 16) out[n++] = empty;
        }
    }
    return n;
}

struct Ctx {
    Pos b;
    Scratch s;
    int max_nodes;
    int max_ply;
    int nodes;
    int first_move = -1;   // ply==0 时取胜的那手（cell index），-1 = 未定
};

int search(Ctx& c, int ply, int att, int def) {
    if (c.nodes >= c.max_nodes || ply >= c.max_ply) return 0;
    ++c.nodes;

    scan_board(c.b, att, def, c.s);
    if (c.s.n_att_five > 0) {              // 我方立刻成五
        // 根层必须记下这一手：否则 vcf_find_move 会「返回 1 但没写回 out_cell」，
        // 调用方拿到的是 untouched 的哨兵值，只能当成未命中（白算一次搜索）。
        if (ply == 0) c.first_move = pad_to_cell(c.s.att_five[0]);
        return 1;
    }
    if (c.s.n_def_five >= 2) return 0;     // 对手双五，堵不住

    int cand[N_CELLS];
    int ncand;
    if (c.s.n_def_five == 1) {
        ncand = 1;
        cand[0] = c.s.def_five[0];
    } else {
        ncand = c.s.n_rush;
        for (int i = 0; i < ncand; ++i) cand[i] = c.s.rush[i];
    }

    for (int i = 0; i < ncand; ++i) {
        const int p = cand[i];
        if (c.b.p[p] != 0) continue;
        const int px = p % PAD_SIZE - PAD_OFF;
        const int py = p / PAD_SIZE - PAD_OFF;
        c.b.place(px, py, att);

        int result = 0;
        if (five_at(c.b, p, att)) {
            result = 1;  // 这手直接成五
        } else {
            int five[16];
            const int nf = five_points_at(c.b, p, att, five);
            if (nf >= 2) {
                result = 1;  // 双威胁
            } else if (nf == 1) {
                const int q = five[0];
                if (c.b.p[q] == 0) {
                    const int qx = q % PAD_SIZE - PAD_OFF;
                    const int qy = q / PAD_SIZE - PAD_OFF;
                    c.b.place(qx, qy, def);
                    // 对手被迫挡的这一手如果让他自己成五，那这条链就输了。
                    if (!five_at(c.b, q, def)) result = search(c, ply + 1, att, def);
                    c.b.unplace(qx, qy, def);
                }
            }
        }
        c.b.unplace(px, py, att);
        if (result) {
            if (ply == 0) c.first_move = pad_to_cell(p);
            return 1;
        }
    }
    return 0;
}

}  // namespace

int vcf_find_move(const Pos& pos, int* out_cell, const VCFOptions& opt) {
    const int att = pos.player;

    // 廉价预筛：攻方至少要有 3 颗子，才可能连续冲四取胜
    // （3 子 → 冲四 → 成五）。直接数子，不靠奇偶推（测试里手摆局面不交替）。
    int att_stones = 0;
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x)
            if (pos.p[pad_index(x, y)] == att) ++att_stones;
    if (att_stones < 3) return 0;

    Ctx c;
    c.b = pos;
    c.max_nodes = opt.max_nodes;
    c.max_ply = opt.max_ply;
    c.nodes = 0;
    const int ok = search(c, 0, att, 3 - att) ? 1 : 0;
    if (ok && out_cell && c.first_move >= 0) *out_cell = c.first_move;
    return ok;
}

int vcf_win(const Pos& pos, const VCFOptions& opt) {
    return vcf_find_move(pos, nullptr, opt);
}

}  // namespace gk
