// test_vcf.cpp — VCF 静态搜索的正确性与成本自测
//
// 正确性判据：返回 1 必须是「攻方真的必胜」，因此这里既测正例（应当返回 1），
// 也测反例（守方有四在先、攻方无四），并用穷举小棋盘交叉验证若干局面。
#include <chrono>
#include <cstdio>
#include <cstdlib>   // std::atoi（命令行迭代数）
#include <cstring>
#include <string>
#include <vector>

#include "../src/board.h"
#include "../src/vcf.h"

using namespace gk;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("[FAIL] %s\n", what.c_str());
        ++g_fail;
    } else {
        std::printf("[ ok ] %s\n", what.c_str());
    }
}

// 从字符串棋盘摆子：'B' 黑、'W' 白、'.' 空。只用于 19x19 的前 n 行。
void load(Pos& b, const std::vector<std::string>& rows) {
    for (int y = 0; y < (int)rows.size(); ++y)
        for (int x = 0; x < (int)rows[y].size(); ++x) {
            const char c = rows[y][x];
            if (c == 'B') b.place(x, y, BLACK);
            else if (c == 'W') b.place(x, y, WHITE);
        }
}

// 用「落子后谁成五」的朴素递归穷举，判断攻方在纯冲四约束下是否必胜。
// 仅用于极小的验证局面（深度很浅），与 vcf_win 相互印证。
bool brute_has_five_point(const Pos& b, int who, int& out_pad) {
    const int8_t w = (int8_t)who;
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x) {
            if (b.p[pad_index(x, y)] != 0) continue;
            if (b.check_win_after(y * SIZE + x, who)) {
                out_pad = pad_index(x, y);
                return true;
            }
        }
    (void)w;
    return false;
}

// 找该方所有「冲四点」：落子后自身成四（即存在一个成五点）
void rush_points(const Pos& b, int who, std::vector<int>& out) {
    out.clear();
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x) {
            if (b.p[pad_index(x, y)] != 0) continue;
            Pos t = b;
            t.place(x, y, who);
            int pad = 0;
            if (brute_has_five_point(t, who, pad)) out.push_back(y * SIZE + x);
        }
}

// 盘上是否已有五连（用于跳过已分胜负的局面）
// 注意：check_win_after 只对**空点**有效（board.h 第一句就是 p[base]!=0 -> false），
// 所以这里绝不能写成「先判 p[..]==who 再调 check_win_after」——那个条件恒为 false，
// 会让本函数永远返回 false、第 295 行的过滤静默失效。直接扫 5 连窗口。
bool board_has_five(const Pos& b, int who) {
    static const int DX[4] = {1, 0, 1, 1};
    static const int DY[4] = {0, 1, 1, -1};
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x)
            for (int d = 0; d < 4; ++d) {
                const int ex = x + 4 * DX[d], ey = y + 4 * DY[d];
                if (ex < 0 || ex >= SIZE || ey < 0 || ey >= SIZE) continue;
                int k = 0;
                for (; k < 5; ++k)
                    if (b.at(x + k * DX[d], y + k * DY[d]) != who) break;
                if (k == 5) return true;
            }
    return false;
}

// 朴素 VCF：攻方只看冲四点，守方只走唯一挡点。返回是否存在必胜链。
bool brute_vcf(Pos& b, int att, int depth) {
    if (depth <= 0) return false;
    int pad = 0;
    if (brute_has_five_point(b, att, pad)) return true;  // 攻方一步成五
    int defpad = 0;
    const bool def_must = brute_has_five_point(b, 3 - att, defpad);

    std::vector<int> cands;
    if (def_must) {
        cands.push_back((defpad / PAD_SIZE - PAD_OFF) * SIZE + (defpad % PAD_SIZE - PAD_OFF));
    } else {
        rush_points(b, att, cands);
    }
    for (int c : cands) {
        const int x = c % SIZE, y = c / SIZE;
        if (b.p[pad_index(x, y)] != 0) continue;
        b.place(x, y, att);
        int f1 = 0;
        const bool five = brute_has_five_point(b, att, f1);
        if (five) {
            // 数一下有几个成五点：>=2 守方挡不住
            int cnt = 0;
            for (int yy = 0; yy < SIZE; ++yy)
                for (int xx = 0; xx < SIZE; ++xx) {
                    if (b.p[pad_index(xx, yy)] != 0) continue;
                    if (b.check_win_after(yy * SIZE + xx, att)) ++cnt;
                }
            if (cnt >= 2) {
                b.unplace(x, y, att);
                return true;
            }
            const int qx = f1 % PAD_SIZE - PAD_OFF, qy = f1 / PAD_SIZE - PAD_OFF;
            if (b.p[f1] == 0) {
                b.place(qx, qy, 3 - att);
                const bool r = brute_vcf(b, att, depth - 1);
                b.unplace(qx, qy, 3 - att);
                if (r) {
                    b.unplace(x, y, att);
                    return true;
                }
            }
        }
        b.unplace(x, y, att);
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // 随机交叉验证的迭代数。原来写死 300 且 main() 无参，
    // 所以 `test_vcf.exe 400` 里的 400 是被静默忽略的 —— 想加样本量必须从命令行传进来。
    int n_rand = argc > 1 ? std::atoi(argv[1]) : 300;
    if (n_rand <= 0) {
        std::printf("用法: %s [随机局面迭代数>0]   默认 300\n", argv[0]);
        return 2;
    }
    // ---------------- 1) 空盘 / 子太少 ----------------
    {
        Pos b;
        check(vcf_win(b) == 0, "空盘 → 0");
        b.place(9, 9, BLACK);
        b.place(9, 10, WHITE);
        b.place(10, 10, BLACK);
        b.place(8, 10, WHITE);
        b.place(11, 10, BLACK);
        b.place(7, 10, WHITE);
        check(vcf_win(b) == 0, "子少于 5 → 0");
    }

    // ---------------- 2) 活三：两步必胜，应当命中 ----------------
    {
        Pos b;
        load(b, {"...................",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 ".....BBB...........",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 "................W..",
                 "..................."});
        b.place(0, 0, WHITE);
        check(b.player == BLACK, "行棋方为黑");
        check(vcf_win(b) == 1, "活三（BBB 两端皆空）→ 1");
    }

    // ---------------- 3) 守方有四在先：攻方必须先堵，堵了不成四 → 0 ----------------
    {
        Pos b;
        load(b, {"...................",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 ".....BBB...........",
                 "...................",
                 "...................",
                 "...................",
                 ".....WWWW..........",
                 "...................",
                 "..................."});
        // 让白先走完，黑行棋（黑方此刻面对白的冲四）
        check(b.player == BLACK, "行棋方为黑");
        // 白 WWWW 在 (5..8,9)，黑必须先堵 (4,9) 或 (9,9)。
        // 黑堵完之后自己的 BBB 还在，但白还有另一个五点在等着 → 黑堵不过来 → 0。
        check(vcf_win(b) == 0, "守方活四在先 → 0");
    }

    // ---------------- 4) 隔断的四：黑能一步成五 → 1 ----------------
    {
        Pos b;
        load(b, {"...................",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 "....BBBB...........",
                 "...................",
                 "...................",
                 "...................",
                 "..W.W.......W......",
                 "..................."});
        check(vcf_win(b) == 1, "黑 BBBB 一端空 → 1");
    }

    // ---------------- 5) 无四无三：0 ----------------
    {
        Pos b;
        load(b, {"...................",
                 "...................",
                 "...................",
                 "...................",
                 "...................",
                 ".....B.W...........",
                 "......W.B..........",
                 "...................",
                 "..................."});
        check(vcf_win(b) == 0, "无任何威胁 → 0");
    }

    // ---------------- 6) 与朴素穷举交叉验证 ----------------
    {
        // 构造几个中等复杂的局面，用朴素 VCF 做参照
        const std::vector<std::vector<std::string>> cases = {
            {"...................",
             "...................",
             "...................",
             "...................",
             "...................",
             "....BBB............",
             ".......W...........",
             "...................",
             "...................",
             "...................",
             "..................."},
            {"...................",
             "...................",
             "...................",
             "...................",
             ".....B.............",
             "......B.W..........",
             ".......B...........",
             "........W..........",
             "...................",
             "...................",
             "..................."},
            {"...................",
             "...................",
             "...................",
             "...................",
             "...................",
             "....BB.......W.....",
             "......B............",
             "..........W........",
             "..................."},
        };
        for (size_t i = 0; i < cases.size(); ++i) {
            Pos b;
            load(b, cases[i]);
            Pos copy = b;
            const bool ref = brute_vcf(copy, b.player, 4);
            const int got = vcf_win(b);
            std::printf("        case %zu: brute=%d vcf=%d\n", i, (int)ref, got);
            check(!ref || got == 1, "交叉验证 case " + std::to_string(i) + "：穷举必胜则 VCF 必须也报胜");
        }
    }

    // ---------------- 6b) 随机局面双向交叉验证 ----------------
    // 单向检查（brute=1 → vcf=1）只能抓漏报。VCF 多报胜更致命（把必败判胜），
    // 所以这里做双向：两边深度都用 6，结论必须完全一致。
    {
        uint64_t seed = 0x2545F4914F6CDD1Dull;
        auto rnd = [&seed]() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            return seed;
        };

        int n_cases = 0, n_win = 0, n_mismatch = 0;
        for (int iter = 0; iter < n_rand; ++iter) {
            Pos b;
            const int stones = 10 + (int)(rnd() % 6);  // 10..15 子
            // 一半局面用「中心 5x5 密集」生成，专门制造三连/四连，
            // 否则随机稀疏局面上几乎不会出现 VCF 必胜，交叉验证等于没测。
            const bool dense = (iter % 2) == 0;
            const int span = dense ? 5 : 9;
            const int lo = dense ? 7 : 5;
            for (int k = 0; k < stones; ++k) {
                const int who = (k % 2 == 0) ? BLACK : WHITE;
                for (int tries = 0; tries < 64; ++tries) {
                    const int x = lo + (int)(rnd() % span);
                    const int y = lo + (int)(rnd() % span);
                    if (b.p[pad_index(x, y)] != 0) continue;
                    b.place(x, y, who);
                    break;
                }
            }
            // 让黑行棋（子数偶数时 place 后 player 已是黑）
            if (b.player != BLACK) continue;
            // 已分胜负的局面没有意义
            if (board_has_five(b, BLACK) || board_has_five(b, WHITE)) continue;

            Pos copy = b;
            const bool ref = brute_vcf(copy, BLACK, 6);
            VCFOptions o;
            o.max_ply = 6;
            o.max_nodes = 4096;
            const bool got = vcf_win(b, o) == 1;
            ++n_cases;
            if (ref) ++n_win;
            if (ref != got) {
                ++n_mismatch;
                if (n_mismatch <= 3) {
                    std::printf("        MISMATCH #%d brute=%d vcf=%d\n", iter, (int)ref, (int)got);
                    for (int y = 0; y < SIZE; ++y) {
                        std::printf("        ");
                        for (int x = 0; x < SIZE; ++x) {
                            const int v = b.p[pad_index(x, y)];
                            std::printf("%c", v == BLACK ? 'B' : v == WHITE ? 'W' : '.');
                        }
                        std::printf("\n");
                    }
                }
            }
        }
        // 把「迭代数 → 有效局面数」的损耗显式打出来：过滤掉的是 player 不是黑、
        // 以及（board_has_five 修好后才真正生效的）已分胜负局面。
        std::printf("        random: 迭代 %d 次 → 有效 %d 局面（跳过 %d）, brute 报胜 %d, 不一致 %d\n",
                    n_rand, n_cases, n_rand - n_cases, n_win, n_mismatch);
        check(n_mismatch == 0, "随机局面双向一致");
    }

    // ---------------- 7) 成本实测 ----------------
    {
        // 用自对弈产生的典型局面（无四）和一个有大量冲四的局面分别计时
        Pos quiet;
        load(quiet, {"...................",
                     "...................",
                     "...................",
                     "...................",
                     "...................",
                     ".....B.W.B.........",
                     "......W.B..........",
                     ".....B...W.........",
                     "...................",
                     "..................."});
        Pos busy;
        load(busy, {"...................",
                    "...................",
                    "...B...B...B.......",
                    "...W...W...W.......",
                    "...B...B...B.......",
                    "....W.W.W..........",
                    "...B...B...B.......",
                    "....W.W.W..........",
                    "...................",
                    "..................."});

        auto bench = [](const char* name, const Pos& p, int iters) {
            VCFOptions o;
            const auto t0 = std::chrono::steady_clock::now();
            int hits = 0;
            for (int i = 0; i < iters; ++i) hits += vcf_win(p, o);
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            std::printf("        %-8s %8.3f us/call   hits=%d\n", name, us, hits);
        };
        bench("quiet", quiet, 20000);
        bench("busy", busy, 5000);
    }

    std::printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
