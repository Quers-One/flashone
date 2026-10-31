// test_rules.cpp — 规则层自测（对标 PYTHON/gomoku_env.py main() 的 12 个用例）
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "../src/board.h"
#include "../src/heuristics.h"

using namespace gk;

static int g_fail = 0;
#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("[FAIL] %s (line %d)\n", msg, __LINE__);     \
            ++g_fail;                                                \
        } else {                                                     \
            std::printf("[PASS] %s\n", msg);                         \
        }                                                            \
    } while (0)

// 对局辅助：记录历史以便悔棋与胜负判定
struct Game {
    Pos pos;
    std::vector<int> hist;   // 依次为 idx
    bool over = false;
    int winner = 0;

    bool play(int x, int y) {
        if (over) return false;
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return false;
        if (!pos.is_empty(x, y)) return false;
        const int who = pos.player;
        const bool win = pos.place(x, y, who);
        hist.push_back(y * SIZE + x);
        if (win) { over = true; winner = who; }
        else if (pos.stones == N_CELLS) { over = true; winner = 0; }
        return true;
    }
    bool undo() {
        if (hist.empty()) return false;
        const int idx = hist.back();
        hist.pop_back();
        const int who = 3 - pos.player;   // place 之后 player 已翻转
        pos.unplace(idx % SIZE, idx / SIZE, who);
        over = false;
        winner = 0;
        return true;
    }
};

static void test_wins() {
    {
        Game g;
        const int black[5][2] = {{9, 9}, {10, 9}, {11, 9}, {12, 9}, {13, 9}};
        const int white[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        for (int i = 0; i < 5; ++i) {
            g.play(black[i][0], black[i][1]);
            if (i < 4) g.play(white[i][0], white[i][1]);
        }
        CHECK(g.over && g.winner == 1, "黑横五连判黑胜");
    }
    {
        Game g;
        const int b[5][2] = {{9, 9}, {9, 10}, {9, 11}, {9, 12}, {9, 13}};
        const int w[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        for (int i = 0; i < 5; ++i) {
            g.play(b[i][0], b[i][1]);
            if (i < 4) g.play(w[i][0], w[i][1]);
        }
        CHECK(g.over && g.winner == 1, "黑竖五连判黑胜");
    }
    {
        Game g;
        const int b[5][2] = {{7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}};
        const int w[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        for (int i = 0; i < 5; ++i) {
            g.play(b[i][0], b[i][1]);
            if (i < 4) g.play(w[i][0], w[i][1]);
        }
        CHECK(g.over && g.winner == 1, "黑正斜五连判黑胜");
    }
    {
        Game g;
        const int b[5][2] = {{7, 11}, {8, 10}, {9, 9}, {10, 8}, {11, 7}};
        const int w[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        for (int i = 0; i < 5; ++i) {
            g.play(b[i][0], b[i][1]);
            if (i < 4) g.play(w[i][0], w[i][1]);
        }
        CHECK(g.over && g.winner == 1, "黑反斜五连判黑胜");
    }
    {
        Game g;
        g.play(0, 0);
        const int w[5][2] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};
        const int b[5][2] = {{0, 5}, {2, 7}, {4, 3}, {6, 9}, {8, 1}};
        for (int i = 0; i < 5; ++i) {
            g.play(w[i][0], w[i][1]);
            if (i < 5 && !g.over) g.play(b[i][0], b[i][1]);
        }
        CHECK(g.over && g.winner == 2, "白横五连判白胜");
    }
    {
        Game g;
        g.play(9, 9);
        g.play(10, 9);
        g.undo();
        CHECK(g.pos.at(9, 9) == 1 && g.pos.at(10, 9) == 0, "undo 撤子正确");
        CHECK(g.pos.player == 2, "undo 后回到白方");
        CHECK(!g.over && g.winner == 0, "undo 后未结束");
    }
}

static void test_legal() {
    Pos p;
    std::vector<int> lm;
    p.legal_moves(lm);
    CHECK(lm.size() == 9, "开局合法点 9 个");
    CHECK(lm[0] == 8 * SIZE + 8 && lm[4] == 9 * SIZE + 9 && lm[8] == 10 * SIZE + 10,
          "开局合法点顺序为行优先升序（天元在中间）");

    p.place(9, 9, BLACK);
    p.legal_moves(lm);
    // 期望：|x-9|<=2, |y-9|<=2, 去掉 (9,9)
    std::set<int> expect;
    for (int y = 7; y <= 11; ++y)
        for (int x = 7; x <= 11; ++x)
            if (!(x == 9 && y == 9)) expect.insert(y * SIZE + x);
    std::set<int> got(lm.begin(), lm.end());
    CHECK(got == expect, "半径2邻域合法点正确（24 个）");
    CHECK(std::is_sorted(lm.begin(), lm.end()), "合法点升序");
}

static void test_tensor() {
    Pos p;
    p.place(9, 9, BLACK);
    std::vector<float> t(3 * N_CELLS);
    p.to_tensor(t.data());
    CHECK(t[9 * SIZE + 9] == 1.0f, "黑通道标记正确");
    CHECK(t[N_CELLS + 9 * SIZE + 9] == 0.0f, "白通道未误标");
    bool all_zero = true;
    for (int i = 0; i < N_CELLS; ++i)
        if (t[2 * N_CELLS + i] != 0.0f) all_zero = false;
    CHECK(all_zero, "落黑子后行棋方通道为白(全0)");
}

static void test_hash() {
    Pos a, b;
    CHECK(a.hash == b.hash, "初始哈希相同");
    b.place(9, 9, BLACK);
    CHECK(a.hash != b.hash, "落子后哈希改变");
    b.unplace(9, 9, BLACK);
    CHECK(a.hash == b.hash, "悔棋后哈希恢复");
    // 只交换行棋方的局面必须不同
    Pos c = a;
    c.player = WHITE;
    c.hash ^= zobrist().side[0];
    c.hash ^= zobrist().side[1];
    CHECK(a.hash != c.hash, "哈希含行棋方键");
}

static void test_heuristics() {
    // 白方活三（黑必须堵）：. O O O .  —— 白在 (7,9)(8,9)(9,9)，两端 (6,9)/(10,9) 该被高亮
    Pos p;
    p.place(0, 0, BLACK);
    p.place(1, 0, BLACK);
    p.place(0, 1, BLACK);
    p.place(7, 9, WHITE);
    p.place(8, 9, WHITE);
    p.place(9, 9, WHITE);
    // 轮到黑：堵活三拿到 TACTICAL_BLOCK_THREE 级别，且应是全场最高分
    std::vector<int> legal;
    p.legal_moves(legal);
    PriorBuf buf;
    move_scores(p, legal, p.player, buf);
    int k6 = -1, k10 = -1;
    double best_other = 0.0;
    for (int k = 0; k < (int)legal.size(); ++k) {
        if (legal[k] == 9 * SIZE + 6) k6 = k;
        if (legal[k] == 9 * SIZE + 10) k10 = k;
        if (legal[k] != 9 * SIZE + 6 && legal[k] != 9 * SIZE + 10)
            best_other = std::max(best_other, buf.raw[k]);
    }
    CHECK(k6 >= 0 && k10 >= 0, "活三两端点都在合法点里");
    if (k6 >= 0) CHECK(buf.raw[k6] >= TACTICAL_BLOCK_THREE, "活三端点拿到堵三级别防守分(>=500)");
    if (k10 >= 0) CHECK(buf.raw[k10] >= TACTICAL_BLOCK_THREE, "活三另一端点同样高亮");
    // 堵点=(6,9)/(10,9)：假如对方占这里就是活四 => 防守分 = 0.8 * 1e6
    if (k6 >= 0) CHECK(buf.raw[k6] >= TACTICAL_OPEN_FOUR * 0.8, "堵点拿到活四级别防守分");
    CHECK(best_other < buf.raw[k6] * 0.5, "堵点分数显著高于其余所有点");

    // 强制点：黑有活四 -> 只保留成五点
    Pos q;
    for (int i = 0; i < 4; ++i) q.place(5 + i, 5, BLACK);
    std::vector<int> lg;
    q.legal_moves(lg);
    PriorBuf b2;
    compute_prior(q, lg, nullptr, 1.0, 1.0, true, b2);
    int wins = 0;
    for (int k = 0; k < (int)lg.size(); ++k)
        if (b2.priors[k] > 0.0f) ++wins;
    CHECK(wins == 2, "活四局面先验只保留两个成五点");
}

int main() {
    std::printf("=== 规则层自测 ===\n");
    test_wins();
    test_legal();
    test_tensor();
    test_hash();
    test_heuristics();
    std::printf("\n%s (%d failure)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
