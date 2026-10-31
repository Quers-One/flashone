// test_mcts.cpp — PUCT 搜索的行为自测（不依赖 libtorch，用一个脚本化评估器）
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/heuristics.h"
#include "../src/mcts.h"

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

// 评估器：策略 = 中心偏好的固定分布，价值 = 0（逼近"无信息"的下界）
class DummyNet : public Evaluator {
public:
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        calls += batch;
        for (int b = 0; b < batch; ++b) {
            float* p = policies + (size_t)b * N_CELLS;
            double s = 0.0;
            for (int i = 0; i < N_CELLS; ++i) {
                const int x = i % SIZE, y = i / SIZE;
                const double cx = (double)x - 9.0, cy = (double)y - 9.0;
                const double w = std::exp(-(cx * cx + cy * cy) / 200.0);
                p[i] = (float)w;
                s += w;
            }
            for (int i = 0; i < N_CELLS; ++i) p[i] = (float)(p[i] / s);
            values[b] = 0.0f;
        }
        (void)states;
    }
    int calls = 0;
};

void put(Pos& p, int x, int y, int who) {
    p.place(x, y, who);
}

}  // namespace

int main() {
    // ---------------- 1) 空盘：合法点在中心 3x3，搜索后访问数正确 ----------------
    {
        Pos b;
        MCTSOptions opt;
        opt.n_sim = 200;
        opt.batch_size = 16;
        DummyNet net;
        MCTS m(&net, opt, 7);
        std::vector<float> pol;
        const int mv = m.sample_move_with_policy(b, pol, 0.0);
        check(mv >= 0, "空盘能选出落点");
        const int cx = mv % SIZE, cy = mv / SIZE;
        check(cx >= 8 && cx <= 10 && cy >= 8 && cy <= 10, "空盘落点在天元 3x3");
        double s = 0.0;
        for (float v : pol) s += v;
        check(std::fabs(s - 1.0) < 1e-4, "策略目标和为 1");
        check(pol.size() == (size_t)N_CELLS, "策略目标长度 361");
        MCTSStats st = m.collect_stats();
        check(st.root_visits == 200, "根访问数 == n_sim (200)");
        std::printf("       空盘: mv=(%d,%d) root_n=%d root_q=%.3f children=%d calls=%d\n", cx, cy,
                    st.root_visits, st.root_value, st.root_children, net.calls);
    }

    // ---------------- 2) 一步成五：必须走成五点 ----------------
    {
        Pos b;
        for (int x = 5; x <= 8; ++x) put(b, x, 5, BLACK);
        put(b, 5, 6, WHITE);
        put(b, 6, 6, WHITE);
        put(b, 7, 6, WHITE);
        MCTSOptions opt;
        opt.n_sim = 200;
        DummyNet net;
        MCTS m(&net, opt, 11);
        std::vector<float> pol;
        const int mv = m.sample_move_with_policy(b, pol, 0.0);
        const bool wins = (mv == 4 * 1 + 5 * SIZE) || (mv == 9 + 5 * SIZE);
        check(wins, "黑有活四时选择成五点");
        std::printf("       成五: mv=(%d,%d) policy[mv]=%.3f\n", mv % SIZE, mv / SIZE, pol[(size_t)mv]);
    }

    // ---------------- 3) 必挡：对方活四必须堵 ----------------
    {
        Pos b;
        for (int x = 5; x <= 8; ++x) put(b, x, 8, WHITE);
        put(b, 3, 3, BLACK);
        put(b, 4, 4, BLACK);
        MCTSOptions opt;
        opt.n_sim = 300;
        DummyNet net;
        MCTS m(&net, opt, 13);
        std::vector<float> pol;
        const int mv = m.sample_move_with_policy(b, pol, 0.0);
        const bool blocks = (mv == 4 + 8 * SIZE) || (mv == 9 + 8 * SIZE);
        check(blocks, "对方活四时选择堵点");
        std::printf("       必挡: mv=(%d,%d)\n", mv % SIZE, mv / SIZE);
    }

    // ---------------- 4) 终局：无合法手 ----------------
    {
        Pos b;
        // 造一个黑已五连、但仍有空点的局面（棋盘状态为终局由上层判断）
        for (int x = 0; x < 5; ++x) put(b, x, 0, BLACK);
        for (int y = 1; y < 3; ++y)
            for (int x = 0; x < 5; ++x) put(b, x, y, WHITE);
        MCTSOptions opt;
        opt.n_sim = 50;
        DummyNet net;
        MCTS m(&net, opt, 17);
        std::vector<float> pol;
        const int mv = m.sample_move_with_policy(b, pol, 0.0);
        check(mv >= 0, "五连已成的局面仍能给出兜底落点");
    }

    // ---------------- 5) 树复用：set_root_move 命中后继续搜索 ----------------
    {
        Pos b;
        MCTSOptions opt;
        opt.n_sim = 120;
        DummyNet net;
        MCTS m(&net, opt, 19);
        std::vector<float> pol;
        const int mv = m.sample_move_with_policy(b, pol, 0.0);
        put(b, mv % SIZE, mv / SIZE, BLACK);
        const bool hit = m.set_root_move((int16_t)mv);
        check(hit, "set_root_move 命中刚搜索的子节点");
        std::vector<float> pol2;
        const int mv2 = m.sample_move_with_policy(b, pol2, 0.0);
        check(mv2 >= 0 && b.at(mv2 % SIZE, mv2 / SIZE) == 0, "复用后落点是空点");
        MCTSStats st = m.collect_stats();
        std::printf("       复用: mv2=(%d,%d) root_n=%d\n", mv2 % SIZE, mv2 / SIZE, st.root_visits);
    }

    // ---------------- 6) 虚拟损失不进 n/w 的不变式 ----------------
    {
        Pos b;
        put(b, 9, 9, BLACK);
        put(b, 10, 10, WHITE);
        MCTSOptions opt;
        opt.n_sim = 400;
        opt.batch_size = 16;
        DummyNet net;
        MCTS m(&net, opt, 23);
        std::vector<float> pol;
        m.sample_move_with_policy(b, pol, 0.0);
        Node* r = m.root();
        check(r != nullptr, "根存在");
        int sum_n = 0;
        bool vl_clean = true;
        for (const auto& c : r->children) {
            sum_n += c.node->n;
            if (c.node->vl != 0) vl_clean = false;
            for (const auto& g : c.node->children)
                if (g.node->vl != 0) vl_clean = false;
        }
        check(vl_clean, "搜索结束后所有节点 vl 归零");
        // 根的前 batch_size 次模拟叶子都是根自身 => 子节点之和 = n_sim - batch_size
        check(sum_n == 400 - 16, "子节点访问数之和 == n_sim - batch_size");
        check(r->n == 400, "根访问数 == n_sim");
        std::printf("       不变式: sum(child.n)=%d root.n=%d q=%.4f\n", sum_n, r->n, r->q);
    }

    // ---------------- 7) 「四的争夺」点不被宽度剪枝 ----------------
    // 盘上放 4 组互不相连的白三，每组两端都是「白落此处即成四」的点 → 8 个 tier-1。
    // 关掉保护时非根节点的子节点数必须 <= max_children(6)（能被剪掉）；
    // 打开时必须有节点超过 6（说明这些点被无条件保住了）。
    {
        auto max_children_over_nonroot = [](MCTS& m) {
            int best = 0;
            std::vector<Node*> stack;
            for (auto& c : m.root()->children) stack.push_back(c.node);
            while (!stack.empty()) {
                Node* n = stack.back();
                stack.pop_back();
                if ((int)n->children.size() > best) best = (int)n->children.size();
                for (auto& c : n->children) stack.push_back(c.node);
            }
            return best;
        };
        auto count_nodes = [](MCTS& m) {
            int n = 0;
            std::vector<Node*> stack{m.root()};
            while (!stack.empty()) {
                Node* x = stack.back();
                stack.pop_back();
                ++n;
                for (auto& c : x->children) stack.push_back(c.node);
            }
            return n;
        };

        auto build = []() {
            Pos b;
            // 黑先落（保证最后落的是白 → 轮到黑走）
            put(b, 15, 2, BLACK);
            put(b, 15, 4, BLACK);
            put(b, 15, 11, BLACK);
            put(b, 2, 13, BLACK);
            // 4 组白三，横向排开，彼此至少隔 2 格
            put(b, 2, 2, WHITE);
            put(b, 3, 2, WHITE);
            put(b, 4, 2, WHITE);
            put(b, 8, 4, WHITE);
            put(b, 9, 4, WHITE);
            put(b, 10, 4, WHITE);
            put(b, 3, 11, WHITE);
            put(b, 4, 11, WHITE);
            put(b, 5, 11, WHITE);
            put(b, 12, 13, WHITE);
            put(b, 13, 13, WHITE);
            put(b, 14, 13, WHITE);
            return b;
        };

        Pos b0 = build();
        check(b0.player == BLACK, "「四的争夺」用例：轮到黑走");

        // 先确认盘面上确实有 8 个 tier-1 点
        {
            std::vector<int> legal;
            b0.legal_moves(legal);
            PriorBuf buf;
            move_scores(b0, legal, BLACK, buf);
            int n_tier1 = 0, n_forced = 0;
            for (size_t i = 0; i < legal.size(); ++i) {
                if (buf.tier[i] == 1) ++n_tier1;
                if (buf.forced[i]) ++n_forced;
            }
            std::printf("       tier1=%d forced=%d legal=%zu\n", n_tier1, n_forced, legal.size());
            check(n_tier1 > 6, "盘面上有超过 6 个 tier-1 点");
            check(n_forced == 0, "该局面没有一步成五/堵五点（否则走的是另一条分支）");
        }

        int wid_on = 0, wid_off = 0;
        {
            DummyNet net;
            MCTSOptions opt;
            opt.n_sim = 300;
            opt.max_children = 6;
            opt.root_max_children = 16;
            opt.vcf = false;  // 排除 VCF 干扰，只看宽度保护
            opt.guarantee_rush = true;
            MCTS m(&net, opt, 31);
            Pos b = build();
            std::vector<float> pol;
            m.sample_move_with_policy(b, pol, 0.0);
            wid_on = max_children_over_nonroot(m);
        }
        {
            DummyNet net;
            MCTSOptions opt;
            opt.n_sim = 300;
            opt.max_children = 6;
            opt.root_max_children = 16;
            opt.vcf = false;
            opt.guarantee_rush = false;
            MCTS m(&net, opt, 31);
            Pos b = build();
            std::vector<float> pol;
            m.sample_move_with_policy(b, pol, 0.0);
            wid_off = max_children_over_nonroot(m);
        }
        std::printf("       非根最大子节点数: 保护开=%d 关=%d\n", wid_on, wid_off);
        check(wid_off <= 6, "关闭保护时非根宽度不超过 max_children");
        check(wid_on > 6, "开启保护时非根宽度被撑开（tier-1 未被剪掉）");
    }

    std::printf(g_fail ? "\n%d 项失败\n" : "\n全部通过\n", g_fail);
    return g_fail ? 1 : 0;
}
