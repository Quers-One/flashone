// probe_mcts.cpp — 一次性诊断：逐次单模拟，观察评估/缓存/节点增长
#include <cstdio>
#include <vector>

#include "mcts.h"

using namespace gk;

class Dummy : public Evaluator {
public:
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        (void)states;
        for (int i = 0; i < batch; ++i) {
            float* p = policies + (size_t)i * N_CELLS;
            for (int k = 0; k < N_CELLS; ++k) p[k] = 1.0f / (float)N_CELLS;
            values[i] = 0.0f;
        }
    }
};

int main() {
    Dummy net;
    MCTSOptions o;
    o.n_sim = 1;
    o.batch_size = 1;
    o.max_children = 6;
    o.root_max_children = 16;
    o.heuristic_weight = 0.5;
    o.policy_weight = 1.0;
    MCTS m(&net, o, 7);
    Pos board;
    std::vector<float> pol(N_CELLS);
    MCTSStats prev{};
    for (int i = 0; i < 40; ++i) {
        const int mv = m.sample_move_with_policy(board, pol, 0.0, 1, -1.0, -1.0);
        const MCTSStats s = m.collect_stats();
        printf("sim %2d mv=%4d  d_states=%lld d_cache=%lld d_nodes=%lld d_term=%lld  root_n=%lld "
               "root_child=%d\n",
               i, mv, (long long)(s.states_evaluated - prev.states_evaluated),
               (long long)(s.cache_hits - prev.cache_hits),
               (long long)(s.nodes_created - prev.nodes_created),
               (long long)(s.terminal_hits - prev.terminal_hits), (long long)s.root_visits,
               s.root_children);
        prev = s;
        if (mv < 0) break;
    }
    return 0;
}
