// mcts.h — PUCT 搜索（对标 PYTHON/mcts.py，规格见 docs/spec_mcts.md）
//
// 与 Python 版对齐的语义要点（改这里要同步改 PYTHON 的测试）：
//   * 虚拟损失只进 score 分母 (1+n+vl)，绝不写 n/w —— 这是修 bug 的核心
//   * score = -q + c_puct * P * sqrt(1+N_parent) / (1+n+vl)，q 是「子节点视角」
//   * 平局取插入序第一个；children 插入序 = 先验降序、等值按 legal 下标降序
//   * 每条模拟恰好一个叶子 => leaf_visits == n_sim
//   * 根的前 batch_size 次模拟叶子全是根自身（真实行为，不要"优化"掉）
//   * 终局：先判我方一步成五 +1，再判对方 -1；一步活四的子节点直接 is_terminal
//   * 缓存键 = 含行棋方的 Zobrist，值 = 该局面「执子方视角」，写满 150k 后静默停写
//   * 统计量（n/w/q）与 PUCT 一律用 double，float 会在几百次回溯后翻转落子
//
// 与 Python 版的有意差异（见 docs/PORTING_NOTES.md）：
//   1. RNG 用 std::mt19937_64 + gamma 采样，不复刻 numpy PCG64（数据不可逐字节对齐）
//   2. 棋盘导航改为「从根 ply 撤子 + 沿链重放」，不做快照前缀核对：
//      本实现的 board_ 永远只从根 ply 之上重放，混合前缀在结构上不可能出现
//   3. 批内相同局面合并成一次网络前向（值完全相同，数值等价，仅省算力）
#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

#include "board.h"
#include "heuristics.h"

namespace gk {

// 网络评估器接口：一次给 B 个局面，回 B 个策略分布 + B 个价值。
// states:   B*3*361 float（ch0 黑 / ch1 白 / ch2 行棋方=1 表示黑走）
// policies: B*361 float（softmax 后，未 mask，索引 y*19+x）
// values:   B float（tanh 后 [-1,1]，该局面执子方视角）
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void evaluate_batch(const float* states, int batch, float* policies, float* values) = 0;
};

struct MCTSOptions {
    int n_sim = 400;               // <=0 表示按手数动态（400/800/1200）
    double c_puct = 1.0;
    double virtual_loss = 3.0;
    int batch_size = 16;           // 叶子批量
    int max_children = 6;          // 非根展开宽度
    int root_max_children = 16;    // 根展开宽度（不截断，<=0 表示全展开）
    double fpu_reduction = 0.20;
    double prior_fpu_weight = 1.0;
    bool use_heuristics = true;
    double heuristic_weight = 6.0;
    double policy_weight = 0.3;
    bool open_four_win = true;
    bool immediate_win_loss = true;
    // 正向 VCF（连续冲四）静态搜索：在叶子交给网络之前先证明「这里有必杀」。
    // 只证明必胜，不证明必败；命中就跳过网络评估。
    // 默认关闭：400 局自战胜率 0.390（Wilson 上界 0.439 < 0.5），实测显著变弱。
    bool vcf = false;
    int vcf_max_ply = 6;      // 冲四链深度（1 层 = 我冲四 + 对方挡）
    int vcf_max_nodes = 128;  // 单次 VCF 搜索的节点预算
    // 【2026-10-31 加】根节点 VCF：落子前先在根算一次连续冲四，命中就直接走那一手，不跑 MCTS。
    // 与上面的 vcf 是**两种不同的接法**：vcf 在每个叶子命中后替换网络评估（实测变弱），
    // vcf_root 完全不进 MCTS 内部，每手只多一次 VCF 调用，不改变搜索行为与训练数据分布。
    bool vcf_root = false;
    // 宽度剪枝时无条件保留「四的争夺」点（我能做出四 / 对方能做出四）。
    // 这些分数（1e5 / 1e4）在六子宽度下通常排得进，但没有保证；打开后必进候选。
    // 默认关闭：400 局自战胜率 0.450，实测无收益。
    bool guarantee_rush = false;
};

struct Node {
    int16_t move = -1;       // 父 -> 本节点所落之手（扁平下标）；根为 -1
    int8_t player = 0;       // 本节点局面的行棋方
    int8_t is_terminal = 0;  // 已判终局：选择时不再下钻
    int8_t expanded = 0;
    int8_t over = 0;         // 本局面已结束（五连或满盘）
    int8_t winner = 0;       // over 时的胜方：0 和棋 / 1 黑 / 2 白
    int8_t win_computed = 0; // win_moves 是否已算
    int16_t vl = 0;          // 虚拟损失计数（只进 score 分母）
    int32_t ply = 0;         // 绝对手数
    int32_t n = 0;           // 完成回溯的访问数
    uint64_t hash = 0;       // 含行棋方的 Zobrist
    double prior = 1.0;
    double w = 0.0;
    double q = 0.0;          // = w/n，每次回溯重算（不独立维护）
    double value = 0.0;      // 诊断用，搜索内部不读
    Node* parent = nullptr;

    struct Child {
        int16_t move;
        Node* node;
    };
    std::vector<Child> children;      // 插入有序，顺序即平局规则
    std::vector<int16_t> win_moves;   // 惰性缓存：本节点执子方的一步成五点

    Node* child(int16_t mv) const {
        for (const auto& c : children)
            if (c.move == mv) return c.node;
        return nullptr;
    }
};

struct MCTSStats {
    int64_t simulations = 0;
    int64_t network_calls = 0;
    int64_t states_evaluated = 0;
    int64_t cache_hits = 0;
    int64_t terminal_hits = 0;
    int64_t vcf_hits = 0;      // VCF 证明必胜而跳过网络评估的次数
    int64_t nodes_created = 0;
    int32_t root_visits = 0;
    int32_t root_children = 0;
    double root_value = 0.0;
    int32_t max_depth = 0;
    double elapsed_ms = 0.0;
};

class MCTS {
public:
    MCTS(Evaluator* net, const MCTSOptions& opt, uint64_t seed = 0);

    // ---- 可变属性（train.py 会在运行中直接改写，必须立即生效）----
    double heuristic_weight = 6.0;
    double policy_weight = 0.3;
    double root_noise_alpha = 0.0;  // <=0 关闭
    double root_noise_eps = 0.25;

    // ---- 树的生命周期 ----
    Node* set_root(const Pos& board);
    bool set_root_move(int16_t move);    void reset_tree();
    Node* root() const { return root_; }

    // ---- 搜索 ----
    // 跑搜索并输出 361 长度的访问分布（n/Σn；终局全 0）
    void get_action_probs(const Pos& board, std::vector<float>& out, double temp = -1.0,
                          int n_sim = -1);
    // 采样落手并输出策略目标（n/Σn，与温度无关）；终局返回 -1
    int sample_move_with_policy(const Pos& board, std::vector<float>& policy, double temp = -1.0,
                                int n_sim = -1, double noise_alpha = -1.0,
                                double noise_eps = -1.0);
    int get_best_action(const Pos& board, int n_sim = -1);

    int n_sim() const { return opt_.n_sim; }
    void set_n_sim(int v) { opt_.n_sim = v; }
    const MCTSOptions& options() const { return opt_; }
    MCTSOptions& options() { return opt_; }
    Evaluator* net() const { return net_; }
    MCTSStats collect_stats() const;
    void clear_stats() { stats_ = MCTSStats(); }

private:
    struct SimRec {
        Node* leaf;
        int path_begin, path_end;  // path_pool_ 上的区间
        int mark_begin, mark_end;  // mark_pool_ 上的区间
        double value;
    };

    void run_search(int budget);
    void reset_board(const Pos& b);
    void seek(Node* n);

    Node* new_node();
    Node* select_child(Node* node) const;
    double score(const Node* c, double sqrt_parent, double fpu_base) const;
    void expand(Node* node, const float* policy);
    void child_priors(Node* node, const float* policy, std::vector<int16_t>& moves,
                      std::vector<float>& priors);
    void flush();
    void backup(const Node* const* path, int count, double value);
    void apply_root_noise();
    bool terminal_value(Node* node, double& v);
    bool node_win_moves(Node* node);
    bool opponent_has_immediate_win(Node* node);
    // 根必须总是可展开：清掉 expand 期打上的捷径标记（活四/一步杀）
    void reopen_if_shortcut(Node* n);

    int effective_n_sim(int depth) const;
    double effective_temp(int depth) const;

    Evaluator* net_;
    MCTSOptions opt_;
    Node* root_ = nullptr;
    std::deque<Node> arena_;
    std::unordered_map<uint64_t, double> value_cache_;
    std::unordered_map<uint64_t, std::vector<float>> policy_cache_;
    static constexpr size_t MAX_CACHE = 150000;

    // 共享棋盘 + 自根 ply 起的落子栈
    Pos board_;
    int32_t cur_ply_ = 0;
    std::vector<std::pair<int16_t, int8_t>> placed_;

    // 每批暂存
    std::vector<SimRec> batch_;
    std::vector<Node*> path_pool_;
    std::vector<Node*> mark_pool_;
    std::vector<Node*> chain_;
    std::vector<float> state_buf_;
    std::vector<float> policy_buf_;
    std::vector<float> value_buf_;

    // 先验计算缓冲
    PriorBuf prior_buf_;
    std::vector<int> legal_;
    std::vector<float> netp_;
    std::vector<int16_t> tmp_moves_;
    std::vector<float> tmp_priors_;
    std::vector<uint8_t> keep_;

    bool root_noise_done_ = false;
    std::mt19937_64 rng_;
    mutable MCTSStats stats_;
};

}  // namespace gk
