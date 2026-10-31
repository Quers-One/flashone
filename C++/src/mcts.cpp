// mcts.cpp — PUCT 搜索实现（对标 PYTHON/mcts.py）
#include "mcts.h"

#include "vcf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace gk {
namespace {
int g_dbg_cache = 0;
long long g_hash_mismatch = 0;
constexpr double kBig = std::numeric_limits<double>::infinity();

inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

// 复刻 np.argsort(priors)[::-1]：先验降序；等值按下标升序排序后反转 => 下标降序
void sort_desc_ties_high(const std::vector<float>& priors, std::vector<int>& order) {
    order.resize(priors.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (priors[a] != priors[b]) return priors[a] < priors[b];
        return a < b;
    });
    // 反转：先验降序、等值下标降序
    std::reverse(order.begin(), order.end());
}
}  // namespace

MCTS::MCTS(Evaluator* net, const MCTSOptions& opt, uint64_t seed)
    : heuristic_weight(opt.heuristic_weight), policy_weight(opt.policy_weight), net_(net), opt_(opt),
      rng_(seed ? seed : 0x9E3779B97F4A7C15ULL) {
    opt_.batch_size = std::max(1, opt_.batch_size);
    opt_.max_children = std::max(1, opt_.max_children);
    opt_.virtual_loss = std::max(0.0, opt_.virtual_loss);
    cur_ply_ = 0;
    batch_.reserve((size_t)opt_.batch_size + 4);
    path_pool_.reserve(512);
    mark_pool_.reserve(512);
    chain_.reserve(128);
}

Node* MCTS::new_node() {
    arena_.emplace_back();
    stats_.nodes_created++;
    return &arena_.back();
}

Node* MCTS::set_root(const Pos& board) {
    if (root_ && root_->ply == board.stones && root_->hash == board.hash) {
        reopen_if_shortcut(root_);
        seek(root_);
        return root_;
    }
    reset_tree();
    board_ = board;
    cur_ply_ = board.stones;
    placed_.clear();
    root_ = new_node();
    root_->move = -1;
    root_->player = board.player;
    root_->ply = board.stones;
    root_->hash = board.hash;
    root_->prior = 1.0;
    root_->parent = nullptr;
    if (board.stones == N_CELLS) {
        root_->over = 1;
        root_->winner = 0;
    }
    return root_;
}

void MCTS::reset_tree() {
    arena_.clear();
    root_ = nullptr;
    value_cache_.clear();
    policy_cache_.clear();
    batch_.clear();
    path_pool_.clear();
    mark_pool_.clear();
    placed_.clear();
    board_ = Pos();
    cur_ply_ = 0;
}

bool MCTS::set_root_move(int16_t move) {
    if (!root_) return false;
    Node* child = root_->child(move);
    if (!child) return false;
    // 关键：board_ / placed_ 是相对「当前根」的。提升新根之前先把棋盘落到这一手之后，
    // 再把落子栈清零，否则 seek() 会以为棋盘还在旧根上，哈希与节点记录不一致。
    seek(root_);
    const int x = move % SIZE;
    const int y = move / SIZE;
    board_.place(x, y, root_->player);
    root_->children.clear();
    child->parent = nullptr;
    child->prior = 1.0;
    root_ = child;
    placed_.clear();
    cur_ply_ = root_->ply;
    reopen_if_shortcut(root_);
    return true;
}

// 「活四 = 对方必败」这类捷径会在 expand 时把子节点直接标成 is_terminal/expanded。
// 那个节点一旦被提升为搜索根，就永远不再展开、子节点访问数恒为 0、策略目标全零。
// 根必须总是展开（Python 的根也是无条件下钻的），所以这里把捷径标记清掉重来。
void MCTS::reopen_if_shortcut(Node* n) {
    if (!n || n->over) return;
    if (!n->is_terminal) return;
    n->is_terminal = 0;
    n->expanded = 0;
    n->win_computed = 0;
    n->value = 0.0;
}

// ---------------------------------------------------------------------------
// 棋盘导航：把共享棋盘定位到 n 的局面
//   不变式：board_ 的前 root_->ply 手始终等于根局面，所以先撤到根 ply，
//   再沿「根 -> n」的链逐手重放。混合分支前缀在结构上不可能出现。
// ---------------------------------------------------------------------------
void MCTS::seek(Node* n) {
    if (!n) return;
    // 构造从根到 n 的链：chain_[0] = 根，chain_.back() = n
    chain_.clear();
    for (Node* c = n; c; c = c->parent) chain_.push_back(c);
    std::reverse(chain_.begin(), chain_.end());
    // 与当前落子栈求最长公共前缀后再回退。
    // 只按 ply 回退是错的：目标节点可能在另一个分支上，回退到相同深度会得到别的局面。
    size_t common = 0;
    while (common < placed_.size() && common + 1 < chain_.size() &&
           placed_[common].first == chain_[common + 1]->move) {
        ++common;
    }
    while (placed_.size() > common) {
        const auto& last = placed_.back();
        board_.unplace(last.first % SIZE, last.first / SIZE, last.second);
        placed_.pop_back();
    }
    // 沿链把剩余部分重放下去
    for (size_t i = placed_.size() + 1; i < chain_.size(); ++i) {
        Node* c = chain_[i];
        const int x = c->move % SIZE;
        const int y = c->move / SIZE;
        // 落子方 = 该节点父节点的 player
        const int8_t who = c->player == BLACK ? WHITE : BLACK;
        board_.place(x, y, who);
        placed_.emplace_back(c->move, who);
    }
    cur_ply_ = chain_.front()->ply + (int32_t)placed_.size();
    if (n->hash != board_.hash) {
        if (g_hash_mismatch++ == 0 && std::getenv("GK_DEBUG_CACHE")) {
            std::printf("[seek] mismatch ply=%d mv=%d node=%llu board=%llu | root_ply=%d "
                        "chain=%lld placed=%lld cur_ply=%d | parent ply=%d hash=%llu mv=%d\n",
                        (int)n->ply, (int)n->move, (unsigned long long)n->hash,
                        (unsigned long long)board_.hash, root_ ? (int)root_->ply : -1,
                        (long long)chain_.size(), (long long)placed_.size(), (int)cur_ply_,
                        n->parent ? (int)n->parent->ply : -1,
                        n->parent ? (unsigned long long)n->parent->hash : 0ULL,
                        n->parent ? (int)n->parent->move : -1);
        }
    }
}

// ---------------------------------------------------------------------------
// 选择
// ---------------------------------------------------------------------------
double MCTS::score(const Node* c, double sqrt_parent, double fpu_base) const {
    const double eff = double(c->n) + double(c->vl);
    if (eff <= 0.0) return fpu_base + opt_.prior_fpu_weight * opt_.c_puct * c->prior * sqrt_parent;
    const double q = c->n > 0 ? c->w / double(c->n) : 0.0;
    return -q + opt_.c_puct * c->prior * sqrt_parent / (1.0 + eff);
}

Node* MCTS::select_child(Node* node) const {
    if (node->children.empty()) return nullptr;
    const double sqrt_parent = std::sqrt(1.0 + double(node->n));
    const double fpu_base = -node->q - opt_.fpu_reduction;
    Node* best = nullptr;
    double best_score = -kBig;
    for (const auto& c : node->children) {
        const double s = score(c.node, sqrt_parent, fpu_base);
        if (s > best_score) {
            best_score = s;
            best = c.node;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// 先验：合法点顺序 = 扁平下标升序（规则层保证），宽度裁剪 + 强制点保护
// ---------------------------------------------------------------------------
void MCTS::child_priors(Node* node, const float* policy, std::vector<int16_t>& moves,
                        std::vector<float>& priors) {
    moves.clear();
    priors.clear();
    board_.legal_moves(legal_);
    const int n = (int)legal_.size();
    if (n == 0) return;
    netp_.resize((size_t)n);
    for (int i = 0; i < n; ++i) netp_[(size_t)i] = policy ? policy[legal_[(size_t)i]] : 0.f;

    prior_buf_.resize(n);
    compute_prior(board_, legal_, policy ? netp_.data() : nullptr, heuristic_weight, policy_weight,
                  opt_.use_heuristics, prior_buf_);

    const int width = (node->parent == nullptr) ? opt_.root_max_children : opt_.max_children;
    if (width <= 0 || n <= width) {
        moves.resize((size_t)n);
        priors.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            moves[(size_t)i] = (int16_t)legal_[(size_t)i];
            priors[(size_t)i] = prior_buf_.priors[(size_t)i];
        }
        return;
    }

    int n_forced = 0;
    int n_rush = 0;
    for (int i = 0; i < n; ++i) {
        if (prior_buf_.forced[(size_t)i]) ++n_forced;
        else if (opt_.guarantee_rush && prior_buf_.tier[(size_t)i] == 1) ++n_rush;
    }
    // limit 必须容纳所有被保护的点，否则下面的 picked 循环会把它们截断掉。
    const int limit = std::max(width, n_forced + n_rush);

    std::vector<int> order;
    sort_desc_ties_high(prior_buf_.priors, order);

    keep_.assign((size_t)n, 0);
    int kept = 0;
    for (int i = 0; i < n; ++i) {
        if (prior_buf_.forced[(size_t)i]) {
            keep_[(size_t)i] = 1;
            ++kept;
        }
    }
    // 强制点（成五 / 堵五）之外的「四的争夺」点也保证入选：漏掉一个冲四点，
    // 在六子宽度下几乎必输。这里的 tier==1 只有在 opt_.guarantee_rush 打开时生效。
    if (opt_.guarantee_rush) {
        for (int i = 0; i < n; ++i) {
            if (!keep_[(size_t)i] && prior_buf_.tier[(size_t)i] == 1) {
                keep_[(size_t)i] = 1;
                ++kept;
            }
        }
    }
    for (int idx : order) {
        if (kept >= limit) break;
        if (!keep_[(size_t)idx]) {
            keep_[(size_t)idx] = 1;
            ++kept;
        }
    }

    std::vector<int> picked;
    picked.reserve((size_t)limit);
    for (int idx : order) {
        if (keep_[(size_t)idx]) picked.push_back(idx);
        if ((int)picked.size() >= limit) break;
    }

    const int m = (int)picked.size();
    moves.resize((size_t)m);
    priors.resize((size_t)m);
    double s = 0.0;
    for (int k = 0; k < m; ++k) s += (double)prior_buf_.priors[(size_t)picked[(size_t)k]];
    for (int k = 0; k < m; ++k) {
        const int idx = picked[(size_t)k];
        moves[(size_t)k] = (int16_t)legal_[(size_t)idx];
        priors[(size_t)k] = s > 0.0 ? (float)((double)prior_buf_.priors[(size_t)idx] / s)
                                    : (float)(1.0 / m);
    }
    if (n_forced > 0) {
        double tot = 0.0;
        for (int k = 0; k < m; ++k) {
            const int idx = picked[(size_t)k];
            if (prior_buf_.forced[(size_t)idx]) priors[(size_t)k] = std::max(priors[(size_t)k], 0.5f);
            tot += priors[(size_t)k];
        }
        if (tot > 0.0)
            for (int k = 0; k < m; ++k) priors[(size_t)k] = (float)(priors[(size_t)k] / tot);
    }
}

// ---------------------------------------------------------------------------
// 扩展
// ---------------------------------------------------------------------------
void MCTS::expand(Node* node, const float* policy) {
    if (node->expanded) return;
    seek(node);
    child_priors(node, policy, tmp_moves_, tmp_priors_);
    if (tmp_moves_.empty()) {
        node->is_terminal = 1;
        node->expanded = 1;
        node->value = 0.0;
        return;
    }
    const int8_t mover = node->player;
    const int8_t other = (int8_t)(3 - mover);
    const bool of_win = opt_.open_four_win;

    for (size_t k = 0; k < tmp_moves_.size(); ++k) {
        const int mv = tmp_moves_[k];
        const int x = mv % SIZE, y = mv / SIZE;
        const bool open_four = of_win && board_.open_four_after(mv, mover);
        const bool won = board_.place(x, y, mover);
        const bool full = board_.stones == N_CELLS;
        const uint64_t h = board_.hash;

        Node* c = new_node();
        c->move = (int16_t)mv;
        c->player = other;
        c->parent = node;
        c->prior = (double)tmp_priors_[k];
        c->ply = node->ply + 1;
        c->hash = h;
        if (won || full) {
            c->over = 1;
            c->is_terminal = 1;
            c->expanded = 1;
            c->winner = won ? mover : 0;
            c->value = won ? 1.0 : 0.0;
        } else if (open_four) {
            c->is_terminal = 1;
            c->expanded = 1;
            c->value = -1.0;
        }
        node->children.push_back({(int16_t)mv, c});
        board_.unplace(x, y, mover);
    }
    node->expanded = 1;
}

// ---------------------------------------------------------------------------
// 叶子评估
// ---------------------------------------------------------------------------
bool MCTS::node_win_moves(Node* node) {
    if (node->win_computed) return !node->win_moves.empty();
    node->win_computed = 1;
    seek(node);
    board_.legal_moves(legal_);
    const int8_t me = node->player;
    for (int mv : legal_)
        if (board_.check_win_after(mv, me)) node->win_moves.push_back((int16_t)mv);
    return !node->win_moves.empty();
}

bool MCTS::opponent_has_immediate_win(Node* node) {
    seek(node);
    board_.legal_moves(legal_);
    const int8_t op = (int8_t)(3 - node->player);
    for (int mv : legal_)
        if (board_.check_win_after(mv, op)) return true;
    return false;
}

bool MCTS::terminal_value(Node* node, double& v) {
    if (node->over) {
        v = node->winner == 0 ? 0.0 : (node->winner == node->player ? 1.0 : -1.0);
        return true;
    }
    // 根节点不做「一步杀 / 必被一步杀」的捷径：否则根永远不展开，访问数全 0，
    // 策略目标会退化成全零（Python 的根总是展开的）。深层的叶子仍然走捷径。
    if (opt_.immediate_win_loss && node != root_) {
        if (node_win_moves(node)) {
            v = 1.0;
            return true;
        }
        if (opponent_has_immediate_win(node)) {
            v = -1.0;
            return true;
        }
    }
    // 正向 VCF：网络之前先用便宜的穷举证明「行棋方有连续冲四必胜」。
    // 只证明胜，不证明负（返回 0 只代表未证明），所以 v=-1 分支不存在。
    // 同样跳过根节点，理由同上（根必须展开以获得策略目标）。
    if (opt_.vcf && node != root_) {
        seek(node);
        VCFOptions vo;
        vo.max_ply = opt_.vcf_max_ply;
        vo.max_nodes = opt_.vcf_max_nodes;
        if (vcf_win(board_, vo)) {
            ++stats_.vcf_hits;
            v = 1.0;
            return true;
        }
    }
    return false;
}

void MCTS::flush() {
    if (batch_.empty()) return;
    // 1) 定值 + 收集需要网络的叶子
    std::vector<Node*> need;
    need.reserve(batch_.size());
    std::vector<uint64_t> need_hash;
    need_hash.reserve(batch_.size());
    std::vector<int> slot(batch_.size(), -1);
    std::vector<double> values(batch_.size(), 0.0);
    std::vector<const float*> pols(batch_.size(), nullptr);

    for (size_t i = 0; i < batch_.size(); ++i) {
        Node* leaf = batch_[i].leaf;
        seek(leaf);
        const uint64_t h = board_.hash;
        auto it = value_cache_.find(h);
        if (it != value_cache_.end()) {
            values[i] = it->second;
            auto pit = policy_cache_.find(h);
            if (pit != policy_cache_.end()) pols[i] = pit->second.data();
            stats_.cache_hits++;
            if (g_dbg_cache < 30 && std::getenv("GK_DEBUG_CACHE")) {
                std::printf("[cache] ply=%d h_ok=%d exp=%d kids=%lld term=%d over=%d n=%lld\n",
                            (int)leaf->ply, (int)(h == leaf->hash), (int)leaf->expanded,
                            (long long)leaf->children.size(), (int)leaf->is_terminal,
                            (int)leaf->over, (long long)leaf->n);
                ++g_dbg_cache;
            }
            continue;
        }
        double tv = 0.0;
        if (terminal_value(leaf, tv)) {
            values[i] = tv;
            stats_.terminal_hits++;
            if (value_cache_.size() < MAX_CACHE) value_cache_.emplace(h, tv);
            continue;
        }
        slot[i] = (int)need.size();
        need.push_back(leaf);
        need_hash.push_back(h);
    }

    // 2) 批量前向
    if (!need.empty()) {
        const int B = (int)need.size();
        state_buf_.assign((size_t)B * 3 * N_CELLS, 0.f);
        policy_buf_.assign((size_t)B * N_CELLS, 0.f);
        value_buf_.assign((size_t)B, 0.f);
        for (int k = 0; k < B; ++k) {
            seek(need[(size_t)k]);
            board_.to_tensor(state_buf_.data() + (size_t)k * 3 * N_CELLS);
        }
        net_->evaluate_batch(state_buf_.data(), B, policy_buf_.data(), value_buf_.data());
        stats_.network_calls++;
        stats_.states_evaluated += B;
        for (int k = 0; k < B; ++k) {
            const float* pol = policy_buf_.data() + (size_t)k * N_CELLS;
            const double v = (double)value_buf_[(size_t)k];
            const uint64_t h = need_hash[(size_t)k];
            if (value_cache_.size() < MAX_CACHE) {
                value_cache_.emplace(h, v);
                policy_cache_.emplace(h, std::vector<float>(pol, pol + N_CELLS));
            }
            for (size_t i = 0; i < batch_.size(); ++i) {
                if (slot[i] == k) {
                    values[i] = v;
                    pols[i] = policy_cache_.find(h) != policy_cache_.end()
                                  ? policy_cache_.find(h)->second.data()
                                  : pol;
                }
            }
        }
    }

    // 3) 扩展（此时棋盘在叶子处，expand 内部会再 seek 一次，代价小）
    for (size_t i = 0; i < batch_.size(); ++i) expand(batch_[i].leaf, pols[i]);

    // 4) 根噪声：只在根的第一次扩展之后施加一次
    if (root_ && root_->expanded && !root_->children.empty() && root_noise_alpha > 0.0 &&
        !root_noise_done_) {
        apply_root_noise();
        root_noise_done_ = true;
    }

    // 5) 回溯 + 收虚拟损失
    for (size_t i = 0; i < batch_.size(); ++i) {
        const SimRec& r = batch_[i];
        backup(path_pool_.data() + r.path_begin, r.path_end - r.path_begin, values[i]);
        for (int k = r.mark_begin; k < r.mark_end; ++k) mark_pool_[(size_t)k]->vl -= 1;
    }
    batch_.clear();
    path_pool_.clear();
    mark_pool_.clear();
}

void MCTS::backup(const Node* const* path, int count, double value) {
    double v = value;
    if (v < -1.0) v = -1.0;
    if (v > 1.0) v = 1.0;
    for (int i = count - 1; i >= 0; --i) {
        Node* n = const_cast<Node*>(path[i]);
        n->n += 1;
        n->w += v;
        n->q = n->w / double(n->n);
        v = -v;
    }
}

void MCTS::apply_root_noise() {
    const int K = (int)root_->children.size();
    if (K <= 0) return;
    std::gamma_distribution<double> gamma(root_noise_alpha, 1.0);
    std::vector<double> noise((size_t)K, 0.0);
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        noise[(size_t)i] = gamma(rng_);
        s += noise[(size_t)i];
    }
    if (!(s > 0.0)) return;
    for (int i = 0; i < K; ++i) {
        Node* c = root_->children[(size_t)i].node;
        c->prior = (1.0 - root_noise_eps) * c->prior + root_noise_eps * (noise[(size_t)i] / s);
    }
}

// ---------------------------------------------------------------------------
// 主循环
// ---------------------------------------------------------------------------
int MCTS::effective_n_sim(int depth) const {
    if (opt_.n_sim > 0) return opt_.n_sim;
    if (depth < 8) return 400;
    if (depth < 24) return 800;
    return 1200;
}

double MCTS::effective_temp(int depth) const {
    if (depth < 12) return 1.0;
    if (depth < 30) return 0.5;
    return 0.0;
}

void MCTS::run_search(int budget) {
    root_noise_done_ = false;
    for (int s = 0; s < budget; ++s) {
        Node* node = root_;
        const int path_begin = (int)path_pool_.size();
        const int mark_begin = (int)mark_pool_.size();
        path_pool_.push_back(node);
        while (!node->children.empty() && !node->is_terminal) {
            Node* nxt = select_child(node);
            if (!nxt) break;
            nxt->vl += 1;
            mark_pool_.push_back(nxt);
            path_pool_.push_back(nxt);
            node = nxt;
        }
        SimRec r;
        r.leaf = node;
        r.path_begin = path_begin;
        r.path_end = (int)path_pool_.size();
        r.mark_begin = mark_begin;
        r.mark_end = (int)mark_pool_.size();
        r.value = 0.0;
        batch_.push_back(r);
        if ((int)batch_.size() >= opt_.batch_size) flush();
    }
    flush();
    stats_.simulations += budget;
    if (root_) {
        stats_.root_visits = root_->n;
        stats_.root_children = (int)root_->children.size();
        stats_.root_value = root_->q;
    }
}

void MCTS::get_action_probs(const Pos& board, std::vector<float>& out, double temp, int n_sim_req) {
    out.assign(N_CELLS, 0.f);
    if (!board.has_legal()) return;
    Node* r = set_root(board);
    (void)r;
    const int budget = n_sim_req > 0 ? n_sim_req : effective_n_sim((int)board.stones);
    run_search(budget);
    if (!root_) return;
    double total = 0.0;
    for (const auto& c : root_->children)
        if (c.node->n > 0) total += (double)c.node->n;
    if (total <= 0.0) return;
    const double t = temp < 0.0 ? effective_temp((int)board.stones) : temp;
    if (t <= 0.0) {
        Node* best = nullptr;
        int bestn = -1;
        for (const auto& c : root_->children) {
            if (c.node->n > bestn) {
                bestn = c.node->n;
                best = c.node;
            }
        }
        if (best && bestn > 0) out[(size_t)best->move] = 1.f;
        return;
    }
    double s = 0.0;
    std::vector<double> w;
    w.reserve(root_->children.size());
    for (const auto& c : root_->children) {
        const double x = c.node->n > 0 ? std::pow((double)c.node->n, 1.0 / t) : 0.0;
        w.push_back(x);
        s += x;
    }
    for (size_t i = 0; i < root_->children.size(); ++i) {
        out[(size_t)root_->children[i].move] = s > 0.0 ? (float)(w[i] / s) : 0.f;
    }
}

int MCTS::sample_move_with_policy(const Pos& board, std::vector<float>& policy, double temp,
                                  int n_sim_req, double noise_alpha, double noise_eps) {
    policy.assign(N_CELLS, 0.f);
    if (!board.has_legal()) return -1;

    const double keep_alpha = root_noise_alpha;
    const double keep_eps = root_noise_eps;
    if (noise_alpha >= 0.0) root_noise_alpha = noise_alpha;
    if (noise_eps >= 0.0) root_noise_eps = noise_eps;

    set_root(board);
    const int budget = n_sim_req > 0 ? n_sim_req : effective_n_sim((int)board.stones);
    run_search(budget);

    int best = -1;
    if (root_) {
        double total = 0.0;
        for (const auto& c : root_->children)
            if (c.node->n > 0) total += (double)c.node->n;
        if (total > 0.0) {
            for (const auto& c : root_->children)
                if (c.node->n > 0) policy[(size_t)c.move] = (float)((double)c.node->n / total);
            const double t = temp < 0.0 ? effective_temp((int)board.stones) : temp;
            if (t <= 0.0) {
                int bestn = -1;
                for (const auto& c : root_->children)
                    if (c.node->n > bestn) {
                        bestn = c.node->n;
                        best = c.move;
                    }
            } else {
                std::vector<double> w;
                w.reserve(root_->children.size());
                double s = 0.0;
                for (const auto& c : root_->children) {
                    double x = 0.0;
                    if (c.node->n > 0) x = std::pow((double)c.node->n, 1.0 / t);
                    w.push_back(x);
                    s += x;
                }
                if (s > 0.0) {
                    std::uniform_real_distribution<double> ud(0.0, s);
                    double r = ud(rng_);
                    size_t pick = 0;
                    double acc = 0.0;
                    for (size_t i = 0; i < w.size(); ++i) {
                        acc += w[i];
                        if (r <= acc) {
                            pick = i;
                            break;
                        }
                        pick = i;
                    }
                    best = root_->children[pick].move;
                }
            }
        }
    }
    if (best < 0) {
        // 兜底：必胜点 > 必挡点 > argmax(raw)，与 Python 的 _heuristic_fallback 同序
        seek(root_);
        board_.legal_moves(legal_);
        if (!legal_.empty()) {
            prior_buf_.resize((int)legal_.size());
            compute_prior(board_, legal_, nullptr, heuristic_weight, policy_weight, true,
                          prior_buf_);
            int pick = 0;
            bool got = false;
            for (size_t i = 0; i < legal_.size(); ++i)
                if (prior_buf_.win[i]) {
                    pick = (int)i;
                    got = true;
                    break;
                }
            if (!got)
                for (size_t i = 0; i < legal_.size(); ++i)
                    if (prior_buf_.block[i]) {
                        pick = (int)i;
                        got = true;
                        break;
                    }
            if (!got) {
                double bv = -1e300;
                for (size_t i = 0; i < legal_.size(); ++i)
                    if (prior_buf_.raw[i] > bv) {
                        bv = prior_buf_.raw[i];
                        pick = (int)i;
                    }
            }
            best = legal_[(size_t)pick];
        } else {
            best = -1;
        }
    }

    root_noise_alpha = keep_alpha;
    root_noise_eps = keep_eps;
    return best;
}

int MCTS::get_best_action(const Pos& board, int n_sim_req) {
    // 根节点 VCF：先在根算一次「我方有连续冲四必胜」，命中就直接落那一手，完全跳过 MCTS。
    // 只在这里用，不碰叶子评估，所以不会破坏 MCTS 的价值回传与探索。
    if (opt_.vcf_root) {
        VCFOptions vo;
        vo.max_ply = opt_.vcf_max_ply;
        vo.max_nodes = opt_.vcf_max_nodes;
        int cell = -1;
        if (vcf_find_move(board, &cell, vo) && cell >= 0 && cell < N_CELLS) {
            ++stats_.vcf_hits;
            return cell;
        }
    }
    std::vector<float> out;
    get_action_probs(board, out, 0.0, n_sim_req);
    int best = -1;
    float bv = -1.f;
    for (int i = 0; i < N_CELLS; ++i)
        if (out[(size_t)i] > bv) {
            bv = out[(size_t)i];
            best = i;
        }
    return bv > 0.f ? best : -1;
}

MCTSStats MCTS::collect_stats() const {
    MCTSStats s = stats_;
    s.root_visits = root_ ? root_->n : 0;
    s.root_children = root_ ? (int)root_->children.size() : 0;
    s.root_value = root_ ? root_->q : 0.0;
    return s;
}

}  // namespace gk
