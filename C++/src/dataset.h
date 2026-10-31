// dataset.h — 自对弈样本、8 变换增广、回放池（对标 PYTHON/train.py 的 ReplayBuffer/augment_batch）
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "board.h"

namespace gk {

// ---------------------------------------------------------------------------
// 单样本：state 是 3x19x19 的 uint8 平面（与 Python 的 states 完全同布局）
//   ch0 = 黑子、ch1 = 白子、ch2 = 行棋方（轮黑全 1 / 轮白全 0）
// policy 是 361 维 float（索引 y*19+x），value 是该步行棋方视角的 +1/-1/0
// ---------------------------------------------------------------------------
struct Sample {
    uint8_t state[3 * N_CELLS];
    float policy[N_CELLS];
    float value;
};

// 坐标映射：目标格 dst_idx 在变换后取自源格 src_idx（rot 逆时针 90 度次数 0..3，flip 沿 x 镜像）
int transform_index(int dst_idx, int rot, int flip);
// 平面工具：rot 逆时针 90 度次数（0..3），flip 之后沿 x 轴左右镜像
void transform_plane(const uint8_t* src, uint8_t* dst, int rot, int flip);
// 把拉直成 361 的策略图旋转/翻转（与局面张量同一几何）
void transform_policy(const float* src, float* dst, int rot, int flip);
// 单样本变换：dst = transform(src, rot, flip)
void transform_sample(const Sample& src, Sample& dst, int rot, int flip);

// 整批随机增强：每样本独立抽 rot~U{0,3}、flip~U{0,1}（先抽满 rot 再抽满 flip，
// 与 Python 的 RNG 消耗顺序一致，便于对照复现）
void augment_batch(const Sample* src, Sample* dst, int n, std::mt19937_64& rng);

// ---------------------------------------------------------------------------
// 回放池：按「代」整块淘汰，永不裁半代（对标 Python ReplayBuffer）
//   容量：总条数 <= capacity 且代数 <= gen_limit
//   抽样：先按代抽，概率 ∝ 该代条数 × 2^(-代龄/半衰期)，再代内均匀抽
// ---------------------------------------------------------------------------
class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity, int gen_limit, double decay_half_life)
        : capacity_(capacity), gen_limit_(gen_limit), half_life_(decay_half_life) {}

    void add_generation(int gen, std::vector<Sample>&& samples);
    // 抽 n 条（有放回）。exclude_latest=true 时排除最后入池的一代
    void sample(int n, std::vector<Sample>& out, std::mt19937_64& rng, bool exclude_latest);
    // 混合抽样：n_fresh 条来自 fresh（当代数据，有放回），n_hist 条来自历史池
    void sample_mixed(const std::vector<Sample>& fresh, int n_fresh, int n_hist,
                      std::vector<Sample>& out, std::mt19937_64& rng);

    size_t total() const { return total_; }
    int generations() const { return (int)gens_.size(); }
    size_t capacity() const { return capacity_; }
    int gen_limit() const { return gen_limit_; }
    double half_life() const { return half_life_; }
    void clear();

    // 诊断：最近 20 代样本占比（期望值 vs 实际抽到）
    struct SampleStats {
        int n_hist = 0;
        int gens_drawn = 0;
        double mean_age_exp = 0.0;
        double mean_age_seen = 0.0;
        double share_recent20_exp = 0.0;
        double share_recent20_seen = 0.0;
    };
    SampleStats last_stats() const { return last_stats_; }
    void reset_stats() { last_stats_ = SampleStats(); }

private:
    struct Gen {
        int gen;
        std::vector<Sample> samples;
    };
    std::vector<int> draw_gen_indices(int n, int pool_size, std::mt19937_64& rng) const;

    std::vector<Gen> gens_;
    size_t total_ = 0;
    size_t capacity_;
    int gen_limit_;
    double half_life_;
    mutable SampleStats last_stats_;
};

// ---------------------------------------------------------------------------
// 落盘 / 读盘：与 Python 的 data/gen_XXX.npz 对齐的自定义二进制格式
//   magic "GKD1" + count + 每样本 (state 1083B + policy 1444B + value 4B)
// ---------------------------------------------------------------------------
bool save_samples(const std::string& path, const std::vector<Sample>& samples);
bool load_samples(const std::string& path, std::vector<Sample>& out);

}  // namespace gk
