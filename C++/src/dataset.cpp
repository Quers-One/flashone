// dataset.cpp — 样本变换、回放池、样本落盘
#include "dataset.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace gk {
namespace {

// np.rot90 在 (y,x) 平面是逆时针：dst(y,x) = src(x, N-1-y)
// 翻转沿 x 轴（左右镜像）且发生在旋转之后：dst(y,x) = rot(y, N-1-x)
inline void src_coords_impl(int y, int x, int rot, int flip, int& sy, int& sx) {
    const int xx = flip ? (SIZE - 1 - x) : x;
    switch (rot & 3) {
        case 0:
            sy = y;
            sx = xx;
            break;
        case 1:
            sy = xx;
            sx = SIZE - 1 - y;
            break;
        case 2:
            sy = SIZE - 1 - y;
            sx = SIZE - 1 - xx;
            break;
        default:
            sy = SIZE - 1 - xx;
            sx = y;
            break;
    }
}

}  // namespace

int transform_index(int dst_idx, int rot, int flip) {
    const int y = dst_idx / SIZE, x = dst_idx % SIZE;
    int sy, sx;
    src_coords_impl(y, x, rot, flip, sy, sx);
    return sy * SIZE + sx;
}

void transform_plane(const uint8_t* src, uint8_t* dst, int rot, int flip) {
    rot &= 3;
    if (rot == 0 && !flip) {
        std::memcpy(dst, src, (size_t)N_CELLS);
        return;
    }
    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            int sy, sx;
            src_coords_impl(y, x, rot, flip, sy, sx);
            dst[y * SIZE + x] = src[sy * SIZE + sx];
        }
    }
}

void transform_policy(const float* src, float* dst, int rot, int flip) {
    rot &= 3;
    if (rot == 0 && !flip) {
        std::memcpy(dst, src, sizeof(float) * N_CELLS);
        return;
    }
    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            int sy, sx;
            src_coords_impl(y, x, rot, flip, sy, sx);
            dst[y * SIZE + x] = src[sy * SIZE + sx];
        }
    }
}

void transform_sample(const Sample& src, Sample& dst, int rot, int flip) {
    for (int c = 0; c < 3; ++c)
        transform_plane(src.state + (size_t)c * N_CELLS, dst.state + (size_t)c * N_CELLS, rot, flip);
    transform_policy(src.policy, dst.policy, rot, flip);
    dst.value = src.value;
}

void augment_batch(const Sample* src, Sample* dst, int n, std::mt19937_64& rng) {
    if (n <= 0) return;
    // 先抽满 rot，再抽满 flip（与 Python 的 RNG 消耗顺序一致）
    std::vector<uint8_t> rots((size_t)n), flips((size_t)n);
    std::uniform_int_distribution<int> dr(0, 3), df(0, 1);
    for (int i = 0; i < n; ++i) rots[(size_t)i] = (uint8_t)dr(rng);
    for (int i = 0; i < n; ++i) flips[(size_t)i] = (uint8_t)df(rng);
    for (int i = 0; i < n; ++i)
        transform_sample(src[i], dst[i], rots[(size_t)i], flips[(size_t)i]);
}

// ---------------------------------------------------------------------------
// ReplayBuffer
// ---------------------------------------------------------------------------
void ReplayBuffer::clear() {
    gens_.clear();
    total_ = 0;
    last_stats_ = SampleStats();
}

void ReplayBuffer::add_generation(int gen, std::vector<Sample>&& samples) {
    if (samples.empty()) return;
    total_ += samples.size();
    gens_.push_back({gen, std::move(samples)});
    while ((int)gens_.size() > gen_limit_ && gens_.size() > 1) {
        total_ -= gens_.front().samples.size();
        gens_.erase(gens_.begin());
    }
    while (total_ > capacity_ && gens_.size() > 1) {
        total_ -= gens_.front().samples.size();
        gens_.erase(gens_.begin());
    }
}

std::vector<int> ReplayBuffer::draw_gen_indices(int n, int pool_size, std::mt19937_64& rng) const {
    std::vector<int> out;
    out.reserve((size_t)n);
    if (pool_size <= 0 || n <= 0) return out;

    std::vector<double> w((size_t)pool_size, 0.0);
    double sum = 0.0;
    double sum_age = 0.0, sum_recent = 0.0;
    for (int i = 0; i < pool_size; ++i) {
        const double age = double(pool_size - 1 - i);  // 最新代 age=0（按池中位置，不看 gen 标签）
        double wi = double(gens_[(size_t)i].samples.size());
        if (half_life_ > 0.0) wi *= std::pow(2.0, -age / half_life_);
        w[(size_t)i] = wi;
        sum += wi;
        sum_age += wi * age;
        if (age < 20.0) sum_recent += wi;
    }
    if (!(sum > 0.0)) return out;
    last_stats_.mean_age_exp = sum_age / sum;
    last_stats_.share_recent20_exp = sum_recent / sum;

    std::vector<double> cum((size_t)pool_size);
    double acc = 0.0;
    for (int i = 0; i < pool_size; ++i) {
        acc += w[(size_t)i] / sum;
        cum[(size_t)i] = acc;
    }
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    for (int k = 0; k < n; ++k) {
        const double r = ud(rng);
        int idx = (int)(std::upper_bound(cum.begin(), cum.end(), r) - cum.begin());
        if (idx >= pool_size) idx = pool_size - 1;
        out.push_back(idx);
    }
    return out;
}

void ReplayBuffer::sample(int n, std::vector<Sample>& out, std::mt19937_64& rng,
                          bool exclude_latest) {
    out.clear();
    const int pool = (int)gens_.size() - (exclude_latest ? 1 : 0);
    if (pool <= 0 || n <= 0) {
        last_stats_.n_hist = 0;
        return;
    }
    const std::vector<int> gidx = draw_gen_indices(n, pool, rng);
    if (gidx.empty()) return;
    out.reserve((size_t)n);
    double seen_age = 0.0;
    int seen = 0, seen_recent = 0;
    for (int gi : gidx) {
        const auto& g = gens_[(size_t)gi].samples;
        if (g.empty()) continue;
        std::uniform_int_distribution<size_t> ui(0, g.size() - 1);
        out.push_back(g[ui(rng)]);
        const double age = double(pool - 1 - gi);
        seen_age += age;
        ++seen;
        if (age < 20.0) ++seen_recent;
    }
    last_stats_.n_hist = (int)out.size();
    {
        std::vector<int> uniq = gidx;
        std::sort(uniq.begin(), uniq.end());
        last_stats_.gens_drawn = (int)(std::unique(uniq.begin(), uniq.end()) - uniq.begin());
    }
    if (seen > 0) {
        last_stats_.mean_age_seen = seen_age / seen;
        last_stats_.share_recent20_seen = double(seen_recent) / seen;
    }
}

void ReplayBuffer::sample_mixed(const std::vector<Sample>& fresh, int n_fresh, int n_hist,
                                std::vector<Sample>& out, std::mt19937_64& rng) {
    out.clear();
    out.reserve((size_t)(n_fresh + n_hist));
    if (!fresh.empty() && n_fresh > 0) {
        std::uniform_int_distribution<size_t> ui(0, fresh.size() - 1);
        for (int i = 0; i < n_fresh; ++i) out.push_back(fresh[ui(rng)]);
    }
    if (n_hist > 0) {
        std::vector<Sample> hist;
        sample(n_hist, hist, rng, /*exclude_latest=*/true);
        out.insert(out.end(), hist.begin(), hist.end());
    }
}

// ---------------------------------------------------------------------------
// 落盘
// ---------------------------------------------------------------------------
namespace {
constexpr char kMagic[4] = {'G', 'K', 'D', '1'};
struct Header {
    char magic[4];
    uint32_t count;
    uint32_t stride;  // sizeof(Sample) 的记录，用于自检
};
}  // namespace

bool save_samples(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    Header h{kMagic[0], kMagic[1], kMagic[2], kMagic[3], (uint32_t)samples.size(),
             (uint32_t)sizeof(Sample)};
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!samples.empty())
        f.write(reinterpret_cast<const char*>(samples.data()),
                (std::streamsize)(samples.size() * sizeof(Sample)));
    return f.good();
}

bool load_samples(const std::string& path, std::vector<Sample>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || std::memcmp(h.magic, kMagic, 4) != 0) return false;
    if (h.stride != (uint32_t)sizeof(Sample)) return false;
    out.resize(h.count);
    if (h.count > 0)
        f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(out.size() * sizeof(Sample)));
    return f.good();
}

}  // namespace gk
