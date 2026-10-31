// test_dataset.cpp — 8 变换与 numpy 逐字节对齐 + 回放池抽样/淘汰自测
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../src/dataset.h"

using namespace gk;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf(ok ? "[ ok ] %s\n" : "[FAIL] %s\n", what.c_str());
    if (!ok) ++g_fail;
}

std::vector<int> read_row(std::istream& is, int n) {
    std::vector<int> v((size_t)n);
    for (int i = 0; i < n; ++i) is >> v[(size_t)i];
    return v;
}
}  // namespace

int main() {
    // ---------------- 1) 8 变换：与 numpy 的 np.rot90 + fliplr 完全一致 ----------------
    {
        std::ifstream f("_tmp/aug_ref.txt");
        if (!f) {
            std::printf("[SKIP] 缺少 _tmp/aug_ref.txt\n");
            return 0;
        }
        std::vector<int> srcv = read_row(f, N_CELLS);
        Sample s{};
        for (int i = 0; i < N_CELLS; ++i) {
            s.state[i] = (uint8_t)srcv[(size_t)i];
            s.state[N_CELLS + (size_t)i] = (uint8_t)(255 - srcv[(size_t)i]);
            s.state[2 * N_CELLS + (size_t)i] = 1;
            s.policy[i] = (float)i * 0.5f;
        }
        bool all_ok = true;
        for (int t = 0; t < 8; ++t) {
            const int rot = t / 2, flip = t % 2;
            std::vector<int> ref = read_row(f, N_CELLS);
            uint8_t dst[N_CELLS];
            transform_plane(s.state, dst, rot, flip);
            bool ok = true;
            for (int i = 0; i < N_CELLS; ++i)
                if (dst[i] != (uint8_t)ref[(size_t)i]) {
                    ok = false;
                    std::printf("       t=%d 首个不一致 i=%d got=%d want=%d\n", t, i, (int)dst[i],
                                ref[(size_t)i]);
                    break;
                }
            if (!ok) all_ok = false;
            // 策略图与局面同构
            float pdst[N_CELLS];
            transform_policy(s.policy, pdst, rot, flip);
            bool okp = true;
            for (int i = 0; i < N_CELLS; ++i) {
                const float want = s.policy[(size_t)ref[(size_t)i] == 0 ? i : i];
                (void)want;
            }
            (void)okp;
        }
        check(all_ok, "8 个变换与 numpy rot90+fliplr 逐格一致");

        // 策略图应与 transform_index 的坐标映射一致
        bool pol_ok = true;
        for (int t = 0; t < 8; ++t) {
            const int rot = t / 2, flip = t % 2;
            float base[N_CELLS], got[N_CELLS];
            for (int i = 0; i < N_CELLS; ++i) base[i] = (float)i;
            transform_policy(base, got, rot, flip);
            for (int i = 0; i < N_CELLS; ++i) {
                const float want = (float)transform_index(i, rot, flip);
                if (got[i] != want) { pol_ok = false; break; }
            }
        }
        check(pol_ok, "策略图变换与局面同构（结构检查）");
    }

    // ---------------- 2) 变换是置换：值域不变、单射 ----------------
    {
        Sample a{}, b{};
        for (int i = 0; i < N_CELLS; ++i) {
            a.state[i] = (uint8_t)(i % 251);
            a.policy[i] = (float)i;
        }
        bool perm_ok = true;
        for (int t = 0; t < 8; ++t) {
            transform_sample(a, b, t / 2, t % 2);
            std::vector<int> s1(a.state, a.state + N_CELLS);
            std::vector<int> s2(b.state, b.state + N_CELLS);
            std::sort(s1.begin(), s1.end());
            std::sort(s2.begin(), s2.end());
            if (s1 != s2) perm_ok = false;
        }
        check(perm_ok, "8 变换都是置换（多重集不变）");
        // 逆变换：t 与 (4-t)%4 + 同样的 flip 复合应回到原样（rot 180）
        Sample r{};
        transform_sample(a, r, 1, 0);
        Sample back{};
        transform_sample(r, back, 3, 0);
        check(std::memcmp(a.state, back.state, N_CELLS) == 0, "rot90 后 rot270 复原");
    }

    // ---------------- 3) 回放池：淘汰与抽样 ----------------
    {
        ReplayBuffer buf(1000, 5, 0.0);
        std::mt19937_64 rng(42);
        for (int g = 0; g < 8; ++g) {
            std::vector<Sample> v;
            for (int i = 0; i < 200; ++i) {
                Sample s{};
                s.value = (float)g;
                v.push_back(s);
            }
            buf.add_generation(g, std::move(v));
        }
        check(buf.generations() == 5, "代数上限 5 生效");
        check(buf.total() <= 1000, "容量上限 1000 生效");
        std::vector<Sample> out;
        buf.sample(500, out, rng, false);
        check(out.size() == 500, "抽样条数正确");
        bool vals_ok = true;
        for (const auto& s : out)
            if (s.value < 3.0f || s.value > 7.0f) vals_ok = false;
        check(vals_ok, "只抽到仍在池中的代数");

        ReplayBuffer::SampleStats st = buf.last_stats();
        check(st.mean_age_exp >= 0.0 && st.mean_age_exp < 5.0, "平均代龄在 [0,代数) 内");

        std::vector<Sample> fresh(50);
        std::vector<Sample> mix;
        buf.sample_mixed(fresh, 20, 30, mix, rng);
        check(mix.size() == 50, "混合抽样条数 = fresh + hist");
    }

    // ---------------- 4) 落盘往返 ----------------
    {
        std::vector<Sample> v(7);
        for (int i = 0; i < 7; ++i) {
            v[(size_t)i].value = (float)i * 0.25f;
            v[(size_t)i].policy[0] = (float)i;
            v[(size_t)i].state[5] = (uint8_t)i;
        }
        check(save_samples("_tmp/samples_test.bin", v), "写出样本文件");
        std::vector<Sample> back;
        check(load_samples("_tmp/samples_test.bin", back), "读回样本文件");
        check(back.size() == v.size(), "条数一致");
        check(std::memcmp(back.data(), v.data(), v.size() * sizeof(Sample)) == 0, "内容逐字节一致");
    }

    std::printf(g_fail ? "\n%d 项失败\n" : "\n全部通过\n", g_fail);
    return g_fail ? 1 : 0;
}
