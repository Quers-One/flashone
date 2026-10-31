// elo.h — 版本间 Elo 估计（Bradley–Terry / 对数几率回归，KataGo 风格）
//
// 动机：门控只会回答「这一代 > 0.55 吗」，回答不了「43 代里一共涨了多少」。
// 连续 19 代被拒时，无法区分「模型没进步」和「20 局测不出来」。Elo 曲线把整条
// 历史压成一条单调可比的纵轴。
//
// 口径（对齐 KataGo 的 elo 图）：
//   * 一对版本间打 N 局，记 score = (胜 + 0.5*平) / N；
//   * 用 Bradley–Terry 拟合 γ_i：p_ij = γ_i / (γ_i + γ_j)，MM 算法迭代；
//   * Elo_i = 400/ln10 * ln(γ_i / γ_ref)，锚点（通常是第一代或最强代）恒为 0；
//   * 置信区间用参数化 bootstrap（对每对的胜负重采样），给 5%–95% 分位；
//   * 只有一个连通分量才算得出来：链式相邻对打 + 每代都与锚点打一局，图必连通。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gk {

struct EloModel {
    std::string name;   // 显示名，如 "gen_024"
    std::string path;   // 权重路径
    int gen = 0;
    double elo = 0.0, lo = 0.0, hi = 0.0;
    int games = 0;      // 参与的对局数（用于判断该点有多可信）
};

struct EloEdge {
    int a = 0, b = 0;      // models 下标
    int games = 0;
    double score_a = 0.0;  // A 视角得分：胜=1、平=0.5、负=0
    int wins_a = 0, draws = 0;
    double elapsed_s = 0.0;
};

// 拟合。anchor 指向的那个模型 Elo 恒为 0；bootstrap>0 时给出 lo/hi。
void fit_elo(std::vector<EloModel>& models, const std::vector<EloEdge>& edges, int anchor,
             int bootstrap = 200, uint64_t seed = 12345);

// 生成 elo.json（看板直接读）。
std::string elo_json(const std::vector<EloModel>& models, const std::vector<EloEdge>& edges,
                     int anchor, const std::string& generated_at);

}  // namespace gk
