// elo_run.h — 「打一批 Elo 对局」与「自动更新曲线」
//
// 这一层把 cmd_elo 里那段「构造版本 → 生成对局对 → 打 → 收边」抽出来，
// 目的是让**训练循环也能调用同一段逻辑**（--elo-every）。两份实现迟早会对不上，
// 而 Elo 曲线的可信度全靠口径一致。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "elo_store.h"

namespace gk {

struct EloRunOptions {
    int games = 30;          // 每对局数
    int opening_plies = 2;   // 配对开局手数（0 = 关）
    int max_ref_span = 30;   // 参照边最大跨度（超过就跳，避免饱和边浪费局数）
    bool with_ref = true;    // 是否打参照边
    bool verbose = true;
    // 边库里这一对已经打过至少 skip_if_games 局就跳过（>0 生效）。
    // 看板按钮每次都送同一个窗口，不跳过的话滚动更新会反复重打同样的对局。
    int skip_if_games = 0;
    const std::vector<EloStoreEdge>* existing = nullptr;  // 边库快照（skip 时必填）
};

// 打一批：gens 内相邻代链式边 + 每个版本对锚点 anchor_gen 的参照边。
// 返回带名字的边（未写入边库）。权重文件不存在的代自动跳过。
std::vector<EloStoreEdge> elo_play(const TrainConfig& cfg, const std::string& dir,
                                   const std::vector<int>& gens, int anchor_gen,
                                   const EloRunOptions& o);

struct EloAutoOptions {
    int window = 5;          // 滚动窗口代数（还要再往前延到「最新已评代-2」保证重叠）
    int games = 30;          // 滚动边每对局数
    int calib_every = 15;    // 每 N 代打一条**跨段标定边**（老代直打当前代）；0 = 不做
    int calib_games = 400;   // 标定边局数（要能分辨 ±17 Elo，别低于 200）
    int max_ref_span = 30;
    int bootstrap = 200;
    std::string out;         // elo.json 路径；空 = <dir>/elo.json
    std::string store;       // 边库路径；空 = <dir>/elo_edges.jsonl
    bool verbose = true;
};

// 一次完整的自动更新：定窗口 → 打新边 → 追加边库 → **全量重拟合** → 重写 elo.json。
// 训练循环在 gen % elo_every == 0 时调用它；失败时只打印警告，绝不中断训练。
// 返回 0 = 成功。
int elo_auto_update(const TrainConfig& cfg, const std::string& dir, int cur_gen,
                    const EloAutoOptions& o);

struct EloCommitOptions {
    int bootstrap = 200;
    int anchor_gen = 0;      // 0 = 用「边库里最新已评代」当锚点
    std::string out;         // elo.json；空 = <dir>/elo.json
    std::string store;       // 边库；空 = <dir>/elo_edges.jsonl
    bool verbose = true;
};

// 把一批新边并入边库 → 用**全部边**重拟合 → 重写 elo.json。
// 这是边库模式的收口动作：不管边是滚动窗口打出来的、还是门控采纳后补打的、
// 还是命令行手打的，最后都走这里，保证"曲线 = 边库全部信息的自洽解"。
int elo_commit(const TrainConfig& cfg, const std::string& dir,
               const std::vector<EloStoreEdge>& fresh, const EloCommitOptions& o);

}  // namespace gk
