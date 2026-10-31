// vcf.h — 正向 VCF（连续冲四）静态搜索
//
// 用途：在 MCTS 叶子进网络评估之前，用便宜的穷举证明「当前行棋方能靠连续冲四
// 直接取胜」。命中就不花 GPU 评估了（Rapfi 的 SimpleVCF 就是这么接的）。
//
// 语义边界（与 Rapfi 一致）：
//   * 只做**正向** VCF —— 返回 1 表示「我方有必杀」，0 表示「未证明」，
//     **不表示必败**。搜索里拿 0 当成「这叶子还需要评估」用。
//   * 只考虑冲四（成四后必须被堵），不处理活三/VCT。深度因此很短即可覆盖常见杀。
//   * 不修改传入的 Pos。
#pragma once

#include "board.h"

namespace gk {

struct VCFOptions {
    // 攻击方最多连续冲四几手（1 手 = 我冲四 + 对方唯一挡点）。默认 6 手够覆盖
    // 绝大多数实战 VCF，再深代价陡增而收益很小。
    int max_ply = 6;
    // 搜索节点预算上限。超了立刻返回 0（未证明），保证单次调用有硬上限。
    int max_nodes = 128;
};

// 返回 1 = 当前行棋方存在连续冲四必胜；0 = 未证明。
int vcf_win(const Pos& pos, const VCFOptions& opt = VCFOptions{});

// 同上，但额外把「取胜的第一手」写进 *out_cell（0..N_CELLS-1 的扁平下标）。
// out_cell 传 nullptr 即等价于 vcf_win()。用于根节点直接落子，不经过 MCTS。
// 注意：只保证是**一条**必胜路径的首手，不保证是最短的那条。
int vcf_find_move(const Pos& pos, int* out_cell, const VCFOptions& opt = VCFOptions{});

}  // namespace gk
