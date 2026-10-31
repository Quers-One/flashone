"""
test_eval_parallel.py —— 门控并行化（`evaluate_models(..., n_workers>1)`）的正确性验收。

【为什么必须测这个】门控决定"采纳/拒绝"和"降不降 lr"，历史上它踩过一个
致命坑：颜色记账写错，导致"20 局胜率"实际记的是黑方胜率 —— 症状是胜率只出现
0.000 / 0.500 这类值，训练被噪声做决定（train.py 里那段注释记着）。
并行版本把"局号 -> 进程"重新分了一遍，因此必须证明：

  A. 同网络自战 -> 胜率不会退化成 0 或 1（说明颜色记账正常、双方对称）
  B. 强网络 vs 随机网络 -> 胜率必须**高**（若颜色记反，会得到 ~0）
  C. 进程数 > 局数 -> 自动退回串行，不崩
  D. `_mcts_kwargs_from` 抽出的参数能重建等价搜索器（否则子进程搜的不是同一套参数）

用法:
    python test_eval_parallel.py            # 全通过则 exit 0
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from network import GomokuNet
from mcts import MCTS
from train import Config, load_checkpoint, evaluate_models, _mcts_kwargs_from

fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        fails.append(name)


def mk(net, cfg, hw, pw, n_sim):
    return MCTS(net, n_sim=n_sim, c_puct=cfg.c_puct,
                virtual_loss=cfg.virtual_loss,
                batch_size=cfg.batch_size_mcts,
                max_children=cfg.max_children,
                root_max_children=cfg.root_max_children,
                heuristic_weight=hw, policy_weight=pw,
                root_noise_alpha=cfg.eval_noise_alpha,
                root_noise_eps=cfg.eval_noise_eps)


def main():
    cfg = Config()
    n_sim = 60
    print("=" * 74)
    print("门控并行化验收 test_eval_parallel.py")
    print("=" * 74)
    print(f"  n_sim={n_sim}　eval_noise={cfg.eval_noise_alpha}/"
          f"{cfg.eval_noise_eps}　eval_games={cfg.eval_games}")

    good = load_checkpoint("models/run3/gen_60.pt", verbose=False)
    good.eval()

    # ---- D. kwargs 往返 ----
    m = mk(good, cfg, 6.0, 0.3, n_sim)
    kw = _mcts_kwargs_from(m)
    m2 = MCTS(good, **kw, seed=1)
    check("_mcts_kwargs_from 往返（参数一致）",
          all(getattr(m2, k) == v for k, v in kw.items()),
          f"{len(kw)} 个参数")

    # ---- A. 同网络自战：不能退化成 0 / 1 ----
    print("\n[A] 同一网络自战（12 局）—— 胜率必须严格落在 (0,1)")
    a_serial = evaluate_models(good, good, 12, mk(good, cfg, 6.0, 0.3, n_sim),
                               mk(good, cfg, 6.0, 0.3, n_sim), n_workers=1)
    a_par = evaluate_models(good, good, 12, mk(good, cfg, 6.0, 0.3, n_sim),
                            mk(good, cfg, 6.0, 0.3, n_sim), n_workers=4)
    check("串行胜率在 (0,1) 内", 0.0 < a_serial < 1.0, f"{a_serial:.3f}")
    check("并行胜率在 (0,1) 内", 0.0 < a_par < 1.0, f"{a_par:.3f}")

    # ---- B. 强 vs 弱：并行版必须也是"强的高胜率" ----
    print("\n[B] 强(gen_60) vs 弱(随机初始化)，hw/pw=0/1（让网络质量主导）"
          "（8 局）")
    weak = GomokuNet()
    weak.eval()
    b_serial = evaluate_models(good, weak, 8, mk(good, cfg, 0.0, 1.0, n_sim),
                               mk(weak, cfg, 0.0, 1.0, n_sim), n_workers=1)
    b_par = evaluate_models(good, weak, 8, mk(good, cfg, 0.0, 1.0, n_sim),
                            mk(weak, cfg, 0.0, 1.0, n_sim), n_workers=4)
    check("串行：强的胜率高", b_serial >= 0.5, f"{b_serial:.3f}")
    check("并行：强的胜率高（颜色没记反）", b_par >= 0.5, f"{b_par:.3f}")

    # ---- C. 进程数 > 局数 -> 退回串行 ----
    print("\n[C] n_workers(8) > n_games(4) —— 必须退回串行且不崩")
    try:
        c = evaluate_models(good, good, 4, mk(good, cfg, 6.0, 0.3, n_sim),
                            mk(good, cfg, 6.0, 0.3, n_sim), n_workers=8)
        check("退回串行后仍给出合法胜率", 0.0 <= c <= 1.0, f"{c:.3f}")
    except Exception as e:
        check("退回串行后仍给出合法胜率", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 74)
    if fails:
        print(f"结果：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结果：全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
