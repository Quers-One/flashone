"""
test_lr_ratchet.py —— 门控的 lr 决策（含 `--no-lr-ratchet`）单元验收。

背景：run3 的门控在 20 局下假拒率 74.8%，平台期里 lr 被统计噪声一路压低
（13 次 → 初始值的 5.5%）—— 这是"训练自己把自己锁死"的典型。`--no-lr-ratchet`
把它改成"必须连续 N 次显著退化才降一次"。这个判断是纯函数，必须钉死行为。

用法：python test_lr_ratchet.py   （全通过则 exit 0）
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train import Config, lr_decay_decision

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}　got={got} want={want}")
    if not ok:
        fails.append(name)


def main():
    cfg = Config()          # lr_decay_min_win_rate=0.30, lr_decay_streak=3
    print("=" * 70)
    print("lr 棘轮决策验收 test_lr_ratchet.py")
    print("=" * 70)
    print(f"  阈值 lr_decay_min_win_rate={cfg.lr_decay_min_win_rate}　"
          f"streak={cfg.lr_decay_streak}")

    print("\n[1] 旧行为（lr_ratchet=True）：低于阈值立刻降")
    cfg.lr_ratchet = True
    check("0.25 -> 降", lr_decay_decision(0.25, False, cfg, 0), (True, 0))
    check("0.40 -> 不降", lr_decay_decision(0.40, False, cfg, 0), (False, 0))
    check("0.30 -> 不降（边界取 >）", lr_decay_decision(0.30, False, cfg, 0),
          (False, 0))

    print("\n[2] --no-lr-ratchet：必须连续 3 次")
    cfg.lr_ratchet = False
    check("第 1 次 0.25 -> 不降(计数1)", lr_decay_decision(0.25, False, cfg, 0),
          (False, 1))
    check("第 2 次 0.25 -> 不降(计数2)", lr_decay_decision(0.25, False, cfg, 1),
          (False, 2))
    check("第 3 次 0.25 -> 降(计数归零)", lr_decay_decision(0.25, False, cfg, 2),
          (True, 0))

    print("\n[3] --no-lr-ratchet：中间被打断就清零")
    check("0.25 -> (False,1)", lr_decay_decision(0.25, False, cfg, 0), (False, 1))
    check("0.40 -> 清零", lr_decay_decision(0.40, False, cfg, 1), (False, 0))
    check("再 0.25 -> 从 1 重新数", lr_decay_decision(0.25, False, cfg, 0),
          (False, 1))

    print("\n[4] 采纳 / 不门控")
    check("采纳 -> 不降且清零", lr_decay_decision(0.20, True, cfg, 2), (False, 0))
    check("win_rate=None（不门控）-> 计数不动",
          lr_decay_decision(None, False, cfg, 2), (False, 2))

    print("\n[5] 自定义 streak=2")
    cfg.lr_decay_streak = 2
    check("第 1 次 0.10 -> 不降", lr_decay_decision(0.10, False, cfg, 0),
          (False, 1))
    check("第 2 次 0.10 -> 降", lr_decay_decision(0.10, False, cfg, 1),
          (True, 0))

    print("\n" + "=" * 70)
    if fails:
        print(f"结果：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结果：全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
