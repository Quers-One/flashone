"""
test_buffer_sample.py —— 回放池采样整形（任务 4）的验收门槛。

【为什么需要这个测试】2026-10-13 的 V4 白跑了 15 代，根因就在这里：
    `prepare_run.ps1` 把继承来的回放池代数**重编号成 1..N**，而新 run 又从 **1**
    重新计数；采样当时用 `最新gen - 本条gen` 当代龄 ⇒ 代龄全成了**负数**，
    `2^(-代龄/半衰期)` 于是把**最老**的数据抬到最高权重 —— 恰好是任务 4 要治的
    那个病的**反向**（症状：日志 `mean_age_exp` 从 -37.63 爬到 -23.69、
    `share_recent20_seen` 恒为 1.0，设计值 ≈0.55）。

    当时漏掉它的原因很具体：单元测试与 80 代集成跑里，池子都是 train.py 自己写
    的、编号连续一致，"代龄为负"这条路径**一次都没被走到**。而热启动
    （prepare_run 重编号 + 本 run 从 1 计数）是 V4 唯一走的路，却完全没覆盖。

这个测试钉死六件事：
    ① 关掉整形 ⇒ 与旧算法**逐位一致**（同种子、同 rng 调用序列）；
    ② 代龄按**池中位置**算 ⇒ 最近 20 代占比 ≈0.55、平均代龄为正且与手算一致；
    ③ **撞号场景**（池 = 1..40 + 新代 1、2、3）：代龄非负、最近 20 代占比仍在
       [0.4, 0.7]、权重偏向新数据 —— 这就是 V4 那个 bug 的回归测试；
    ④ `weight_distribution()` 单调偏向新数据、且归一化到 1；
    ⑤ 熵轴 `exp(β·H/Hmax)` 与手算一致；旧条目缺 `sample_cumw` 能当场补齐；
    ⑥ 关掉整形时不写任何统计、也不建任何缓存。

用法:
    python test_buffer_sample.py       # 全部通过则退出码 0
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import train as T

fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ----------------------------------------------------------------------
def make_gen(n, seed):
    """造一代合成数据（形状与真实 npz 一致：states uint8 / policies f32 / values f32）。"""
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 3, size=(n, 3, 19, 19), dtype=np.uint8)
    logits = rng.normal(0.0, 4.0, size=(n, 361))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    pol = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    val = rng.normal(0.0, 0.5, size=(n,)).astype(np.float32)
    return states, pol, val


def old_sample(gens, rng, n, exclude_latest):
    """逐字复刻"整形开关引入之前"的 sample()，用于 ① 的逐位对照。"""
    pool = gens[:-1] if exclude_latest else gens
    if not pool or n <= 0:
        return None
    sizes = np.array([len(g["states"]) for g in pool], dtype=np.float64)
    probs = sizes / sizes.sum()
    gen_idx = rng.choice(len(pool), size=n, p=probs)
    out_s = np.empty((n, 3, 19, 19), dtype=np.uint8)
    out_p = np.empty((n, 361), dtype=np.float32)
    out_v = np.empty((n,), dtype=np.float32)
    for gi in np.unique(gen_idx):
        mask = gen_idx == gi
        k = int(mask.sum())
        g = pool[int(gi)]
        sel = rng.integers(0, len(g["states"]), size=k)
        out_s[mask] = g["states"][sel]
        out_p[mask] = g["policies"][sel]
        out_v[mask] = g["values"][sel]
    return out_s, out_p, out_v


def build(labels, n, half_life=0.0, entropy=0.0, seed=7):
    """按给定 gen 标签造一个回放池（gen_limit 放大，免得代被淘汰掉）。"""
    buf = T.ReplayBuffer(capacity=10_000_000, gen_limit=1000,
                         rng=np.random.default_rng(seed),
                         decay_half_life=half_life, entropy_weight=entropy)
    for i, lb in enumerate(labels):
        s, p, v = make_gen(n, 1000 + i)
        buf.add_generation(s, p, v, lb)
    return buf


def main():
    print("=" * 74)
    print("回放池采样整形验收 test_buffer_sample.py")
    print("=" * 74)

    # ---- 1. 关掉整形 ⇒ 与旧算法逐位一致 ----
    print("\n[1] 关掉整形时与旧算法逐位一致（同种子、同 rng 调用序列）")
    labels = list(range(1, 13))
    for n_draw in (1, 7, 77, 256, 5000):
        for excl in (False, True):
            b = build(labels, 500, seed=7)
            got = b.sample(n_draw, exclude_latest=excl)
            ref = old_sample(b.generations, np.random.default_rng(7), n_draw, excl)
            ok = (got is not None and ref is not None
                  and np.array_equal(got[0], ref[0])
                  and np.array_equal(got[1], ref[1])
                  and np.array_equal(got[2], ref[2]))
            check(f"n={n_draw} exclude_latest={excl} 三个数组逐位一致", ok)
    b = build(labels, 500, seed=7)
    b.sample(64)
    check("关掉整形时不写统计", b.last_sample_stats == {}, str(b.last_sample_stats))
    check("关掉整形时不建 sample_cumw 缓存",
          all("sample_cumw" not in g for g in b.generations))

    # ---- 2. 代龄按池中位置算 ----
    print("\n[2] 代龄按池中位置算（60 代池、半衰期 20）")
    buf = build(list(range(1, 61)), 1000, half_life=20.0, seed=11)
    ages = np.arange(59, -1, -1, dtype=np.float64)      # 全池：最新 0、最老 59
    w = np.power(2.0, -ages / 20.0)
    exp_share20 = float(w[ages < 20.0].sum() / w.sum())
    exp_mean = float((w * ages).sum() / w.sum())
    buf.sample(20000, exclude_latest=False)
    st = buf.last_sample_stats
    # 注意：last_sample_stats 里的数都按 4 位小数存（round(...,4)），容差必须留到 1e-3
    check("share_recent20_exp 与手算一致",
          abs(st["share_recent20_exp"] - exp_share20) < 1e-3,
          f"{st['share_recent20_exp']} vs {exp_share20:.4f}")
    check("share_recent20_exp 落在设计区间 [0.5, 0.65]",
          0.5 <= st["share_recent20_exp"] <= 0.65, str(st["share_recent20_exp"]))
    check("mean_age_exp 为正且与手算一致",
          st["mean_age_exp"] > 0 and abs(st["mean_age_exp"] - exp_mean) < 0.01,
          f"{st['mean_age_exp']} vs {exp_mean:.2f}")
    check("实测最近 20 代占比贴近期望（±0.04）",
          abs(st["share_recent20_seen"] - exp_share20) < 0.04,
          f"{st['share_recent20_seen']} vs {exp_share20:.4f}")
    check("实测平均代龄贴近期望（±0.6）",
          abs(st["mean_age_seen"] - exp_mean) < 0.6,
          f"{st['mean_age_seen']} vs {exp_mean:.2f}")

    # ---- 3. 撞号场景（V4 那个 bug 的回归测试）----
    print("\n[3] 撞号：池 = 1..40（prepare_run 重编号）+ 新代 1、2、3（本 run 从 1 计数）")
    buf = build(list(range(1, 41)), 800, half_life=20.0, seed=13)
    for lb in (1, 2, 3):
        s, p, v = make_gen(800, 2000 + lb)
        buf.add_generation(s, p, v, lb)
    check("标签不递增被识别出来（gen_labels_monotone() 为 False）",
          buf.gen_labels_monotone() is False,
          f"{len(buf.generations)} 代")
    newest_label = int(buf.generations[-1]["gen"])
    label_ages = [newest_label - int(g["gen"]) for g in buf.generations]
    check("旧算法（按标签算代龄）在此场景下确实会算出负代龄",
          min(label_ages) < 0, f"最负 {min(label_ages)}")
    buf.sample(20000, exclude_latest=True)
    st = buf.last_sample_stats
    check("撞号下 mean_age_exp 非负（修前是 -37.6 那种）",
          st["mean_age_exp"] >= 0.0, str(st["mean_age_exp"]))
    check("撞号下 share_recent20_exp ∈ [0.4, 0.7]",
          0.4 <= st["share_recent20_exp"] <= 0.7, str(st["share_recent20_exp"]))
    check("撞号下 share_recent20_seen ∈ [0.4, 0.7]（修前恒为 1.0）",
          0.4 <= st["share_recent20_seen"] <= 0.7, str(st["share_recent20_seen"]))
    check("撞号下 mean_age_seen 非负", st["mean_age_seen"] >= 0.0, str(st["mean_age_seen"]))

    shares = [x[1] for x in buf.weight_distribution()]
    check("权重单调偏向新数据（越新占比越大）",
          all(b >= a - 1e-12 for a, b in zip(shares, shares[1:])),
          f"最老 {shares[0]:.4f} → 最新 {shares[-1]:.4f}")
    check("weight_distribution() 归一化到 1（误差 < 1e-9）",
          abs(sum(shares) - 1.0) < 1e-9, f"sum={sum(shares):.12f}")

    # ---- 4. 熵轴 ----
    print("\n[4] 熵轴 exp(β·H/Hmax)")
    uni = np.full((3, 361), 1.0 / 361, dtype=np.float32)
    onehot = np.zeros((3, 361), dtype=np.float32)
    onehot[:, 0] = 1.0
    wu = T._policy_entropy_weights(uni, 2.0)
    wo = T._policy_entropy_weights(onehot, 2.0)
    check("均匀策略 ⇒ e^β = 7.389", np.allclose(wu, np.exp(2.0), rtol=1e-5),
          f"{wu[0]:.4f}")
    check("独热策略 ⇒ 1.0", np.allclose(wo, 1.0, rtol=1e-6), f"{wo[0]:.4f}")
    check("β=0 ⇒ 恒为 1.0", np.allclose(T._policy_entropy_weights(uni, 0.0), 1.0))
    be = build([1, 2, 3], 200, entropy=2.0, seed=17)
    check("开熵轴后 add_generation 缓存了 sample_cumw",
          all("sample_cumw" in g for g in be.generations))
    check("sample_cumw 严格递增",
          all(np.all(np.diff(g["sample_cumw"]) > 0) for g in be.generations))

    # ---- 5. 旧 resume.pt 里的代条目（没有 sample_cumw）----
    print("\n[5] 旧 resume.pt 里的代条目（缺 sample_cumw）照常抽样并补齐")
    old = T.ReplayBuffer(capacity=10_000_000, gen_limit=100,
                         rng=np.random.default_rng(23),
                         decay_half_life=20.0, entropy_weight=2.0)
    for i in range(5):
        s, p, v = make_gen(300, 3000 + i)
        old.generations.append({"gen": i + 1, "states": s, "policies": p, "values": v})
        old.total += len(s)
    check("手工塞进去的旧条目确实没有 sample_cumw",
          "sample_cumw" not in old.generations[0])
    out = old.sample(50, exclude_latest=True)
    check("旧条目照常抽样（返回 50 条）",
          out is not None and out[0].shape == (50, 3, 19, 19))
    check("抽样后当场补齐 sample_cumw", "sample_cumw" in old.generations[0])

    # ---- 6. 熵轴确实改变代内样本占比 ----
    print("\n[6] 熵轴确实改变代内占比（后半代是均匀策略 = 高熵）")
    for beta, lo, hi in ((0.0, 0.45, 0.55), (2.0, 0.83, 0.93)):
        bb = T.ReplayBuffer(capacity=10_000_000, gen_limit=100,
                            rng=np.random.default_rng(31),
                            decay_half_life=0.0, entropy_weight=beta)
        s, p, v = make_gen(4000, 41)
        half = len(p) // 2
        p = p.copy()
        p[:half] = 0.0
        p[:half, 0] = 1.0                    # 前半：独热（零熵，权重 1.0）
        p[half:] = 1.0 / 361                 # 后半：均匀（满熵，权重 e^β）
        bb.add_generation(s, p, v, 1)
        s2, p2, v2 = make_gen(100, 42)
        bb.add_generation(s2, p2, v2, 2)
        got = bb.sample(20000, exclude_latest=True)
        flat = float((got[1][:, 0] < 0.5).mean())
        check(f"熵轴 β={beta}：高熵样本占比 {flat:.3f} ∈ [{lo}, {hi}]",
              lo <= flat <= hi)

    print("\n" + "=" * 74)
    if fails:
        print(f"结果：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结果：全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
