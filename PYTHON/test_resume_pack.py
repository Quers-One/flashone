"""
test_resume_pack.py —— resume.pt 回放池压缩存储（2026-10-16）的验收门槛。

【为什么需要这个测试】resume.pt 每代写一份，而它的 97% 是**未压缩**的回放池：
    58 代 / 49.4 万条 = 1249.9 MB（numpy 数组被 pickle 当 bytes 内联进 data.pkl，
    torch.save 的 zip 又是 ZIP_STORED）。改成"一代一个 npz 压缩块 + 缓存"后，
    写盘从 ≈1231 MB/代 降到 ≈60 MB/代。

压缩存储最怕**静默改变训练语义**，所以这个测试钉死五件事：
    ① 往返逐位一致：states(uint8)/policies(float32)/values 与条目顺序、gen 全等
       —— 代龄按**池中位置**算（`_ages()`），顺序一变，采样权重就变；
    ② 真的变小：落在盘上的文件 < 原始池字节的一半；
    ③ **旧格式 resume.pt 仍能读**（现存 models/*/resume.pt 全是旧格式，读坏了
       等于毁掉全部可续训状态）；
    ④ 熵轴缓存 `sample_w`/`sample_cumw` 不丢（丢了，开着熵轴的 run 一 resume 就炸）；
    ⑤ 加载后的池还能正常吃新数据（`add_generation` + 逐出按 gen_limit 生效）、
       采样整形（`--buffer-decay-half-life`）在加载后仍按位置算代龄。

用法:
    python test_resume_pack.py        # 全部通过则退出码 0
"""

import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

from network import GomokuNet
import train as T

TMP = os.path.join("checkpoints", "_resumepacktest")

fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        fails.append(name)


def make_gen(n, gen, seed):
    """造一代**像真的**样本：0/1 棋盘 uint8 + 稀疏 MCTS 策略 float32。"""
    rng = np.random.default_rng(seed)
    states = (rng.random((n, 3, 19, 19)) < 0.5).astype(np.uint8)
    pol = np.zeros((n, 361), dtype=np.float32)
    for i in range(n):
        idx = rng.choice(361, size=8, replace=False)
        pol[i, idx] = rng.random(8).astype(np.float32)
        pol[i] /= pol[i].sum()
    return {"gen": int(gen), "states": states, "policies": pol,
            "values": rng.standard_normal(n).astype(np.float32)}


def gen_equal(a, b):
    """只比"训练语义相关"的字段：形状/dtype/逐位内容。"""
    return (a["states"].dtype == b["states"].dtype
            and a["states"].shape == b["states"].shape
            and a["policies"].dtype == b["policies"].dtype
            and a["values"].dtype == b["values"].dtype
            and np.array_equal(a["states"], b["states"])
            and np.array_equal(a["policies"], b["policies"])
            and np.array_equal(a["values"], b["values"]))


def raw_bytes(gens):
    return sum(g["states"].nbytes + g["policies"].nbytes + g["values"].nbytes
               for g in gens)


def main():
    os.makedirs(TMP, exist_ok=True)
    print("=" * 74)
    print("resume.pt 回放池压缩存储验收 test_resume_pack.py")
    print("=" * 74)

    gens = [make_gen(60, 1, 1), make_gen(60, 2, 2), make_gen(60, 3, 3)]

    # ---- 1. pack/unpack 往返 ----
    print("\n[1] pack_buffer / unpack_buffer 往返")
    buf = T.ReplayBuffer()
    for g in gens:
        buf.add_generation(g["states"], g["policies"], g["values"], g["gen"])
    # 模拟"熵轴"写进条目的缓存字段（train.py `_ensure_sample_weights`）
    buf.generations[0]["sample_w"] = np.linspace(0.1, 0.2, len(gens[0]["states"]))
    buf.generations[0]["sample_cumw"] = np.cumsum(buf.generations[0]["sample_w"])

    packed = T.pack_buffer(buf)
    back = T.unpack_buffer(packed)
    check("① 代数/顺序/gen 一致",
          len(back) == 3 and [b["gen"] for b in back] == [1, 2, 3],
          str([b["gen"] for b in back]))
    check("① states/policies/values 逐位一致",
          all(gen_equal(a, b) for a, b in zip(buf.generations, back)))
    check("① dtype 原样（uint8 / float32）",
          back[0]["states"].dtype == np.uint8
          and back[0]["policies"].dtype == np.float32
          and back[0]["values"].dtype == np.float32)
    check("④ 熵轴缓存 sample_w/sample_cumw 不丢",
          np.array_equal(back[0]["sample_w"], buf.generations[0]["sample_w"])
          and np.array_equal(back[0]["sample_cumw"],
                             buf.generations[0]["sample_cumw"]))
    check("② 压缩：blob 明显小于原始字节",
          sum(len(p["blob"]) for p in packed) < raw_bytes(buf.generations) * 0.5,
          f"{raw_bytes(buf.generations)/1e6:.2f} MB → "
          f"{sum(len(p['blob']) for p in packed)/1e6:.2f} MB")
    # 同一代只压一次：再压一次必须命中缓存（否则每代写盘都会重压整池）
    b1 = T.pack_buffer(buf)
    check("② 二代压缩命中缓存（字节相同）",
          [p["blob"] for p in b1] == [p["blob"] for p in packed])

    # 条目后来多出字段（真实场景：sample() 给缺权重的旧条目补 sample_cumw）⇒ 必须重压
    buf.generations[1]["sample_cumw"] = np.arange(len(gens[1]["states"]),
                                                 dtype=np.float64)
    b2 = T.pack_buffer(buf)
    back2 = T.unpack_buffer(b2)
    check("④ 条目新增字段 ⇒ 缓存失效重压（字段不丢）",
          "sample_cumw" in back2[1] and b2[1]["blob"] != packed[1]["blob"]
          and np.array_equal(back2[1]["sample_cumw"],
                             buf.generations[1]["sample_cumw"]))

    # ---- 2. 全链路：save_resume_state → load_resume_state ----
    print("\n[2] 全链路 save/load（新格式）")
    net = GomokuNet(n_blocks=3, channels=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # 先把优化器状态"喂"出来：没 step 过的 Adam，state 是空的，读回来自然是 0 个
    for _p in net.parameters():
        _p.grad = torch.randn_like(_p)
    opt.step()
    rng = np.random.default_rng(1234)
    rp = os.path.join(TMP, "resume_new.pt")
    T.save_resume_state(rp, gen=3, net=net, ema=None, best_net=None, optimizer=opt,
                        lr=7e-4, lr_decay_count=2, rng=rng, buffer=buf,
                        summary={"final_gen": 3, "accepted": 1, "rejected": 2},
                        lr_bad_streak=1, name_offset=9)
    size = os.path.getsize(rp)
    # 用"同一份状态 + 空池"作基线：两个文件大小之差就是池的落盘成本。
    # 不这么比会被网络权重淹没 —— 测试用的 3×64 网只有 ≈1 MB，真实 6×128 是 36 MB。
    p0 = os.path.join(TMP, "resume_nopool.pt")
    T.save_resume_state(p0, gen=3, net=net, ema=None, best_net=None, optimizer=opt,
                        lr=7e-4, lr_decay_count=2, rng=rng, buffer=T.ReplayBuffer(),
                        summary={}, lr_bad_streak=1, name_offset=9)
    pool_mb = (size - os.path.getsize(p0)) / 1e6
    check("② 池的落盘成本 < 原始字节的一半",
          pool_mb < raw_bytes(buf.generations) / 1e6 * 0.5,
          f"池原始 {raw_bytes(buf.generations)/1e6:.2f} MB → 落盘 +{pool_mb:.2f} MB"
          f"（{raw_bytes(buf.generations)/1e6/max(pool_mb, 1e-9):.0f}×）")
    st_raw = torch.load(rp, map_location="cpu", weights_only=False)
    check("② 池以压缩块落盘（buffer_packed 标记）",
          int(st_raw.get("buffer_packed", 0)) == 2
          and isinstance(st_raw["buffer"][0].get("blob"), (bytes, bytearray)))

    net2 = GomokuNet(n_blocks=3, channels=64)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    buf2 = T.ReplayBuffer()
    rng2 = np.random.default_rng(0)
    st = T.load_resume_state(rp, net2, None, opt2, buf2, rng2)
    check("③ gen/lr/name_offset/lr_decay_count/lr_bad_streak 读回",
          (int(st["gen"]), float(st["lr"]), int(st["name_offset"]),
           int(st["lr_decay_count"]), int(st["lr_bad_streak"])) == (3, 7e-4, 9, 2, 1),
          f"gen={st['gen']} name_offset={st['name_offset']}")
    check("③ 池逐位一致 + total 正确",
          buf2.total == buf.total and len(buf2.generations) == 3
          and all(gen_equal(a, b) for a, b in
                  zip(buf.generations, buf2.generations)),
          f"total={buf2.total}")
    check("③ 网络权重逐位读回",
          all(torch.equal(p1, p2) for p1, p2 in
              zip(net.state_dict().values(), net2.state_dict().values())))
    check("③ 优化器状态读回（动量逐位一致）",
          opt2.state.get(next(iter(net2.parameters()))) is not None
          and torch.equal(opt2.state[next(iter(net2.parameters()))]["exp_avg"],
                          opt.state[next(iter(net.parameters()))]["exp_avg"]),
          f"{len(opt2.state)} 个参数")
    check("③ rng 状态读回",
          repr(rng2.bit_generator.state) == repr(rng.bit_generator.state))

    # ---- 3. 旧格式兼容（原始数组条目、无 blob）----
    print("\n[3] 旧格式 resume.pt 兼容")
    net3 = GomokuNet(n_blocks=3, channels=64)
    old_entries = [{k: v for k, v in g.items() if not k.startswith("_")}
                   for g in buf.generations]
    old_path = os.path.join(TMP, "resume_old.pt")
    torch.save({"gen": 5, "arch": net3.arch(), "lr": 1e-3,
                "net": net3.state_dict(), "ema": None, "best_net": None,
                "optimizer": torch.optim.Adam(net3.parameters(), lr=1e-3).state_dict(),
                "buffer": old_entries, "summary": {"final_gen": 5},
                "name_offset": 1, "rng": rng.bit_generator.state}, old_path)
    buf3 = T.ReplayBuffer()
    st3 = T.load_resume_state(old_path, net3, None,
                              torch.optim.Adam(net3.parameters(), lr=1e-3), buf3,
                              np.random.default_rng(0))
    check("③ 旧格式能读：gen + 池逐位一致 + total",
          int(st3["gen"]) == 5 and len(buf3.generations) == 3
          and buf3.total == buf.total
          and all(gen_equal(a, b) for a, b in
                  zip(buf.generations, buf3.generations)),
          f"total={buf3.total}")
    check("③ 旧格式能读：熵轴缓存字段保留",
          np.array_equal(buf3.generations[0]["sample_w"],
                         buf.generations[0]["sample_w"]))

    # ---- 4. 加载后的池还能用：新数据、逐出、代龄 ----
    print("\n[4] 加载后：新数据进池 / 逐出 / 代龄仍按位置算")
    buf4 = T.ReplayBuffer(capacity=10_000_000, gen_limit=3, decay_half_life=20.0,
                          rng=np.random.default_rng(7))
    for g in gens:
        buf4.add_generation(g["states"], g["policies"], g["values"], g["gen"])
    p4 = os.path.join(TMP, "resume_pool.pt")
    T.save_resume_state(p4, gen=3, net=net, ema=None, best_net=None, optimizer=opt,
                        lr=1e-3, lr_decay_count=0, rng=rng, buffer=buf4, summary={})
    buf5 = T.ReplayBuffer(capacity=10_000_000, gen_limit=3, decay_half_life=20.0,
                          rng=np.random.default_rng(8))
    T.load_resume_state(p4, net2, None, opt2, buf5, np.random.default_rng(0))
    check("⑤ 加载后 _ages() 按池中位置 = [2,1,0]",
          list(np.asarray(buf5._ages(), dtype=float)) == [2.0, 1.0, 0.0])
    buf5.sample(60)
    stats = dict(buf5.last_sample_stats)
    check("⑤ 加载后采样整形统计正常（平均代龄 > 0）",
          float(stats.get("mean_age_exp", -1)) > 0,
          f"mean_age_exp={stats.get('mean_age_exp')}")
    before = [g["gen"] for g in buf5.generations]
    gnew = make_gen(60, 4, 4)
    buf5.add_generation(gnew["states"], gnew["policies"], gnew["values"], 4)
    check("⑤ 加载后新数据能进池、逐出按 gen_limit",
          [g["gen"] for g in buf5.generations] == before[1:] + [4]
          and buf5.total == int(sum(len(g["states"]) for g in buf5.generations)),
          str([g["gen"] for g in buf5.generations]))
    # 加载后再存一次：新代只压一次，旧代命中缓存（字节可复用）
    p5 = os.path.join(TMP, "resume_pool2.pt")
    T.save_resume_state(p5, gen=4, net=net, ema=None, best_net=None, optimizer=opt,
                        lr=1e-3, lr_decay_count=0, rng=rng, buffer=buf5, summary={})
    st5 = torch.load(p5, map_location="cpu", weights_only=False)
    check("⑤ 加载后再存：格式仍是压缩块",
          all("blob" in e for e in st5["buffer"]) and len(st5["buffer"]) == 3)

    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
    print()
    if fails:
        print(f"结果: {len(fails)} 项失败 -> " + "; ".join(fails))
        sys.exit(1)
    print("结果: 全部通过")
