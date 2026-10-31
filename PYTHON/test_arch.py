"""
test_arch.py —— 骨干架构读写的验收门槛（放大网络的**前置**保障）。

【为什么需要这个测试】放大骨干（3×64 → 6×128）时最容易踩的坑是：
    每个旧加载路径都写着 `net = GomokuNet(); net.load_state_dict(sd, strict=False)`
  —— 这会把 6×128 的权重**整包当成"多余层"丢掉**，得到一个随机初始化的 3×64
  网络，而且**不报错**：compare/ladder/arena 照样跑完、照样打印比分，
  结论却全是噪声。这个测试就是钉死这条：
  ① 大网络必须被完整加载（missing/unexpected 都是 0）；
  ② 绝不允许"静默降级"成 3×64；
  ③ 老 checkpoint 的行为必须一字不变（严格加载 0 缺失）；
  ④ 所有入口（train/compare/play/val_value/probe_infer_speed）走同一个加载器。

用法:
    python test_arch.py            # 全部通过则退出码 0
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

TMP = os.path.join("checkpoints", "_archtest")
OLD = ["models/run3/gen_1.pt", "models/run3/gen_30.pt",
       "models/run3/gen_60.pt", "models/run3/best.pt",
       "models/run3/init_gen9.pt"]

fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    os.makedirs(TMP, exist_ok=True)
    print("=" * 74)
    print("架构读写验收 test_arch.py")
    print("=" * 74)

    # ---- 1. infer_arch 反推 ----
    print("\n[1] infer_arch 从权重反推规模")
    small = GomokuNet(n_blocks=3, channels=64)
    big = GomokuNet(n_blocks=6, channels=128)
    check("3x64 反推", GomokuNet.infer_arch(small.state_dict()) == (3, 64, 64),
          str(GomokuNet.infer_arch(small.state_dict())))
    check("6x128 反推", GomokuNet.infer_arch(big.state_dict()) == (6, 128, 64),
          str(GomokuNet.infer_arch(big.state_dict())))

    # ---- 2. 老 checkpoint：架构不变 + 严格加载 0 缺失 ----
    print("\n[2] 老 checkpoint 行为不变（3×64、严格加载 0 缺失）")
    for f in OLD:
        if not os.path.exists(f):
            print(f"  [跳过] {f} 不存在")
            continue
        net, ck = T.load_checkpoint_ex(f, verbose=False)
        check(f"{os.path.basename(f)} 架构 = 3×64",
              (net.n_blocks, net.channels) == (3, 64),
              f"{net.n_blocks}×{net.channels}")
        try:
            net.load_state_dict(ck["state_dict"], strict=True)
            check(f"{os.path.basename(f)} 严格加载 0 缺失", True,
                  f"{sum(p.numel() for p in net.parameters()) / 1e4:.1f} 万参数")
        except Exception as e:
            check(f"{os.path.basename(f)} 严格加载 0 缺失", False,
                  f"{type(e).__name__}: {str(e)[:80]}")

    # ---- 3. 大网络往返 ----
    print("\n[3] 6×128 存取往返（含 arch 元数据）")
    fp = os.path.join(TMP, "big.pt")
    torch.save({"state_dict": big.state_dict(), "gen": 0,
                "arch": big.arch()}, fp)
    net2, ck2 = T.load_checkpoint_ex(fp, verbose=False)
    check("往返后架构一致", (net2.n_blocks, net2.channels) == (6, 128))
    try:
        net2.load_state_dict(ck2["state_dict"], strict=True)
        check("往返后严格加载 0 缺失", True)
    except Exception as e:
        check("往返后严格加载 0 缺失", False,
              f"{type(e).__name__}: {str(e)[:80]}")
    # 【2026-10-13 任务 6】arch 字典新增了 aux/aux_ch 两个键（辅助头）。这里只断言
    # "主干规模被写进了 arch"，不再要求键集合完全等于这三项 —— 否则每加一个
    # 可选字段都要改这条断言，而它真正要守的是"加载方能据此建出同规模的网"。
    _arch = ck2.get("arch") or {}
    _want_arch = {"n_blocks": 6, "channels": 128, "value_hidden": 64}
    check("checkpoint 里带 arch 字段", all(_arch.get(k) == v
                                          for k, v in _want_arch.items()),
          str(_arch))
    n_par = sum(p.numel() for p in net2.parameters())
    check("参数量 = 182 万（7.4×）", abs(n_par / 1e4 - 182.0) < 0.5,
          f"{n_par / 1e4:.1f} 万")

    # ---- 4. 绝不静默降级（本测试的核心）----
    print("\n[4] 各入口都不许把 6×128 静默降级成 3×64")
    import compare
    from play import load_model
    import val_value
    import probe_infer_speed
    _t = T.load_checkpoint(fp, verbose=False)
    got = {
        "train.load_checkpoint": (_t.n_blocks, _t.channels),
        "compare.load_net": (compare.load_net(fp).n_blocks,
                             compare.load_net(fp).channels),
        "play.load_model": (load_model(fp).n_blocks, load_model(fp).channels),
        "val_value.load_net": (val_value.load_net(fp).n_blocks,
                               val_value.load_net(fp).channels),
        "probe_infer_speed.load_net": (probe_infer_speed.load_net(fp).n_blocks,
                                       probe_infer_speed.load_net(fp).channels),
    }
    for k, v in got.items():
        check(f"{k} → 6×128", v == (6, 128), str(v))

    # ---- 5. 两个独立加载器输出逐位相同 ----
    print("\n[5] 不同入口加载同一权重 → 输出逐位相同")
    d = np.load("models/run3/data/gen_59.npz")
    x = torch.from_numpy(d["states"][:2].astype(np.float32))
    a = T.load_checkpoint(fp, verbose=False).eval()
    b = compare.load_net(fp).eval()
    with torch.no_grad():
        pa, va = a.forward(x.to(a.device))
        pb, vb = b.forward(x.to(b.device))
    check("policy 逐位相同", bool(np.array_equal(pa.cpu().numpy(), pb.cpu().numpy())))
    check("value 逐位相同", bool(np.array_equal(va.cpu().numpy(), vb.cpu().numpy())))

    # ---- 6. resume 架构不一致必须报错（而不是静默跑错）----
    print("\n[6] resume.pt 架构不一致时必须显式报错")
    rp = os.path.join(TMP, "fake_resume.pt")
    torch.save({"gen": 1, "arch": {"n_blocks": 3, "channels": 64,
                                   "value_hidden": 64},
                "net": small.state_dict()}, rp)
    try:
        T.load_resume_state(rp, net2, None, torch.optim.Adam(net2.parameters()),
                            T.ReplayBuffer(), np.random.default_rng(0))
        check("架构不匹配时抛 ValueError", False, "没有抛异常！")
    except ValueError as e:
        check("架构不匹配时抛 ValueError", True, str(e)[:60] + "…")
    except Exception as e:
        check("架构不匹配时抛 ValueError", False,
              f"抛的是 {type(e).__name__}: {e}")

    # ---- 7. 热启动 + 续训：resume.pt 没有辅助头、当前网络新挂辅助头 ----
    # 【2026-10-13 加，为**两次真实崩溃**】V4 的起法就是"从 V3 的 resume.pt 续训"，
    # 而 V3 的网络没有辅助头。先按当前结构建好辅助头、再读旧 resume.pt，参数集合就对不上：
    #   ① optimizer.load_state_dict → ValueError: loaded state dict contains a
    #      parameter group that doesn't match the size of optimizer's group
    #   ② ema.load_state_dict（严格）→ 缺 aux_heads.* 键
    #   ③ best_net.load_state_dict（严格）→ RuntimeError: Missing key(s) …
    # 三条都会让训练**根本起不来**（不是跑错，是启动即崩）。修法是
    # "优化器按参数顺序部分加载 + EMA/best_net 只容忍辅助头键差"。
    # 这条测试把它钉死：必须能起，且主干权重与主干动量逐位保留。
    print("\n[7] 新挂辅助头后从没有辅助头的旧 resume.pt 续训必须能起")
    rp2 = os.path.join(TMP, "resume_noaux.pt")
    opt_o = torch.optim.Adam(small.parameters(), lr=1e-3)
    _ps, _vs = small.forward(torch.rand(2, 3, 19, 19, device=small.device))
    (_ps.sum() + _vs.sum()).backward()      # 让优化器里真的有动量
    opt_o.step()
    torch.save({"gen": 5, "arch": small.arch(), "lr": 1e-3,
                "net": small.state_dict(), "ema": small.state_dict(),
                "best_net": small.state_dict(),
                "optimizer": opt_o.state_dict()}, rp2)

    hot = GomokuNet(n_blocks=3, channels=64, aux="threat,ownership", aux_ch=16)
    opt_h = torch.optim.Adam(hot.parameters(), lr=1e-3)
    ema_h = T.EMAWeights(hot, decay=0.99)
    names = [n for n, _ in hot.named_parameters()]
    n_all = len(names)
    n_aux = sum(1 for n in names if n.startswith("aux_heads."))
    try:
        st2 = T.load_resume_state(rp2, hot, ema_h, opt_h, T.ReplayBuffer(),
                                  np.random.default_rng(0))
        check("① 新挂辅助头后 load_resume_state 不抛异常", True)
        cur = hot.state_dict()
        check("② 主干权重逐位来自 resume.pt",
              all(torch.equal(cur[k].cpu(), st2["net"][k].cpu())
                  for k in st2["net"] if not k.startswith("aux_heads.")))
        n_old = len(opt_o.state)
        check("③ 主干动量被保留（部分加载，不因参数个数不同而整份丢弃）",
              len(opt_h.state) >= n_old > 0,
              f"旧 {n_old} 个 → 新 {len(opt_h.state)} 个")
        # ③b 是这条 bug 的真正要害：`aux_heads` 在 net.parameters() 里排在**最前面**，
        # 所以"按位置对齐"会把动量灌到错误的张量上 —— 形状巧合相同的那些（各种 64 维
        # BN）会**静默**错位，训练照样跑、只是动量脏了。只有按名字核对才抓得住。
        _nm = [n for n, _ in hot.named_parameters()]
        _pmap = dict(hot.named_parameters())
        _old_new_names = [k for k in st2["net"].keys() if k in set(_nm)]
        _oids = [i for g in opt_o.state_dict()["param_groups"]
                 for i in g["params"]]
        _ea_old = opt_o.state_dict()["state"][_oids[_old_new_names.index("stem.weight")]]["exp_avg"]
        # 注意：`optimizer.state` 的键是 **Parameter 对象**（不是索引）—— load_state_dict
        # 之后仍然如此。按索引查会静默拿到 None，把这条检查变成永远通过的假测试。
        _ea_new = opt_h.state.get(_pmap["stem.weight"], {}).get("exp_avg")
        check("③b 动量按**名字**对齐（不是按位置 —— 位置对齐会静默错位）",
              _ea_new is not None and torch.equal(_ea_new.cpu(), _ea_old.cpu()))
        _aux_params = {p for n, p in _pmap.items() if n.startswith("aux_heads.")}
        check("④ 新增的辅助头参数从零动量开始",
              not (_aux_params & set(opt_h.state.keys())),
              f"参数总数 {n_all}（其中辅助头 {n_aux}）")
        with torch.no_grad():
            _p2, _v2, _a2 = hot.forward_with_aux(
                torch.rand(2, 3, 19, 19, device=hot.device))
        check("⑤ 续训后前向可用（辅助头输出形状正确）",
              tuple(_p2.shape) == (2, 361) and tuple(_v2.shape) == (2,) and
              set(_a2) == {"threat", "ownership"},
              f"threat{tuple(_a2['threat'].shape)} ownership{tuple(_a2['ownership'].shape)}")
        # 门控对手（best_net）也必须能按旧存档灌回
        best_h = GomokuNet(n_blocks=3, channels=64, aux="threat,ownership",
                           aux_ch=16)
        n_new_keys = T.load_best_net_into(best_h, st2["best_net"])
        check("⑥ best_net 按旧存档灌回（只容忍辅助头键差）",
              n_new_keys == n_aux and
              all(torch.equal(best_h.state_dict()[k].cpu(), st2["best_net"][k].cpu())
                  for k in st2["best_net"]),
              f"未加载 {n_new_keys} 个新键")
        # 反面：骨干真的不一致时必须抛错，不许静默降级
        try:
            T.load_best_net_into(big, st2["best_net"])
            check("⑦ best_net 骨干不一致时必须抛 ValueError", False, "没有抛异常！")
        except ValueError:
            check("⑦ best_net 骨干不一致时必须抛 ValueError", True)
        except Exception as e:
            check("⑦ best_net 骨干不一致时必须抛 ValueError", False,
                  f"抛的是 {type(e).__name__}")
        # ⑧ 同一套辅助头、**相反注册顺序** —— 参数个数相同（39 对 39），
        # 所以"个数相同就直接原生加载"的捷径会按位置错位。实测的崩溃点是
        # **optimizer.step()**（`_multi_tensor_adam` 里 ownership(1,32,1,1) 的
        # exp_avg 撞上 threat(2,32,1,1) 的梯度），只检查"load 有没有抛异常"是抓
        # 不住的 —— 静默错位必须靠"按名字逐位比对动量"才看得出来。
        rp3 = os.path.join(TMP, "resume_aux_order.pt")
        _ph, _vh, _ah = hot.forward_with_aux(
            torch.rand(2, 3, 19, 19, device=hot.device))
        (_ph.sum() + _vh.sum() + _ah["threat"].sum()
         + _ah["ownership"].sum()).backward()
        opt_h.step()
        torch.save({"gen": 6, "arch": hot.arch(), "lr": 1e-3,
                    "net": hot.state_dict(), "ema": hot.state_dict(),
                    "best_net": hot.state_dict(),
                    "optimizer": opt_h.state_dict()}, rp3)
        _sd3 = torch.load(rp3, map_location="cpu", weights_only=False)["optimizer"]
        _ids3 = [i for g in _sd3["param_groups"] for i in g["params"]]
        _by_name = {names[j]: _sd3["state"][_ids3[j]]["exp_avg"]
                    for j in range(len(_ids3))}
        rev = GomokuNet(n_blocks=3, channels=64,
                        aux="ownership,threat", aux_ch=16)    # ← 顺序相反
        opt_r = torch.optim.Adam(rev.parameters(), lr=1e-3)
        try:
            T.load_resume_state(rp3, rev, None, opt_r, T.ReplayBuffer(),
                                np.random.default_rng(0))
            _pm_r = dict(rev.named_parameters())
            _bad = [k for k in _by_name
                    if not torch.equal(opt_r.state[_pm_r[k]]["exp_avg"].cpu(),
                                       _by_name[k].cpu())]
            _pr, _vr, _ar = rev.forward_with_aux(
                torch.rand(2, 3, 19, 19, device=rev.device))
            (_pr.sum() + _vr.sum() + _ar["threat"].sum()
             + _ar["ownership"].sum()).backward()
            opt_r.step()                        # ← 旧代码在这里崩
            check("⑧ 等个数但顺序相反时动量仍按名字对齐，且能真的 step",
                  not _bad, f"错位 {len(_bad)}/{len(_by_name)} 个动量")
        except Exception as e:
            check("⑧ 等个数但顺序相反时动量仍按名字对齐，且能真的 step", False,
                  f"{type(e).__name__}: {e}")
    except Exception as e:
        check("① 新挂辅助头后 load_resume_state 不抛异常", False,
              f"{type(e).__name__}: {e}")

    shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 74)
    if fails:
        print(f"结果：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结果：全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
