#!/usr/bin/env python
r"""Python 侧 checkpoint 与 C++ 引擎权重文件（GKAZ1）互转，并可选导出参考前向结果。

用法：
    python tools/convert_ckpt.py --ckpt ..\PYTHON\models\V4\gen_37.pt --out _tmp\v4_gen37.bin
    python tools/convert_ckpt.py --ckpt ... --out ... --ref-dir _tmp\ref
    python tools/convert_ckpt.py --bin ..\_tmp\out\gen_003.bin --pt ..\PYTHON\models\cpp_gen3.pt

--ref-dir 会写三个文件（float32 裸数据，供 C++ 逐值比对）：
    states.bin  (4,3,19,19)
    logits.bin  (4,361)   网络 eval 模式下的策略 logits（softmax 前）
    values.bin  (4,)      tanh 后的价值
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch


def load_state(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("arch")
    raise SystemExit("checkpoint 里没有 state_dict: %s" % ckpt_path)


def infer_arch(sd):
    channels = int(sd["stem.weight"].shape[0])
    blocks = 0
    while ("backbone.%d.conv1.weight" % blocks) in sd:
        blocks += 1
    value_hidden = int(sd["value_fc1.weight"].shape[0])
    return blocks, channels, value_hidden


def write_bin(path, sd, arch):
    blocks, channels, value_hidden = arch
    names = list(sd.keys())
    with open(path, "wb") as f:
        f.write(b"GKAZ1")
        f.write(struct.pack("<3i", blocks, channels, value_hidden))
        f.write(struct.pack("<i", len(names)))
        for name in names:
            t = sd[name].detach().cpu().contiguous()
            if t.dtype == torch.float32:
                code = 0
                raw = t.numpy().astype(np.float32).tobytes()
            elif t.dtype == torch.int64:
                code = 1
                raw = t.numpy().astype(np.int64).tobytes()
            else:
                raise SystemExit("不支持的 dtype: %s (%s)" % (t.dtype, name))
            nb = name.encode("utf-8")
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", code))
            f.write(struct.pack("<i", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<q", int(d)))
            f.write(raw)
    print("写出 %s: %d 个张量, arch=(%d,%d,%d)" % (path, len(names), blocks, channels, value_hidden))


def dump_ref(ref_dir, sd, arch):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "PYTHON"))
    from network import GomokuNet  # noqa: E402

    blocks, channels, value_hidden = arch
    net = GomokuNet(n_blocks=blocks, channels=channels, value_hidden=value_hidden)
    net.load_state_dict(sd, strict=True)
    net = net.cpu()
    net.eval()

    rng = np.random.RandomState(20251025)
    N = 19
    states = np.zeros((4, 3, N, N), dtype=np.float32)
    for b in range(4):
        k = 0
        while k < 12:
            x, y = rng.randint(0, N), rng.randint(0, N)
            if states[b, 0, y, x] or states[b, 1, y, x]:
                continue
            states[b, 0 if k % 2 == 0 else 1, y, x] = 1.0
            k += 1
        states[b, 2, :, :] = 1.0 if (12 % 2 == 0) else 0.0
    os.makedirs(ref_dir, exist_ok=True)
    with torch.no_grad():
        x = torch.from_numpy(states)
        policy, value = net(x)
        logits = torch.log(policy + 1e-30)
    states.tofile(os.path.join(ref_dir, "states.bin"))
    logits.numpy().astype(np.float32).tofile(os.path.join(ref_dir, "logits.bin"))
    value.numpy().astype(np.float32).tofile(os.path.join(ref_dir, "values.bin"))
    print("写出参考前向: %s (states/logits/values)" % ref_dir)


def read_bin(path):
    """读 C++ 引擎的 GKAZ1 权重文件，返回 (state_dict, (blocks, channels, value_hidden))。"""
    sd = {}
    with open(path, "rb") as f:
        magic = f.read(5)
        if magic != b"GKAZ1":
            raise SystemExit("不是 GKAZ1 文件: %s" % path)
        blocks, channels, value_hidden = struct.unpack("<3i", f.read(12))
        count = struct.unpack("<i", f.read(4))[0]
        for _ in range(count):
            nlen = struct.unpack("<i", f.read(4))[0]
            name = f.read(nlen).decode("utf-8")
            code = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<i", f.read(4))[0]
            shape = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
            n = int(np.prod(shape)) if shape else 1
            if code == 0:
                arr = np.fromfile(f, dtype=np.float32, count=n).copy()
                sd[name] = torch.from_numpy(arr).view(shape)
            elif code == 1:
                arr = np.fromfile(f, dtype=np.int64, count=n).copy()
                sd[name] = torch.from_numpy(arr).view(shape)
            else:
                raise SystemExit("不支持的 dtype code=%d (%s)" % (code, name))
    return sd, (blocks, channels, value_hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Python 侧 .pt（转成 C++ 的 .bin）")
    ap.add_argument("--out", default=None, help="输出路径（.bin 或 .pt）")
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--bin", default=None, help="反向模式：C++ 的 .bin 转回 Python 可读的 .pt")
    ap.add_argument("--pt", default=None, help="反向模式的输出 .pt")
    ap.add_argument("--gen", type=int, default=0, help="反向模式写进 checkpoint 的代数标签")
    args = ap.parse_args()

    # 反向模式：C++ checkpoint -> Python checkpoint（让 ladder.py / analysis.py 能直接读）
    if args.bin:
        out_pt = args.pt or (args.out or "model.pt")
        sd, arch = read_bin(args.bin)
        blocks, channels, value_hidden = arch
        torch.save(
            {
                "state_dict": sd,
                "gen": args.gen,
                "lr": 0.0,
                "win_rate": 0.0,
                "arch": {"n_blocks": blocks, "channels": channels,
                         "value_hidden": value_hidden, "aux": "", "aux_ch": 0},
            },
            out_pt,
        )
        print("写出 %s: %d 个张量, arch=(%d,%d,%d)" % (out_pt, len(sd), blocks, channels, value_hidden))
        return

    if not args.ckpt or not args.out:
        raise SystemExit("正向模式需要 --ckpt 与 --out；或反向用 --bin 与 --pt")

    sd, arch_hint = load_state(args.ckpt)
    arch = infer_arch(sd)
    if arch_hint:
        hint = (arch_hint.get("n_blocks"), arch_hint.get("channels"), arch_hint.get("value_hidden"))
        print("checkpoint arch=%s 反推 arch=%s" % (hint, arch))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_bin(args.out, sd, arch)
    if args.ref_dir:
        dump_ref(args.ref_dir, sd, arch)


if __name__ == "__main__":
    main()
