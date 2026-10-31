"""
register_models.py —— 把一个 run 目录里的 checkpoint 补登记进 `checkpoints/registry.csv`。

**为什么需要**：登记表是后加的（2026-10-11），而 V1 这个 run 是在那之前启动的进程在跑，
它不会自己写登记行；另外手工建目录/复制模型也不会登记。这个脚本做**幂等补登**：
扫描 `--dir` 下所有 `gen_*.pt`，把还没登记过的行追加进去（按 version+name 去重）。

用法：
    python register_models.py --dir checkpoints/pilot/big --version V1
    python register_models.py --dir checkpoints --version run3
    python register_models.py --dir checkpoints/pilot/big --version V1 --dry-run
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from network import GomokuNet

REG = os.path.join("checkpoints", "registry.csv")
FIELDS = ["version", "name", "run_dir", "model_path", "arch", "params_wan",
          "local_gen", "engine_mcts", "created_at"]


def engine_fp():
    try:
        h = hashlib.sha1()
        with open("mcts.py", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run 目录")
    ap.add_argument("--version", required=True, help="版本标签，例如 V1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    if os.path.exists(REG):
        with open(REG, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r.get("version", ""), r.get("name", "")) for r in rows}

    files = []
    for p in glob.glob(os.path.join(args.dir, "gen_*.pt")):
        m = re.match(r"gen_(\d+)\.pt$", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()

    added = 0
    for g, p in files:
        name = f"gen_{g}"
        if (args.version, name) in seen:
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        nb, ch, vh = GomokuNet.infer_arch(sd)
        n_par = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        row = {
            "version": args.version, "name": name, "run_dir": args.dir,
            "model_path": p, "arch": f"{nb}x{ch}", "params_wan": f"{n_par / 1e4:.1f}",
            "local_gen": ck.get("gen", g) if isinstance(ck, dict) else g,
            "engine_mcts": engine_fp(),
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))),
        }
        print(f"  + {args.version} {name}  {row['arch']}  {row['params_wan']}万  "
              f"local_gen={row['local_gen']}  {row['created_at']}")
        rows.append(row)
        seen.add((args.version, name))
        added += 1

    if not added:
        print(f"  （{args.dir} 下没有需要补登的 checkpoint）")
        return 0
    if args.dry_run:
        print(f"  dry-run：将新增 {added} 行")
        return 0

    os.makedirs(os.path.dirname(REG), exist_ok=True)
    with open(REG, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  已写入 {REG}（新增 {added} 行，共 {len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
