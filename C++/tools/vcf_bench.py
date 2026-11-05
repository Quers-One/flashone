"""VCF 战术基准：在"已被证明存在必胜"的局面里，看引擎能不能走出胜手。

真值来自 `gk vcfsuite` 的**穷举证明**（连续冲四必胜），不是启发式判断 ——
所以这个数字不依赖任何人的主观评分，也不依赖引擎自己的评估。

两种用法：

  1) 复跑已发布的题库（不需要训练日志，只要权重）：
       python vcf_bench.py --replay ../../docs/tactics/vcf_puzzles.tsv

  2) 重新生成题库（会跑若干局自对弈，慢）：
       python vcf_bench.py --gen 16 --gen-sims 400

环境变量：
  GK_EXE     引擎路径，默认 ../../build/gk.exe
  GK_NET     权重路径，默认 ../../model/run_v2/gen_092.bin
  TORCHROOT  torch 目录，需要时才设（Windows 下 gk.exe 依赖其 lib 子目录）
  GK_SIMS    作答用的模拟数，默认 600
"""
import argparse
import json
import os
import random
import subprocess
import sys

SIZE = 19
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("GK_EXE", os.path.join(ROOT, "build", "gk.exe"))
NET = os.environ.get("GK_NET", os.path.join(ROOT, "model", "run_v2", "gen_092.bin"))
TORCH_LIB = os.environ.get("TORCHROOT", "")
SIMS = int(os.environ.get("GK_SIMS", "600"))

# 证明预算：max_ply 越大能证明的杀法越长，max_nodes 是单次调用的节点上限
PROOF_PLY, PROOF_NODES = 8, 2000


def env():
    e = dict(os.environ)
    if TORCH_LIB:
        e["PATH"] = os.path.join(TORCH_LIB, "lib") + os.pathsep + e.get("PATH", "")
    e.pop("GK_HALF", None)
    return e


def fmt(ms):
    return " ".join("%d %d" % (m % SIZE, m // SIZE) for m in ms)


def label(positions, max_ply, max_nodes):
    """一次进程批量标注，返回 [(vcf, x, y), ...]。"""
    txt = "\n".join(fmt(m) for m in positions) + "\n"
    r = subprocess.run([EXE, "vcfsuite", "--vcf-max-ply", str(max_ply),
                        "--vcf-max-nodes", str(max_nodes)],
                       input=txt, capture_output=True, text=True, env=env())
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            out.append((d.get("vcf", 0), d.get("x", -1), d.get("y", -1)))
    return out


def check_one(moves, x, y):
    r = subprocess.run([EXE, "vcfsuite", "--check-x", str(x), "--check-y", str(y),
                        "--vcf-max-ply", str(PROOF_PLY), "--vcf-max-nodes", str(PROOF_NODES)],
                       input=fmt(moves) + "\n", capture_output=True, text=True, env=env())
    for line in r.stdout.splitlines():
        if line.strip().startswith("{"):
            return bool(json.loads(line).get("winning", 0))
    return False


def engine_moves(positions, sims):
    """起一个 play 进程，逐个局面问 top-1 着法。"""
    p = subprocess.Popen([EXE, "play", "--net", NET, "--sims", str(sims)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env())
    if not p.stdout.readline().startswith("READY"):
        raise SystemExit("引擎没起来，检查 GK_EXE / GK_NET / TORCHROOT")
    out = []
    for ms in positions:
        p.stdin.write("GO %d 0.5 1.0 %d %s\n" % (sims, len(ms), fmt(ms)))
        p.stdin.flush()
        line = p.stdout.readline().strip()
        if line.startswith("MOVE"):
            q = line.split()
            out.append((int(q[1]), int(q[2])))
        else:
            out.append((-1, -1))
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return out


def is_five(board, x, y, who):
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            cx, cy = x + dx * s, y + dy * s
            while 0 <= cx < SIZE and 0 <= cy < SIZE and board.get(cy * SIZE + cx) == who:
                n += 1
                cx += dx * s
                cy += dy * s
        if n >= 5:
            return True
    return False


def gen_games(n_games, sims):
    """随机开局 + 自对弈，返回每局的完整手顺。"""
    p = subprocess.Popen([EXE, "play", "--net", NET, "--sims", str(sims)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env())
    if not p.stdout.readline().startswith("READY"):
        raise SystemExit("引擎没起来")
    games = []
    for gi in range(n_games):
        rnd = random.Random(1000 + gi)
        cells = [y * SIZE + x for y in range(7, 12) for x in range(7, 12)]
        rnd.shuffle(cells)
        moves = cells[: rnd.choice([2, 3, 4])]
        board = {m: (1 if k % 2 == 0 else 2) for k, m in enumerate(moves)}
        while len(moves) < 200:
            p.stdin.write("GO %d 0.5 1.0 %d %s\n" % (sims, len(moves), fmt(moves)))
            p.stdin.flush()
            line = p.stdout.readline().strip()
            if not line.startswith("MOVE"):
                break
            q = line.split()
            x, y = int(q[1]), int(q[2])
            m = y * SIZE + x
            who = 1 if len(moves) % 2 == 0 else 2
            if m in board:
                break
            board[m] = who
            moves.append(m)
            if is_five(board, x, y, who):
                break
        games.append(moves)
        print("  第 %d 局：%d 手" % (gi + 1, len(moves)), flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return games


def report(rows):
    """rows: [{'tier','hit'}]"""
    import math

    def wilson(k, n, z=1.96):
        if n == 0:
            return 0.0, 0.0
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return max(0.0, c - h), min(1.0, c + h)

    for t, name in ((2, "短杀 ≤2 手"), (4, "中杀 ≤4 手"), (6, "长杀 ≤6 手"), (8, "极长 ≤8 手")):
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        k = sum(1 for r in sub if r["hit"])
        lo, hi = wilson(k, len(sub))
        print("  %-10s %3d 题  命中 %3d = %3.0f%%   Wilson [%.0f%%, %.0f%%]"
              % (name, len(sub), k, 100.0 * k / len(sub), lo * 100, hi * 100))
    k = sum(1 for r in rows if r["hit"])
    lo, hi = wilson(k, len(rows))
    print("  %-10s %3d 题  命中 %3d = %3.0f%%   Wilson [%.0f%%, %.0f%%]"
          % ("合计", len(rows), k, 100.0 * k / len(rows), lo * 100, hi * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="复跑一个题库 TSV")
    ap.add_argument("--gen", type=int, default=0, help="重新生成：跑多少局自对弈")
    ap.add_argument("--gen-sims", type=int, default=400, help="生成对局用的模拟数")
    a = ap.parse_args()

    if a.replay:
        rows = []
        with open(a.replay, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("ply\t") or not line.strip():
                    continue
                f_ply, f_tier, f_pos, f_first, _mv, _hit = line.rstrip("\n").split("\t")
                ms = [int(v) for v in f_pos.split()]
                ms = [b * SIZE + a2 for a2, b in zip(ms[0::2], ms[1::2])]
                rows.append({"tier": int(f_tier), "moves": ms})
        print("题库 %d 题，作答中（%d 模拟）…" % (len(rows), SIMS), flush=True)
        ans = engine_moves([r["moves"] for r in rows], SIMS)
        for k, r in enumerate(rows):
            x, y = ans[k]
            r["hit"] = check_one(r["moves"], x, y) if x >= 0 else False
            r["move"] = (x, y)
        print()
        report(rows)
        return

    if a.gen:
        print("生成对局（%d 局 × %d 模拟）…" % (a.gen, a.gen_sims), flush=True)
        games = gen_games(a.gen, a.gen_sims)
        cands = []
        for g in games:
            for k in range(6, len(g) - 1):
                cands.append(g[:k])
        print("候选局面 %d 个，标注中…" % len(cands), flush=True)
        lab = label(cands, PROOF_PLY, PROOF_NODES)
        pos = [cands[i] for i, d in enumerate(lab) if d[0]]
        print("有必胜：%d 个（%.1f%%）" % (len(pos), 100.0 * len(pos) / max(1, len(cands))), flush=True)
        if not pos:
            print("题库为空，加大 --gen")
            return
        t2 = label(pos, 2, PROOF_NODES)
        t4 = label(pos, 4, PROOF_NODES)
        t6 = label(pos, 6, PROOF_NODES)
        tiers = []
        for k in range(len(pos)):
            tiers.append(2 if t2[k][0] else 4 if t4[k][0] else 6 if t6[k][0] else 8)
        print("分档：≤2 %d · ≤4 %d · ≤6 %d · ≤8 %d"
              % (tiers.count(2), tiers.count(4), tiers.count(6), tiers.count(8)), flush=True)
        print("作答中 …", flush=True)
        ans = engine_moves(pos, SIMS)
        rows = []
        for k, ms in enumerate(pos):
            x, y = ans[k]
            rows.append({"tier": tiers[k], "hit": check_one(ms, x, y) if x >= 0 else False})
        print()
        report(rows)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
