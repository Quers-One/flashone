"""selfplay_totals.py —— 精确统计各 run 的自对弈局数（用 train_log.jsonl 的 n_games）。

为什么不用"代数 × 每代局数"：早期几代的 n_games 并不都相同，
而且有的 run 目录是**同一批数据的副本**（models/V1 与 checkpoints/pilot/big 样本数
完全一样、最后一条日志时间戳也完全一样），直接相加会重复计数。这里按指纹去重。
"""
import glob
import hashlib
import json
import os

found = []
for log in glob.glob('**/train_log.jsonl', recursive=True):
    if '\\_autotest\\' in log or '/_autotest/' in log:
        continue
    rows = []
    for line in open(log, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        continue
    gens = [r['gen'] for r in rows if isinstance(r.get('gen'), int)]
    games = sum(int(r.get('n_games') or 0) for r in rows)
    per = sorted({int(r.get('n_games') or 0) for r in rows})
    samples = sum(int(r.get('data_samples') or 0) for r in rows)
    steps = sum(int(r.get('train_steps') or 0) for r in rows)
    found.append(dict(log=log, n=len(rows), g0=min(gens), g1=max(gens), games=games,
                      per=per, samples=samples, steps=steps,
                      first=rows[0].get('timestamp'), last=rows[-1].get('timestamp')))

print('=' * 96)
print(f'{"train_log":44s} {"条":>4s} {"代数范围":>11s} {"自对弈局":>9s} {"每代局数":>10s} {"训练步":>8s}')
print('-' * 96)
for r in sorted(found, key=lambda x: x['g0']):
    per = ','.join(str(p) for p in r['per'])
    print(f'{r["log"]:44s} {r["n"]:>4d} {str(r["g0"])+".."+str(r["g1"]):>11s} '
          f'{r["games"]:>9,d} {per:>10s} {r["steps"]:>8,d}')
print('-' * 96)

# 去重：样本数 + 时间范围 + 局数都一样 = 同一批（迁移出来的副本）
seen, uniq = {}, []
for r in found:
    key = (r['samples'], r['games'], r['n'], r['first'], r['last'])
    if key in seen:
        print(f'[去重] {r["log"]} 与 {seen[key]} 是同一批数据（样本/局数/时间戳全同），不计入合计')
        continue
    seen[key] = r['log']
    uniq.append(r)

print('-' * 96)
print(f'{"总计（去重后）":44s} {"":>4s} {"":>11s} {sum(r["games"] for r in uniq):>9,d} '
      f'{"":>10s} {sum(r["steps"] for r in uniq):>8,d}')
print('=' * 96)

print('\n各 run 明细（按时间顺序）：')
for r in sorted(uniq, key=lambda x: (x['first'] or '')):
    span = f'{r["first"]}  ->  {r["last"]}'
    print(f'  {r["log"]:42s} gen {r["g0"]:>3d}..{r["g1"]:<3d}  {r["games"]:>7,d} 局   {span}')
