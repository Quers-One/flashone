import math
import re

RUNS = [
    ('(1) 整合重测 400 局', 'checkpoints/history/v2_cons_vs_orig_fixed.log'),
    ('(2) 纯网络vs融合 200 局', 'checkpoints/history/purenet_vs_fused.log'),
    ('(3) 自战基线 200 局', 'checkpoints/history/selfmatch_baseline.log'),
]

print('执黑红利统计（黑胜 = 黑方获胜的局数 / 黑方总对局数）')
print('-' * 74)
for tag, path in RUNS:
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except FileNotFoundError:
        continue
    bw = bwin = ww = wwin = 0
    # ladder 逐局行： 第123局: 候选执黑 候选胜 42 手
    for line in txt.splitlines():
        m = re.search(r'第\s*(\d+)\s*局[:：]\s*候选执(黑|白)\s+(\S+?)\s*胜', line)
        if m:
            side, winner = m.group(2), m.group(3)
            if side == '黑':
                bw += 1
                bwin += 1 if winner == '候选' else 0
            else:
                ww += 1
                wwin += 1 if winner != '候选' else 0
            continue
        # compare 逐局行： 123: A  B  38 手
        m2 = re.match(r'\s*(\d+):\s+(\S+)\s+(\S+)\s+(\d+)\s*手?\s*$', line)
        if m2:
            aw, bw2 = m2.group(2), m2.group(3)
            if aw == 'A':      # A 执黑
                bw += 1
                bwin += 0 if aw in ('B',) else 1
            elif bw2 == 'A':   # A 执白 → B 执黑
                ww += 1
                wwin += 1 if bw2 == 'B' else 0
    tot, tb = bw + ww, bwin + wwin
    if not tot:
        print(f'{tag}: 没解析到逐局行')
        continue
    p = tb / tot
    elo = 400 * math.log10(p / (1 - p)) if 0 < p < 1 else float('nan')
    print(f'{tag}')
    print(f'    黑方胜 {tb}/{tot} = {p:.3f}   → 执黑约 +{elo:.0f} Elo')
    print(f'    （黑方视角拆分：候选执黑时 {bwin}/{bw}，锚点执黑时 {wwin}/{ww}）')
print('-' * 74)
