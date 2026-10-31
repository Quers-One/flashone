# promote_manual.py —— 人工晋升：把某个 gen 复制成 champion，并复刻 auto_eval.ps1 的记账。
#
# 为什么要有这个脚本：auto_eval.ps1 的晋升分支（第 190-212 行）只有在**跑完一整场
# 200 局 ladder** 后才会执行。当那场对局已经跑过、数字已经进了曲线账本时，重跑一遍
# 既慢又没新信息。这个脚本用**已经落盘的那份 JSON** 做同样的记账，字段格式与
# auto_eval.ps1 完全一致（name/elo_vs_root/ci_lo/ci_hi/hops/history.csv）。
#
# 用法：
#   python promote_manual.py --cand models/V2/gen_120.pt --run V2 --gen 120 \
#       --measure-json models/V2/eval/gen120_vs_champ.json --dry-run
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

CHAMP_PT = 'checkpoints/champions/current.pt'
CHAMP_JS = 'checkpoints/champions/current.json'
CHAMP_HIST = 'checkpoints/champions/history.csv'
ARCHIVE = 'checkpoints/champions/archive'


def sha8(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True, help='要晋升的模型文件')
    ap.add_argument('--run', required=True, help='run 名（如 V2）')
    ap.add_argument('--gen', type=int, required=True, help='代数')
    ap.add_argument('--measure-json', required=True, help='那场 候选 vs champion 的 ladder JSON')
    ap.add_argument('--chain-weights', default='6.0/0.3',
                    help='用来接链条的读数档位（曲线账本统一在 6.0/0.3）')
    ap.add_argument('--prod-note', default='', help='另一个档位的读数，只写进 note 不进链条')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.cand):
        sys.exit(f'候选模型不存在: {a.cand}')
    if not os.path.exists(a.measure_json):
        sys.exit(f'测量 JSON 不存在: {a.measure_json}')

    champ = json.load(open(CHAMP_JS, encoding='utf-8'))
    j = json.load(open(a.measure_json, encoding='utf-8'))

    # 链条口径检查：这一场必须真是"该候选 vs 当前 champion"
    if j.get('candidate', '').replace('\\', '/').split('Flashone/')[-1] != a.cand.replace('\\', '/'):
        print(f"[警告] 测量 JSON 里的 candidate 是 {j.get('candidate')}，与你给的 --cand 不一致 —— 请确认")
    if j.get('verdict') != 'stronger':
        print(f"[警告] 该场判定是 {j.get('verdict')}，不是 stronger —— 晋升应人工确认")

    elo = float(j['elo'])
    elo_root = float(champ.get('elo_vs_root', 0.0)) + elo
    wc = (float(champ.get('ci_hi', 0.0)) - float(champ.get('ci_lo', 0.0))) / 2.0
    wm = (float(j['elo_hi']) - float(j['elo_lo'])) / 2.0
    w = (wc * wc + wm * wm) ** 0.5
    ci_lo, ci_hi = elo_root - w, elo_root + w
    hops = int(champ.get('hops', 0)) + 1

    h8 = sha8(a.cand)
    name = f'{a.run}_g{a.gen}'
    meta = {
        'name': name,
        'path': CHAMP_PT,
        'sha1_8': h8,
        'elo_vs_root': round(elo_root, 1),
        'ci_lo': round(ci_lo, 1),
        'ci_hi': round(ci_hi, 1),
        'hops': hops,
        'source': a.cand.replace('\\', '/'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': (f'从 {champ.get("name")} 晋升；本场 {elo:.1f} 分（相对 champion:{champ.get("name")}，'
                 f'口径 {a.chain_weights}，{j.get("games")} 局 {j.get("wins")}/{j.get("losses")}/{j.get("draws")}）。'
                 + (f' {a.prod_note}' if a.prod_note else '')
                 + ' 【人工晋升】由 promote_manual.py 执行，用已落盘的测量 JSON 记账，未重跑对局。'),
    }

    print('== 将要写入 ==')
    print(f'  {CHAMP_PT}  <-  {a.cand}   (sha1_8 {sha8(CHAMP_PT)} -> {h8})')
    print(f'  {CHAMP_JS}:')
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'  {CHAMP_HIST} += 一行（old={champ.get("name")} -> new={name}）')

    if a.dry_run:
        print('== --dry-run：什么都没写 ==')
        return

    os.makedirs(ARCHIVE, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(CHAMP_JS, os.path.join(ARCHIVE, f'{stamp}_pre_{name}.json'))
    # 旧 champion 若就是 root（anchors 里那份还在），就不必再存一份模型
    old_pt = os.path.join('checkpoints/anchors', f'{champ.get("name")}.pt')
    if os.path.exists(old_pt) and sha8(old_pt) == sha8(CHAMP_PT):
        print(f'[备份] 旧 champion 与 {old_pt} 同一份（sha1_8 相同），模型不另存，只归档了 current.json')
    else:
        shutil.copy2(CHAMP_PT, os.path.join(ARCHIVE, f'{stamp}_pre_{name}_model.pt'))
        print('[备份] 旧 champion 模型已归档')

    shutil.copy2(a.cand, CHAMP_PT)
    with open(CHAMP_JS, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    if not os.path.exists(CHAMP_HIST):
        with open(CHAMP_HIST, 'w', encoding='utf-8', newline='') as fh:
            fh.write('at,old_champion,new_champion,source_gen,elo_vs_root,ci_lo,ci_hi,gain_vs_old,games,run\n')
    with open(CHAMP_HIST, 'a', encoding='utf-8', newline='') as fh:
        fh.write('{},{},{},{},{:.1f},{:.1f},{:.1f},{:.1f},{},{}\n'.format(
            meta['updated_at'], champ.get('name'), name, a.gen,
            elo_root, ci_lo, ci_hi, elo, j.get('games'), a.run))
    print(f'[完成] champion: {champ.get("name")} -> {name}（相对 root {elo_root:.1f} 分，本场 +{elo:.1f}）')


if __name__ == '__main__':
    main()
