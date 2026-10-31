"""
plot_progress.py —— 把"训练有没有进步"画成一张图（Pillow 手绘，无需 matplotlib）。

为什么要有它
------------
* **Elo 曲线分辨率有限**：200 局 ≈ ±50 Elo，单点常常落在噪声里 —— 得看**趋势**。
* **留出集指标灵敏得多**：同一批冻结局面上，policy CE / top1 一致率 / 价值方向
  准确率会随代数明显变好（实测 CE 2.29→1.31、方向 49.6%→65.5%）。它是"有没有在学"
  的主指标，Elo 是"有没有变成棋力"的结果指标。
* 两个一起画，才能区分"没在学"和"学了但还不值分"。

内容
----
    A 上图：ΔElo（相对**当时**的锚点）+ 95% 区间。**换锚点=换零点**，按锚点分段着色，
      跨段不能直接连（这是设计如此：晋升就换尺子）。
    B 中图：留出集上的 top1 一致率 / 价值方向准确率（%，越高越好）
    C 下图：留出集上的 policy CE / value MSE（越低越好，各自按最大值归一）

数据来源
--------
    checkpoints/history/elo_history.csv     由 auto_eval.ps1 自动追加
    checkpoints/history/learning_metrics.csv 本脚本计算并缓存（key = 冻结集 + gen）

用法
----
    python plot_progress.py                          # 用默认留出集与代数列表
    python plot_progress.py --gens 1 5 10 20 30 40 50 58 60
    python plot_progress.py --no-refresh             # 只用缓存，不重算
"""

import argparse
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

from train import load_checkpoint

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIST = os.path.join("checkpoints", "history")
CACHE = os.path.join(HIST, "learning_metrics.csv")
ELO = os.path.join(HIST, "elo_history.csv")
OUT = os.path.join(HIST, "progress.png")

FONTS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhl.ttc",
         "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"]


def get_font(size):
    from PIL import ImageFont
    for f in FONTS:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
    return ImageFont.load_default()


def metrics_for(gen, frozen, model_dir="checkpoints", batch=256):
    """在冻结留出集上算 (policy CE, top1%, value MSE, 方向准确率%)。"""
    d = np.load(frozen)
    s, p, v = d["states"], d["policies"].astype(np.float32), d["values"].astype(np.float32)
    net = load_checkpoint(os.path.join(model_dir, f"gen_{gen}.pt"),
                          verbose=False)
    net.eval()
    ce = top1 = 0
    se = 0.0
    ok = tot = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(s), batch):
            st = torch.from_numpy(np.ascontiguousarray(s[i:i + batch])).float().to(DEV)
            pt = torch.from_numpy(p[i:i + batch]).to(DEV)
            vt = torch.from_numpy(v[i:i + batch]).to(DEV)
            po, va = net.forward(st)
            ce += float(-(pt * torch.log(po + 1e-9)).sum(dim=1).sum())
            top1 += int((po.argmax(1) == pt.argmax(1)).sum())
            se += float(((va - vt) ** 2).sum())
            nz = vt != 0
            ok += int(((va[nz] * vt[nz]) > 0).sum())
            tot += int(nz.sum())
            n += st.shape[0]
    return (ce / n, 100.0 * top1 / n, se / n, 100.0 * ok / max(tot, 1))


def cache_rows(cache, frozen=None):
    """把缓存摊平成「每个 run 一组」，供 B/C 面板**按 run 分线**画。
    【2026-10-12 加】以前 B/C 只画 `--dir` 那一个目录的点（用户会以为"只有当前 run 在学"）。
    冻结局面对所有 run 是同一批题，所以横向比较才有意义。
    run 名从 `model_dir` 推：models/V2 → V2；checkpoints/pilot/big → V1（那是 V1 的 junction 正本）。
    """
    groups = {}
    for r in cache.values():
        if frozen and r.get("frozen") != frozen:
            continue
        md = r.get("model_dir") or "checkpoints"
        groups.setdefault(md, []).append(r)
    for md in groups:
        groups[md].sort(key=lambda r: r["gen"])
    return groups


def load_cache():
    """缓存以 `目录|冻结集|代数` 为键 —— 不同 run（不同骨干）不能混在一条曲线上。"""
    if not os.path.exists(CACHE):
        return {}
    out = {}
    with open(CACHE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = int(row["gen"])
            row["gen"] = g
            for k in ("policy_ce", "top1_pct", "value_mse", "direction_pct"):
                row[k] = float(row[k])
            out[f"{row.get('model_dir', 'checkpoints')}|{row['frozen']}|{g}"] = row
    return out


def save_cache(cache, frozen):
    os.makedirs(HIST, exist_ok=True)
    with open(CACHE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_dir", "frozen", "gen", "policy_ce", "top1_pct",
                    "value_mse", "direction_pct"])
        for k in sorted(cache, key=lambda k: (cache[k].get("model_dir", ""),
                                              cache[k]["gen"])):
            r = cache[k]
            w.writerow([r.get("model_dir", "checkpoints"), r["frozen"], r["gen"],
                        r["policy_ce"], r["top1_pct"], r["value_mse"],
                        r["direction_pct"]])


def read_elo():
    """
    读曲线账本。**优先用 `elo_vs_root`**（相对固定 root 的分数）—— 这样曲线是
    连续的、跨 champion 晋升也能连；只有在没有该列时才退回旧的 `elo`（相对当时锚点）。
    """
    if not os.path.exists(ELO):
        return []
    rows = []
    with open(ELO, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if "elo_vs_root" in r and r.get("elo_vs_root") not in (None, ""):
                    elo = float(r["elo_vs_root"])
                    lo, hi = float(r["ci_lo_root"]), float(r["ci_hi_root"])
                    fixed = True
                else:
                    elo, lo, hi = float(r["elo"]), float(r["elo_lo"]), float(r["elo_hi"])
                    fixed = False
                rows.append({"gen": int(r["gen"]), "elo": elo, "lo": lo, "hi": hi,
                             "anchor": ("root:" + (r.get("root") or "?")) if fixed
                                       else (r.get("anchor") or "?"),
                             "champion": r.get("champion", ""),
                             "measured_vs": r.get("measured_vs", ""),
                             "run": r.get("run", ""), "games": r.get("games", ""),
                             "at": r.get("at", ""),
                             # 【直测/链式】hops=0 ⇒ 这一行是**直接对 root 测的**（真值锚，与曲线零点同一坐标系）；
                             # hops>=1 ⇒ "champion 基准 + 本场增量"的链式推算，链上每跳各带 ±40 噪声、
                             # 会随链条变长随机游走（实测同一 gen30：直测 149.3 vs 链式 219.8；gen110：251.9 vs 358.9）。
                             # 空值（老行/坏行）按链式处理 —— 画成实心，宁可少标也不要把不确定的当锚。
                             "hops": (r.get("hops") or "").strip(),
                             "fixed_root": fixed})
            except (ValueError, KeyError):
                continue
    # 【保持 CSV 的原始顺序】账本是按行追加的 ⇒ 文件顺序就是时间顺序。
    # 不要按 gen 重排：多 run 的 gen 区间重叠（V1 1–75 / V2 76–175 / V3 1–30…），
    # 按 gen 排会把不同 run 的点交错在一起、连出假尖峰（2026-10-12 修）。
    return rows


def draw(learn, elo, path, frozen, learn_all=None):
    from PIL import Image, ImageDraw
    W, H = 1080, 960
    PAD_L, PAD_R, PAD_T, PAD_B = 90, 40, 108, 46
    GAP = 46
    panel_h = (H - PAD_T - PAD_B - 2 * GAP) // 3
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_small = get_font(20), get_font(14), get_font(12)

    gens = [r["gen"] for r in learn] + [r["gen"] for r in elo]
    if not gens:
        return False
    gmin, gmax = min(gens), max(gens)

    # ---- x 轴 =「每个 run 一段横向区段」----
    # 【2026-10-12 改】旧写法把所有点按 gen 排序连成**一条线** ⇒ 多 run 混排时出现假尖峰
    # （V3 的 gen 30 与 V2 的 gen 135 被连起来，图上凭空 +200，用户实测指出）。
    # 新布局：每个 run 占一段宽度（比例 = 它的测量点数），段内按**该 run 自己的代数**推进；
    # 段与段之间留缝 ⇒ 一眼能看出"这是不同的轮次"，也不会跨轮次连线。
    PALETTE = [(30, 90, 200), (200, 90, 30), (30, 150, 80), (150, 60, 180),
               (180, 140, 20), (40, 160, 180)]
    elo_group = {}
    for r in elo:
        elo_group.setdefault(r.get("run") or "?", []).append(r)
    # ⚠️ learn_all 是 cache_rows() 的返回：**dict[model_dir, list[row]]**（不是扁平的 list）。
    # 直接 for r in learn_all 会遍历到**键名（字符串）** ⇒ 报 'str' object has no attribute 'get'
    # （2026-10-12 实测踩到）。
    learn_group = {}
    if isinstance(learn_all, dict):
        learn_group = {k: list(v) for k, v in learn_all.items()}
    else:
        for r in (learn_all if learn_all is not None else learn):
            learn_group.setdefault(r.get("model_dir") or "?", []).append(r)
    # run 的先后：以 elo 账本的出现顺序为准，把只出现在 learn 里的 run 接在后面
    run_order = list(elo_group.keys())
    for rn in learn_group.keys():
        if rn not in run_order:
            run_order.append(rn)
    # 【2026-10-12 加】learn 缓存里用目录名（models/V3、checkpoints/pilot/big…），
    # 而 elo 账本里用 run 名（V3、V2…）—— 这是**同一个 run**，必须归一，
    # 否则同一个 run 会出两条线、颜色也乱。映射表在这里写死（就这几个历史目录）。
    DIR_TO_RUN = {
        "models/V3": "V3", "models/V2": "V2", "models/V1": "V1",
        "models/run3": "run3", "checkpoints/pilot/big": "V1",   # junction 正本 = V1
        "checkpoints/pilot/small": "pilot/small",
    }
    learn_group = {DIR_TO_RUN.get(k, k): v for k, v in learn_group.items()}
    for rn in learn_group.keys():
        if rn not in run_order:
            run_order.append(rn)
    # 给 learn 的每一行打上所属 run（x 定位要用）
    for _rn, _rows in learn_group.items():
        for _r in _rows:
            _r["_run"] = _rn
    RUNCOL = {rn: PALETTE[i % len(PALETTE)] for i, rn in enumerate(run_order)}
    # 每个 run 的显示名（models/V2 → V2；checkpoints/pilot/big → pilot/big）
    def short_run(rn):
        return rn.split("/")[-1] or rn
    # 段宽：按该 run 的点数分配（最少给 1 份），段间留 10px 缝
    counts = {rn: max(len(elo_group.get(rn, [])), len(learn_group.get(rn, [])), 1)
              for rn in run_order}
    total = sum(counts.values()) or 1
    avail = (W - PAD_L - PAD_R) - (len(run_order) - 1) * 10
    SEG = {}
    _x = PAD_L
    for rn in run_order:
        wdt = avail * counts[rn] / total
        SEG[rn] = (_x, _x + wdt)
        _x += wdt + 10
    gspan = {}
    for rn in run_order:
        gs = [r["gen"] for r in elo_group.get(rn, [])] + [r["gen"] for r in learn_group.get(rn, [])]
        gspan[rn] = (min(gs), max(gs)) if gs else (0, 1)

    def xr(rn, gen):
        x0, x1 = SEG.get(rn, (PAD_L, W - PAD_R))
        g0, g1 = gspan.get(rn, (0, 1))
        if g1 <= g0 or x1 <= x0:
            return (x0 + x1) / 2
        return x0 + (gen - g0) / (g1 - g0) * (x1 - x0)

    def xmap_elo(r):
        return xr(r.get("run") or "?", r["gen"])

    def xmap_learn(r):
        # learn 组的键已归一成 run 名（见 DIR_TO_RUN）
        return xr(r.get("_run") or (r.get("model_dir") or "?"), r["gen"])

    def panel(idx, title, ymin, ymax, ylabel):
        top = PAD_T + idx * (panel_h + GAP)
        bottom = top + panel_h
        d.rectangle([PAD_L, top, W - PAD_R, bottom], outline=(180, 180, 180))
        # y 刻度
        for k in range(5):
            y = bottom - k / 4 * panel_h
            v = ymin + k / 4 * (ymax - ymin)
            d.line([PAD_L, y, W - PAD_R, y], fill=(235, 235, 235))
            d.text((PAD_L - 8, y), f"{v:.2f}" if ymax - ymin <= 5 else f"{v:.0f}",
                   font=f_small, fill=(90, 90, 90), anchor="rm")
        d.text((PAD_L, top - 26), title, font=f_title, fill=(20, 20, 20))
        d.text((W - PAD_R, top - 24), ylabel, font=f_small, fill=(120, 120, 120),
               anchor="ra")
        # x 刻度：只在最下面那块写字。
        # 【2026-10-12 改】轴现在是"每个 run 一段"：每段画它的起/终点刻度 + 段名，
        # 段内再等距加两个刻度，便于读该 run 的代数位置。
        for rn in run_order:
            x0, x1 = SEG.get(rn, (PAD_L, W - PAD_R))
            g0, g1 = gspan.get(rn, (0, 1))
            col = RUNCOL.get(rn, (90, 90, 90))
            marks = [(x0, g0), (x1, g1)]
            if g1 - g0 >= 30:
                marks.insert(1, (xr(rn, g0 + (g1 - g0) / 2), int(g0 + (g1 - g0) / 2)))
            for x, gv in marks:
                d.line([x, bottom, x, bottom - 5], fill=(150, 150, 150))
                if idx == 2:
                    # 只标"起点"代数，段名放下面一行 —— 多 run 的代数会重复（V1 到 75、
                    # V2 从 76 起…），全标会挤成一团（2026-10-12 实测）。
                    d.text((x0, bottom + 8), str(g0), font=f_small, fill=col, anchor="ma")
                    break
            if idx == 2:
                # 段名（缩短：checkpoints/pilot/big → pilot/big）
                _nm = short_run(rn)
                if _nm.startswith("big"):
                    _nm = "big6x128"
                d.text(((x0 + x1) / 2, bottom + 23), _nm, font=f_small,
                       fill=col, anchor="ma")
        return top, bottom

    def ymap(v, top, bottom, ymin, ymax):
        if ymax <= ymin:
            ymax = ymin + 1
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    # ---- A: ΔElo（相对固定 root，连续曲线）----
    ymin = -200
    ymax = max(200, (max(r["hi"] for r in elo) + 20) if elo else 200)
    top, bottom = panel(0, "A. Elo 相对固定基准（root）—— 连续曲线",
                        ymin, ymax,
                        "越高越好 · 0 分 = root；◆ 标记 = 当时的 champion")
    d.line([PAD_L, ymap(0, top, bottom, ymin, ymax),
            W - PAD_R, ymap(0, top, bottom, ymin, ymax)],
           fill=(200, 60, 60), width=1)
    if not elo:
        d.text(((PAD_L + W - PAD_R) / 2, (top + bottom) / 2),
               "尚无 Elo 数据 —— 首轮测量会在 pilot 结束后由 auto_eval.ps1 写入",
               font=f_lab, fill=(150, 60, 60), anchor="mm")
    else:
        # 【2026-10-12 改】**按 run 分成多条曲线**（分组/配色/区段在函数上方已算好：
        # elo_group / run_order / RUNCOL / xmap_elo）。旧写法把所有点按 gen 排序连成一条线，
        # 多 run 混排时会出现假尖峰（V3 gen 30 与 V2 gen 135 被连起来 ⇒ 凭空 +200）。
        for rn in run_order:
            g = elo_group.get(rn)
            if not g:
                continue
            col = RUNCOL[rn]
            # 误差棒
            for r in g:
                x = xmap_elo(r)
                d.line([x, ymap(r["lo"], top, bottom, ymin, ymax),
                        x, ymap(r["hi"], top, bottom, ymin, ymax)],
                       fill=tuple(min(255, c + 110) for c in col), width=2)
            pts = [(xmap_elo(r), ymap(r["elo"], top, bottom, ymin, ymax)) for r in g]
            if len(pts) > 1:
                d.line(pts, fill=col, width=3)
            prev_ch = None
            for r, (x, y) in zip(g, pts):
                # 【直测 vs 链式 · 2026-10-13】同一代**可能有两个点**：一个直测 root、一个链式推算
                # （实测 gen110 差 107 Elo）。必须一眼分清哪个准 —— **空心圆 = 直测锚**
                # （与曲线零点同坐标系、无链条误差），**实心圆 = 链式推算**（champion 基准 + 本场增量，
                # 随链长漂移）。用户 2026-10-13 正是被这两个点问住的："+252 和 +359 哪个准"。
                direct = (r.get("hops") == "0")
                if direct:
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255),
                              outline=col, width=3)
                else:
                    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
                if r["champion"] and r["champion"] != prev_ch:
                    d.polygon([(x, y - 12), (x + 8, y), (x, y + 12), (x - 8, y)],
                              outline=(200, 120, 20))
                    prev_ch = r["champion"]
                d.text((x, y - 14), f"{r['elo']:+.0f}", font=f_small, fill=col,
                       anchor="mb")
        # 图例：每条 run 一行（颜色 + run 名 + 点数 + 相对 root 的最新读数）。
        # ⚠️ 排版：不能像旧版那样贴右上角 —— 曲线爬到 +220 时会和图例撞字（2026-10-12 实测）。
        # 放在 A 面板**左半边空白处**（曲线都在右半），并加白底框保证可读。
        _lines = ["◆ = champion 晋升（零点不变）",
                  f"基准 root = {elo[0]['anchor']}（各 run 独立成线）",
                  "○ 空心 = 直测 root（真值锚）　● 实心 = 链式推算"]
        _cols = [(200, 120, 20), (120, 120, 120), (90, 90, 90)]
        for rn in run_order:
            g = elo_group.get(rn)
            if not g:
                continue
            _lines.append(f"{short_run(rn)}: {len(g)} 点，最新 {g[-1]['elo']:+.0f}")
            _cols.append(RUNCOL.get(rn, (60, 60, 60)))
        _lw = max(d.textlength(x, font=f_small) for x in _lines) + 18
        _lh = 15 * len(_lines) + 8
        _lx, _ly = PAD_L + 6, top + 6
        d.rectangle([_lx, _ly, _lx + _lw, _ly + _lh], fill=(255, 255, 255),
                    outline=(200, 200, 200))
        for _i, _t in enumerate(_lines):
            # ⚠️ 颜色必须与 _lines **并行**取，不能再按 `_i-2` 反查 run_order ——
            # 图例里插进固定行（◆ / 基准 / 空心实心）之后下标就和 run_order 错位了。
            d.text((_lx + 8, _ly + 4 + 15 * _i), _t, font=f_small, fill=_cols[_i])

    # ---- B: 准确率（**按 run 分线**）----
    # 【2026-10-12 改】以前这里只画 --dir 那一个目录的点，别的 run 看不见。
    # 冻结局面对所有 run 是同一批题 ⇒ 分线画才能横向比较"各代/各轮次在同一批题上的表现"。
    top, bottom = panel(1, "B. 留出集：top1 一致率（实线）/ 价值方向准确率（虚线），按 run 分色",
                        0, 100, "越高越好")
    for rn in run_order:
        rows = learn_group.get(rn)
        if not rows:
            continue
        col = RUNCOL.get(rn, (60, 60, 60))
        for key, dash in (("top1_pct", False), ("direction_pct", True)):
            pts = [(xmap_learn(r), ymap(float(r[key]), top, bottom, 0, 100))
                   for r in rows]
            if dash and len(pts) > 1:
                # PIL 没有虚线，手动画：每两段画一段
                for i in range(0, len(pts) - 1, 2):
                    d.line([pts[i], pts[i + 1]], fill=col, width=2)
            else:
                d.line(pts, fill=col, width=3)
            for x, y in pts:
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    # 图例放**面板外右上角**（面板内放会挡住曲线 —— 实测 V2 的绿线正好压在图例上）。
    # 紧凑版：只给"run → 颜色"的对应，细节读数留在 A 面板的图例里。
    _lg = "　".join(short_run(rn) for rn in run_order if learn_group.get(rn))
    _lw = d.textlength("实线=top1/CE  虚线=方向/MSE  " + _lg, font=f_small) + 10
    # ⚠️ 位置：**面板内**左上角（top+2 起）。放 top-26 会和右上角的"越高越好"轴标签撞字；
    # 放 top+6 而图例较高时又会压住中段的曲线（都实测过）。现在压成一行 + 白底。
    d.rectangle([PAD_L + 3, top + 2, PAD_L + 3 + _lw, top + 18],
                fill=(255, 255, 255), outline=(210, 210, 210))
    _xx = PAD_L + 8
    d.text((_xx, top + 4), "实线=top1/CE　虚线=方向/MSE", font=f_small, fill=(60, 60, 60))
    _xx += d.textlength("实线=top1/CE　虚线=方向/MSE", font=f_small) + 10
    for rn in run_order:
        if not learn_group.get(rn):
            continue
        d.line([_xx, top + 10, _xx + 12, top + 10], fill=RUNCOL.get(rn, (60, 60, 60)), width=3)
        d.text((_xx + 15, top + 4), short_run(rn), font=f_small,
               fill=RUNCOL.get(rn, (60, 60, 60)))
        _xx += 15 + d.textlength(short_run(rn), font=f_small) + 8

    # ---- C: 损失（**按 run 分线**）----
    top, bottom = panel(2, "C. 留出集：policy CE（实线）/ value MSE（虚线）",
                        0, 1, "越低越好（按全局最大值归一，跨 run 可比）")
    _allv = [float(r[k]) for rows in learn_group.values() for r in rows
             for k in ("policy_ce", "value_mse")]
    _mx = max(_allv) if _allv else 1.0
    for rn in run_order:
        rows = learn_group.get(rn)
        if not rows:
            continue
        col = RUNCOL.get(rn, (60, 60, 60))
        for key, dash in (("policy_ce", False), ("value_mse", True)):
            pts = [(xmap_learn(r), ymap(float(r[key]) / _mx, top, bottom, 0, 1))
                   for r in rows]
            if dash and len(pts) > 1:
                for i in range(0, len(pts) - 1, 2):
                    d.line([pts[i], pts[i + 1]], fill=col, width=2)
            else:
                d.line(pts, fill=col, width=3)
            for x, y in pts:
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    _lg2 = "　".join(short_run(rn) for rn in run_order if learn_group.get(rn))
    _lbl2 = f"实线=CE　虚线=MSE（÷{_mx:.2f}）"
    _lw2 = d.textlength(_lbl2 + "  " + _lg2, font=f_small) + 10
    d.rectangle([PAD_L + 3, top + 2, PAD_L + 3 + _lw2, top + 18],
                fill=(255, 255, 255), outline=(210, 210, 210))
    _xx2 = PAD_L + 8
    d.text((_xx2, top + 4), _lbl2, font=f_small, fill=(60, 60, 60))
    _xx2 += d.textlength(_lbl2, font=f_small) + 10
    for rn in run_order:
        if not learn_group.get(rn):
            continue
        d.line([_xx2, top + 10, _xx2 + 12, top + 10], fill=RUNCOL.get(rn, (60, 60, 60)), width=3)
        d.text((_xx2 + 15, top + 4), short_run(rn), font=f_small,
               fill=RUNCOL.get(rn, (60, 60, 60)))
        _xx2 += 15 + d.textlength(short_run(rn), font=f_small) + 8

    d.text((PAD_L, 22), "训练进步图（留出集 = 模型从未训练过的对局）",
           font=get_font(22), fill=(10, 10, 10))
    runs = sorted({r["run"] for r in elo if r.get("run")})
    d.text((PAD_L, 54), f"留出集 {frozen}　世代 {gmin}–{gmax}　"
                        f"Elo 点 {len(elo)} 个"
                        + (f"（run: {'、'.join(runs)}）" if runs else ""),
           font=f_small, fill=(120, 120, 120))
    d.text((PAD_L, 74), "读法：B/C 是「有没有在学」（灵敏）；A 是「学成了多少棋力」",
           font=f_small, fill=(120, 120, 120))
    # 【2026-10-12 加】把"单点不可比"钉在图上（同口径 gen_170 +34.9 vs gen_171 −20.9
    # 差 55.7 Elo 却只有 1.16σ）。
    # ⚠️ 排版：顶部 74~100 那段已经被"读法/A 面板标题/右上图例"占满，实测写在那里必撞字；
    # A 面板**左下空白区**（曲线与文字都在右侧）才是安全位置 —— 加浅底框保证可读。
    _bx, _by = PAD_L + 6, top + (bottom - top) * 3 // 4
    _msg = ["⚠️ 尺子噪声：200 局 ⇒ CI ±34 Elo",
            "相邻两次测量差 <50 属噪声，单点之间不可比",
            "（实例：gen_170 +34.9 vs gen_171 −20.9）"]
    _wmax = max(d.textlength(m, font=f_small) for m in _msg)
    d.rectangle([_bx - 4, _by - 4, _bx + _wmax + 6, _by + 14 * len(_msg) + 2],
                fill=(255, 250, 235), outline=(220, 190, 140))
    for _i, _m in enumerate(_msg):
        d.text((_bx, _by + 14 * _i), _m, font=f_small, fill=(170, 90, 40))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="checkpoints/frozen/gen_59.npz",
                    help="冻结留出集（默认用 frozen/ 下的副本，避免被新训练覆盖）")
    ap.add_argument("--dir", dest="model_dir", default="models/run3",
                    help="checkpoint 所在目录（例如 checkpoints/pilot/big）")
    ap.add_argument("--gens", nargs="*", type=int, default=None,
                    help="要画的代数；默认自动扫描该目录里的 gen_N.pt")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-refresh", action="store_true",
                    help="只用缓存，不重算留出集指标")
    args = ap.parse_args()

    if not args.gens:
        import glob
        import re as _re
        args.gens = sorted(
            int(m.group(1)) for m in
            (_re.match(r"gen_(\d+)\.pt$", os.path.basename(p))
             for p in glob.glob(os.path.join(args.model_dir, "gen_*.pt")))
            if m)

    cache = load_cache()
    todo = [] if args.no_refresh else [
        g for g in args.gens
        if os.path.exists(os.path.join(args.model_dir, f"gen_{g}.pt"))
        and (f"{args.model_dir}|{args.frozen}|{g}" not in cache)]
    if todo:
        print(f"计算留出集指标（{args.model_dir}）：{todo}")
        for g in todo:
            m = metrics_for(g, args.frozen, args.model_dir)
            cache[f"{args.model_dir}|{args.frozen}|{g}"] = {
                "model_dir": args.model_dir, "frozen": args.frozen, "gen": g,
                "policy_ce": float(f"{m[0]:.4f}"), "top1_pct": float(f"{m[1]:.2f}"),
                "value_mse": float(f"{m[2]:.4f}"),
                "direction_pct": float(f"{m[3]:.2f}")}
            print(f"  gen {g:>3}: CE {m[0]:.4f}  top1 {m[1]:.1f}%  "
                  f"MSE {m[2]:.4f}  方向 {m[3]:.1f}%")
        save_cache(cache, args.frozen)

    learn = [cache[k] for k in cache
             if k.startswith(f"{args.model_dir}|{args.frozen}|")
             and cache[k]["gen"] in args.gens]
    learn.sort(key=lambda r: r["gen"])
    elo = read_elo()
    out = args.out if args.model_dir == "checkpoints" else os.path.join(
        args.model_dir, "progress.png")
    ok = draw(learn, elo, out, args.frozen, learn_all=cache_rows(cache, args.frozen))
    print(f"\n{'已生成' if ok else '无数据'}：{out}　"
          f"（留出集点 {len(learn)} 个，Elo 点 {len(elo)} 个）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
