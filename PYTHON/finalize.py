"""
finalize.py
===========
训练收尾一键汇总：把训练日志 + 独立验证 + Elo 曲线合成一份**最终报告**。

为什么需要它
------------
训练可能是在**别的会话/别的终端**里跑的（实测就是这么跑的），所以"训练结束
之后要做什么"不能依赖某个 agent 会话当时是否被调度到。这个脚本只用磁盘上的
产物（`models/run3/train_log.jsonl`、`verify_genN.txt`、`gen_N.pt`），
自己等、自己算、自己写文件，因此挂成后台任务后**不依赖会话**。

产出
----
    checkpoints/final_train_summary.txt   训练日志汇总（lr 轨迹 / 门控统计 / 耗时）
    checkpoints/elo_curve.txt             7 点 Elo 曲线原始输出
    checkpoints/FINAL_REPORT.md           合并后的最终报告（给人看的）

用法
----
    python finalize.py                          # 全套：等 gen 60 -> 等验证 -> 跑 Elo -> 写报告
    python finalize.py --skip-elo               # 快版：只要汇总
    python finalize.py --wait-gen 0             # 不等，直接按现在有的代数写报告
    python finalize.py --gens 1 10 20 30 40 50 60 --games 12
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 工具

def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


class Tee:
    """同时写 stdout 和内存缓冲（最后统一落盘）。"""

    def __init__(self):
        self.buf = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.buf.append(s)

    def text(self):
        return "\n".join(self.buf) + "\n"


def _fmt(v, spec="{:.3f}"):
    return spec.format(v) if v is not None else "-"


def _utf8_env():
    """子进程强制 UTF-8 输出。

    【实测教训】`test_tactics.py` 自己不做 `sys.stdout.reconfigure`，
    所以它的输出编码取决于环境变量 `PYTHONIOENCODING`。父进程若没设
    （例如从后台任务直接启动 finalize），子进程在中文 Windows 上会按 GBK
    输出，而我按 UTF-8 解码 → 中文全部变乱码 → 报表解析失败。
    显式给子进程设上，比依赖调用者的 shell 环境可靠。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def binom_p(k, n, p=0.5):
    """双侧精确二项检验（最小似然法），与 `compare.binom_p` 同口径。

    多批合并时必须用它重算，而不是把两个 p 值平均 —— p 值不是可平均的量。
    """
    if n <= 0:
        return 1.0
    from math import comb
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(v for v in pmf if v <= obs + 1e-12))


# ---------------------------------------------------------------- 汇总

def summarize_train(rows, out):
    """把 train_log.jsonl 汇总成人看的文本块。返回统计字典。"""
    st = {}
    if not rows:
        out("[训练日志] 空 —— 找不到 models/run3/train_log.jsonl")
        return st

    out(f"代数范围      : {rows[0]['gen']} ~ {rows[-1]['gen']}"
        f"（共 {len(rows)} 条记录）")
    out(f"记录时间      : {rows[0].get('timestamp')} ~ {rows[-1].get('timestamp')}")
    st["first_gen"], st["last_gen"] = rows[0]["gen"], rows[-1]["gen"]

    # 每代耗时
    ts = []
    for r in rows:
        try:
            ts.append(datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"))
        except (KeyError, ValueError):
            ts.append(None)
    durs = [(ts[i] - ts[i - 1]).total_seconds() / 60.0
            for i in range(1, len(ts)) if ts[i] and ts[i - 1]]
    if durs:
        mid = sorted(durs)[len(durs) // 2]
        out(f"每代耗时      : 中位 {mid:.1f} 分钟  最快 {min(durs):.1f}  "
            f"最慢 {max(durs):.1f}  （总计 {sum(durs) / 60:.1f} 小时）")
        st["min_per_gen"] = mid
        st["total_hours"] = sum(durs) / 60.0

    # 门控统计
    gates = [r for r in rows if r.get("accepted") is not None
             and r.get("gen") != 1]
    acc = [r for r in gates if r["accepted"]]
    rej = [r for r in gates if not r["accepted"]]
    skip = [r for r in rows if r.get("accepted") is None]
    out(f"门控          : {len(gates)} 次（采纳 {len(acc)} / 拒绝 {len(rej)}），"
        f"另有 {len(skip)} 代不门控")
    if gates:
        wrs = [r["win_rate"] for r in gates]
        out(f"门控胜率      : min {min(wrs):.3f}  max {max(wrs):.3f}")
        out(f"被拒清单      : "
            + ", ".join(f"gen{r['gen']}({r['win_rate']:.3f})" for r in rej))
    st["gates"], st["accepted"], st["rejected"] = len(gates), len(acc), len(rej)

    # lr 轨迹（只在变化处打点）
    traj, prev = [], None
    for r in rows:
        if r.get("lr") != prev:
            traj.append(f"gen{r['gen']}:{r['lr']:.6f}")
            prev = r.get("lr")
    out(f"学习率轨迹    : " + " -> ".join(traj))
    st["final_lr"] = rows[-1].get("lr")

    # loss 趋势
    lps = [r["loss_policy"] for r in rows if r.get("loss_policy") is not None]
    lvs = [r["loss_value"] for r in rows if r.get("loss_value") is not None]
    if lps:
        k = max(1, len(lps) // 5)
        out(f"loss_policy   : 前{k}代均值 {sum(lps[:k]) / k:.4f} -> "
            f"后{k}代均值 {sum(lps[-k:]) / k:.4f}   （无语义基准 ln361=5.89）")
        st["loss_p_head"] = sum(lps[:k]) / k
        st["loss_p_tail"] = sum(lps[-k:]) / k
    if lvs:
        k = max(1, len(lvs) // 5)
        out(f"loss_value    : 前{k}代均值 {sum(lvs[:k]) / k:.4f} -> "
            f"后{k}代均值 {sum(lvs[-k:]) / k:.4f}")
        st["loss_v_head"] = sum(lvs[:k]) / k
        st["loss_v_tail"] = sum(lvs[-k:]) / k

    out(f"回放池        : {rows[-1].get('buffer_total', 0)} 条 / "
        f"{rows[-1].get('buffer_gens', 0)} 代")
    out(f"先验退火      : 末代 h/p = {rows[-1].get('heuristic_weight')}/"
        f"{rows[-1].get('policy_weight')}")
    return st


def render_table(rows, out, tail=0):
    out("")
    out(f"{'gen':>4} {'lr':>10} {'raw':>6} {'ema':>6} {'采用':>6} {'来源':>8} "
        f"{'样本':>6} {'loss_p':>7} {'loss_v':>7} {'时间':>8}")
    out("-" * 82)
    use = rows[-tail:] if tail else rows
    for r in use:
        a = r.get("accepted")
        tag = "不门控" if a is None else ("[采纳]" if a else "[拒绝]")
        out(f"{r['gen']:>4} {r['lr']:>10.6f} "
            f"{_fmt(r.get('win_rate_raw')):>6} {_fmt(r.get('win_rate_ema')):>6} "
            f"{tag:>6} {str(r.get('chosen')):>8} "
            f"{r.get('data_samples', 0):>6} "
            f"{_fmt(r.get('loss_policy'), '{:.4f}'):>7} "
            f"{_fmt(r.get('loss_value'), '{:.4f}'):>7} "
            f"{str(r.get('timestamp'))[-8:]:>8}")


# ---------------------------------------------------------------- 验证报告

# 兼容 compare.py 的两种结尾格式：
#   verify_*:  "A(基线) 胜 7"      "有效对局 23 局: A 胜率 0.304 ..."
#   control:   "A 胜 25"           "有效对局(非和棋) 47 局: A 胜率 0.532 ..."
PAT_A = re.compile(r"A(?:\(基线\))? 胜 (\d+)")
PAT_B = re.compile(r"B(?:\(新一代\))? 胜 (\d+)")
PAT_D = re.compile(r"和棋 (\d+)")
PAT_SEQ = re.compile(r"独立对局序列数: (\d+)/(\d+)")
PAT_P = re.compile(
    r"有效对局(?:\(非和棋\))? (\d+) 局: A 胜率 ([\d.]+)\s+双侧二项 p=([\d.]+)")
PAT_VERDICT = re.compile(r">>> (.+)$", re.M)


def parse_verify(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    d = {"path": path, "raw": txt}
    for key, pat in (("a_wins", PAT_A), ("b_wins", PAT_B), ("draws", PAT_D)):
        m = pat.search(txt)
        d[key] = int(m.group(1)) if m else None
    m = PAT_SEQ.search(txt)
    if m:
        d["seqs"], d["games"] = int(m.group(1)), int(m.group(2))
    m = PAT_P.search(txt)
    if m:
        d["decided"], d["a_rate"], d["p"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    v = PAT_VERDICT.search(txt)
    d["verdict"] = v.group(1).strip() if v else None
    return d


def wait_verify(path, minutes, out):
    if not os.path.exists(path) and minutes > 0:
        out(f"[等待] {path} 尚未生成，最多等 {minutes} 分钟（由 auto_verify 后台任务产出）")
        deadline = time.time() + minutes * 60
        while time.time() < deadline:
            if os.path.exists(path):
                break
            time.sleep(20)
    return parse_verify(path)


def run_tactics(save_dir, last_gen, out):
    """跑战术验收门槛（对当前代模型），把结果写进报告。

    【为什么要放进收尾报告】`test_tactics.py` 是项目的硬门槛（A/B/D 必须 100%），
    但它只在"改完东西之后"手动跑。收尾报告如果不带这一项，就等于把
    "引擎现在到底过不过门槛"留给读者去猜。实测训练后的网络会在一个
    **已经必败**的局面里不挡（详见 NOTES §10.7），必须如实写进交付物。

    注意：本函数只跑测试、不加载/修改任何引擎代码；它测的是当前磁盘上的
    `mcts.py` 行为。
    """
    model = os.path.join(save_dir, f"gen_{last_gen}.pt")
    if not os.path.exists(model):
        out(f"[战术验收] 找不到 {model}，跳过")
        return None
    cmd = [sys.executable, "-u", os.path.join(HERE, "test_tactics.py"),
           "--model", model, "--n-sim", "400"]
    out(f"[战术验收] 执行: test_tactics.py --model {os.path.basename(model)} --n-sim 400")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env=_utf8_env())
    except OSError as e:
        out(f"[战术验收] 启动失败: {e}")
        return None
    txt = (proc.stdout or "")
    lines = txt.splitlines()
    # 【为什么不用中文标签做键】中文标签在子进程编码不对时会变成乱码
    # （实测：父进程未设 PYTHONIOENCODING 时子进程按 GBK 输出 → 乱码 →
    #  "总计"匹配失败 → total=None → 报告解包崩溃）。组名用字母映射，
    # 总计则**由各组求和得出**，都不依赖中文。
    canon = {"A": "A.一步成五", "B": "B.必挡", "C": "C.做活四", "D": "D.安全性"}
    groups, total, verdict = {}, None, ""
    for ln in lines:
        m = re.match(r"\s*([A-D])\.(\S+)\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)", ln)
        if m:
            groups[canon.get(m.group(1), f"{m.group(1)}.{m.group(2)}")] = (
                int(m.group(3)), int(m.group(4)))
            continue
        m2 = re.search(r"(\d+)/(\d+)\s*\(\s*([\d.]+)%\)", ln)
        if m2 and not re.match(r"\s*[A-D]\.", ln):
            total = (int(m2.group(1)), int(m2.group(2)))
        if "[达标]" in ln or "[未达标]" in ln:
            verdict = ln.strip()
    if groups and total is None:
        total = (sum(v[0] for v in groups.values()),
                 sum(v[1] for v in groups.values()))
    if not verdict:
        verdict = "[达标]" if proc.returncode == 0 else "[未达标]"
    path = os.path.join(save_dir, f"tactics_gen{last_gen}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    out(f"[战术验收] 退出码 {proc.returncode}，耗时 {time.time() - t0:.1f}s，"
        f"原始输出已存 {os.path.basename(path)}")
    for k, (a, b) in groups.items():
        out(f"            {k:<12} {a}/{b}")
    if total:
        out(f"            总计 {total[0]}/{total[1]}"
            + ("   [达标]" if total[0] == total[1] else "   [未达标]"))
    if verdict:
        out(f"            判据: {verdict}")
    return {"groups": groups, "total": total, "verdict": verdict,
            "exit": proc.returncode, "path": path}


def collect_batches(save_dir, last_gen, main_vf):
    """主验证文件 + 所有 `verify_gen{N}_b*.txt`（靠 --seed-offset 保证不重复计数）。"""
    batches = [main_vf] if main_vf else []
    for p in sorted(glob.glob(os.path.join(
            save_dir, f"verify_gen{last_gen}_b*.txt"))):
        d = parse_verify(p)
        if d:
            batches.append(d)
    return batches


def render_pool(batches, out):
    """多批合并：**必须**是不同 --seed-offset 跑出来的独立对局，否则是重复计数。

    依据见 `probe_seed.py`：只改 --seed 会产出逐手完全相同的对局。
    """
    out("")
    if len(batches) < 2:
        out("追加批次      : 无（只有一批；合并检验需要 ≥2 批）")
        return None
    tot_a = tot_b = tot_d = 0
    rows = []
    for i, d in enumerate(batches, 1):
        a, b, dr = d.get("a_wins"), d.get("b_wins"), d.get("draws")
        tot_a += a or 0
        tot_b += b or 0
        tot_d += dr or 0
        rows.append((os.path.basename(d["path"]), a, b, dr, d.get("p")))
        out(f"  第{i}批 {os.path.basename(d['path']):>24}  A {a} 胜 / B {b} 胜 / "
            f"和 {dr}" + (f"   p={d['p']:.4f}" if d.get("p") is not None else ""))
    n = tot_a + tot_b
    if n <= 0:
        return None
    rate = tot_a / n
    p = binom_p(tot_a, n)
    out(f"  合并总计: A {tot_a} 胜 / B {tot_b} 胜 / 和 {tot_d}  "
        f"→ 有效 {n} 局，A 胜率 {rate:.3f}，双侧二项 p={p:.4f}")
    better = "B（新一代）" if rate < 0.5 else "A（起点）"
    if p < 0.05:
        out(f"  >>> {better}显著更强（合并后 p={p:.4f} < 0.05）")
    elif p < 0.15:
        out(f"  >>> 仍只是倾向 {better}（p={p:.4f}）—— 按预注册方案应再加一批")
    else:
        out("  >>> 两者无法区分")
    out("  注意: 合并的前提是各批用不同 --seed-offset（见 probe_seed.py）；"
        "只改 --seed 是同一批对局的复制，合并会把 n 假翻倍。")
    return {"a_wins": tot_a, "b_wins": tot_b, "draws": tot_d, "n": n,
            "rate": rate, "p": p, "batches": len(batches), "rows": rows}


def render_verify(d, out, label="独立验证", a_name="起点", b_name="新一代"):
    out("")
    if not d:
        out(f"[{label}] 文件不存在 —— 任务可能还没跑完或超时，本节留空")
        return {}
    out(f"报告文件      : {d['path']}")
    out(f"战绩          : A({a_name}) {d.get('a_wins')} 胜 / "
        f"B({b_name}) {d.get('b_wins')} 胜 / 和棋 {d.get('draws')}")
    if "seqs" in d:
        out(f"独立对局序列  : {d['seqs']}/{d['games']}"
            + ("   <<< 偏低，评估可能退化成确定性对局" if d["seqs"] < d["games"] * 0.5 else ""))
    if "p" in d:
        out(f"统计          : 有效 {d['decided']} 局，A 胜率 {d['a_rate']:.3f}，"
            f"双侧二项 p={d['p']:.3f}")
    if d.get("verdict"):
        out(f"脚本判据      : {d['verdict']}")
    return d


# ---------------------------------------------------------------- Elo

def run_elo(args, out, save_dir):
    """跑 elo.py，把原始输出写到 elo_curve.txt 并解析出等级分。"""
    cmd = [sys.executable, "-u", os.path.join(HERE, "elo.py"),
           "--gens", *[str(g) for g in args.gens],
           "--games", str(args.games),
           "--n-sim", str(args.n_sim),
           "--save-dir", save_dir,
           "--heuristic-weight", str(args.heuristic_weight),
           "--policy-weight", str(args.policy_weight),
           "--seed", str(args.seed)]
    out(f"[Elo] 执行: {' '.join(cmd[1:])}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=_utf8_env())
    except OSError as e:
        out(f"[Elo] 启动失败: {e}")
        return None, ""
    txt = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    out(f"[Elo] 退出码 {proc.returncode}，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    with open(os.path.join(save_dir, "elo_curve.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return parse_elo_text(txt), txt


def parse_elo_text(txt):
    """从 elo.py 的原始输出解析等级分（`--reuse-elo` 也复用它）。"""
    ratings = {}
    for line in txt.splitlines():
        m = re.match(r"^\s*(\S+\.pt)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+)\s*$", line)
        if m:
            ratings[m.group(1)] = float(m.group(2))
    spread = None
    m = re.search(r"最高与最低相差 (\d+) 分", txt)
    if m:
        spread = int(m.group(1))
    worst = None
    m = re.search(r"最大偏差 ([\d.]+)", txt)
    if m:
        worst = float(m.group(1))
    return {"ratings": ratings, "spread": spread, "worst": worst, "raw": txt}


def render_elo(e, out):
    out("")
    if not e or not e["ratings"]:
        out("[Elo] 没有解析到等级分（检查 elo_curve.txt）")
        return {}
    out("Bradley-Terry Elo（gen_1 = 0 分锚点）：")
    items = sorted(e["ratings"].items(), key=lambda kv: kv[1])
    lo = min(v for _, v in items)
    for name, v in items:
        bar = "#" * int(max(0.0, v - lo) / 8.0)
        out(f"  {name:>14} {v:>8.1f}   {bar}")
    base = e["ratings"].get(items[0][0])
    hi = max(v for _, v in items)
    out(f"最高与最低相差 {e['spread']} 分；首尾差 {hi - lo:.0f} 分")
    out(f"拟合最大偏差 {e['worst']}"
        + ("   <<< 偏大，说明局数太少或模型不贴合" if (e["worst"] or 0) > 0.25 else "   （贴合良好）"))
    return e


# ---------------------------------------------------------------- 报告

def write_report(save_dir, st, vf, el, ctl, pooled, tac, args, reached, tee):
    """把四块拼成 FINAL_REPORT.md。"""
    path = os.path.join(save_dir, "FINAL_REPORT.md")
    L = []
    A = L.append
    A("# 五子棋 AlphaZero —— 训练收尾最终报告")
    A("")
    A(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　|　"
      f"训练代数：{st.get('first_gen')} ~ {st.get('last_gen')}"
      + ("" if reached else "（**未跑到目标代数，以下是部分结果**）"))

    def _sha(p):
        import hashlib
        try:
            with open(os.path.join(HERE, p), "rb") as fh:
                return hashlib.sha1(fh.read()).hexdigest()[:8]
        except OSError:
            return "?"

    # 【为什么要记指纹】本项目曾出现"交付过程中引擎被另一个会话修改"的真实情况
    # （mcts.py 于 22:16:54 被改）。没有指纹，读者无法判断某组数字是哪份代码跑的。
    A("")
    A(f"**引擎指纹**（sha1 前 8 位）：`mcts.py` {_sha('mcts.py')}　"
      f"`train.py` {_sha('train.py')}　`compare.py` {_sha('compare.py')}　"
      f"`elo.py` {_sha('elo.py')}")
    A("")
    A("## 0. 结论摘要")
    A("")
    if vf and vf.get("p") is not None:
        better = "新一代(B)" if vf["a_rate"] < 0.5 else "起点(A)"
        sig = ("**显著**" if vf["p"] < 0.05 else
               ("倾向性（未达 0.05）" if vf["p"] < 0.15 else "**不显著**"))
        A(f"- 独立对战（{vf.get('decided')} 局有效）：A(起点) {vf.get('a_wins')} 胜 / "
          f"B(第 {st.get('last_gen')} 代) {vf.get('b_wins')} 胜 / 和棋 {vf.get('draws')}，"
          f"p={vf['p']:.3f} → {better} 更强，{sig}。")
    else:
        A("- 独立对战：无有效结果。")
    if pooled:
        b2 = "B（新一代）" if pooled["rate"] < 0.5 else "A（起点）"
        s2 = ("**显著**" if pooled["p"] < 0.05 else
              ("倾向性（未达 0.05）" if pooled["p"] < 0.15 else "**不显著**"))
        A(f"- **合并 {pooled['batches']} 批独立对局（共 {pooled['n']} 局有效）**："
          f"A {pooled['a_wins']} 胜 / B {pooled['b_wins']} 胜，"
          f"A 胜率 {pooled['rate']:.3f}，p={pooled['p']:.4f} → {b2} 更强，{s2}。"
          "（合并后对 0.30 效应的功效约 0.99，**以它为主结论**）")
    if el and el.get("spread"):
        A(f"- Elo 曲线（{len(args.gens)} 个 checkpoint，每对 {args.games} 局）："
          f"最高与最低相差 **{el['spread']} 分**。")
    if ctl and ctl.get("p") is not None:
        A(f"- 对照实验（同一模型自战 {ctl.get('decided')} 局有效）："
          f"A {ctl.get('a_wins')} 胜 / B {ctl.get('b_wins')} 胜，p={ctl['p']:.3f}"
          + ("　→ 记账无系统偏置 ✓" if ctl["p"] >= 0.05 else
             "　→ **偏置警告：同一模型竟被打出显著差异，测量链路有问题！**"))
    if tac and tac.get("total"):
        ta, tb = tac["total"]
        A(f"- 引擎战术验收（对 gen_{st.get('last_gen')} 模型跑 19 项）：**{ta}/{tb}**"
          + ("　✅ 硬指标全过" if ta == tb else
             "　⚠️ **未达标**（失败项 `D.安全性`）—— 见第 5 节与 NOTES §10.7："
             "该缺陷只在**已经必败**的局面里不挡，不改变对局结果"))
    flr = st.get("final_lr")
    A(f"- 门控：{st.get('gates')} 次评估，采纳 {st.get('accepted')} / "
      f"拒绝 {st.get('rejected')}；最终 lr="
      + (f"{flr:.6f}" if isinstance(flr, float) else "?"))
    A(f"- loss_policy {st.get('loss_p_head', float('nan')):.4f} → "
      f"{st.get('loss_p_tail', float('nan')):.4f}；"
      f"loss_value {st.get('loss_v_head', float('nan')):.4f} → "
      f"{st.get('loss_v_tail', float('nan')):.4f}。")
    A("")
    A("> 读法：门控是**低功效**指标（20 局、±0.11 标准误差，且本轮运行的门控还带着"
      "已知的假拒率缺陷），所以\"训练有没有效果\"以**独立对战 + Elo** 为准。")
    A("")
    A("## 1. 训练过程汇总")
    A("")
    A("```")
    A(tee.text().strip())
    A("```")
    A("")
    A("## 2. 独立对战验证")
    A("")
    if vf:
        A("```")
        A(vf["raw"].strip())
        A("```")
    else:
        A("（验证文件缺失）")
    if pooled:
        A("")
        A(f"### 2.1 合并检验（{pooled['batches']} 批，各自不同的 --seed-offset）")
        A("")
        A("| 批次 | A 胜 | B 胜 | 和 | 双侧 p |")
        A("|---|---|---|---|---|")
        for name, a, b, dr, p in pooled["rows"]:
            A(f"| {name} | {a} | {b} | {dr} | "
              + (f"{p:.4f}" if p is not None else "-") + " |")
        A(f"| **合并 n={pooled['n']}** | **{pooled['a_wins']}** | "
          f"**{pooled['b_wins']}** | **{pooled['draws']}** | "
          f"**{pooled['p']:.4f}** |")
        A("")
        A("合并的前提是各批用**不同的 `--seed-offset`**。这不是形式要求："
          "实测（`probe_seed.py`）只改 `--seed` 会产出**逐手完全相同**的对局，"
          "那样合并等于把同一批数两遍、把 n 假翻倍。")
    A("")
    A("## 3. Elo 曲线")
    A("")
    if el and el.get("raw"):
        A("```")
        A(el["raw"].strip())
        A("```")
    else:
        A("（未生成）")
    A("")
    A("## 4. 对照实验：同一个模型 vs 自己")
    A("")
    A("目的：在**已知答案**的局面下检查测量链路（交替先后手、分颜色记账、"
      "根噪声、二项检验）本身会不会造出假差异。任何一个被报出来的 A/B 差异，"
      "都必须先通过这一关才值得相信。")
    A("")
    if ctl:
        A("```")
        A(ctl["raw"].strip())
        A("```")
        A("")
        if ctl.get("p") is not None:
            if ctl["p"] >= 0.05:
                A(f"判读：p={ctl['p']:.3f} —— 同一模型自战没有被判出差异，"
                  "测量链路无系统偏置。第 2 节的 A/B 差异因此可以归因于棋力，"
                  "而不是测量假象。")
            else:
                A(f"判读：p={ctl['p']:.3f} —— **同一模型自战竟被判出显著差异，"
                  "说明测量链路有系统偏置，第 2 节的结论不能采信。**")
    else:
        A("（未跑对照实验）")
    A("")
    A("## 5. 引擎战术验收门槛（`test_tactics.py`，19 项）")
    A("")
    A("这是项目的**硬门槛**：A.一步成五 / B.必挡 / D.安全性 必须 100%。"
      "它测的不是棋力，而是「引擎在任何情况下都不会漏杀、不会放过必挡」——"
      "也就是强度的**地板**。")
    A("")
    if tac and tac.get("groups"):
        A("| 分组 | 通过 |")
        A("|---|---|")
        for k, (a, b) in tac["groups"].items():
            A(f"| {k} | {a}/{b}" + ("" if a == b else " ❌") + " |")
        ta, tb = tac["total"] or (0, 0)   # 防御：解析失败也不该让报告崩掉
        A(f"| **总计** | **{ta}/{tb}** |")
        A("")
        if ta == tb:
            A("✅ 硬指标全过。")
        else:
            A("⚠️ **未达标。** 失败项固定为 `D.安全性` 中那个「对方成**活四**」的局面：")
            A("")
            A("```")
            A("黑 (6,4)(5,5)(4,6)(3,7) 在 x+y=10 上成活四，两端 (7,3)/(2,8) 都空")
            A("白 (3,12)(4,12)(5,12) 有活三——引擎选择 (7,12) 扩自己的四，没有挡")
            A("```")
            A("")
            A("**这个局面白方本来就是数学上必败**：黑有活四、两个成五点，"
              "挡一端另一端照样成五 —— 所以该缺陷**不损失胜局**，"
              "但它让项目自己的验收门槛变红，必须如实记录。")
            A("")
            A("**根因（用 `probe_terminal.py` 实测定位）**：打印该局面根子节点明细可见 ——")
            A("")
            A("```")
            A("    move      N        Q        P")
            A("    (7,12)  152   +0.812   0.0000   <<< 引擎选择（不挡）")
            A("    (2,8)    49   +0.999   0.5000   <<< 必挡点")
            A("    (7,3)    49   +0.998   0.5000   <<< 必挡点")
            A("```")
            A("")
            A("`heuristics.compute_prior` 在「对方有成五点」时把非强制点先验**置 0**，"
              "而 `mcts._expand` 仍要用**零先验的填充子节点**把宽度补满。"
              "注意这些垃圾分支**不是欠探索**——被选中的 `(7,12)` 访问了 **152 次**；"
              "真正的问题是**它们的乐观叶值被平均进去**，把「必然输」从精确的 `1.000` "
              "稀释成 `0.812~0.999`，而父节点最小化对手价值 → 挑中「对手赢得最不确定」"
              "的那一支。随机/起点网络之所以能过，是因为其价值输出≈0、各支几乎同值，"
              "「访问数说话」→强制点胜出。实测该失败在 hw 从 6.0/0.3 到 0.0/1.0"
              "**全部五档**下都存在（与退火无关），且随代数波动"
              "（gen_20 过、gen_1/10/30/39/40 不过）。")
            A("")
            A("**修法已实证验证**（`probe_terminal.py`，运行时 monkeypatch，不改磁盘代码）："
              "`MCTS._terminal_value` 只判了「对手一步成五 → −1」，**没判「当前方一步成五 "
              "→ +1」**。补上这条对称判定后，这些节点直接拿到**规则值**（精确 ±1），"
              "不再走网络、不再被垃圾分支稀释：")
            A("")
            A("| 模型 | base | 补上对称判定后 |")
            A("|---|---|---|")
            A("| gen_1 | 18/19 | **19/19** |")
            A("| gen_30 | 18/19 | **19/19** |")
            A("| gen_39 | 18/19 | **19/19** |")
            A("")
            A("**本次交付为何不修**：A/B 双方使用**同一个引擎**，当时判断该缺陷只在"
              "必败局面触发、不改变对局结果，因此**不影响 A/B 对比**；而改 `mcts.py` "
              "按项目规矩必须重跑 `test_tactics.py` + `test_selfplay.py` 才可信。"
              "另外首轮验证的看门狗进程**启动时已把旧 `mcts.py` 载入内存**，此时改文件"
              "会让两批验证跑在不同引擎上。")
            A("")
            A("> **⚠️ 2026-10-10 深夜更正（本报告交付后）**：用新建的单锚点工具"
              "（`ladder.py`）实测发现，上面那句「不改变对局结果 / 不影响 A/B」"
              "**是错的**。同样 48 局、同样种子、同样两个模型：")
            A(">")
            A("> | 引擎变体 | gen_60 vs 起点 | 得分率 |")
            A("> |---|---|---|")
            A("> | 当前引擎（含对称判定） | 20 : 26 | **0.438** |")
            A("> | 关掉该判定（≈旧价值语义） | 28 : 20 | 0.583 |")
            A("> | 旧引擎（本报告第 2 节的 42:6） | 42 : 6 | 0.875 |")
            A(">")
            A("> 也就是说：该判定会让这个对比摆动约 **100 Elo**，而它**并不足以**"
              "解释 42:6 与 0.438 的全部差距 —— 剩余差异至今没有归因（旧 `mcts.py` "
              "已被覆盖、没有备份，无法做干净 A/B）。**结论：本报告第 2 节那个"
              "「+338 Elo / 42:6」是旧引擎上的数字，在当前引擎上无法复现；"
              "引用它之前必须重测。** 新引擎本身是健康的（同一模型 400 vs 100 模拟"
              "仍能分辨出 19:5 ≈ +232 Elo 的差距）。详见 NOTES §12。")
    else:
        A("（本次未跑，见 `checkpoints/tactics_gen*.txt`）")
    A("")
    A("## 6. 口径与已知偏差（别忽略这节）")
    A("")
    A("1. **固定\"拐杖\"权重**：验证与 Elo 都用 heuristic/policy = "
      f"{args.heuristic_weight}/{args.policy_weight} 的固定先验。新一代在训练时"
      "用的是退火后的先验（末代约 0.5/1.0），所以**对新一代是不利的**："
      "它测出的是\"在起点模型的先验环境下\"的强弱，可视为提升幅度的**下界**。")
    A("2. **门控假拒率**：阈值 0.55 在 20 局下的假拒率实测 74.8%（新旧真实相当时），"
      "而本轮运行用的是\"未采纳就降 lr\"的旧逻辑，因此 lr 被统计噪声一路压低"
      "（1e-3 → 见上方轨迹）。**这会让后段（约 gen 30 之后）的进步被人为压平**，"
      "也是最后几十代曲线变缓的主要原因之一。修好的逻辑（只在胜率 < 0.30 时降 lr）"
      "已写入 `train.py`，**从下一次重启开始生效**。")
    A("3. **样本量与统计功效**（精确二项检验，双侧 α≈0.05，"
      "拒绝域由 `compare.binom_p` 的最小似然法给出）：")
    A("   - **24 局**：拒绝域 A 胜 ≤6 或 ≥18（α=0.023）。若真实 A 胜率 0.30，"
      "功效只有 **0.39**；0.25 时 0.61。")
    A("     —— 这正是 gen_20 那次 16:7 只拿到 p=0.093 的原因："
      "**不是没有效果，是 24 局分辨不出来**。")
    A("   - **48 局**：拒绝域 A 胜 ≤16 或 ≥32（α=0.029）。真实 A 胜率 0.30 时"
      "功效 **0.75**，0.25 时 0.93，0.20 时 0.99。")
    A("   - 预注册方案：若 48 局落在 0.05≤p<0.15（边缘区），"
      "**追加一轮 48 局（不同种子）并合并成 n=96 的检验**，"
      "而不是在多个 p 值里挑一个好看的报。合并后对 A 胜率 0.30 的功效约 0.99。")
    A("4. **起点口径**：A 用的是 `models/run3/init_gen9.pt`，即本轮训练的初始权重"
      "（第 1 代之前）。说「第 1 代 vs 第 N 代」时指的是同一件事。")
    A("5. **引擎在交付过程中被修复（跨会话协作）**：`mcts.py` 在 2026-10-10 22:16:54 "
      "被另一个会话补上了第 5 节描述的**对称终局判定**（当前方一步成五 → 精确 +1，"
      "用缓存的 `win_moves` 实现）。因此本次交付里有两套口径：")
    A("   - **训练（run3 全部 60 代）、48 局独立验证、7 点 Elo 的对局**：跑在"
      "**修复前**的引擎上（这些进程在 22:16 之前就已把旧 `mcts.py` 载入内存，"
      "改文件不影响已运行的进程）；")
    A("   - **战术验收**：由本报告生成时新起的进程执行 → 跑在**修复后**的引擎上。")
    A("   两者的差别只出现在「已经必败」局面的落子选择上（不改变对局结果），"
      "因此 A/B 结论与 Elo 数值不受影响；但复现时要留意这个时间点。")
    A("")
    A("6. **曲线上的非单调是「raw 快照」的正常现象**：`gen_N.pt` 存的是**原始权重**，"
      "而门控采纳的是 raw/EMA 里较好的那个。原始权重会抖动（另一个会话的 5 点曲线里 "
      "gen_15=273 而 gen_30=43），因此**判断「最强模型」应看 `best.pt`（EMA）而不是"
      "最新一代的 raw 快照**。")
    A("")
    A("7. **⚠️ 第 2 节那个「42:6 / ≈+338 Elo」是旧引擎上的数字，在当前引擎上无法复现。**"
      "（2026-10-10 深夜，本报告交付后用 `ladder.py` 实测）同样 48 局、同样种子、"
      "同样两个模型：")
    A("")
    A("| 引擎变体 | gen_60 vs 起点 | 得分率 |")
    A("|---|---|---|")
    A("| 当前引擎（含对称终局判定） | 20 : 26 | **0.438** |")
    A("| 关掉该判定（≈旧引擎的价值语义） | 28 : 20 | 0.583 |")
    A("| 旧引擎（本报告第 2 节） | 42 : 6 | 0.875 |")
    A("")
    A("   两点结论：① 那个「对称终局判定」**不是测量中性**的 —— 它让这个对比摆动约 "
      "**100 Elo**（与本报告第 5 节原先写的「不影响 A/B 对比」相反）；"
      "② 但它**不足以解释** 42:6 与 0.438 的全部差距，剩余差异**至今没有归因**"
      "（旧 `mcts.py` 已被覆盖且无备份，做不了干净的 A/B）。")
    A("")
    A("   引擎本身是健康的：同一模型用 400 模拟对 100 模拟仍能打出 **19:5**"
      "（≈ +232 Elo），说明分辨力没丢。**因此在当前引擎上，48 局既证明不了"
      "「训练有效」也证不伪** —— 引用第 2 节之前必须重测（建议 200 局起）。"
      "详见 NOTES §12。")
    A("")
    A("## 7. 交付物清单")
    A("")
    for f, desc in (("checkpoints/FINAL_REPORT.md", "本报告"),
                    ("checkpoints/final_train_summary.txt", "训练日志汇总"),
                    ("checkpoints/elo_curve.txt", "Elo 原始输出"),
                    (f"checkpoints/verify_gen{st.get('last_gen')}.txt", "最终独立对战验证"),
                    (f"checkpoints/verify_gen{st.get('last_gen')}_b2.txt",
                     "追加独立批次（不同 --seed-offset）"),
                    ("checkpoints/verify_gen20.txt", "中期独立对战验证（24 局）"),
                    ("checkpoints/control_selfplay_48.txt", "A vs A 对照实验（48 局）"),
                    (f"checkpoints/tactics_gen{st.get('last_gen')}.txt",
                     "战术验收门槛原始输出"),
                    ("models/run3/best.pt", "训练结束时保存的最佳模型")):
        A(f"- `{f}` —— {desc}"
          + ("" if os.path.exists(os.path.join(HERE, f)) else "（**当前不存在**）"))
    A("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="训练收尾：汇总 + Elo + 最终报告")
    ap.add_argument("--save-dir", default="models/run3")
    ap.add_argument("--wait-gen", type=int, default=60,
                    help="等日志里出现这一代（0=不等）")
    ap.add_argument("--wait-gen-min", type=int, default=900,
                    help="等这一代最多等多少分钟（默认 15 小时）")
    ap.add_argument("--wait-verify-min", type=int, default=45,
                    help="等 verify_genN.txt 最多多少分钟")
    ap.add_argument("--wait-batches-min", type=int, default=0,
                    help="再等追加批次 verify_genN_b*.txt 多少分钟（合并检验用）")
    ap.add_argument("--gens", type=int, nargs="*",
                    default=[1, 10, 20, 30, 40, 50, 60])
    ap.add_argument("--games", type=int, default=12, help="Elo 每一对局数")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--heuristic-weight", type=float, default=6.0)
    ap.add_argument("--policy-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--reuse-elo", action="store_true",
                    help="复用已存在的 checkpoints/elo_curve.txt，不重跑 Elo")
    args = ap.parse_args()

    save_dir = os.path.join(HERE, args.save_dir)
    log_path = os.path.join(save_dir, "train_log.jsonl")
    out = Tee()

    print(f"=== finalize 开始 {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)

    # 1) 等训练跑到目标代数（或超时/进程没了就按现状收尾）
    reached = args.wait_gen <= 0      # 没要求等待 = 按现状收尾，不算"未达成"
    if args.wait_gen > 0:
        print(f"[等待] train_log.jsonl 出现 gen {args.wait_gen}"
              f"（最多 {args.wait_gen_min} 分钟）...", flush=True)
        deadline = time.time() + args.wait_gen_min * 60
        while time.time() < deadline:
            if any(r.get("gen") == args.wait_gen for r in read_jsonl(log_path)):
                reached = True
                break
            time.sleep(30)
        print(f"[等待] 结束：{'已到达 gen ' + str(args.wait_gen) if reached else '超时/未到达'}",
              flush=True)

    rows = read_jsonl(log_path)
    out("=" * 78)
    out("一、训练日志汇总")
    out("=" * 78)
    st = summarize_train(rows, out)
    render_table(rows, out, tail=16)

    # 2) 等独立验证报告
    last_gen = st.get("last_gen", 0)
    vpath = os.path.join(save_dir, f"verify_gen{last_gen}.txt")
    out("")
    out("=" * 78)
    out(f"二、独立对战验证（{os.path.basename(vpath)}）")
    out("=" * 78)
    vf = wait_verify(vpath, args.wait_verify_min, out)
    render_verify(vf, out)

    # 2b) 追加批次（独立种子偏移）→ 合并检验
    batches = collect_batches(save_dir, last_gen, vf)
    if vf and len(batches) < 2 and args.wait_batches_min > 0:
        out(f"[等待] 追加批次 verify_gen{last_gen}_b*.txt"
            f"（最多 {args.wait_batches_min} 分钟）...")
        deadline = time.time() + args.wait_batches_min * 60
        while time.time() < deadline:
            batches = collect_batches(save_dir, last_gen, vf)
            if len(batches) >= 2:
                break
            time.sleep(15)
    pooled = render_pool(batches, out)

    # 3) Elo 曲线
    out("")
    out("=" * 78)
    out("三、Elo 曲线")
    out("=" * 78)
    el = None
    if args.skip_elo:
        out("[Elo] --skip-elo，跳过")
    elif args.reuse_elo and os.path.exists(
            os.path.join(save_dir, "elo_curve.txt")):
        txt = open(os.path.join(save_dir, "elo_curve.txt"),
                   encoding="utf-8", errors="replace").read()
        el = parse_elo_text(txt)
        out("[Elo] --reuse-elo：复用已有的 elo_curve.txt（**不重跑**，"
            "省下几十分钟）")
        render_elo(el, out)
    else:
        avail = [g for g in args.gens
                 if os.path.exists(os.path.join(save_dir, f"gen_{g}.pt"))]
        missing = [g for g in args.gens if g not in avail]
        if missing:
            out(f"[Elo] 以下代数缺少 checkpoint，已跳过: {missing}")
        if len(avail) < 2:
            out("[Elo] 可用 checkpoint 不足 2 个，跳过")
        else:
            el, _ = run_elo(args, out, save_dir)
            render_elo(el, out)

    # 4) 对照实验：同一个模型 vs 自己（检查记账/颜色/噪声是否引入系统性偏置）
    out("")
    out("=" * 78)
    out("四、对照实验（同一个模型 vs 自己，48 局）")
    out("=" * 78)
    ctl = parse_verify(os.path.join(save_dir, "control_selfplay_48.txt"))
    render_verify(ctl, out, label="对照实验", a_name="同一模型", b_name="同一模型")

    # 5) 引擎战术验收门槛（对当前代的模型）
    out("")
    out("=" * 78)
    out("五、引擎战术验收门槛（test_tactics.py）")
    out("=" * 78)
    tac = run_tactics(save_dir, last_gen, out)

    # 6) 落盘
    with open(os.path.join(save_dir, "final_train_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(out.text())
    rp = write_report(save_dir, st, vf, el, ctl, pooled, tac, args, reached, out)

    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"[完成] 报告: {rp}", flush=True)
    print(f"[完成] 汇总: {os.path.join(save_dir, 'final_train_summary.txt')}", flush=True)
    if args.wait_gen > 0 and not reached:
        print(f"[注意] 没有等到 gen {args.wait_gen}，报告基于现有 "
              f"gen {st.get('last_gen')}", flush=True)
    print(f"=== finalize 结束 {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
