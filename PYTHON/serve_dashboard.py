# serve_dashboard.py —— 看板的 HTTP 服务：发页面、发曲线图，并提供两个动作接口
#
# 用法：python serve_dashboard.py [端口]     默认 8791，只监听 127.0.0.1
#
# 能发的路径（白名单，其它一律 404）：
#   GET  /  /index.html  /progress_live.html   → 看板页面（no-store，永远取最新）
#   GET  /models/<run>/progress.png            → 各 run 的训练曲线图（带版本号参数时可用强缓存）
#   GET  /api/last-params                      → 当前/上次的训练参数（网页表单用它预填）
#   GET  /api/v4-preset                        → V4 一键配方（按钮用它显示"将要跑什么"，只读）
#   GET  /api/state                            → 【2026-10-14 加】全量看板快照（只读 JSON）
#   POST /api/start-train                      → 启动训练（参数经白名单校验）
#   POST /api/start-v4                         → 一键开 V4：先 prepare_run.ps1 再按 V4 配方启动
#   POST /api/stop-train                       → 停止训练（调用 stop_train.ps1）
#
# 【安全边界】这是本机单机服务，但仍按"最小能力"设计：
#   · 只监听 127.0.0.1
#   · 启动训练时 **SaveDir 必须是 models/<名字> 形状**，其余（含绝对路径、
#     `..`、以及 checkpoints/pilot/big 那个指向 models/V1 的 junction）一律拒绝
#   · 所有数值参数有范围检查；不认识 / 越界的字段直接拒绝，不拼接进命令行
#   · POST 只接受 Content-Type: application/json（跨站表单无法伪造这个头，
#     且不实现 OPTIONS 预检 ⇒ 外部网页发不出能生效的请求）
#   · 训练已在跑时拒绝再次启动
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "progress_live.html")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
ALLOWED_PAGE = {"/", "/index.html", "/progress_live.html"}
IMG_RE = re.compile(r"^/models/([^/]+)/progress\.png$")

# 【2026-10-25 加】对弈页（web_play.py）由看板按需拉起，供 iframe 嵌入。
# 只做「端口没开就启动」这一件事：不转发、不代理，两个进程各管各的页面。
PLAY_PORT = int(os.environ.get("FLASH_PLAY_PORT", "8770"))

# ---- C++ 引擎（gk.exe）的路径约定 ----
# 两套实现完全独立：C++ 的权重/回放池/日志都在 C++\model\<run> 下，
# 所以「切引擎」= 换一套启动命令 + 换一套进度来源，不是同一个目录换解析方式。
CXX_DIR = os.path.join(os.path.dirname(ROOT), "C++")
GK_EXE = os.path.join(CXX_DIR, "build", "gk.exe")
CPP_RUN_DIR_DEFAULT = "model/run"
CPP_DIR_RE = re.compile(r"^model/[A-Za-z0-9_.\-]{1,40}$")


# ------------------------------------------------------------------ C++ 引擎
def cpp_pids():
    """正在跑的 gk.exe loop 的 PID 列表（play 那种常驻对弈进程不算训练）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'gk.exe' -and "
             "$_.CommandLine -match '\\bloop\\b' } | Select-Object -ExpandProperty ProcessId) -join ','"],
            capture_output=True, text=True, timeout=20)
        raw = (out.stdout or "").strip()
        return [int(x) for x in raw.split(",") if x.strip().isdigit()]
    except Exception:
        return []


def _cpp_manifest(rd):
    """读 run.json（gk.exe loop 第一次进目录时写的自述文件）。

    没有它就只能靠目录名猜「这是主线还是实验」，而目录名恰好是最容易混乱的东西
    （run_v2 是从 run_v4amp 第 7 代接过来的，它自己的 gen 从 1 重开）。
    """
    path = os.path.join(CXX_DIR, rd, "run.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cpp_run_dirs():
    """C++\\model 下所有像 run 的目录（有 train_log.jsonl 或 gen_*.bin）。

    broken* 那些是「已知跑坏的配方」产物：照样列出来（标成损坏、勿用），
    但**不会**成为默认选项，免得下次一点就接着练坏的。
    """
    base = os.path.join(CXX_DIR, "model")
    found = []
    if not os.path.isdir(base):
        return found
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        has_log = os.path.isfile(os.path.join(full, "train_log.jsonl"))
        try:
            has_gen = any(f.startswith("gen_") and f.endswith(".bin")
                          for f in os.listdir(full))
        except OSError:
            has_gen = False
        if has_log or has_gen:
            found.append("model/" + name)
    return found


def _cpp_runs_detail():
    """每个 C++ run 的累计训练量 + champion 时间 + 自述（角色/备注），给 C++ 面板用。"""
    out = []
    for rd in _cpp_run_dirs():
        entries, mtime = _cpp_read_log(rd, tail=2000)
        gens = _cpp_gen_files(rd)
        best = os.path.join(CXX_DIR, rd, "best.bin")
        man = _cpp_manifest(rd)
        last_gen = (entries[-1].get("gen") if entries else (gens[-1] if gens else None))
        try:
            offset = int(man.get("gen_offset") or 0)
        except Exception:
            offset = 0
        out.append({
            "dir": rd,
            "role": str(man.get("role") or ""),
            "note": str(man.get("note") or ""),
            "gen_offset": offset,
            # total_gen = 这个目录自己的第几代 + 接手时前面的代数
            # （run_v2 的 gen_004 其实是项目的第 11 代）
            "total_gen": ((last_gen or 0) + offset) if last_gen else None,
            "gens": len(entries),
            "gen_files": len(gens),
            "last_gen": last_gen,
            "samples": sum(int(e.get("samples") or 0) for e in entries),
            "games": sum(int(e.get("games") or 0) for e in entries),
            "steps": sum(int(e.get("steps") or 0) for e in entries),
            "selfplay_s": round(sum(float(e.get("selfplay_s") or 0) for e in entries), 1),
            "train_s": round(sum(float(e.get("train_s") or 0) for e in entries), 1),
            "log_mtime": mtime,
            "best_mtime": (os.path.getmtime(best) if os.path.isfile(best) else 0.0),
            "best_mb": (round(os.path.getsize(best) / 1048576.0, 2)
                        if os.path.isfile(best) else 0.0),
        })
    out.sort(key=lambda r: (r["log_mtime"], r["best_mtime"]), reverse=True)
    # 推荐续训的目录：优先 role 里写着「主线」的；否则取最新、且确实训过的那个
    rec = ""
    for r in out:
        if "主线" in r["role"]:
            rec = r["dir"]
            break
    if not rec:
        for r in out:
            if (r["gen_files"] or 0) > 0 and "损坏" not in r["role"]:
                rec = r["dir"]
                break
    for r in out:
        r["recommended"] = (r["dir"] == rec)
        r["scratch"] = (0 == (r["gen_files"] or 0) and 0.0 == (r["best_mtime"] or 0.0))
    return out


def cpp_recommended_dir():
    """默认该续训的 C++ 目录（给「开始训练」的默认值和 C++ 面板的推荐项）。"""
    for r in _cpp_runs_detail():
        if r["recommended"]:
            return r["dir"]
    return CPP_RUN_DIR_DEFAULT


def _cpp_read_log(rundir, tail=60):
    """读 C++ 的 train_log.jsonl，返回 (entries, mtime)。条目就是 loop 写的那些键。"""
    path = os.path.join(CXX_DIR, rundir, "train_log.jsonl")
    if not os.path.isfile(path):
        return [], 0.0
    try:
        mtime = os.path.getmtime(path)
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-tail:]
    except OSError:
        return [], 0.0
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out, mtime


def _cpp_gen_files(rundir):
    """某 run 里已落盘的代数（取 gen_NNN.bin 的编号）。"""
    d = os.path.join(CXX_DIR, rundir)
    gens = []
    try:
        for f in os.listdir(d):
            m = re.fullmatch(r"gen_(\d+)\.bin", f)
            if m:
                gens.append(int(m.group(1)))
    except OSError:
        return []
    return sorted(gens)


def _cpp_stdout_tail(rundir, n=40):
    """gk loop 的 stdout 尾巴（start_train_cpp 会把它写进 loop_stdout.log）。"""
    path = os.path.join(CXX_DIR, rundir, "loop_stdout.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [l.rstrip() for l in fh if l.strip()]
    except OSError:
        return []
    return lines[-n:]


def _cpp_phase(rundir, running):
    """从 stdout 尾巴推断「现在这一代在干什么」，看板顶部那句状态就靠它。"""
    if not running:
        return {"phase": "idle", "text": "空闲", "gen": None, "total": None}
    lines = _cpp_stdout_tail(rundir, 40)
    gen = total = None
    phase, text = "unknown", "训练中"
    for line in reversed(lines):
        m = re.search(r"===\s*gen\s*(\d+)\s*/\s*(\d+)", line)
        if m and gen is None:
            gen, total = int(m.group(1)), int(m.group(2))
        if "[门控]" in line:
            phase, text = "gate", "门控：候选 vs 现任 best"
            break
        if "[训练]" in line:
            phase, text = "train", "训练：前向 + 反向"
            break
        if "[自对弈]" in line:
            phase, text = "selfplay", "自对弈：采样"
            break
        if re.search(r"===\s*gen\s*\d+", line):
            phase, text = "selfplay", "自对弈：采样"
            break
    return {"phase": phase, "text": text, "gen": gen, "total": total}


def _cpp_elo(rundir):
    """读 gk.exe elo 生成的 elo.json（KataGo 式棋力曲线）。缺文件返回 None。"""
    if not rundir:
        return None
    p = os.path.join(CXX_DIR, rundir, "elo.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("versions") else None


# ---------------------------------------------------------------- Elo 对战任务
# 【2026-10-30 加】用户要求「在实时看板上增加 elo 对战功能，方便我看 elo 提升没有」。
#
# 为什么必须做成"后台任务 + 轮询状态"，而不是像 /api/start-train 那样同步调用：
# 一次 gk.exe elo 要真下棋（每局 ~0.75 s @600 sims），"最近 5 代 + 锚点"就是
# 11 对 × 12 局 ≈ 130 局 ≈ 2 分钟，HTTP 请求根本等不起，而浏览器也不该卡在
# 一个两分钟的白屏请求上。
#
# 三条设计决定（都踩过坑或与既有代码对齐）：
#   ① 【2026-10-31 改】边库模式下 gk 直接写 elo.json（它内部已用全部边重拟合，
#      输出永远是全量曲线），看板不再做 .new→replace 的原子替换。
#      原子性现在由 gk 侧 + 这条兜底保证：_cpp_elo() 解析失败会安静地显示
#      "还没有 Elo 数据"，而边库 elo_edges.jsonl 是**只追加**的，重跑一次就能重建曲线。
#      （旧做法）gk 是最后一次性写整份 JSON 的，中途被杀/写一半会留下半截文件，
#      而 _cpp_elo() 解析失败会安静地显示"还没有 Elo 数据" ⇒ 用户会以为从没测过。
#      os.replace 在 Windows 上对同目录文件是原子的，不会出现"读到半个 JSON"。
#   ② 只有 rc == 0 且新文件能被 json 解析，才替换。否则保留旧曲线并在状态里报错。
#   ③ 状态落盘（写成 JSON 文件）而不是只放内存：看板服务重启后仍然能看到
#      "上一次对战是什么时候跑的、结果如何"，否则一点重启就丢失全部上下文。
ELO_DIR = os.path.join(ROOT, "checkpoints", "history")
ELO_STATE = os.path.join(ELO_DIR, "elo_job.json")
ELO_LOG = os.path.join(ELO_DIR, "elo_job.log")
ELO_LOCK = threading.Lock()
_ELO_PROC = {"proc": None, "started": 0.0, "cmd": [], "pairs": 0, "games": 0}


def _elo_write_state(d):
    try:
        os.makedirs(ELO_DIR, exist_ok=True)
        tmp = ELO_STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, ELO_STATE)
    except OSError:
        pass


def _elo_read_state():
    try:
        with open(ELO_STATE, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _elo_tail(n=14):
    try:
        with open(ELO_LOG, encoding="utf-8", errors="replace") as fh:
            lines = [l.rstrip() for l in fh]
    except OSError:
        return []
    return [l for l in lines if l.strip()][-n:]


def _elo_pairs_from_log():
    """从日志里数「已经打完的对」——gk 每打完一对就打印 '  gen_A vs gen_B：…'"""
    done = 0
    try:
        with open(ELO_LOG, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if " vs " in line and "：" in line and line.startswith("  "):
                    done += 1
    except OSError:
        return 0
    return done


def _elo_job_alive():
    p = _ELO_PROC.get("proc")
    return p is not None and p.poll() is None


def elo_plan(rundir, recent=5, games=30, anchor=0):
    """算这次要打哪些代、多少对。纯函数 —— 页面「预览」和真启动用同一套，防止漂移。

    gens 规则：
      · 锚点（默认 10 = 现有 elo.json 的锚点 gen_010，保持历史可比）+ 最近 N 代；
      · 若锚点本身落在"最近 N 代"里，不重复；
      · 少于 2 个可载入的版本就没得打。
    pairs 规则与 gk.exe elo 一致（main.cpp:1051-1057）：
      链式相邻 (n-1) + 每代 vs 锚点 (n-1) = 2n-2 对（锚点自己是 0，要扣掉）。
    """
    gens_all = _cpp_gen_files(rundir) if rundir else []
    if not gens_all:
        return {"ok": False, "errors": ["%s 里还没有 gen_*.bin，没有可对战的版本" % (rundir or "(未选目录)")]}
    recent = max(1, min(40, int(recent or 5)))
    games = max(2, min(200, int(games or 30)))
    tail = gens_all[-recent:]
    a = int(anchor or 0)
    if a and a not in gens_all:
        return {"ok": False, "errors": ["锚点 gen_%03d 不在 %s 里（现有 %d–%d）" %
                                        (a, rundir, gens_all[0], gens_all[-1])]}
    anchor_note = ""

    # 【2026-10-31 改·滚动锚点（用户提的方案）】没显式给锚点时，不再拿一个远古锚点（gen_010），
    # 而是拿「旧曲线里**已评过分的最新一代**」当尺子：新模型跟它打，测出的差值直接加到它的 Elo 上。
    # 为什么这才对：Elo 只有**差值**有意义，绝对刻度只是人为定死的常数。而远锚点对 61 代之后的
    # 模型必然全胜（12:0），这条边得分恒 1、提供 0 信息，拟合反而被它拽住 —— 2026-10-31 实测
    # 曲线据此给出「gen_075 比 gen_065 低 100 Elo」，而 400 局定点对打是**高 109.5**。
    # 窗口下界同时满足两件事：① 至少 recent 代；② 覆盖「最新已评代 − 2」，
    # 这样合并时至少有 3 代共有，可用中位偏移稳健地把新段对齐到旧刻度（见 _elo_merge）。
    if not a:
        _elo = _cpp_elo(rundir) or {}
        rated = []
        for _v in (_elo.get("versions") or []):
            try:
                rated.append(int(_v["gen"]))
            except (KeyError, TypeError, ValueError):
                continue
        rated = sorted(set(rated))
        if rated:
            # 边库模式下**不需要**拿锚点去对齐刻度（那是旧做法，靠平移拼接）：
            # 新边并进 elo_edges.jsonl 后是**全部边一起**重新拟合，没有"两段刻度"这回事。
            # 这里保留锚点只是为了决定窗口下界——往前多盖 2 代，让新边能和已测部分接上。
            lo = min(rated[-1] - 2, gens_all[-recent])
            tail = [g for g in gens_all if g >= lo]
            a = rated[-1] if rated[-1] in tail else tail[-1]
            overlap = len(set(tail) & set(rated))
            anchor_note = ("窗口 %d 代（含已评代 %d 个，接得上现有曲线）；"
                           "锚点由边库决定 = 最老一代，曲线向上即变强。边库里已打够 %d 局的"
                           "对会自动跳过" % (len(tail), overlap, games))

    # 显式给了锚点、但离最新代太远 → 自动降级成窗口最早一代（该锚点的对局必然饱和）。
    if a and not anchor_note and tail and abs(gens_all[-1] - a) > 30:
        anchor_note = ("锚点 gen_%03d 距最新代 %d 代，对局会饱和（全胜/全负），"
                       "已自动改用窗口最早一代 gen_%03d" % (a, abs(gens_all[-1] - a), tail[0]))
        a = tail[0]
    gens = sorted(set(tail + ([a] if a else [])))
    if len(gens) < 2:
        return {"ok": False, "errors": ["只选出 %d 个版本，至少需要 2 个" % len(gens)]}
    n = len(gens)
    # 【必须照抄 main.cpp:1051-1057，不要自己推公式】实测曾按 (n-1)+n 算，
    # 而 gk 给的 3 版本实际只有 4 对。逐行读源码：
    #   for i in [0, n-2]:            pairs.push(i, i+1)      → n-1 对
    #   if with_ref: for i != anchor: pairs.push(i, anchor)   → n-1 对（**跳过锚点自己**）
    # 所以总数恒为 2(n-1)，与锚点是不是在列表里无关。
    # 自对局被排除，也就不会出现 [0,0] 那种无意义的 0 分边。
    pairs = 2 * (n - 1)
    total_games = pairs * games
    return {"ok": True, "dir": rundir, "gens": gens, "anchor": a, "games": games,
            # anchor_explicit：只有用户**自己填了**锚点才传给 gk。边库模式默认不传，
            # 让 gk 用「最老一代」当零点（曲线向上 = 变强）。
            "anchor_explicit": bool(int(anchor or 0)),
            "pairs": pairs, "total_games": total_games,
            # 0.75 s/局是早期口径；2026-10-31 实测最近这些代约 1.7 s/局（对局更长、棋力更强），
            # 低估一倍会让 ETA 提前变成 0（页面显示"预计还要 0 秒"）。取 1.6 保守一点，
            # 真实进度仍由 elo_status() 按已完成对数外推。
            "est_seconds": int(total_games * 1.6),
            "anchor_note": anchor_note,
            "available": gens_all}


def elo_defaults(rundir):
    """页面默认值。

    【2026-10-31 改】锚点默认**留空（0）**：边库模式下零点由 gk 决定 = 曲线里最老的一代，
    于是"曲线向上 = 变强"，不需要用户操心选谁。旧版本这里会回填 elo.json 的 anchor_gen，
    那是分段拼接时代的产物（要跟旧刻度对齐），现在反而会把零点钉在一个中间代上。
    """
    gens = _cpp_gen_files(rundir) if rundir else []
    return {"anchor": 0,
            "anchor_from": "留空（最老一代 = 0 Elo，曲线向上即变强）",
            "gens": gens, "max_gen": gens[-1] if gens else 0}


def _as_float(v, dflt=0.0):
    """把可能是数字、也可能是字符串的值安全转成 float。

    【为什么需要这个】started 在两处存在，类型还不一样：
      · _ELO_PROC["started"]   = time.time()  → float 时间戳
      · 状态文件里的 started     = "%Y-%m-%d %H:%M:%S"  → 字符串
    第一版直接 float(两边的 or)，服务重启后读回带时间戳的旧状态就抛
    ValueError: could not convert string to float —— 而且异常发生在
    HTTP 处理线程里，浏览器只会看到 "Empty reply from server"，
    排查难度远大于普通报错。
    """
    if isinstance(v, bool) or v is None:
        return dflt
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return dflt


def elo_status():
    """给页面轮询的任务状态。running 以"进程还活着"为准，不信状态文件（可能写失败）。"""
    st = _elo_read_state()
    running = _elo_job_alive()
    pairs = int(_as_float(_ELO_PROC.get("pairs") or st.get("pairs")))
    done = _elo_pairs_from_log() if running else int(_as_float(st.get("pairs_done")))
    started = _as_float(_ELO_PROC.get("started"))
    if running and started:
        elapsed = round(time.time() - started, 1)
    else:
        elapsed = _as_float(st.get("elapsed"))
    # 【2026-10-31 修】`eta` 以前只在启动时写一次（= 计划里的 est_seconds，按 0.75 s/局估的），
    # 运行中从不更新。实际速率约 1.7 s/局，跑到一半 elapsed 就超过这个常数 ⇒
    # 页面（progress_live.html:1836）算 `job.eta - job.elapsed` 得到负数、显示"预计还要 0 秒"。
    # 现在按已完成对数外推，快慢都能给出合理估计。
    eta = _as_float(st.get("eta"))
    if running and done > 0 and pairs > 0 and elapsed:
        eta = elapsed + elapsed / float(done) * float(max(0, pairs - done))
    out = {
        "running": running,
        "ok": st.get("ok"),
        "stage": "running" if running else (st.get("stage") or "idle"),
        "pairs": pairs or st.get("pairs"),
        "pairs_done": done,
        "games": _ELO_PROC.get("games") or st.get("games"),
        "gens": st.get("gens"),
        "anchor": st.get("anchor"),
        "dir": st.get("dir"),
        "started": st.get("started"),
        "finished": st.get("finished"),
        "elapsed": elapsed,
        "eta": eta,
        "rc": st.get("rc"),
        "error": st.get("error"),
        "summary": st.get("summary"),
        "delta": st.get("delta"),
        "delta_sig": st.get("delta_sig"),
        # 【2026-10-30 补】上面是白名单式构造，漏了两个字段 ⇒ 后端明明合并了、
        # 状态文件里也有，页面上却看不到"本次只测 N 代、其余保留旧值"。
        # 这种"后端对、前端缺"的 bug 不会报错，只会让用户误以为整条曲线都重算了。
        "merge_note": st.get("merge_note"),
        "merged_versions": st.get("merged_versions"),
        "tail": _elo_tail(),
    }
    return out


def _gk_env():
    """给 gk.exe 用的环境：PATH 里补上 torch\\lib。

    【为什么必须有这个函数】gk.exe 静态链了 torch 的 DLL，桌面双击时 PATH 里没有
    torch\\lib 就会直接 0xC0000135（STATUS_DLL_NOT_FOUND）**静默**退出 ——
    stdout/stderr 一个字都不留。实测：裸跑 `gk.exe elo ...` 返回 -1073741515。
    start_train_cpp() 早就踩过这个坑并修了（见其中注释），这里抽出公共版本，
    免得 elo 这条路再踩一遍。
    """
    env = dict(os.environ)
    tlib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    parts = [tlib]
    # MinGW 运行时（libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll）
    for c in (os.path.join(CXX_DIR, "tools", "mingw64", "bin"),):
        if os.path.isdir(c):
            parts.append(c)
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def elo_start(body):
    """启动一次 Elo 对战。body: {dir, recent, games, anchor, dry_run}"""
    dr = body.get("dry_run")
    if not isinstance(dr, bool):
        return {"ok": False, "errors": [
            "必须显式带布尔字段 dry_run（true=只预览计划，false=真打）"]}
    rundir = str(body.get("dir") or "").strip().replace("\\", "/") or cpp_recommended_dir()
    if not CPP_DIR_RE.fullmatch(rundir):
        return {"ok": False, "errors": ["运行目录必须形如 model/名字（不能写到 C++\\model 之外）"]}
    if not os.path.isfile(GK_EXE):
        return {"ok": False, "errors": ["找不到 {0}，先构建 C++（C++\\build.bat）".format(GK_EXE)]}

    plan = elo_plan(rundir, body.get("recent"), body.get("games"), body.get("anchor"))
    if not plan.get("ok"):
        return plan
    # 【2026-10-31 改：边库模式】不再 --out 到 elo.json.new：gk 现在自己把新边并入
    # elo_edges.jsonl 后全量重拟合、直接写 elo.json（那里永远是全量曲线）。
    # 再写到 .new 反而会让边库更新了、曲线却没换。
    # --skip-existing：滚动窗口每次都送同一批代，这里让 gk 跳过边库里已经打够局数的对，
    # 否则同一对会被反复重打（局数白白翻倍，曲线却没更准）。
    args = [GK_EXE, "elo", "--dir", rundir,
            "--gens", ",".join(str(g) for g in plan["gens"]),
            "--eval-games", str(plan["games"]),
            "--skip-existing", str(plan["games"])]
    if plan.get("anchor_explicit"):
        args += ["--anchor", str(plan["anchor"])]
    plan["cmd"] = " ".join(['gk.exe'] + args[1:])
    if dr:
        plan["dry_run"] = True
        plan["note"] = ("预览：不会真打。%d 对 × %d 局 = %d 局，约 %d 分 %d 秒"
                        % (plan["pairs"], plan["games"], plan["total_games"],
                           plan["est_seconds"] // 60, plan["est_seconds"] % 60))
        plan["current_elo"] = _cpp_elo(rundir)
        return plan

    if not ELO_LOCK.acquire(blocking=False):
        return {"ok": False, "errors": ["已有一个 Elo 对战在跑，等它结束或先点「停止」"]}
    try:
        if _elo_job_alive():
            return {"ok": False, "errors": ["已有一个 Elo 对战在跑（进程还活着）"]}
        os.makedirs(ELO_DIR, exist_ok=True)
        # 【顺序要紧】旧曲线必须在**启动 gk 之前**快照。
        # 旧流程里 gk 写 .new、elo.json 还是旧的，可以等跑完再读；现在 gk 直接覆盖
        # elo.json，跑完再读拿到的就是新文件 ⇒ 同代相减恒为 0（这个坑踩过一次）。
        _ELO_PROC["old"] = _elo_load(os.path.join(CXX_DIR, rundir, "elo.json"))
        # 旧日志清空：_elo_pairs_from_log() 靠行数数进度，不清会把上次的算进来
        with open(ELO_LOG, "w", encoding="utf-8") as fh:
            fh.write("=== {0} Elo 对战 {1} ===\n{2}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), rundir, plan["cmd"]))
            fh.flush()
            proc = subprocess.Popen(args, cwd=CXX_DIR, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    env=_gk_env(),
                                    creationflags=0x00000008 | 0x00000200)
        _ELO_PROC.update({"proc": proc, "started": time.time(), "cmd": args,
                          "pairs": plan["pairs"], "games": plan["games"]})
        _elo_write_state({
            "ok": None, "stage": "running", "dir": rundir, "gens": plan["gens"],
            "anchor": plan["anchor"], "games": plan["games"], "pairs": plan["pairs"],
            "pairs_done": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": 0, "eta": plan["est_seconds"], "cmd": plan["cmd"],
            "pid": proc.pid,
        })
        threading.Thread(target=_elo_watch, args=(proc, rundir, plan), daemon=True).start()
        plan["pid"] = proc.pid
        plan["started"] = True
        return plan
    finally:
        ELO_LOCK.release()


def _elo_load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _elo_merge(old, new):
    """【已废弃，仅存档】2026-10-31 起不再被调用 —— 边库模式下没有"合并"这一步了。

    gk.exe elo 现在把新边并入 <run>/elo_edges.jsonl，然后用**全部边**重新全局拟合、
    直接写 elo.json。旧流程的"平移对齐"被证明是误差来源（新旧差值散在 +60~+218，
    不存在常数偏移，等于在猜）。保留这个函数是为了能读懂历史代码和旧备份文件。
    原文如下：
    把"这次跑出来的几代"并进"旧的全量曲线"，返回 (合并后的 dict, 说明)。

    【为什么必须有这一步】gk.exe elo 是**按 --gens 点名拟合**的：给它 3 个代，
    它就只解这 3 个未知数、只写这 3 项。看板上的按钮按设计只测"最近 N 代 + 锚点"
    （全量 60 代要 25 分钟），所以它的输出天然比现有曲线小。
    直接拿它覆盖 = 每做一次快速对比，就把历史曲线削短一截。

    规则：
      · 锚点不同 → 用**共有代**估一个整体偏移（中位数），把新段平移对齐旧刻度后再合并；
        两边没有共有代才拒绝（那才是真的没法比）。
      · versions：以 gen 为键，旧的在先、新值覆盖同名项，按 gen 排序。
      · edges：按 (a,b) 判重（gk 每个邻居对只出一条边，重跑就是覆盖），取并集。
      · generated 用新的（它描述的是"最近一次测算"）。
    """
    if not new.get("versions"):
        return None, "新生成的 elo.json 里没有 versions，已保留旧曲线"
    o_anchor = (old.get("anchor_gen") if old.get("versions") else None)
    n_anchor = new.get("anchor_gen")
    # 【2026-10-31 改】锚点不同**不再直接拒绝**，改成用共有代估一个整体偏移、把新段平移对齐。
    # 为什么可以这样做：BT 的刻度只差一个常数，段内的相对关系（也就是我们真正关心的
    # "这几代涨了多少"）不受平移影响。原来一律拒绝，是因为 2026-10-31 之前锚点固定 gen_010；
    # 而现在 gk 的默认锚点改成了"窗口内最早一代"（远锚点会饱和，见 main.cpp 里 cmd_elo 的注释），
    # 拒绝就会让看板每次都无法合并。
    # 找不到共有代时仍然拒绝——那才是真的没有共同刻度。
    shift = 0.0
    if old.get("versions") and o_anchor is not None and n_anchor is not None \
            and int(o_anchor) != int(n_anchor):
        old_by_gen = {}
        for v in old["versions"]:
            try:
                old_by_gen[int(v["gen"])] = float(v["elo"])
            except (KeyError, TypeError, ValueError):
                continue
        diffs = []
        for v in new["versions"]:
            try:
                g = int(v["gen"])
                if g in old_by_gen:
                    diffs.append(old_by_gen[g] - float(v["elo"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not diffs:
            return None, ("锚点变了（旧 gen_%s → 新 gen_%s）且两边没有共有代，无法对齐刻度；"
                          "请用命令行全量重跑一次 `gk.exe elo --dir <run> --gens <全部代>`"
                          % (o_anchor, n_anchor))
        diffs.sort()
        shift = diffs[len(diffs) // 2]        # 中位数，抗单点噪声
        for v in new["versions"]:
            for k in ("elo", "lo", "hi"):
                if isinstance(v.get(k), (int, float)):
                    v[k] = float(v[k]) + shift
    by_gen = {}
    for v in (old.get("versions") or []):
        try:
            by_gen[int(v["gen"])] = v
        except (KeyError, TypeError, ValueError):
            continue
    kept = len(by_gen)
    for v in new["versions"]:
        try:
            by_gen[int(v["gen"])] = v
        except (KeyError, TypeError, ValueError):
            continue
    versions = [by_gen[g] for g in sorted(by_gen)]
    edges, seen = [], set()
    for e in (new.get("edges") or []) + (old.get("edges") or []):
        key = (e.get("a"), e.get("b"))
        if key in seen:
            continue
        seen.add(key)
        edges.append(e)
    out = dict(new)
    out["versions"] = versions
    out["edges"] = edges
    # 平移过就把锚点标回旧的那个（数值已经对齐到旧刻度，再标新锚点会误导）
    if shift != 0.0 and o_anchor is not None:
        out["anchor_gen"] = int(o_anchor)
        if old.get("anchor_name"):
            out["anchor_name"] = old.get("anchor_name")
    note = ""
    if kept > len(new["versions"]):
        note = ("已合并：本次测 %d 代，保留了旧曲线其它 %d 代 → 共 %d 代（旧曲线另存 elo.json.bak）"
                % (len(new["versions"]), len(versions) - len(new["versions"]), len(versions)))
        if shift:
            note += "；新段按共有代平移 %+.1f Elo 以对齐旧刻度" % shift
    return out, note


def _elo_summary(old, new):
    """从"新曲线"里取最新一点，并给出它和"旧曲线里同一代"的差。

    【参数为什么要显式传 old/new】第一版是自己在函数里读 elo.json 当"新"，
    再调 _cpp_elo() 当"旧"—— 结果调用点在 os.replace **之后**，
    于是 _cpp_elo() 读到的已经是新文件，"旧值"和"新值"一模一样 ⇒ delta 恒为 0。
    更坏的是外层又拿"新值 − 旧曲线末位代"报了个 -188.9 的假跌幅：
    两个不同代相减，纯属造数。差值只用"同一代、替换前后"比较。
    """
    vs = new.get("versions") or []
    if not vs:
        return None

    def pick(d, gen):
        for v in (d.get("versions") or []):
            try:
                if int(v.get("gen")) == gen:
                    return v
            except (TypeError, ValueError):
                continue
        return None

    top = vs[-1]
    last_gen = int(top.get("gen") or 0)
    out = {
        "gen": last_gen,
        "elo": round(float(top.get("elo") or 0), 1),
        "lo": round(float(top.get("lo") or 0), 1),
        "hi": round(float(top.get("hi") or 0), 1),
        "anchor_name": new.get("anchor_name"),
        "anchor_gen": new.get("anchor_gen"),
        "versions": len(vs),
    }
    prev = pick(old, last_gen)
    if prev is not None:
        try:
            out["prev_elo"] = round(float(prev.get("elo") or 0), 1)
        except (TypeError, ValueError):
            out["prev_elo"] = None
        out["prev_at"] = old.get("generated")
    return out


def _elo_diff(new_sum):
    """同代前后差值 + 显著性。

    只比"同一代"：上次跑过 gen_61 才能说 gen_61 涨了没。
    上一版没比较代就相减，把 gen_61 的新值减去旧曲线末位 gen_43 的值，
    报出 -188.9 —— 两个不同代相减，纯属造数。
    """
    if not new_sum or new_sum.get("prev_elo") is None:
        return None, False
    d = round(new_sum["elo"] - new_sum["prev_elo"], 1)
    # 显著性：被替换掉的那次测量（一个点）是否落在本次 95% 区间之外。
    sig = new_sum["lo"] > new_sum["prev_elo"] or new_sum["hi"] < new_sum["prev_elo"]
    return d, sig


def _elo_watch(proc, rundir, plan):
    """后台看守：等 gk.exe 结束 → 校验新文件 → 原子替换 → 落盘结果。"""
    t0 = time.time()
    rc = proc.wait()
    elapsed = round(time.time() - t0, 1)
    jpath = os.path.join(CXX_DIR, rundir, "elo.json")
    npath = jpath + ".new"
    st = {
        "stage": "done", "ok": rc == 0, "rc": rc, "dir": rundir,
        "gens": plan["gens"], "anchor": plan["anchor"], "games": plan["games"],
        "pairs": plan["pairs"], "pairs_done": _elo_pairs_from_log(),
        "started": _elo_read_state().get("started"), "elapsed": elapsed,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eta": 0, "cmd": plan["cmd"],
    }
    if rc != 0:
        st["error"] = "gk.exe 退出码 %d（日志：checkpoints/history/elo_job.log）" % rc
        _elo_write_state(st)
        _ELO_PROC["proc"] = None
        return
    # 【2026-10-31 改：边库模式】gk.exe elo 现在自己完成整套动作：
    #   打新边 → 并入 <run>/elo_edges.jsonl → 用**全部边**重新全局拟合 → 直接写 elo.json。
    # 所以看板**不再**做「平移合并 + 原子替换」：
    #   · 旧做法的前提是「新文件只含本次那几个代」，而现在 elo.json 永远是全量曲线；
    #   · 平移对齐本身是误差来源（新旧差值散在 +60~+218，不存在常数偏移），
    #     全局拟合把这件事彻底消掉了。
    # 这里只做两件事：校验 elo.json 可解析、拿「跑之前」的快照算差值。
    old = _ELO_PROC.get("old") or {}
    try:
        with open(jpath, encoding="utf-8") as fh:
            probe = json.load(fh)
        if not (isinstance(probe, dict) and probe.get("versions")):
            raise ValueError("elo.json 里没有 versions")
    except (OSError, ValueError) as e:
        st["ok"] = False
        st["error"] = "跑完后的 elo.json 不可用（{0}）".format(e)
        _elo_write_state(st)
        _ELO_PROC["proc"] = None
        return
    new = probe
    st["merged_versions"] = len(new.get("versions") or [])
    st["merge_note"] = ("边库模式：新边已并入 elo_edges.jsonl 并全量重拟合，"
                       "看板不再做平移合并")
    st["summary"] = _elo_summary(old, new)
    # 差值只用"同一代、替换前后"比：summary.prev_elo 就是旧曲线里这一代的值。
    st["delta"], st["delta_sig"] = _elo_diff(st["summary"])
    _elo_write_state(st)
    _ELO_PROC["proc"] = None


def elo_stop():
    """停掉正在跑的 Elo 对战（只杀我们自己起的这一个进程，不碰训练）。"""
    p = _ELO_PROC.get("proc")
    if p is None or p.poll() is not None:
        return {"ok": True, "note": "没有在跑的 Elo 对战"}
    try:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       capture_output=True, timeout=20)
    except Exception as e:                          # noqa: BLE001
        return {"ok": False, "errors": ["停止失败：{0}".format(e)]}
    st = _elo_read_state()
    st.update({"stage": "stopped", "ok": False, "error": "已被手动停止",
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    _elo_write_state(st)
    _ELO_PROC["proc"] = None
    return {"ok": True, "note": "已停止（未打完的对局不会进边库；已打完的边已落盘，"
                               "曲线是完整的）"}


def _cpp_peifang():
    """C++\\peifang 里的配方档案（配方01-*.json …），只取看板要展示的字段。"""
    d = os.path.join(CXX_DIR, "peifang")
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({
            "file": name, "id": j.get("id"), "name": j.get("name"),
            "status": j.get("status"), "created": j.get("created"),
            "hypothesis": j.get("hypothesis"),
            "args": j.get("args") or {}, "lesson": j.get("lesson") or [],
            "changed": j.get("changed_vs_recipe_1") or {},
        })
    return out


def cpp_state(rundir=None):
    """C++ 引擎的全量快照：给网页里「版本 = C++」的那套面板用。"""
    runs = _cpp_run_dirs()
    pids = cpp_pids()
    running = bool(pids)
    if rundir and rundir not in runs:
        rundir = None
    if not rundir and runs:
        best = None
        for rd in runs:
            _e, mtime = _cpp_read_log(rd, tail=1)
            d = os.path.join(CXX_DIR, rd)
            dtime = os.path.getmtime(d) if os.path.isdir(d) else 0
            score = (mtime, dtime)
            if best is None or score > best[0]:
                best = (score, rd)
        rundir = best[1] if best else None

    if not rundir:
        return {"ok": True, "running": running, "pids": pids, "runs": [],
                "dir": None, "last": None, "recent": [], "gens": [], "history": [],
                "next_gen": 1, "totals": {}, "phase": _cpp_phase("", running),
                "tail": [], "runs_detail": _cpp_runs_detail(), "best": None,
                "elo": None, "elo_job": elo_status(), "elo_defaults": {"anchor": 0, "gens": []}}

    entries, _mtime = _cpp_read_log(rundir, tail=2000)
    gens = _cpp_gen_files(rundir)
    totals = {
        "gens": len(entries),
        "samples": sum(int(e.get("samples") or 0) for e in entries),
        "games": sum(int(e.get("games") or 0) for e in entries),
        "steps": sum(int(e.get("steps") or 0) for e in entries),
        "selfplay_s": round(sum(float(e.get("selfplay_s") or 0) for e in entries), 1),
        "train_s": round(sum(float(e.get("train_s") or 0) for e in entries), 1),
    }
    if entries:
        totals["avg_gen_s"] = round(
            (totals["selfplay_s"] + totals["train_s"]) / len(entries), 1)
    best = None
    bpath = os.path.join(CXX_DIR, rundir, "best.bin")
    if os.path.isfile(bpath):
        best = {"path": rundir + "/best.bin",
                "mtime": os.path.getmtime(bpath),
                "mb": round(os.path.getsize(bpath) / 1048576.0, 2),
                "gen": (entries[-1].get("gen") if entries else None)}
    return {
        "ok": True,
        "running": running,
        "pids": pids,
        "runs": runs,
        "dir": rundir,
        "last": entries[-1] if entries else None,
        "recent": entries[-6:],
        "history": entries,
        "gens": gens,
        "next_gen": (gens[-1] + 1) if gens else 1,
        "totals": totals,
        "phase": _cpp_phase(rundir, running),
        "tail": _cpp_stdout_tail(rundir, 8),
        "runs_detail": _cpp_runs_detail(),
        "best": best,
        "elo": _cpp_elo(rundir),
        "elo_job": elo_status(),
        "elo_defaults": elo_defaults(rundir),
        "peifang": _cpp_peifang(),
    }


def start_train_cpp(p, dry_run=False):
    """用看板的表单参数启动 gk.exe loop（C++ 引擎）。只映射真正同义的字段。"""
    errs = []
    # 默认续训「主线」目录（run.json 里 role 含「主线」的那个），而不是一个可能不存在的新名字。
    # 旧默认 model/run 在磁盘上并不存在 ⇒ 一点「开始训练」就静默从随机权重开训。
    rundir = str(p.get("CppDir") or "").strip().replace("\\", "/") or cpp_recommended_dir()
    if not CPP_DIR_RE.fullmatch(rundir):
        errs.append("C++ 运行目录必须形如 model/名字（不能写到 C++\\model 之外）")
    if not os.path.isfile(GK_EXE):
        errs.append("找不到 {0}，先构建 C++（C++\\build.bat）".format(GK_EXE))

    full = os.path.join(CXX_DIR, rundir)
    try:
        _files = os.listdir(full) if os.path.isdir(full) else []
    except OSError:
        _files = []
    has_gen = any(f.startswith("gen_") and f.endswith(".bin") for f in _files)
    has_best = os.path.isfile(os.path.join(full, "best.bin"))
    scratch = not (has_gen or has_best)

    def _int(name, lo, hi, dflt):
        try:
            v = int(p.get(name, dflt))
        except Exception:
            errs.append("{0} 必须是整数".format(name))
            return dflt
        if not (lo <= v <= hi):
            errs.append("{0} 超出范围 [{1}, {2}]：{3}".format(name, lo, hi, v))
            return dflt
        return v

    gens = _int("GensPerRound", 1, 500, 10)
    games = _int("Games", 1, 2000, 320)
    workers = _int("Workers", 1, 64, 16)
    steps = _int("MinSteps", 1, 200000, 160)
    egames = _int("EvalGames", 1, 2000, 20)
    eevery = _int("EvalEvery", 1, 100, 2)

    # 【2026-10-26 加】C++ 面板专属字段（对应 gk.exe 自己的开关）。
    # 以前只映射 6 个 int，用户在 C++ 面板里改模拟数/学习率/评估批都是**静默丢弃**的。
    OPT = [
        ("Sims", "--sims", int, 1, 100000),
        ("BatchMcts", "--batch-mcts", int, 1, 1024),
        ("Batch", "--batch", int, 1, 8192),
        ("LR", "--lr", float, 1e-6, 1.0),
        # 【2026-10-28 加】配方 2/3 引入的三个开关 + 配方编号（写进 train_log 供看板标断点）
        ("LrGenDecay", "--lr-gen-decay", float, 0.0, 1.0),
        ("LrFloor", "--lr-floor", float, 0.0, 1.0),
        ("OpeningPlies", "--opening-plies", int, 0, 40),
        ("GateThreshold", "--gate-threshold", float, 0.0, 1.0),
        ("Recipe", "--recipe", int, 0, 99),
        ("Fresh", "--fresh-ratio", float, 0.0, 1.0),
        ("EvalBatch", "--eval-batch", int, 1, 4096),
        ("EvalThreads", "--eval-threads", int, 1, 32),
        ("HalfLife", "--half-life", float, 1.0, 1000.0),
        ("Procs", "--procs", int, 0, 64),
        ("KeepGens", "--keep-gens", int, 0, 100000),
        ("KeepBufferGens", "--keep-buffer-gens", int, 0, 100000),
        ("BufferGens", "--buffer-gens", int, 1, 10000),
        ("HeuristicWeight", "--hw", float, 0.0, 100.0),
        ("PolicyWeight", "--pw", float, 0.0, 100.0),
        ("Seed", "--seed", int, 0, 2 ** 31 - 1),
    ]
    opt_args = []
    for name, flag, typ, lo, hi in OPT:
        raw = p.get(name)
        if raw is None or str(raw).strip() == "":
            continue                      # 没填 = 用 gk.exe 的默认，不传
        try:
            v = typ(float(raw)) if typ is int else typ(raw)
        except Exception:
            errs.append("{0} 必须是数字".format(name))
            continue
        if not (lo <= v <= hi):
            errs.append("{0} 超出范围 [{1}, {2}]：{3}".format(name, lo, hi, v))
            continue
        opt_args.extend([flag, ("%d" % v) if typ is int else ("%g" % v)])
    if errs:
        return {"ok": False, "errors": errs}
    # 防「静默从零开训」：目录里既没有 gen_*.bin 也没有 best.bin 时，gk.exe 会用**随机权重**
    # 起步（相当于把已经练了十几代的东西扔掉）。这种操作必须显式确认。
    if scratch and not dry_run and not bool(p.get("confirm_new")):
        return {"ok": False, "error_kind": "scratch",
                "errors": [
                    "目录 {0} 里没有 gen_*.bin / best.bin：这会用**随机权重从零开始**训练，"
                    "而不是接着主线练。要续训主线请选 {1}；确实要从零开训请确认。".format(
                        rundir, cpp_recommended_dir())],
                "dir": rundir,
                "recommended": cpp_recommended_dir()}
    if cpp_pids() and not dry_run:
        return {"ok": False, "errors": ["已有 C++ 训练在跑（PID {0}），先停止再启动".format(cpp_pids())]}

    args = [GK_EXE, "loop", "--dir", rundir, "--gens", str(gens),
            "--games", str(games), "--workers", str(workers),
            "--min-steps", str(steps), "--eval-games", str(egames),
            "--eval-every", str(eevery)] + opt_args + ["--nice"]
    extra = str(p.get("ExtraArgs") or "").strip()
    # ExtraArgs 是给 train.py 的开关，语义不同；C++ 只接受它自己认识的 --key value。
    # 为了不把一个引擎的参数偷偷塞给另一个引擎，这里只在显式是 --xxx 形状时透传。
    if extra and extra.startswith("--"):
        args.extend(extra.split())

    if dry_run:
        return {"ok": True, "dry_run": True, "engine": "cpp", "dir": rundir,
                "scratch": scratch,
                "recommended": cpp_recommended_dir(),
                "cmd": " ".join('"{0}"'.format(a) if " " in a else a for a in args),
                "note": ("只预览，没有启动" if not scratch else
                         "只预览：该目录里没有权重，真启动会从随机权重开始")}

    logdir = os.path.join(CXX_DIR, rundir)
    os.makedirs(logdir, exist_ok=True)
    outpath = os.path.join(logdir, "loop_stdout.log")
    # gk.exe 依赖 torch 的 DLL：不把 torch\lib 放进 PATH 会**静默**退出（0xC0000135，
    # 日志里一个字都没有），所以这里必须显式继承一份带 PATH 的环境。
    env = dict(os.environ)
    env["PATH"] = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib") \
        + os.pathsep + env.get("PATH", "")
    try:
        with open(outpath, "ab") as fh:
            fh.write("\n=== {0} start from dashboard ===\n".format(_now()).encode("utf-8"))
            fh.write((" ".join(args) + "\n").encode("utf-8"))
            fh.flush()
            proc = subprocess.Popen(args, cwd=CXX_DIR, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    creationflags=0x00000008 | 0x00000200)
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "errors": ["启动失败: {0}".format(exc)]}
    st = cpp_state()
    return {"ok": True, "pid": proc.pid, "engine": "cpp", "dir": rundir,
            "gens": gens, "detached": True, "next_gen": st.get("next_gen"),
            "note": "C++ 引擎已启动（{0} 代，日志 {1}）".format(gens, outpath)}


def _play_url():
    return "http://127.0.0.1:{0}/".format(PLAY_PORT)


def _port_alive(port, host="127.0.0.1", timeout=0.4):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def play_status(start=False):
    """返回对弈页状态；start=True 时若没在跑就拉起一个独立进程并等它就绪。"""
    if _port_alive(PLAY_PORT):
        return {"ok": True, "url": _play_url(), "running": True, "started": False}
    if not start:
        return {"ok": True, "url": _play_url(), "running": False, "started": False}
    script = os.path.join(ROOT, "web_play.py")
    if not os.path.isfile(script):
        return {"ok": False, "url": _play_url(), "running": False,
                "errors": ["找不到 web_play.py: {0}".format(script)]}
    try:
        # DETACHED_PROCESS：对弈页的寿命不跟着看板/本会话走
        subprocess.Popen([sys.executable, "-u", script, "--port", str(PLAY_PORT)],
                         cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=0x00000008 | 0x00000200)
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "url": _play_url(), "running": False,
                "errors": ["启动失败: {0}".format(exc)]}
    deadline = time.time() + 60.0          # 首次加载 torch 要十几秒
    while time.time() < deadline:
        if _port_alive(PLAY_PORT):
            return {"ok": True, "url": _play_url(), "running": True, "started": True}
        time.sleep(0.5)
    return {"ok": False, "url": _play_url(), "running": False,
            "errors": ["60 秒内没起来（看 checkpoints/history 或单独跑 web_play.py 看日志）"]}


# 【2026-10-14 加】/api/state 的快照文件。由 watch_progress.ps1 的 -Json 参数每轮原子重写，
# 内容是页面用的**同一个 $s 对象**（见那边主循环的注释）。
# 这里只负责"读文件 + 发出去"，绝不自己算口径 —— 否则就有了第二套数字。
STATE_JSON = os.path.join(ROOT, "checkpoints", "dashboard_state.json")
SSE_MAX_SEC = 1800          # 单条 SSE 连接最长 30 分钟，到点让浏览器自动重连（防资源泄漏）
# 快照超过这么久没更新就报 stale=true（看板正常是 3～5 秒一轮；90 秒足够宽松，
# 不会因为一次门控/落盘的卡顿而误报，但能抓住"写入进程已经死了"）
STATE_STALE_SEC = 90

LAST_PARAMS = os.path.join(ROOT, "checkpoints", "last_train_params.json")
LAUNCH_LOG = os.path.join(ROOT, "checkpoints", "history", "dashboard_launch.log")
STOP_TMP = os.path.join(ROOT, "checkpoints", "history", "dashboard_stop.json")
# 训练是否脱离调用者进程树启动（默认开）。排障时可用 FLASH_NODETACH=1 退回旧行为。
DETACH_TRAIN = os.environ.get("FLASH_NODETACH", "") not in ("1", "true", "yes")

# 展示/传输用的字段定义：名 -> (flag, 类型, 最小, 最大, 默认)
NUM_FIELDS = [
    ("Workers",     "-Workers",     "int", 1, 64,     16),
    ("Games",       "-Games",       "int", 1, 2000,   80),
    ("MinSteps",    "-MinSteps",    "int", 1, 200000, 160),
    ("EvalGames",   "-EvalGames",   "int", 1, 2000,   20),
    ("EvalEvery",   "-EvalEvery",   "int", 1, 100,    2),
    ("EvalWorkers", "-EvalWorkers", "int", 1, 64,     8),
    ("GensPerRound", "-GensPerRound", "int", 1, 500,  15),
    ("StartTarget", "-StartTarget", "int", 1, 100000, 0),   # 0 = 自动（当前代数 + GensPerRound）
    ("RulerGames",  "-RulerGames",  "int", 1, 2000,   200),
    ("Seed",        "-Seed",        "int", 0, 2 ** 31, 12345),
]
STR_FIELDS = [
    ("RunName", "-RunName", r"[A-Za-z0-9_.\-]{1,32}", "V2"),
    ("SaveDir", "-SaveDir", r"models/[A-Za-z0-9_.\-]{1,40}", "models/V2"),
    # ---- 【2026-10-13 加】透传给 train.py 的额外参数（四级透传：看板 → 本服务 → start_train.ps1 → auto_train.ps1 → train.py）----
    # 为什么需要：train.py 的新开关（--aux-heads / --buffer-decay-half-life / --buffer-entropy-weight）
    # 没有具名字段；没有这一格，「开始训练」按钮就永远用不上新开关（静默退回旧行为）。
    # 字符白名单：只允许字母/数字/下划线/短横/=/,/./空格 —— 挡住引号/反引号/$( )/; | & 这类注入字符。
    ("ExtraArgs", "-ExtraArgs", r"[A-Za-z0-9_\-=,./ ]{0,200}", ""),
]
# ---- 【2026-10-12 加】V3 配方用的可选高级参数 ----
# 哨兵值：**负数 = 不传该参数**（保持旧行为），0 是合法值（例如 hw=0 = 纯网络先验）。
# 这就是为什么不能用 0 当"未设置"。范围里的下界写 -1 允许"不传"。
# num: (名, flag, 最小, 最大, 默认)   bool: (名, flag, 默认)
# ⚠️ 第三项 `typ` 必须正确：分数权重（hw 可以是 0.5）用 **float**，
#    写成 int 会把 0.5 直接截成 0 —— 那正好等于"关掉规则先验"的失败配方。
#    2026-10-12 实测踩到：看板启动出的命令行显示 `-HeuristicWeight 0`（传的是 0.5）。
EXTRA_NUM_FIELDS = [
    ("BufferGens",      "-BufferGens",      "int",   -1, 500,  -1),   # >0 才传：回放池保留代数
    ("HeuristicWeight", "-HeuristicWeight", "float", -1, 20,   -1),   # >=0 才传：固定规则先验权重
    ("PolicyWeight",    "-PolicyWeight",    "float", -1, 20,   -1),   # >=0 才传：固定策略先验权重
]
EXTRA_BOOL_FIELDS = [
    ("NoAnneal",       "-NoAnneal",       False),   # 关退火：全程用固定 hw/pw
    ("AllowOverwrite", "-AllowOverwrite", False),   # 冷启动必须：train.py 默认拒绝往已有产物的目录开新训练
]

# ================================================================ 【2026-10-13 加】V4 一键配方
# 为什么要有它（三条都不是"图省事"）：
#   ① 配方一共 18 个字段，手动填容易漏一项 ⇒ 漏了就等于"同时动了两个变量"，
#      出问题无法归因（NOTES §6.3「一次只动一项」）。
#   ② 「先继承源 run 的权重 + 回放池 + 优化器状态」这一步**必须发生在启动之前**：
#      忘了它，train.py 会因为目标目录里没有 resume.pt 而走 `--fresh`（冷启动），
#      那正是 V3 第一次尝试崩坏（−478 Elo）的头号原因。
#   ③ 「填入配方」与「真启动」必须是同一份定义 ⇒ 配方只写在这里一处，
#      网页按钮从 /api/v4-preset 取（两处各写一份必然会漂移）。
#
# V4 = V3 的实配 + **只有一根轴**：回放池最近性 `--buffer-decay-half-life 20`
#   （代权重 = 2 ** (-代龄/20) ⇒ 60 代池里最近 20 代占 ~55%，均匀本应 32%）。
#   为什么先动这一根而不是辅助头：真实 MCTS 策略目标很尖（gen_99 实测熵 p50=0.12），
#   熵轴杠杆弱（β=1 只有 1.42 倍权重跨度）；最近性直接改变"新旧数据谁被训练到"，
#   而本项目的瓶颈是数据量（见 NOTES.md 任务 4 一节）。
V4_FROM_DIR = "models/V3"
# 写死代数（而不是 0=取最新）是刻意的：源 run 又训了新代时要让审计一眼看见，
# 而不是让 V4 的起点悄悄跟着漂。
V4_FROM_GEN = 110
V4_PRESET = {
    "RunName": "V4", "SaveDir": "models/V4",
    "Workers": 16, "Games": 160, "MinSteps": 320,
    "EvalGames": 20, "EvalEvery": 2, "EvalWorkers": 8,
    "GensPerRound": 30, "StartTarget": 0, "RulerGames": 200, "Seed": 12345,
    "AutoEval": True,
    "BufferGens": 60, "HeuristicWeight": 0.5, "PolicyWeight": 1.0,
    "NoAnneal": True, "AllowOverwrite": True,
    "ExtraArgs": "--buffer-decay-half-life 20",
}
V4_AXIS = ("唯一变量 = 回放池最近性 --buffer-decay-half-life 20；"
           "其余每一项都照抄 V3 的实配（160 局/320 步、20 局门控每 2 代、"
           "池 60 代、hw 0.5/pw 1.0、不退火、每轮 30 代 + 200 局直测）")
PREPARE_LOG = os.path.join(ROOT, "checkpoints", "history", "dashboard_prepare.log")
PREPARE_LAST = os.path.join(ROOT, "checkpoints", "history", "dashboard_prepare_last.log")
PREPARE_TIMEOUT = 1800      # 秒。V3 的 resume.pt 有 1.29 GB ⇒ 拷贝 + torch.load + torch.save
# 【危险动作必须显式确认，2026-10-13 事故后加】真启动 V4 要求 body 里带这个令牌。
# 事故：一次 `{"dry_run":true}` 到服务端时 Content-Length 缺失/为 0，旧代码
# `... or "{}"` 把它兜底成空对象 ⇒ `bool({}.get("dry_run"))` = False ⇒ **真启动了一次训练**。
# 教训：不要把"字段缺省"当成"用户想真启动"。缺省一律 400，真启动还要令牌。
V4_CONFIRM = "开始 V4"
# 防重复点击：prepare 阶段训练**还没起来**，training_running() 挡不住第二次点击，
# 而两个 prepare_run 同时往同一个 resume.pt 里写会把它写坏 ⇒ 必须自己上锁。
_V4_LOCK = threading.Lock()
_V4_STATE = {"stage": "idle", "note": "", "at": ""}


# ---------------------------------------------------------------- 参数推断
def _strip_quotes(s):
    return s.strip().strip("'\"")


def parse_ps_args(cmdline):
    """从 auto_train.ps1 的命令行里抽出 -Name Value 形式的参数。"""
    out = {}
    for name, flag, *_ in NUM_FIELDS + STR_FIELDS:
        m = re.search(re.escape(flag) + r"\s+(\S+)", cmdline)
        if m:
            out[name] = _strip_quotes(m.group(1))
    if re.search(r"-AutoEval\b", cmdline):
        out["AutoEval"] = True
    # 【2026-10-13 加】-ExtraArgs 的值可能**含空格**（如 '--buffer-decay-half-life 20'），
    # 上面那条 `\S+` 只能取到第一个词 ⇒ 这里按"带引号优先、否则取到下一个 -开关 为止"重取整串。
    m = re.search(r"-ExtraArgs\s+(?:'([^']*)'|\"([^\"]*)\"|(.+?)(?=\s+-[A-Za-z]|$))", cmdline)
    if m:
        out["ExtraArgs"] = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    return out


def find_supervisor():
    """返回正在跑的监督器命令行（用真实调用形状匹配，别用裸子串）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
             "'-File\\s+\\S*auto_train\\.ps1' } | Select-Object -First 1 -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=20)
        txt = (out.stdout or "").strip()
        return txt or None
    except Exception:
        return None


def max_gen(save_dir):
    d = os.path.join(ROOT, save_dir)
    best = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            m = re.fullmatch(r"gen_(\d+)\.pt", f)
            if m:
                best = max(best, int(m.group(1)))
    return best


def current_params():
    """优先用"正在跑的监督器参数"；否则用上次网页启动时存下的；否则给默认值。"""
    src = "默认值"
    p = {}
    cmd = find_supervisor()
    if cmd:
        p = parse_ps_args(cmd)
        src = "运行中的训练"
    if not p and os.path.exists(LAST_PARAMS):
        try:
            with open(LAST_PARAMS, encoding="utf-8") as fh:
                p = json.load(fh).get("params", {})
            src = "上次启动"
        except Exception:
            p = {}
    base = {}
    for name, flag, typ, lo, hi, dflt in NUM_FIELDS + EXTRA_NUM_FIELDS:
        base[name] = dflt
    for name, flag, pat, dflt in STR_FIELDS:
        base[name] = dflt
    for name, flag, dflt in EXTRA_BOOL_FIELDS:
        base[name] = dflt
    base["AutoEval"] = True
    base.update({k: v for k, v in p.items() if v is not None})
    # 数值字段按各自的类型归一（float 字段必须走 float，否则 0.5 会被截成 0）
    for name, flag, typ, lo, hi, dflt in NUM_FIELDS + EXTRA_NUM_FIELDS:
        try:
            base[name] = float(base[name]) if typ == "float" else int(base[name])
        except Exception:
            base[name] = dflt
    for name, flag, dflt in EXTRA_BOOL_FIELDS:
        base[name] = bool(base.get(name, dflt))
    base["AutoEval"] = bool(base.get("AutoEval", True))
    save_dir = base.get("SaveDir", "models/V2")
    if not re.fullmatch(r"models/[A-Za-z0-9_.\-]{1,40}", str(save_dir)):
        save_dir = "models/V2"
        base["SaveDir"] = save_dir
    base["current_gen"] = max_gen(save_dir)
    base["suggested_start_target"] = base["current_gen"] + int(base.get("GensPerRound") or 15)
    # 【不要把"0=自动"当成"0=填个数"】StartTarget=0 是"自动"的哨兵值。
    # 2026-10-12 修（用户报"网页点开始训练没有用"）：
    #   旧代码把 0 换成"上次启动时算出来的 suggested_start_target"，于是表单里长期躺着
    #   一个**过期的**数字；训练早跑过那个代数之后再点「开始训练」，就会被
    #   start_train.ps1 拒掉（dashboard_launch.log 里连点 4 次全是同一个错）。
    #   现在：规则是"目标代数必须 > 当前最新代数"，不满足就退化成 0（=自动），
    #   并把**自动会算出的那个数**告诉网页 ⇒ 用户看到的永远是能用的值。
    try:
        st = int(base.get("StartTarget") or 0)
    except Exception:
        st = 0
    base["start_target_effective"] = st if st > base["current_gen"] else base["suggested_start_target"]
    if st and st <= base["current_gen"]:
        base["StartTarget"] = 0
        base["start_target_stale"] = True
    base["source"] = src
    return base


# ---------------------------------------------------------------- 校验与启动
def validate(p):
    errs = []
    clean = {}
    for name, flag, pat, dflt in STR_FIELDS:
        v = str(p.get(name, "")).strip()
        if not re.fullmatch(pat, v):
            errs.append(f"{name} 非法：必须是 {pat} 形状（收到 {v!r}）")
        clean[name] = v
    # StartTarget = 0 / 空 ⇒ 视为"自动"：当前最新代数 + GensPerRound。
    # （提示里承诺了这件事，所以必须在范围检查**之前**替换掉，否则会被判越界。）
    try:
        if int(p.get("StartTarget", 0) or 0) == 0:
            gpr = int(p.get("GensPerRound", 15) or 15)
            sd = str(p.get("SaveDir", "") or "models/V2")
            if not re.fullmatch(r"models/[A-Za-z0-9_.\-]{1,40}", sd):
                sd = "models/V2"
            p = dict(p)
            p["StartTarget"] = max_gen(sd) + gpr
    except Exception:
        pass
    for name, flag, typ, lo, hi, dflt in NUM_FIELDS + EXTRA_NUM_FIELDS:
        try:
            # float 字段必须走 float：用 int 会把 0.5 截成 0（= 关掉规则先验的那种配方）
            v = float(p.get(name, dflt)) if typ == "float" else int(p.get(name, dflt))
        except Exception:
            errs.append(f"{name} 必须是{'小数' if typ == 'float' else '整数'}")
            continue
        if not (lo <= v <= hi):
            errs.append(f"{name} 超出范围 [{lo}, {hi}]：{v}")
            continue
        clean[name] = v
    for name, flag, dflt in EXTRA_BOOL_FIELDS:
        clean[name] = bool(p.get(name, dflt))
    clean["AutoEval"] = bool(p.get("AutoEval", True))
    # 明确禁止：pilot/big 与 models/V1 是同一个目录（junction），写进去会污染 V1 那条线
    if clean.get("SaveDir") not in (None, "") and clean["SaveDir"].rstrip("/").lower() in (
            "checkpoints/pilot/big", "models/v1"):
        errs.append("SaveDir 指向 V1 所在目录（junction 陷阱），拒绝启动")
    return clean, errs


def training_running():
    """是否已有训练在跑（监督器 / train.py / C++ 的 gk.exe loop）。"""
    if find_supervisor():
        return True
    if cpp_pids():
        return True
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
             "'-u\\s+train\\.py' }).Count"],
            capture_output=True, text=True, timeout=20)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:
        return False


def start_train(p, dry_run=False):
    clean, errs = validate(p)
    if errs:
        return {"ok": False, "errors": errs}
    # 同 start_v4：dry_run 只是预览命令行，训练期间也必须能用（否则门禁在训练期间跑不了）
    if training_running() and not dry_run:
        return {"ok": False, "errors": ["已有训练在跑，先停止再启动"]}

    # 【第二道防线 · 2026-10-12 加】网页填表期间训练可能已经把代数推过去了
    # （表单里的目标代数是在**上次打开页面**时算的那个数）。这里按**此刻**的
    # 最新代数重判一次：过期就明确告诉用户"留空即自动"，而不是抛一句干巴巴的
    # ps1 报错（那正是"点按钮没用"的现场）。
    sd = clean["SaveDir"]
    cur = max_gen(sd)
    if clean["StartTarget"] <= cur:
        auto = cur + clean["GensPerRound"]
        return {"ok": False, "errors": [
            f"本轮目标代数 {clean['StartTarget']} 已不大于当前最新 gen {cur}，没有可训的代数。"
            f"请把「本轮目标代数」留空（= 自动取 {cur} + 每轮代数 "
            f"{clean['GensPerRound']} = {auto}）或改成更大的值后重试。"],
            "error_kind": "stale_start_target",
            "current_gen": cur,
            "auto_start_target": auto,
        }

    # 【为什么交给 PowerShell 起进程】从 Python 里 DETACHED_PROCESS 直接拉 pwsh，
    # 实测子进程会"写了锁和第一轮日志后静默死掉"；而 Start-Process 拉起的进程在本机一直稳定。
    args = ["pwsh", "-NoLogo", "-NoProfile", "-File", os.path.join(ROOT, "start_train.ps1"),
            "-RunName", clean["RunName"], "-SaveDir", clean["SaveDir"],
            "-Workers", str(clean["Workers"]), "-Games", str(clean["Games"]),
            "-MinSteps", str(clean["MinSteps"]), "-EvalGames", str(clean["EvalGames"]),
            "-EvalEvery", str(clean["EvalEvery"]), "-EvalWorkers", str(clean["EvalWorkers"]),
            "-GensPerRound", str(clean["GensPerRound"]), "-StartTarget", str(clean["StartTarget"]),
            "-RulerGames", str(clean["RulerGames"]), "-Seed", str(clean["Seed"])]
    # ---- V3 配方的高级参数：**哨兵值规则** —— 负数 = 不传（保持旧行为），0 是合法值 ----
    for name, flag, typ, lo, hi, dflt in EXTRA_NUM_FIELDS:
        v = clean[name]
        if v > lo:                      # lo 就是"不传"的哨兵（-1）⇒ 严格大于它才传
            args.append(flag)
            # 整数不带小数点、小数保留（0.5 不能被写成 0）
            args.append(str(int(v)) if float(v).is_integer() else str(v))
    for name, flag, dflt in EXTRA_BOOL_FIELDS:
        if clean[name]:
            args.append(flag)
    # 【2026-10-13 加】额外参数（透传给 train.py）。空串 = 不传，保持旧行为。
    _extra = str(clean.get("ExtraArgs") or "").strip()
    if _extra:
        args.append("-ExtraArgs")
        args.append(_extra)
    if clean["AutoEval"]:
        args.append("-AutoEval")
    # 【脱离调用者进程树】2026-10-12 加：训练必须活得比"起它的那个进程"更久
    # （看板服务、以及 DSH 会话本身）。-Detach 走 WMI 创建进程，
    # 父进程是 WMI 提供程序而不是本进程 ⇒ 上层被杀不会连带杀掉训练。
    # 【2026-10-13 加】-DryRun：只组装命令行并原样打印（不启动、不改任何状态）。
    # V4 一键按钮的自检与门禁都走它 —— 否则"验证按钮"就等于真的开一次训练。
    if dry_run:
        args.append("-DryRun")
    elif DETACH_TRAIN:
        args.append("-Detach")

    os.makedirs(os.path.dirname(LAUNCH_LOG), exist_ok=True)
    # 输出重定向到文件（不用管道，避免受限环境下的 EPERM）
    with open(LAUNCH_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n=== {0} start-train from dashboard ===\n".format(_now()))
        fh.write(" ".join(args) + "\n")
        fh.flush()
        subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=120)

    payload = {}
    last_json = ""
    try:
        with open(LAUNCH_LOG, encoding="utf-8", errors="replace") as fh:
            for line in fh.read().splitlines():
                if line.startswith("FLASHJSON "):
                    last_json = line[len("FLASHJSON "):]      # 只认**最后一行**（本次调用写的）
        if last_json:
            payload = json.loads(last_json)
    except Exception:
        payload = {}
    if not payload:
        return {"ok": False, "errors": ["start_train.ps1 未返回可解析结果，见 checkpoints/history/dashboard_launch.log"]}
    if payload.get("ok"):
        payload.setdefault("note", "已启动；看板顶部的「进度」卡会在几秒内显示当前阶段")
        if payload.get("detached"):
            payload["note"] = ("已**脱离本进程**启动（生命周期独立于看板/DSH 会话）；"
                               "看板顶部的「进度」卡会在几秒内显示当前阶段")
    return payload


def _now():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stop_train(keep_eval=False):
    os.makedirs(os.path.dirname(STOP_TMP), exist_ok=True)
    args = ["pwsh", "-NoLogo", "-NoProfile", "-File", os.path.join(ROOT, "stop_train.ps1"), "-Quiet"]
    # 输出重定向到文件而不是管道（管道在受限沙箱下会 EPERM）
    with open(STOP_TMP, "w", encoding="utf-8") as fh:
        subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    try:
        with open(STOP_TMP, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
    except Exception:
        txt = ""
    for line in txt.splitlines():
        if line.startswith("FLASHJSON "):
            try:
                return json.loads(line[len("FLASHJSON "):])
            except Exception:
                pass
    return {"ok": False, "errors": ["stop_train.ps1 未返回可解析结果"], "raw": txt[-500:]}


# ---------------------------------------------------------------- V4 一键启动
def prepare_run_v4(dry_run=False):
    """把 V3 的「权重 + 回放池 + 优化器状态」继承到 models/V4，并把代数计数清零。

    幂等：prepare_run.ps1 会自己取目标目录里最大的 gen_N.pt 当代数计数，重复跑安全
    （这也是它要求"每次启动前都跑一次"的原因 —— train.py 写进 resume.pt 的 gen 是
    "启动时的计数 + 已训代数"，重启不重置就会重训/覆盖已完成的代）。

    返回 (ok, 本次输出)。
    """
    args = ["pwsh", "-NoLogo", "-NoProfile", "-File", os.path.join(ROOT, "prepare_run.ps1"),
            "-Dir", V4_PRESET["SaveDir"], "-FromDir", V4_FROM_DIR,
            "-FromGen", str(V4_FROM_GEN)]
    if dry_run:
        args.append("-DryRun")
    os.makedirs(os.path.dirname(PREPARE_LOG), exist_ok=True)
    rc = None
    try:
        # 不用管道（受限沙箱下 capture_output 会 EPERM）：每次调用单独一个文件，
        # 读回后再追加进留档日志。
        with open(PREPARE_LAST, "w", encoding="utf-8") as tf:
            with open(PREPARE_LOG, "a", encoding="utf-8") as fh:
                fh.write("\n=== {0} prepare_run from dashboard{1} ===\n".format(
                    _now(), " (dry-run)" if dry_run else ""))
                fh.write(" ".join(args) + "\n")
                fh.flush()
                rc = subprocess.run(args, cwd=ROOT, stdout=tf, stderr=subprocess.STDOUT,
                                    timeout=PREPARE_TIMEOUT).returncode
    except Exception as e:
        return False, "prepare_run.ps1 调用失败：{0}".format(e)
    try:
        with open(PREPARE_LAST, encoding="utf-8", errors="replace") as tf:
            out = tf.read()
    except Exception:
        out = ""
    try:
        with open(PREPARE_LOG, "a", encoding="utf-8") as fh:
            fh.write(out)
    except Exception:
        pass
    return rc == 0, out.strip()


def v4_preset_info():
    """V4 一键配方的只读视图：按钮用它把"将要跑什么"显示给人看；门禁用它做无副作用校验。"""
    clean, errs = validate(dict(V4_PRESET))
    return {
        "ok": not errs,
        "errors": errs,
        "preset": dict(V4_PRESET),      # 原样：StartTarget=0 = 自动
        "effective": clean,             # 已把 StartTarget 换成"此刻算出来的"目标代数
        "axis": V4_AXIS,
        "from": {
            "dir": V4_FROM_DIR,
            "gen": V4_FROM_GEN,
            "weights": "{0}/gen_{1}.pt".format(V4_FROM_DIR, V4_FROM_GEN),
            "resume": "{0}/resume.pt".format(V4_FROM_DIR),
        },
        "prepare_required": True,
        "stage": dict(_V4_STATE),
    }


def start_v4(dry_run=False):
    """一键开 V4 = **先 prepare_run.ps1，再按 V4 配方启动**。

    两件事必须一起做：只启动不继承 ⇒ 走成冷启动（−478 那个坑）；
    只继承不启动 ⇒ 用户还得自己再点一次（且可能点成别的配方）。
    """
    # 【为什么 dry_run 不受"已有训练在跑"的限制】预览既不启动、也不写盘
    # （`prepare_run.ps1 -DryRun` 只打印计划），所以训练期间它也应该可用；否则
    # "训练中想确认这个按钮会下发什么"就做不到，门禁也无法在训练期间照常跑
    # （2026-10-13 实测：V4 一开跑，门禁里所有 dry_run 检查全部挂掉）。
    if training_running() and not dry_run:
        return {"ok": False, "errors": ["已有训练在跑，先点「■ 停止训练」再一键开 V4"],
                "stage": dict(_V4_STATE)}
    if not _V4_LOCK.acquire(blocking=False):
        return {"ok": False, "errors": ["V4 一键启动正在进行中，不要重复点击"],
                "stage": dict(_V4_STATE)}
    try:
        _V4_STATE.update(stage="preparing", at=_now(),
                         note="正在继承 {0} 的权重 + 回放池".format(V4_FROM_DIR))
        ok, out = prepare_run_v4(dry_run=dry_run)
        if not ok:
            _V4_STATE.update(stage="failed", at=_now(), note="prepare_run.ps1 失败")
            return {"ok": False, "stage": dict(_V4_STATE),
                    "errors": ["prepare_run.ps1 失败；详情见 checkpoints/history/dashboard_prepare.log"],
                    "prepare": out[-800:]}
        _V4_STATE.update(stage="starting", at=_now(), note="继承完成，正在启动训练")
        # 【为什么读 INIT.json 而不是回显子进程输出】pwsh 把中文写进重定向文件时用的是
        # 控制台 OEM 代码页（本机 GBK）⇒ 按 UTF-8 读回来是乱码。INIT.json 是
        # prepare_run.ps1 自己用 -Encoding utf8 写的结构化记录，正好是"这次继承了什么"的
        # 权威摘要（dashboard_launch.log 那边一直靠 FLASHJSON 这类纯 ASCII 行规避同一个问题）。
        prepared = ""
        try:
            with open(os.path.join(ROOT, V4_PRESET["SaveDir"], "INIT.json"),
                      encoding="utf-8") as fh:
                ini = json.load(fh)
            n = int(ini.get("counter_reset_to") or 0)
            prepared = "起点已建：{0} → 权重；{1}；代数计数重置为 {2}（下一代 = gen {3}）".format(
                ini.get("weights_from"),
                "回放池 + 优化器状态已继承" if ini.get("resume_inherited")
                else "⚠️ 没继承到回放池（等于冷启动）",
                n, n + 1)
        except Exception:
            prepared = ""
        res = start_train(dict(V4_PRESET), dry_run=dry_run)
        res["prepare"] = out[-800:]
        res["prepare_summary"] = prepared
        res["stage"] = ("dry-run" if dry_run else ("started" if res.get("ok") else "failed"))
        _V4_STATE.update(
            stage=res["stage"], at=_now(),
            note=("已启动 PID {0}，目标 gen {1}".format(res.get("pid"), res.get("start_target"))
                  if res.get("ok") and not dry_run else
                  ("dry-run：只预览了命令行，没有启动" if dry_run else "启动失败")))
        return res
    finally:
        _V4_LOCK.release()


# ---------------------------------------------------------------- HTTP
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FlashUI/1.0"

    def _send(self, data: bytes, ctype: str, cache: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _drain(self):
        """把不打算处理的请求体读干净。

        不读干净会让连接状态错位：客户端还在发 body，服务端已经回包并关连接
        ⇒ 客户端报 `ConnectionAbortedError: [WinError 10053]`（2026-10-13 实测）。
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except Exception:
            n = 0
        if n > 0:
            try:
                self.rfile.read(n)
            except Exception:
                pass

    def _read_body(self):
        """返回 (body_dict, None) 或 (None, 错误说明)。

        【为什么空 body 必须是 400 而不是默认 {}】见 V4_CONFIRM 那段注释：旧代码
        `json.loads(raw or "{}")` 把"body 没收到"变成了"用户没有指定 dry_run"，
        于是 `bool(body.get("dry_run"))` = False ⇒ 一次预览请求真的开了训练。
        """
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._drain()
            return None, f"只接受 application/json（收到 {ctype!r}）"
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except Exception:
            n = 0
        if n <= 0:
            self._drain()
            return None, "空请求体（缺 Content-Length）—— 为安全起见拒绝执行"
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return None, f"请求体解析失败：{e}"
        if not isinstance(body, dict):
            return None, "请求体必须是 JSON 对象"
        return body, None

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/api/start-train", "/api/stop-train", "/api/start-v4",
                        "/api/play/start", "/api/elo-run", "/api/elo-stop"):
            self._drain()
            self.send_error(404, "no such action")
            return
        body, err = self._read_body()
        if err is not None:
            # 只接受 JSON 体：跨站表单伪造不出这个 Content-Type，且我们不实现 OPTIONS 预检
            self._json({"ok": False, "errors": [err]},
                       415 if "application/json" in err else 400)
            return
        if path == "/api/play/start":
            # 只允许「拉起对弈页」这一个动作，不接受任何路径/端口覆盖
            self._json(play_status(start=True))
        elif path == "/api/start-train":
            # engine=cpp → 走 gk.exe loop；其它值（含缺省）→ 原来的 train.py 路径。
            # dry_run 对两边都只是"拼命令行给我看"，不启动任何东西。
            # 【2026-10-26 修】网页表单一律发大写 Engine（那是 select 的字段名），这里两个都认；
            # 只认小写会让「引擎=C++」被静默当成 Python 启动 —— 正是用户报的那个现象。
            eng = str(body.get("engine") or body.get("Engine") or "python").strip().lower()
            if eng == "cpp":
                self._json(start_train_cpp(body, dry_run=bool(body.get("dry_run"))))
            else:
                self._json(start_train(body, dry_run=bool(body.get("dry_run"))))
        elif path == "/api/start-v4":
            # 配方本体不接受网页覆盖 ⇒ 保证「按钮跑的东西」与「文档里写的配方」是同一份。
            # body 只认两个字段：dry_run（**必须显式布尔**）+ confirm（真启动的令牌）。
            dr = body.get("dry_run")
            if not isinstance(dr, bool):
                self._json({"ok": False, "errors": [
                    "必须显式带布尔字段 dry_run（true=只预览不启动，false=真启动）"]}, 400)
                return
            if not dr and body.get("confirm") != V4_CONFIRM:
                self._json({"ok": False, "errors": [
                    "真启动需要 confirm={0!r}（只预览请用 dry_run=true）".format(V4_CONFIRM)]}, 400)
                return
            self._json(start_v4(dry_run=dr))
        elif path == "/api/elo-run":
            # 打 Elo 对战。dry_run 必须显式布尔（照 V4 那条教训：危险动作不能靠字段缺省判断）。
            # 这里不额外要 confirm 令牌：它只读权重下棋 + 写 elo.json，不动任何训练状态。
            self._json(elo_start(body))
        elif path == "/api/elo-stop":
            self._json(elo_stop())
        else:
            res = stop_train(bool(body.get("keep_eval")))
            # C++ 引擎是独立进程，stop_train.ps1 只认 train.py 那一套 ⇒ 这里补一刀。
            killed = []
            for pid in cpp_pids():
                try:
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   capture_output=True, timeout=20)
                    killed.append(pid)
                except Exception:
                    pass
            if killed:
                res = dict(res or {})
                res["ok"] = True
                res.setdefault("killed", {})
                res["killed"]["C++ 训练"] = len(killed)
                res["cpp_killed"] = killed
            self._json(res)

    def _stream(self):
        """SSE：页面文件一被重写就推一条 event:changed，看板据此**原地**同步。

        为什么只推"变了"而不推内容：页面本身是 watch_progress.ps1 生成好的
        74 KB HTML，前端直接 fetch 它再按节点 patch 一次即可；再维护一套
        JSON 序列化只会多出一个会跟模板对不上的真相来源。
        为什么用它：轮询最快也要一个周期才动，而整页 meta refresh 会
        "画面消失再回来"——SSE 让延迟降到百毫秒级，且不重载任何东西。
        """
        self.protocol_version = "HTTP/1.1"
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except Exception:
            return

        def chunk(payload: bytes):
            self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
            self.wfile.flush()

        def stamp():
            page = os.path.getmtime(PAGE) if os.path.isfile(PAGE) else 0.0
            state = os.path.getmtime(STATE_JSON) if os.path.isfile(STATE_JSON) else 0.0
            return page, state

        t0 = time.time()
        last = None
        beat = 0.0
        try:
            chunk(b": connected\n\n")
            while time.time() - t0 < SSE_MAX_SEC:
                try:
                    cur = stamp()
                except OSError:
                    cur = (0.0, 0.0)
                now = time.time()
                if cur != last:
                    last = cur
                    body = json.dumps({"page": round(cur[0], 3), "state": round(cur[1], 3),
                                       "at": round(now, 3)})
                    chunk(("event: changed\ndata: " + body + "\n\n").encode("utf-8"))
                    beat = now
                elif now - beat > 15:
                    chunk(b": ping\n\n")   # 心跳：别让空闲连接被系统/中间层掐掉
                    beat = now
                time.sleep(0.25)
            chunk(b"")                     # 结束分块，浏览器会自动重连
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        except Exception:
            return

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/stream":
            self._stream()
            return

        if path == "/api/last-params":
            self._json(current_params())
            return

        if path == "/api/play/ping":
            self._json(play_status(start=False))
            return

        if path == "/api/cpp-state":
            _q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            _want = (_q.get("dir") or [""])[0].strip() or None
            self._json(cpp_state(_want))
            return

        if path == "/api/elo-status":
            # 「Elo 对战」的轮询口：只读，不启动任何东西（门禁可以随便打）
            self._json(elo_status())
            return

        if path == "/api/v4-preset":
            self._json(v4_preset_info())
            return

        if path == "/api/state":
            # 【2026-10-14 加】只读全量快照，给 RhineLabUI 的三维界面用。
            #
            # 只读文件、不算口径：dashboard_state.json 由 watch_progress.ps1 -Json
            # 每轮原子重写，内容就是页面用的同一个 $s 对象。
            # 为什么不在这里自己重算：那等于第二次实现"当前第几代/ETA/得分率"，
            # 两套迟早对不上，而且出错时无从判断谁对（2026-10-13 刚因为
            # ETA 跨 run 混排报错过一次）。
            try:
                with open(STATE_JSON, encoding="utf-8-sig") as fh:
                    snap = json.load(fh)
            except FileNotFoundError:
                # 文件还没有 = 看板进程没起来（或没带 -Json）。这是**正常状态**，
                # 不是错误：三维界面据此显示"看板离线"，而不是弹报错。
                self._json({"ok": False, "offline": True,
                            "error": "没有看板快照：watch_progress.ps1 未在带 -Json 运行",
                            "hint": "pwsh -NoProfile -File watch_progress.ps1 -Html progress_live.html "
                                    "-Json checkpoints/dashboard_state.json -Interval 3"})
                return
            except Exception as e:
                self._json({"ok": False, "offline": True,
                            "error": f"快照读取失败：{e}"})
                return

            # 新鲜度：快照的 at 是 PowerShell 的本地时间串，用文件修改时间更可靠
            age = max(0.0, time.time() - os.path.getmtime(STATE_JSON))
            if isinstance(snap, dict):
                snap["ok"] = True
                snap["offline"] = False
                snap["age_sec"] = round(age, 1)
                snap["stale"] = age > STATE_STALE_SEC
            self._json(snap)
            return

        if path in ALLOWED_PAGE:
            try:
                with open(PAGE, "rb") as fh:
                    data = fh.read()
            except FileNotFoundError:
                self.send_error(503, "progress_live.html not written yet")
                return
            # 页面必须每次取最新（它每 3 秒被重写）
            self._send(data, "text/html; charset=utf-8",
                       "no-store, no-cache, must-revalidate, max-age=0")
            return

        m = IMG_RE.match(path)
        if m:
            target = os.path.join(ROOT, "models", m.group(1), "progress.png")
            # 双保险：解析后的真实路径必须仍在工作区内
            if not os.path.abspath(target).startswith(os.path.abspath(ROOT)):
                self.send_error(403, "forbidden")
                return
            try:
                with open(target, "rb") as fh:
                    data = fh.read()
            except FileNotFoundError:
                self.send_error(404, "no progress.png for that run")
                return
            # 页面用 ?v=<图片修改时间> 做版本号，所以这里可以放心用强缓存：
            # 图没变就不会重复下载（看板每 3 秒整页刷新，否则会白拉 80KB/3s）
            self._send(data, "image/png", "public, max-age=86400")
            return

        self.send_error(404, "not served here")

    def log_message(self, *args):   # 别把每次刷新都打到日志里
        pass


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"看板地址 http://127.0.0.1:{PORT}/  （页面 {PAGE}）", flush=True)
    print(f"曲线图   http://127.0.0.1:{PORT}/models/V2/progress.png", flush=True)
    print(f"接口     GET /api/last-params  GET /api/v4-preset  GET /api/state  "
          f"POST /api/start-train  POST /api/start-v4  POST /api/stop-train", flush=True)
    srv.serve_forever()
