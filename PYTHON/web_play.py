"""
web_play.py
===========
本地**网页版**人机对战 + **KataGo Analysis 风格分析台**（只用 Python 标准库）。

为什么用它
----------
`play.py` 要在终端里敲坐标（`10 10`）。网页版是**点棋盘**，而且能顺手看到：
AI 的候选点与访问次数、网络对局面的胜率估值、走子记录，还能悔棋/换模型/调强度。

分析模式（2026-10-12 新增，用户指定的下一步）
--------------------------------------------
在同一个页面上把局面"摊开"看，两个模式共用一块棋盘：
* **全盘评估浮层**：每个合法点叠 访问次数（热度）+ 胜率（落子方视角）
  + 网络先验；一眼成五 / 必挡点用金圈、橙圈标出，最佳点蓝圈
* **候选排行**：Top10 的 访问 / 胜率 / 先验 / 相对最佳点的 Δ（KataGo 风格）
* **整盘胜率曲线**：逐手一个点，标出跌幅大的疑似坏棋，**点曲线可直接回到该局面**
* **摆棋/试下**：手动放/拿子后立刻重新分析（edit position）
* **逐手复盘**：点坐标 / 点曲线 / 悔棋都能回到任意局面重算

算法细节全部在 `analysis.py`（只读层，不碰 `mcts.py`/`heuristics.py`/`gomoku_env.py`）。
关键前提：`root_max_children` 是 MCTS 的**构造参数**，分析模式把它调大到 361
⇒ 根节点展开全部合法点，于是每个点都能拿到访问次数，而且**引擎指纹不变**。

用法
----
    python web_play.py                      # 默认 models/V2/gen_150.pt，端口 8770
    python web_play.py --model models/V3/gen_30.pt --n-sim 1600
    python web_play.py --port 8888 --open   # 自动打开浏览器
    python web_play.py --selftest           # 自检（含分析层 6 项），不开服务

然后浏览器访问 http://127.0.0.1:8770

实现要点
--------
* **树复用**：一局棋共用一个 MCTS 实例，每手 `set_root_move`，和 play.py 一样 ——
  比"每手重建搜索树"更强也更省。分析另开独立实例，不污染对局那棵树。
* 前端零依赖：HTML/CSS/JS 全部内联，离线可用。
* 后端线程安全：全局锁保证"一次只有一手 AI 搜索在跑"。
* **棋盘朝向**：`Board.board` 下标是 `[行=y, 列=x]`；前端表格第 0 行 = y18。
  改过任何渲染都要自己核对一遍（开发期的 UI 自动化脚本未随仓库分发）。
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gomoku_env import Board          # noqa: E402
from mcts import MCTS                 # noqa: E402
from play import load_model           # noqa: E402
import analysis                       # noqa: E402  （分析层：只读，不改引擎）

LETTERS = "ABCDEFGHIJKLMNOPQRS"
STAR = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)}

# 分析模式默认模拟次数：根全展开时访问会被摊薄，400 太少（每点 1 次），
# 800 起步才有得看；调更高更准但每次要等更久。
ANALYZE_SIMS_DEFAULT = 800

# --------------------------------------------------------------- 对战配置
# 【为什么必须用这两个数】train.py 的退火终点是 heuristic=0.5 / policy=1.0，
# 而 gen>=45 之后生产自对弈全程就是这个配置。对战要用**同一套权重**，
# 否则你面对的不是"gen_150 本尊"，而是"带规则拐杖"的另一个引擎
# （旧版本这里硬编码 6.0/0.3，那是 gen<5 的启动配置，会明显更强也更不像它）。
PROD_HW, PROD_PW = 0.5, 1.0
# 打开分析台时默认用**当前 champion 对应的模型**（2026-10-12 21:xx 起是 V3_g30）。
# 找不到就回落到 V2 的 gen_150 —— 不写死是怕"champion 换了而这里忘了改"。
DEFAULT_MODEL = "models/V3/gen_30.pt"
if not os.path.isfile(os.path.join(HERE, DEFAULT_MODEL)):
    DEFAULT_MODEL = "models/V2/gen_150.pt"


def discover_models():
    """列出可对战的模型：各 run 目录下的 .pt（新代在前）+ 卫冕 champion。"""
    cands = []
    root = os.path.join(HERE, "models")
    if os.path.isdir(root):
        for run in os.listdir(root):
            d = os.path.join(root, run)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".pt") and fn != "resume.pt":
                    cands.append("models/{}/{}".format(run, fn))
    champ = os.path.join(HERE, "checkpoints", "champions", "current.pt")
    if os.path.isfile(champ):
        cands.append("checkpoints/champions/current.pt")
    cands = list(dict.fromkeys(cands))

    def key(p):
        base = os.path.basename(p)
        m = re.search(r"gen_?(\d+)", base)
        gen = int(m.group(1)) if m else 0
        is_champ = 1 if "champions" in p else 0
        is_best = 1 if base == "best.pt" else 0
        # 默认模型永远第一；默认模型所在的 run 排在其他 run 前面；
        # champion/best 排在同代前面；然后按代数倒序
        default_run = os.path.dirname(DEFAULT_MODEL).replace(os.sep, "/") + "/"
        return (0 if p == DEFAULT_MODEL else 1,
                0 if p.replace(os.sep, "/").startswith(default_run) else 1,
                -is_champ, -is_best, -gen, p)

    cands.sort(key=key)
    return cands


# ------------------------------------------------- 引擎选择：Python 版 / C++ 版
# C++ 版（D:\DHSGZQ\Flashone\C++）不加载在 Python 进程里，而是起一个常驻子进程
# `gk.exe play --net X.bin`，用**单行文本协议**驱动（见 C++\src\main.cpp 的 cmd_play）：
#   GO <n_sim> <hw> <pw> <手数> <x1> <y1> ...  →  MOVE <x> <y> <胜率> <ms> <候选数> [...]
# 这样两套引擎共用同一个网页、同一个棋盘状态机；C++ 那边崩了也只影响它自己。
CXX_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "C++"))
CPP_EXE = os.path.join(CXX_DIR, "build", "gk.exe")
CPP_MODEL_ROOT = os.path.join(CXX_DIR, "model")


def discover_cpp_models():
    """列出 C++ 侧的 .bin 权重（C++/model/**），新代在前。"""
    cands = []
    if os.path.isdir(CPP_MODEL_ROOT):
        for dirpath, _dirs, files in os.walk(CPP_MODEL_ROOT):
            for fn in files:
                if not fn.endswith(".bin"):
                    continue
                full = os.path.join(dirpath, fn)
                if "buffer" in full.replace(os.sep, "/"):
                    continue
                rel = os.path.relpath(full, CXX_DIR).replace(os.sep, "/")
                if "broken" in rel:            # 坏掉的实验目录不进对战线
                    continue
                cands.append(rel)

    def key(p):
        base = os.path.basename(p)
        m = re.search(r"gen_?(\d+)", base)
        gen = int(m.group(1)) if m else 0
        try:
            mt = -os.path.getmtime(os.path.join(CXX_DIR, p))
        except OSError:
            mt = 0.0
        # 只在 run* 目录里挑默认权重：最近改动的 best.bin 排第一
        return (0 if "/run" in p else 1, 0 if base == "best.bin" else 1, mt, -gen, p)

    cands.sort(key=key)
    return cands[:60]


def _gk_spawn(mode, net, graph=False, exe=CPP_EXE):
    """起一个常驻 gk.exe（mode = "play" / "analyze"），一直读到 READY 为止。

    返回 (proc, "")；失败返回 (None, 原因)。CppEngine 与 CppAnalyzer 共用这一份，
    避免"只修了一处"（PATH 里的 torch lib 是硬前提：缺了会静默 0xC0000135）。
    """
    net_abs = net if os.path.isabs(net) else os.path.join(CXX_DIR, net)
    if not os.path.isfile(net_abs):
        return None, "找不到权重 {}".format(net_abs)
    if not os.path.isfile(exe):
        return None, "找不到 {}".format(exe)
    env = dict(os.environ)
    # gk.exe 依赖 torch 的 DLL
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
    if graph:
        # 分析台是常驻单进程、权重不变 ⇒ 正好绕开"C++ 每代重建 CUDA Graph 导致
        # 12 GB 卡 OOM"的坑（C++/HANDOFF.md 坑 17），实测白捡 ~7%。
        env["GK_GRAPH"] = "1"
    argv = [exe, mode, "--net", net_abs, "--nice"]
    # R3：把 C++ 早就有、但 Python 从来没传过的 --batch-mcts 接出来。
    # 默认不传（= config.h 的 16，行为与以前完全一致）；想试就设环境变量
    # GK_BATCH_MCTS=64 再起服务。实测见 _tmp/batch_sweep.py 的口径说明。
    _bm = os.environ.get("GK_BATCH_MCTS", "").strip()
    if _bm.isdigit() and int(_bm) > 0:
        argv += ["--batch-mcts", _bm]
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=CXX_DIR, env=env,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
    except Exception as e:                      # noqa: BLE001
        return None, "启动 gk.exe 失败: {}".format(e)
    line = ""
    # gk.exe 启动时可能先打印 [nice] 之类的一行，读到 READY 为止
    for _ in range(8):
        line = proc.stdout.readline()
        if not line:
            break
        if line.strip().startswith("READY"):
            return proc, ""
    try:
        proc.kill()
    except Exception:                           # noqa: BLE001
        pass
    return None, "gk.exe 未就绪: {}".format(line.strip()[:200])


class CppEngine:
    """gk.exe play 的客户端：一次 GO 一手，进程常驻（模型只加载一次）。"""

    def __init__(self, net, exe=CPP_EXE):
        self.net = net
        self.exe = exe
        self.proc = None
        self.lock = threading.Lock()
        self.last_error = ""

    def _ensure(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        proc, err = _gk_spawn("play", self.net, exe=self.exe)
        if proc is None:
            self.last_error = err
            return False
        self.proc = proc
        return True

    def warm(self):
        """后台预热：拉起进程，并空盘先跑一条 GO。

        第一手慢有两个原因：进程要加载权重（~0.35 s），以及**第一条 GO 要建 CUDA
        图**（~0.42 s）。既然是后台线程，就顺手用空盘跑 400 模拟把这笔账付掉；
        真实第一手再用同样的空盘手顺时，树和 CUDA 图都热着，回包 ~40 ms。
        """
        with self.lock:
            if not self._ensure():
                return False
            try:
                self.proc.stdin.write("GO 400 1.000 0.500 0\n")
                self.proc.stdin.flush()
                self.proc.stdout.readline()
            except Exception:                       # noqa: BLE001
                pass
            return True

    def move(self, moves, n_sim, hw, pw):
        """moves: [(x, y), ...] 从空盘起的全部手顺；返回 dict 或 None。"""
        with self.lock:
            if not self._ensure():
                return None
            parts = ["GO", str(int(n_sim)), "{:.3f}".format(hw), "{:.3f}".format(pw),
                     str(len(moves))]
            for (x, y) in moves:
                parts.append(str(int(x)))
                parts.append(str(int(y)))
            try:
                self.proc.stdin.write(" ".join(parts) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline().strip()
            except Exception as e:                  # noqa: BLE001
                self.last_error = "与 gk.exe 通信失败: {}".format(e)
                self.close()
                return None
        tok = line.split()
        if not tok or tok[0] != "MOVE":
            self.last_error = line[:200] or "空应答"
            return None
        try:
            x, y = int(tok[1]), int(tok[2])
            win = float(tok[3])
            ms = float(tok[4])
            nk = int(tok[5])
            stats = []
            for i in range(nk):
                b = 6 + i * 5
                stats.append({"x": int(tok[b]), "y": int(tok[b + 1]), "n": int(tok[b + 2]),
                              "q": round(1.0 - 2.0 * float(tok[b + 3]), 3),
                              "win": round(float(tok[b + 3]), 4),
                              "p": round(float(tok[b + 4]), 4), "terminal": False})
        except (IndexError, ValueError):
            self.last_error = "应答格式不对: {}".format(line[:200])
            return None
        return {"x": x, "y": y, "win": win, "ms": ms, "stats": stats}

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=3)
        except Exception:                           # noqa: BLE001
            try:
                p.kill()
            except Exception:                       # noqa: BLE001
                pass


class CppAnalyzer:
    """`gk.exe analyze` 的客户端：接口与 `analysis.ContinuousAnalyzer` 完全一致
    （start / stop / running / snapshot），所以 web 路由只按 GAME.engine 二选一，
    **前端与其余路由一行不用改**。

    【为什么必须是独立进程，不能复用 Game._cpp()】CppEngine 有 self.lock 且与对局
    共用一棵搜索树：分析长跑会让对局的 GO 排队，两边交替 reset_tree() 还会让树复用
    失效（batch 变小，速度掉回 Python 水平）。

    【持久化】进程**不随 stop() 退出**：模型重载 7 MB + CUDA 初始化 ~400 ms 太贵，
    而分析台就是"看了又看"的场景，下次 start() 直接发 AN 即可。只在换权重 /
    切回 Python 引擎时 close()（见 cpp_analyzer() / cpp_analyzers_close()）。

    【stop 延迟】约一批（~130 ms），与 Python 版"一个 batch 内返回"的语义一致。
    """

    def __init__(self, net, hw=PROD_HW, pw=PROD_PW):
        self.net = net
        self.hw = float(hw)
        self.pw = float(pw)
        self.proc = None
        self.last_error = ""
        self._io = threading.Lock()      # 串行化「写一行 / 读一行」
        self._th = None
        self._stop = threading.Event()
        self._lock = threading.Lock()    # 只保护 _snap / _meta
        self._snap = None
        self._meta = {"running": False, "total": 0, "batch": 0, "tps": 0.0,
                      "started": 0.0, "max_sims": 0, "full_root": True, "ply": 0}

    # ---- 进程 ----
    def _ensure(self):
        with self._io:
            if self.proc is not None and self.proc.poll() is None:
                return True
            proc, err = _gk_spawn("analyze", self.net, graph=True)
            if proc is None:
                self.last_error = err
                return False
            self.proc = proc
            return True

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=3)
        except Exception:                           # noqa: BLE001
            try:
                p.kill()
            except Exception:                       # noqa: BLE001
                pass

    def _ask(self, moves, batch, full_root):
        """发一条 AN，读一行 JSON。返回 dict；失败返回 None（原因在 last_error）。"""
        if not self._ensure():
            return None
        parts = ["AN", str(int(batch)), "{:.3f}".format(self.hw), "{:.3f}".format(self.pw),
                 "1" if full_root else "0", str(len(moves))]
        for m in moves:
            parts.append(str(int(m[0])))
            parts.append(str(int(m[1])))
            # 第三列是行棋方：摆棋允许不按黑白交替，C++ 侧靠它还原真实局面
            parts.append(str(int(m[2]) if len(m) > 2 else 0))
        line = ""
        with self._io:
            proc = self.proc
            if proc is None:
                return None
            try:
                proc.stdin.write(" ".join(parts) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except Exception as e:                  # noqa: BLE001
                self.last_error = "与 gk.exe 通信失败: {}".format(e)
                line = ""
        if not line:
            self.last_error = self.last_error or "gk.exe 没有应答（进程可能已退出）"
            self.close()
            return None
        line = line.strip()
        if line.startswith("ERR"):
            self.last_error = line
            return None
        try:
            return json.loads(line)
        except ValueError:
            self.last_error = "应答不是 JSON: " + line[:200]
            return None

    # ---- 控制（与 ContinuousAnalyzer 同名同签名）----
    def start(self, board, batch: int = 400, full_root: bool = True,
              max_sims: int = 0, pause: float = 0.12, top_k: int = 10) -> dict:
        """在 `board` 上开始持续搜索：同一棵 C++ 树，访问次数只增不减。"""
        self.stop()
        moves = [(int(m[0]), int(m[1]), int(m[2]))
                 for m in getattr(board, "history", [])]
        # 先把进程拉起来（已预热则瞬间返回）：这样 _loop 里计 tps 的起点就不含冷启动。
        if not self._ensure():
            with self._lock:
                self._meta["running"] = False
                self._meta["error"] = self.last_error or "gk.exe analyze 起不来"
            return self.snapshot()
        with self._lock:
            self._snap = None
            self._meta = {"running": True, "total": 0, "batch": int(batch),
                          "tps": 0.0, "started": time.time(),
                          "max_sims": int(max_sims), "full_root": bool(full_root),
                          "ply": len(moves)}
        self._stop.clear()
        self._th = threading.Thread(target=self._loop,
                                    args=(moves, int(batch), bool(full_root),
                                          int(max_sims), float(pause)),
                                    name="cpp-analysis", daemon=True)
        self._th.start()
        # 等第一份结果再返回（第一批 ≈130 ms）：否则前端第一屏是空浮层。
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                ready = self._snap is not None or bool(self._meta.get("error"))
            if ready:
                break
            time.sleep(0.01)
        return self.snapshot()

    def stop(self, timeout: float = 5.0) -> bool:
        th, self._th = self._th, None
        self._stop.set()
        if th is not None and th.is_alive():
            th.join(timeout=timeout)
        with self._lock:
            self._meta["running"] = False
        return True

    def running(self) -> bool:
        with self._lock:
            return bool(self._meta.get("running"))

    def snapshot(self) -> dict:
        with self._lock:
            meta = dict(self._meta)
            snap = dict(self._snap) if self._snap else None
        if snap is None:
            out = {"running": bool(meta.get("running")), "total": int(meta.get("total", 0)),
                   "batch": int(meta.get("batch", 0)), "tps": float(meta.get("tps", 0.0)),
                   "ply": int(meta.get("ply", 0)), "points": [], "cands": [],
                   "elapsed": 0.0, "root_n": 0, "legal_count": 0}
            if meta.get("error"):
                out["error"] = meta["error"]
            return out
        snap.update({"running": bool(meta.get("running")), "total": int(meta.get("total", 0)),
                     "batch": int(meta.get("batch", 0)), "tps": float(meta.get("tps", 0.0)),
                     "max_sims": int(meta.get("max_sims", 0)),
                     "full_root": bool(meta.get("full_root", True))})
        if meta.get("error"):
            snap["error"] = meta["error"]
        return snap

    # ---- 一次性分析（/api/analyze 与 Game.analyze 用）----
    def analyze_once(self, moves, n_sim, full_root: bool = True) -> dict:
        self.stop()
        r = self._ask(moves, int(n_sim), bool(full_root))
        if r is None:
            return {"error": self.last_error or "C++ 分析失败"}
        r["running"] = False
        # total 的口径同 analysis.analyze_board：报**本次**模拟次数，不是树的累计
        # 访问数（C++ 树跨会话复用，root_n 会一路涨到几十万，直接给出去会骗人）。
        raw = int(r.get("root_n") or 0)
        done = raw if raw <= int(n_sim) else int(n_sim)
        el = float(r.get("elapsed") or 0.0)
        r["total"] = done
        r["tps"] = round(done / el, 1) if el > 0 else 0.0
        return r

    # ---- 后台循环 ----
    def _loop(self, moves, batch, full_root, max_sims, pause) -> None:
        try:
            t0 = None      # 从**第一条 AN 之前**开始计时（不含进程启动，start() 已 _ensure）
            base = None    # C++ 树是跨会话复用的 ⇒ total 只报"本次会话新增的那些"
            while not self._stop.is_set():
                if t0 is None:
                    t0 = time.time()
                r = self._ask(moves, batch, full_root)
                if r is None:
                    with self._lock:
                        self._meta["error"] = self.last_error or "C++ 分析失败"
                    break
                raw = int(r.get("root_n") or 0)
                if base is None:
                    # 首批之后 root_n ≈ base + batch；减掉 base 就与 Python 版
                    # 「total 从 0 开始、每批 +batch」完全同口径。
                    base = max(0, raw - int(batch))
                total = max(0, raw - base)
                elapsed = time.time() - t0
                r["total"] = total
                r["batch"] = int(batch)
                r["tps"] = round(total / max(1e-6, elapsed), 1)
                with self._lock:
                    self._snap = r
                    self._meta["total"] = total
                    self._meta["tps"] = r["tps"]
                if max_sims and total >= max_sims:
                    break
                if pause > 0:
                    self._stop.wait(pause)
        except Exception as exc:                    # noqa: BLE001
            with self._lock:
                self._meta["error"] = "{}: {}".format(type(exc).__name__, exc)
        finally:
            with self._lock:
                self._meta["running"] = False


_CPP_ANALYZERS = {}      # net_abs -> CppAnalyzer（常驻进程池）
_CPP_LOCK = threading.Lock()   # 预热线程与 /api/analyze/start 可能同时进来


def cpp_analyzer(net, hw=PROD_HW, pw=PROD_PW):
    """取（或新建）指向 `net` 的常驻 C++ 分析器；hw/pw 是每条 AN 都带的参数。

    `net` 的口径与 CppEngine 一致：相对 CXX_DIR（如 "model/run_v2/best.bin"）。
    空串时兜底到 discover_cpp_models() 的第一个，避免前端没带 cpp_net 就报
    "找不到权重 D:\\DHSGZQ\\Flashone\\C++"（那其实是目录，不是文件）。
    """
    net = (net or "").strip()
    if not net:
        cands = discover_cpp_models()
        if not cands:
            raise RuntimeError("C++\\model 下没有 .bin 权重")
        net = cands[0]
    key = net if os.path.isabs(net) else os.path.join(CXX_DIR, net)
    with _CPP_LOCK:
        for k in [k for k in _CPP_ANALYZERS if k != key]:
            _CPP_ANALYZERS.pop(k).close()    # 只留最近用的那一份，别攒一堆 GPU 上下文
        a = _CPP_ANALYZERS.get(key)
        if a is None:
            a = CppAnalyzer(net)
            _CPP_ANALYZERS[key] = a
        a.hw, a.pw = float(hw), float(pw)
        return a


def cpp_analyzer_warm(net, hw=PROD_HW, pw=PROD_PW):
    """后台先把 `gk.exe analyze` 拉起来（换到 C++ 引擎时用）。

    进程要加载权重 + 建 CUDA 上下文 ≈ 0.9 s；用户点「分析」时这笔钱就已经付完了，
    第一屏只等一条 AN（~130 ms）。失败静默：真正的错误留到 start() 再报。
    """
    def run():
        try:
            cpp_analyzer(net, hw, pw)._ensure()
        except Exception:                       # noqa: BLE001
            pass
    threading.Thread(target=run, name="cpp-analyzer-warm", daemon=True).start()


def cpp_engine_warm(game):
    """后台把对弈用的 `gk.exe play` 也拉起来（换到 C++ 引擎时用）。

    对弈进程的加载 ~0.35 s、第一条 GO 还要建 CUDA 图（~0.42 s）——不预热的话，
    **用户新开一局后的第一手要等 0.8 s**，很容易被当成"点棋盘没反应"。
    """
    if game is None or game.engine != "cpp":
        return

    def run():
        try:
            game._cpp().warm()
        except Exception:                       # noqa: BLE001
            pass
    threading.Thread(target=run, name="cpp-engine-warm", daemon=True).start()


def cpp_analyzers_close():
    with _CPP_LOCK:
        for a in list(_CPP_ANALYZERS.values()):
            a.close()
        _CPP_ANALYZERS.clear()


def switch_engine(eng, cpp_net=None):
    """换引擎但不重开棋局：前端下拉框一改，服务端立刻跟上（分析与落子都用新引擎）。

    与 `/api/new` 的区别只有一个：**不重置棋盘**。同值调用是 no-op，所以前端可以
    在"下拉框变了"和"点分析"两个时机都送一遍，不用担心中间状态。
    """
    eng = (eng or "").strip().lower()
    if eng not in ("python", "cpp") or eng == GAME.engine:
        return GAME.engine == "cpp"
    GAME.engine = eng
    if cpp_net is not None:
        GAME.cpp_net = (cpp_net or "").strip() or (discover_cpp_models() or [""])[0]
    if eng == "cpp":
        cpp_analyzer_warm(GAME.cpp_net, GAME.hw, GAME.pw)
        cpp_engine_warm(GAME)               # 对弈进程也预热，第一手不等 0.8 s
    else:
        cpp_analyzers_close()
    return True


# ----------------------------------------------------------------- 游戏状态

_LOCK = threading.Lock()
_CALC = threading.Lock()        # 重计算（一次性分析/胜率曲线）互相排队，但**不挡落子**
_NET_CACHE = {}


class Game:
    def __init__(self, model_path=DEFAULT_MODEL, human=1, n_sim=400,
                 hw=PROD_HW, pw=PROD_PW, engine="python", cpp_net=""):
        self.model_path = model_path
        self.engine = engine if engine in ("python", "cpp") else "python"
        # 空权重名要让 Game 自己兜底：以前空串会被拼成 "D:\...\C++\"（目录名），
        # C++ 进程起不来 → 用户落完子 AI 永远不动、棋盘锁死（点棋盘毫无反应）。
        self.cpp_net = cpp_net if cpp_net else (discover_cpp_models() or [""])[0]
        self.cpp_win = None
        self.human = int(human)
        self.n_sim = int(n_sim)
        self.hw = float(hw)
        self.pw = float(pw)
        self.board = Board()
        self.moves = []                # [(x, y, player)]
        self.mcts = None
        self.over = False
        self.winner = None
        self.last_ai = None
        self.last_human = None
        self.root_stats = []           # AI 上一次搜索的候选点统计
        self.ai_time = 0.0
        if self.engine == "python":
            self._build_mcts()

    # ---- 内部 ----
    def _net(self):
        if self.model_path not in _NET_CACHE:
            _NET_CACHE[self.model_path] = load_model(self.model_path)
        return _NET_CACHE[self.model_path]

    def _cpp(self):
        key = ("cpp", self.cpp_net)
        if key not in _NET_CACHE:
            _NET_CACHE[key] = CppEngine(self.cpp_net)
        return _NET_CACHE[key]

    def _build_mcts(self):
        if self.engine != "python":
            self.mcts = None
            return
        self.mcts = MCTS(self._net(), n_sim=self.n_sim, c_puct=1.0,
                         virtual_loss=3.0, batch_size=16, max_children=6,
                         root_max_children=16,
                         heuristic_weight=self.hw, policy_weight=self.pw)
        self.mcts.reset_tree()

    def _collect_stats(self):
        root = getattr(self.mcts, "_root", None)
        if root is None or not root.children:
            self.root_stats = []
            return
        rows = []
        for mv, ch in root.children.items():
            if ch.n <= 0:
                continue
            rows.append({"x": int(mv[0]), "y": int(mv[1]), "n": int(ch.n),
                         "q": round(float(ch.w / ch.n), 3),
                         # q 是**落子后轮到的那一方**（对手）视角；win 才是
                         # "我下这一手"的胜率（= 1 - q 折算）。前端显示 win。
                         "win": round(analysis.mover_win_from_child_q(ch.w / ch.n), 4),
                         "terminal": bool(ch.is_terminal),
                         "p": round(float(ch.prior), 4)})
        rows.sort(key=lambda r: -r["n"])
        self.root_stats = rows[:6]

    def value_estimate(self):
        """网络价值头给出的 AI 胜率估计（与 play.py 同口径）。"""
        if self.engine == "cpp":
            return None if self.cpp_win is None else round(float(self.cpp_win), 4)
        try:
            _, v = self._net().policy_value(self.board.get_state_tensor())
            cur = (float(v) + 1.0) / 2.0            # 当前执子方视角
            ai_win = cur if self.board.current_player != self.human else 1.0 - cur
            return round(ai_win, 4)
        except Exception:
            return None

    # ---- 对外 ----
    def ai_move(self):
        ai = 3 - self.human
        if self.over or self.board.current_player != ai:
            return None
        t0 = time.time()
        if self.engine == "cpp":
            r = self._cpp().move([(m[0], m[1]) for m in self.moves], self.n_sim, self.hw, self.pw)
            if r is None:
                # C++ 进程挂了/权重缺失：以前只写个 note 就 return —— 轮到 AI，
                # 人却还能点（点了被静默忽略），整盘卡死。现在退回 Python 引擎下这一手。
                err = getattr(self._cpp(), "last_error", "")
                try:
                    self.engine = "python"
                    self._build_mcts()
                    self.note_cpp = "C++ 引擎不可用（{}），本手起改用 Python 引擎".format(err or "未知原因")
                    return self.ai_move()
                except Exception as e2:      # noqa: BLE001
                    self.note_cpp = "C++ 引擎不可用：{}；Python 兜底也失败：{}".format(err, e2)
                    return None
            mv = (int(r["x"]), int(r["y"]))
            self.board.move(mv[0], mv[1], ai)
            self.moves.append((mv[0], mv[1], ai))
            self.last_ai = mv
            self.cpp_win = r["win"]
            self.root_stats = r["stats"]
            self.ai_time = r["ms"] / 1000.0
            self._check_over()
            return mv
        best = self.mcts.get_best_action(self.board)
        if best is None:
            self.over = True
            return None
        mv = (int(best[0]), int(best[1]))
        self.board.move(mv[0], mv[1], ai)
        self.moves.append((mv[0], mv[1], ai))
        self.mcts.set_root_move(mv)
        self.last_ai = mv
        self.ai_time = time.time() - t0
        self._collect_stats()
        self._check_over()
        return mv

    def human_move(self, x, y):
        if self.over:
            return False, "对局已结束"
        if self.board.current_player != self.human:
            return False, "现在不是你的回合"
        if not self.board.move(int(x), int(y), self.human):
            return False, "该点非法或已占"
        self.moves.append((int(x), int(y), self.human))
        if self.mcts is not None:
            self.mcts.set_root_move((int(x), int(y)))
        self.last_human = (int(x), int(y))
        self._check_over()
        return True, ""

    def hint(self):
        """给人类一方算一手建议（不改动棋局）。"""
        if self.over or self.board.current_player != self.human:
            return None
        if self.engine == "cpp":
            # 用同一份 C++ 引擎算建议：把当前手顺发过去（C++ 那边不关心轮到谁）
            r = self._cpp().move([(m[0], m[1]) for m in self.moves], self.n_sim, self.hw, self.pw)
            if r is None:
                return None
            return {"x": int(r["x"]), "y": int(r["y"]),
                    "stats": [{"x": s["x"], "y": s["y"], "n": s["n"], "q": s["q"]}
                              for s in r["stats"][:5]]}
        m = MCTS(self._net(), n_sim=self.n_sim, c_puct=1.0, virtual_loss=3.0,
                 batch_size=16, max_children=6, root_max_children=16,
                 heuristic_weight=self.hw, policy_weight=self.pw)
        best = m.get_best_action(self.board)
        if best is None:
            return None
        stats = []
        root = getattr(m, "_root", None)
        if root is not None:
            for mv, ch in root.children.items():
                if ch.n > 0:
                    stats.append({"x": int(mv[0]), "y": int(mv[1]),
                                  "n": int(ch.n), "q": round(float(ch.w / ch.n), 3)})
            stats.sort(key=lambda r: -r["n"])
        return {"x": int(best[0]), "y": int(best[1]), "stats": stats[:5]}

    # ---- 分析模式（KataGo Analysis 风格）----
    # 【为什么不复用 self.mcts】分析要"根全展开 + 指定模拟次数"，与对局那棵树
    # 的参数和代数进度都不同；而且分析只是"看一眼"，不该改变对局搜索树的状态。
    # 模型权重走同一个 _NET_CACHE，所以不会重复加载。
    def analyze(self, n_sim=ANALYZE_SIMS_DEFAULT, full_root=True):
        if self.engine == "cpp":
            # C++ 引擎：走常驻的 gk.exe analyze（一条 AN 拿全 361 格）
            return cpp_analyzer(self.cpp_net, self.hw, self.pw).analyze_once(
                list(self.moves), int(n_sim), bool(full_root))
        return analysis.analyze_board(
            self.board, self._net(), n_sim=int(n_sim),
            heuristic_weight=self.hw, policy_weight=self.pw,
            full_root=bool(full_root))

    def curve(self, n_sim=0):
        """整盘胜率曲线：逐手重放，每手一个点（n_sim=0 = 只用价值头，快）。"""
        return analysis.winrate_curve(
            self.moves, self._net(), n_sim=int(n_sim),
            heuristic_weight=self.hw, policy_weight=self.pw)

    # ---- 逐手复盘 / 摆棋：都落到"按手顺重建棋盘"这一个出口 ----
    def _replay(self, moves):
        """按给定手顺重建棋盘（摆棋允许不按黑白交替，故必要时直接指定执子方）。"""
        b = Board()
        for i, (x, y, p) in enumerate(moves):
            if not (0 <= int(x) < 19 and 0 <= int(y) < 19):
                return False, f"第 {i + 1} 手 ({x},{y}) 越界"
            if b.current_player != int(p):
                b.current_player = int(p)      # 摆棋语义：直接指定这一手是谁下的
            if not b.move(int(x), int(y), int(p)):
                return False, f"第 {i + 1} 手 ({x},{y}) 摆不上（该点已有子或棋局已终）"
        self.board = b
        self.moves = [(int(x), int(y), int(p)) for (x, y, p) in moves]
        self.over, winner = b.is_game_over()
        self.over = bool(self.over)
        self.winner = int(winner) if self.over else None
        self.last_ai = self.last_human = None
        self._build_mcts()
        return True, ""

    def goto(self, ply):
        """回到"走完前 ply 手"的局面（ply=0 = 空盘）—— 逐手复盘用。"""
        ply = max(0, min(int(ply), len(self.moves)))
        return self._replay(self.moves[:ply])

    def edit_undo(self):
        """摆棋专用：拿掉**最后一颗**子。

        【为什么不复用 `undo()`】`undo()` 的语义是"人和 AI 各悔一手"（固定砍 2 手），
        摆棋时那个语义是错的（用户只放了一颗，却退两手 ⇒ 看着像"点了没反应"）。
        """
        if not self.moves:
            return False, "没有可撤销的棋子"
        return self._replay(self.moves[:-1])

    def edit(self, x, y, player=1):
        """摆棋/试下：空点放一颗、有点拿掉一颗，然后立刻可重新分析。

        `player=0` ⇒ **自动**：放"轮到的那一方"（前端勾了「自动交替」时用），
        于是连点就是黑白黑白，不用每颗都去改下拉框。
        """
        x, y = int(x), int(y)
        if not (0 <= x < 19 and 0 <= y < 19):
            return False, "越界"
        if int(player) not in (1, 2):
            player = int(self.board.current_player)
        trial = list(self.moves)
        if self.board.board[y, x]:                 # [行=y, 列=x]
            for i in range(len(trial) - 1, -1, -1):
                if trial[i][0] == x and trial[i][1] == y:
                    del trial[i]
                    break
            else:
                return False, "盘上有子但手顺里找不到（状态不一致）"
        else:
            if self.over:
                return False, "这局面已经成五（终局）—— 要接着摆，先点掉那颗成五的子"
            trial.append((x, y, int(player)))
        return self._replay(trial)

    def undo(self):
        """悔一手（人和 AI 各一手）。"""
        if len(self.moves) < 2:
            return False
        self.moves = self.moves[:-2]
        self.board = Board()
        for (x, y, p) in self.moves:
            self.board.move(x, y, p)
        self.over, self.winner = self.board.is_game_over()
        if self.winner is not None and not self.over:
            self.winner = None
        self.last_ai = None
        self.last_human = self.moves[-1][:2] if self.moves else None
        self._build_mcts()          # 悔棋后重建搜索树（复用树已不可靠）
        return True

    def _check_over(self):
        over, winner = self.board.is_game_over()
        self.over, self.winner = bool(over), (int(winner) if over else None)

    def snapshot(self, note=""):
        if not note and getattr(self, "note_cpp", ""):
            note = self.note_cpp
        rows = []
        for y in range(18, -1, -1):
            # 【P0 已修·2026-10-12】Board.board 是 numpy 数组、下标是 [行=y, 列=x]
            # （gomoku_env.py:167 `self.board[y, x]`）。这里原来写成 board[x][y]，
            # 相当于**把棋盘沿主对角线翻转**：引擎和判定完全正确，但网页上画出来
            # 的位置是你真实落点的转置 → 症状就是"落子位置不对/人和 AI 的棋子镜像"。
            # 只有 x==y 的点（含天元）看不出来，所以很像"偶尔不准"。
            rows.append([int(self.board.board[y, x]) for x in range(19)])
        return {
            "rows": rows,                       # rows[0] 是 y=18 那一行，rows[i][x]=board[18-i, x]
            "human": self.human,
            "turn": int(self.board.current_player),
            "over": self.over,
            "winner": self.winner,
            "moves": [[m[0], m[1], m[2]] for m in self.moves],
            "last_ai": list(self.last_ai) if self.last_ai else None,
            "last_human": list(self.last_human) if self.last_human else None,
            "ai_win": self.value_estimate(),
            "ai_time": round(self.ai_time, 3),
            "stats": self.root_stats,
            "model": self.model_path,
            "engine": self.engine,
            "cpp_net": self.cpp_net,
            "n_sim": self.n_sim,
            "hw": self.hw,
            "pw": self.pw,
            "note": note,
        }


GAME = None
ANALYZER = None          # 持续搜索器（KataGo 式）；None = 没在跑


def stop_analyzer(keep=False):
    """停掉持续搜索。**任何会改变局面的操作前都要先调它**：
    一个局面同时只应有一个搜索在跑，否则两棵树互相抢 CPU/GPU。

    keep=True 时**保留**这个对象（于是 `/api/analyze/state` 还能报出最后一次的
    访问次数与胜率 —— 用户点"停止"后应该看得到最终结果，而不是一片空白）。
    """
    global ANALYZER
    if ANALYZER is not None:
        try:
            ANALYZER.stop()
        except Exception:                           # noqa: BLE001
            pass
    if not keep:
        ANALYZER = None


# ----------------------------------------------------------------- HTTP

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>五子棋 · AlphaZero 人机对战</title>
<style>
 :root{--cell:28px;--line:#5c4526}
 *{box-sizing:border-box}
 body{margin:0;background:#20242b;color:#e8e8e8;font:14px/1.5 "Segoe UI",system-ui,sans-serif}
 .wrap{display:flex;gap:22px;padding:20px;flex-wrap:wrap}
 .boardbox{background:#d9a961;padding:14px 14px 8px;border-radius:8px;
           box-shadow:0 6px 22px #0007}
 table{border-collapse:collapse}
 td{padding:0}
 /* 【棋盘线】每个格子在**自己的中心**画一横一竖 ⇒ 19×19 个格子并起来就是完整线网，
    棋子（同样居中）正好落在**横竖线交点上**；最外圈的线自动停在最外侧交点，
    不会画进左右/上下的坐标标签区（这就是不用"整块背景铺网格"的原因）。 */
 td.c{width:var(--cell);height:var(--cell);position:relative;cursor:pointer;background-image:linear-gradient(var(--line),var(--line)),linear-gradient(var(--line),var(--line));background-size:1px 100%,100% 1px;background-position:50% 50%,50% 50%;background-repeat:no-repeat}
 /* 旧写法 `border:1px solid #00000022` 把棋盘画成"格子"，棋子落在格子里而不是
    线交点上；用户 2026-10-12 明确要求按真棋盘画，已删。 */
 td.c:hover{background-color:#ffffff2e}
 .g{position:absolute;inset:2px;border-radius:50%;box-shadow:0 1px 2px #0006}
 .b{background:radial-gradient(circle at 34% 30%,#6b6b6b,#000)}
 .w{background:radial-gradient(circle at 34% 30%,#fff,#bdbdbd)}
 .last:after{content:"";position:absolute;inset:34%;border-radius:50%;
             background:#ff3b30cc;box-shadow:0 0 6px #ff3b30}
 .hint:after{content:"";position:absolute;inset:26%;border-radius:50%;
             border:2px dashed #1b6cff;background:none;box-shadow:none}
 .star:before{content:"";position:absolute;left:50%;top:50%;width:5px;height:5px;
              margin:-2.5px 0 0 -2.5px;border-radius:50%;background:#00000055}
 .lab{width:19px;height:var(--cell);color:#5a4526;
       font:600 11px/1 "Segoe UI",sans-serif;text-align:center;vertical-align:middle;
       user-select:none}
 .side{min-width:330px;max-width:440px}
 h2{margin:0 0 10px;font-size:16px;font-weight:600}
 .panel{background:#2b313b;border-radius:8px;padding:12px 14px;margin-bottom:11px}
 button{background:#3d7dff;border:0;border-radius:6px;padding:7px 12px;color:#fff;
        cursor:pointer;font-size:13px;margin-right:6px}
 button:hover{filter:brightness(1.12)}
 button.gray{background:#4b5563}
 select{background:#1b1f26;color:#eee;border:1px solid #444;border-radius:5px;
        padding:5px 7px;margin:0 6px 6px 0}
 .status{font-size:15px;font-weight:600;margin-bottom:8px}
 .win{height:16px;background:#1b1f26;border-radius:8px;overflow:hidden;margin:6px 0 4px}
 .win>i{display:block;height:100%;background:linear-gradient(90deg,#41d17a,#d1b441)}
 table.st{width:100%;border-collapse:collapse;font-size:12.5px;color:#c9d1d9}
 table.st td{padding:2px 4px;border:0;width:auto}
 .log{max-height:150px;overflow:auto;font:12px/1.6 ui-monospace,Consolas,monospace;
      color:#a9b4c0;white-space:pre-wrap}
 .muted{color:#8b95a1;font-size:12.5px}
 label.ck{display:inline-flex;align-items:center;gap:5px;margin:0 12px 6px 0;
          font-size:12.5px;color:#c9d1d9;cursor:pointer}
 button.sm{padding:4px 9px;font-size:12px}
 table.st{font-size:12px}
 table.st tr.hl td{background:#3d7dff26}
 .bar{display:inline-block;height:7px;border-radius:4px;vertical-align:middle}
 .svgbox{background:#1b1f26;border-radius:6px;padding:4px;margin-bottom:6px}
 /* ---- 分析浮层：每格最多四层（底色 / 数字 / 访问条 / 圈）---- */
 .netbar{position:fixed;left:0;right:0;top:0;z-index:99;display:none;padding:7px 14px;
         background:#5a1d1d;color:#ffd9d4;border-bottom:1px solid #8a2b2b;
         font:13px/1.4 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
 .ov{position:absolute;inset:0;pointer-events:none}
 .ovn{position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);
      text-align:center;font:700 10px/1 ui-monospace,Consolas,monospace;color:#fff;
      text-shadow:0 0 3px #000,0 0 2px #000}
 .ovb{position:absolute;left:3px;right:3px;bottom:4px;height:3px;border-radius:2px;
      background:#0006;overflow:hidden}
 .ovb>i{display:block;height:100%;background:#fff;border-radius:2px}
 .ring{position:absolute;inset:1px;border-radius:50%;border:2px solid transparent}
 .rbest{border-color:#2f7dff}
 .rw5{border-color:#ffd400}
 .rb5{border-color:#ff7a00}
</style></head><body>
<div class="wrap">
  <div>
    <div class="boardbox"><table id="bd"></table></div>
    <div class="muted" id="boardnote" style="margin-top:7px">棋子落在横竖线交点上</div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>五子棋 · AlphaZero 分析台</h2>
      <div class="status" id="status">加载中…</div>
      <div class="win"><i id="winbar" style="width:0%"></i></div>
      <div class="muted" id="wintxt"></div>
    </div>
    <div class="panel">
      <select id="engine" onchange="engineChanged()">
        <option value="python" selected>Python 引擎</option>
        <option value="cpp">C++ 引擎</option>
      </select>
      <select id="model"></select>
      <select id="cppnet" style="display:none"></select>
      <select id="side"><option value="1">我执黑(先手)</option><option value="2">我执白(后手)</option></select>
      <select id="sims"><option value="200">200 模拟</option><option value="400" selected>400 模拟</option><option value="800">800</option><option value="1600">1600</option></select>
      <div style="margin-top:4px">
        <button onclick="newGame()">新开一局</button>
        <button class="gray" onclick="undo()">悔棋</button>
        <button class="gray" onclick="hint()">提示我</button>
      </div>
    </div>
    <div class="panel">
      <h2>分析模式 <span class="muted">KataGo 风格</span></h2>
      <label class="ck"><input type="checkbox" id="anOn" onchange="anToggle()"> 开启全盘分析</label>
      <label class="ck"><input type="checkbox" id="anFull" checked> 根节点全展开</label>
      <div>
        <select id="anMode" onchange="anModeChange()">
          <option value="cont" selected>持续搜索（KataGo 式）</option>
          <option value="once">一次性</option>
        </select>
        <select id="anSims" onchange="anShowOnly()">
          <option value="200">200</option>
          <option value="400">400</option>
          <option value="800" selected>800</option>
          <option value="1600">1600</option>
          <option value="3000">3000</option>
        </select>
        <span class="muted" id="anHint">每批 800 模拟</span>
      </div>
      <div>
        <select id="anCap">
          <option value="0" selected>不限（一直搜）</option>
          <option value="5000">搜到 5 千次停</option>
          <option value="20000">搜到 2 万次停</option>
          <option value="100000">搜到 10 万次停</option>
        </select>
        <select id="anShow" onchange="draw()">
          <option value="both" selected>胜率+访问</option>
          <option value="win">只看胜率</option>
          <option value="vis">访问热度</option>
          <option value="prior">网络先验</option>
        </select>
      </div>
      <div>
        <button onclick="analyze()">开始分析</button>
        <button class="gray" onclick="stopAnalyze()">停止</button>
      </div>
      <div class="muted" id="anInfo">未分析</div>
      <table class="st" id="anTop"></table>
    </div>
    <div class="panel">
      <h2>整盘胜率曲线</h2>
      <select id="cvMode">
        <option value="0" selected>快速（网络价值头）</option>
        <option value="120">深度（每手 120 模拟）</option>
        <option value="300">深度（每手 300 模拟）</option>
      </select>
      <button class="sm" onclick="curve()">分析全盘</button>
      <div class="svgbox" id="cvChart"></div>
      <div class="muted" id="cvInfo">未分析　（点曲线上的点 = 回到该局面）</div>
    </div>
    <div class="panel">
      <h2>摆棋 / 试下 <span class="muted">edit position</span></h2>
      <label class="ck"><input type="checkbox" id="anEdit" onchange="edHint()"> 摆棋模式（点棋盘放/拿子）</label>
      <div>
        <label class="ck"><input type="checkbox" id="edAuto" checked onchange="edHint()"> 自动交替</label>
        <select id="edColor" onchange="edHint()"><option value="1">放黑子</option><option value="2">放白子</option></select>
        <button class="gray sm" onclick="editUndo()">撤销摆子</button>
        <button class="gray sm" onclick="clearBoard()">清空棋盘</button>
      </div>
      <div class="muted" id="edInfo">勾上「摆棋模式」后：点空点放子（默认自动黑白交替）、点自己的子拿掉。</div>
    </div>
    <div class="panel"><h2>AI 候选（对局口径）</h2><table class="st" id="stats"></table></div>
    <div class="panel"><h2>走子记录</h2><div class="log" id="log"></div></div>
  </div>
</div>
<script>
const L="ABCDEFGHIJKLMNOPQRS"; const STAR=new Set(["3,3","3,9","3,15","9,3","9,9","9,15","15,3","15,9","15,15"]);
let S=null, AN=null, CV=null, hintCell=null, busy=false;
const $=id=>document.getElementById(id);
function anOn(){ return $("anOn").checked; }
// ---- 分析浮层配色：胜率 红→黄→绿；访问 深浅蓝；先验 紫 ----
function winTint(w,a){ const h=Math.round(120*Math.max(0,Math.min(1,w)));
  return "hsla("+h+",75%,42%,"+(a===undefined?0.62:a)+")"; }
function visTint(t){ return "hsla(210,85%,"+Math.round(26+40*t)+"%,"+(0.20+0.55*t).toFixed(2)+")"; }
function priorTint(p){ const t=Math.min(1,p*40); return "hsla(275,60%,"+Math.round(24+46*t)+"%,0.55)"; }
function addRing(td,cls){ const r=document.createElement("div"); r.className="ring "+cls; td.appendChild(r); }
function build(){
  const t=document.getElementById("bd"); t.innerHTML="";
  let h="<tr><td class='lab'></td>";
  for(let x=0;x<19;x++) h+=`<td class='lab'>${L[x]}</td>`;
  h+="<td class='lab'></td></tr>";
  for(let r=0;r<19;r++){ const y=18-r;
    h+=`<tr><td class='lab'>${y+1}</td>`;
    for(let x=0;x<19;x++) h+=`<td class='c' id='c${x}_${y}' onclick='play(${x},${y})'></td>`;
    h+=`<td class='lab'>${y+1}</td></tr>`;
  }
  h+="<tr><td class='lab'></td>";
  for(let x=0;x<19;x++) h+=`<td class='lab'>${L[x]}</td>`;
  h+="<td class='lab'></td></tr>";
  t.innerHTML=h;
}
function draw(){
  if(!S || S.__net) return;    // 请求失败时 S={__net:true}：保留上一帧，别把棋盘刷白
  for(let y=0;y<19;y++) for(let x=0;x<19;x++){
    const td=document.getElementById(`c${x}_${y}`); if(!td) continue;
    td.className = "c" + (STAR.has(x+","+y) ? " star" : "");
    td.innerHTML="";
    const v=S.rows[18-y][x];
    if(v){ const d=document.createElement("div"); d.className="g "+(v===1?"b":"w"); td.appendChild(d); }
  }
  const mark=(m,cls)=>{ if(!m) return; const td=document.getElementById(`c${m[0]}_${m[1]}`);
                        if(td) td.classList.add(cls); };
  mark(S.last_ai,"last"); mark(S.last_human,"last");
  if(hintCell) mark(hintCell,"hint");
  const you=S.human===1?"黑 ●":"白 ○", ai=S.human===1?"白 ○":"黑 ●";
  const editing=$("anEdit").checked;
  let st;
  if(editing && !S.over) st = `摆棋模式　（轮到 ${S.turn===1?"黑 ●":"白 ○"}，共 ${S.moves.length} 颗）`;
  else if(S.over) st = S.winner===0 ? "【平局】棋盘下满" :
      (S.winner===S.human? "🎉 你赢了！" : "AI 赢了这局");
  else st = (S.turn===S.human? "轮到你" : "AI 思考中…") + `　（你 ${you} / AI ${ai}）`;
  $("status").textContent=st+(S.note?("　—　"+S.note):"");
  const aw = S.ai_win==null? null : S.ai_win;
  $("winbar").style.width = aw==null? "0%" : (aw*100).toFixed(1)+"%";
  $("wintxt").textContent =
     aw==null? "" : `网络局面估值：AI 胜率 ${(aw*100).toFixed(1)}%　|　上一手耗时 ${S.ai_time}s　|　第 ${S.moves.length} 手　|　${S.model} / ${S.n_sim} 模拟 / 规则权重 ${S.hw}`;
  const sb=$("stats"); sb.innerHTML="";
  (S.stats||[]).forEach(r=>{
    sb.innerHTML += `<tr><td>${L[r.x]}${r.y+1}</td><td>访问 ${r.n}</td><td>胜率 ${r.win==null?"-":(r.win*100).toFixed(1)+"%"}</td><td>先验 ${r.p}</td></tr>`;
  });
  const lg=document.getElementById("log"); lg.textContent="";
  S.moves.forEach((m,i)=>{
    const who = editing ? (m[2]===1?"黑":"白") : (m[2]===S.human?"你  ":"AI  ");
    lg.textContent += `${String(i+1).padStart(3)}. ${who} ${L[m[0]]}${m[1]+1}\n`;
  });
  lg.scrollTop=lg.scrollHeight;
  edHint();
  paintOverlay(); paintTop(); paintCurve();
}
// ---- 全盘浮层：每个合法空点叠 胜率/访问/先验 ----
// 持续搜索时这里每 ~0.3 s 重画一次，所以**必须先清掉上一轮自己加的节点**，
// 否则数字会叠成一团（而且 DOM 会一直涨）。
let ovCells=[];
function ovClear(td){
  for(let i=td.children.length-1;i>=0;i--){
    const c=td.children[i], cl=c.classList;
    if(cl && (cl.contains("ov")||cl.contains("ovn")||cl.contains("ovb")||cl.contains("ring"))) td.removeChild(c);
  }
}
function paintOverlay(){
  ovCells.forEach(ovClear); ovCells=[];
  if(!AN || AN.error || !anOn()) return;
  if(AN.ply !== S.moves.length) return;      // 局面已变：先不画，等新分析回来
  const mode=$("anShow").value, pts=AN.points||[];
  let nmax=1; pts.forEach(p=>{ if(p.n>nmax) nmax=p.n; });
  const best=AN.best;
  pts.forEach(p=>{
    if(p.occ || !p.legal) return;
    const td=$("c"+p.x+"_"+p.y); if(!td) return;
    ovCells.push(td);
    const ov=document.createElement("div"); ov.className="ov";
    let tint="transparent", txt="";
    if(mode==="vis"){ tint=visTint(p.n/nmax); txt=p.n?String(p.n):""; }
    else if(mode==="prior"){ tint=priorTint(p.netp||0); txt=p.netp?(p.netp*100).toFixed(1):"0"; }
    else if(p.win!=null){ tint=winTint(p.win); txt=String(Math.round(p.win*100)); }
    else { tint="hsla(0,0%,70%,0.07)"; }
    ov.style.background=tint; td.appendChild(ov);
    if(txt){ const n=document.createElement("div"); n.className="ovn"; n.textContent=txt; td.appendChild(n); }
    if((mode==="both"||mode==="vis") && p.n>0){
      const b=document.createElement("div"); b.className="ovb";
      const i=document.createElement("i"); i.style.width=Math.max(4,100*p.n/nmax).toFixed(1)+"%";
      b.appendChild(i); td.appendChild(b);
    }
    if(p.w5) addRing(td,"rw5");                     // 我下这里直接成五
    else if(p.b5) addRing(td,"rb5");                // 对手成五点：必须挡
    else if(best && best.x===p.x && best.y===p.y) addRing(td,"rbest");
  });
}
// ---- 候选排行（KataGo 风格：访问 / 胜率 / 先验 / Δ）----
function paintTop(){
  const tb=$("anTop"); tb.innerHTML="";
  if(!AN) return;
  if(AN.error){ tb.innerHTML="<tr><td class='muted'>"+AN.error+"</td></tr>"; return; }
  if(AN.ply !== S.moves.length){ tb.innerHTML="<tr><td class='muted'>局面已变 —— 点「分析当前局面」</td></tr>"; return; }
  const cs=AN.cands||[];
  if(!cs.length){ tb.innerHTML="<tr><td class='muted'>没有候选点（终局/无合法点）</td></tr>"; return; }
  let h="<tr><td class='muted'>点</td><td class='muted'>访问</td><td class='muted'>胜率</td><td class='muted'>先验</td><td class='muted'>Δ</td></tr>";
  cs.forEach((c,i)=>{
    const w=c.win==null?0:c.win;
    const tag=c.w5?" <span style='color:#ffd400'>成五</span>":(c.b5?" <span style='color:#ff7a00'>必挡</span>":"");
    const d=c.dwin==null?"-":((c.dwin>=0?"+":"")+(c.dwin*100).toFixed(1)+"%");
    h+="<tr"+(i===0?" class='hl'":"")+"><td><b>"+L[c.x]+(c.y+1)+"</b>"+tag+"</td><td>"+c.n+
       "</td><td><span class='bar' style='width:"+Math.round(40*w)+"px;background:"+winTint(w,1)+"'></span> "+
       (c.win==null?"-":(w*100).toFixed(1))+"</td><td>"+c.prior.toFixed(3)+"</td><td>"+d+"</td></tr>";
  });
  tb.innerHTML=h;
}
// ---- 整盘胜率曲线（统一黑方视角；红点=疑似坏棋；点曲线回该局面）----
function paintCurve(){
  const box=$("cvChart"); box.innerHTML="";
  if(!CV || CV.error) return;
  const pts=CV.points||[]; if(!pts.length) return;
  const W=392,H=112,pad=11;
  const ys=pts.map(p=>p.black_win).concat([CV.final_black_win]);
  const n=ys.length;
  const X=i=> n<=1? pad+(W-2*pad)/2 : pad+(W-2*pad)*i/(n-1);
  const Y=w=> H-pad-(H-2*pad)*Math.max(0,Math.min(1,w));
  let g="";
  [[0,"0"],[0.5,"50"],[1,"100"]].forEach(function(g0){
    const w=g0[0], lab=g0[1];
    g+="<line x1='"+pad+"' y1='"+Y(w).toFixed(1)+"' x2='"+(W-pad)+"' y2='"+Y(w).toFixed(1)+"' stroke='#2c3440'/>";
    g+="<text x='1' y='"+(Y(w)+3).toFixed(1)+"' fill='#5d6875' font-size='9'>"+lab+"</text>";
  });
  g+="<polyline points='"+ys.map((w,i)=>X(i).toFixed(1)+","+Y(w).toFixed(1)).join(" ")+"' fill='none' stroke='#41d17a' stroke-width='2'/>";
  const cur=n-1;
  g+="<line x1='"+X(cur).toFixed(1)+"' y1='"+pad+"' x2='"+X(cur).toFixed(1)+"' y2='"+(H-pad)+"' stroke='#3d7dff'/>";
  pts.forEach((p,i)=>{
    const bad=p.loss>CV.threshold;
    const cx=X(i).toFixed(1), cy=Y(p.black_win).toFixed(1);
    g+="<circle cx='"+cx+"' cy='"+cy+"' r='"+(bad?3.5:2)+"' fill='"+(bad?"#ff5a48":"#8fd6a8")+"'/>";
    g+="<circle cx='"+cx+"' cy='"+cy+"' r='7' fill='transparent' style='cursor:pointer' onclick='gotoPly("+i+")'>"+
       "<title>"+L[p.x]+(p.y+1)+"（"+(p.player===1?"黑":"白")+"）这一手丢 "+(p.loss*100).toFixed(1)+"%</title></circle>";
  });
  g+="<circle cx='"+X(cur).toFixed(1)+"' cy='"+Y(ys[cur]).toFixed(1)+"' r='3.5' fill='#3d7dff'/>";
  box.innerHTML="<svg width='"+W+"' height='"+H+"'>"+g+"</svg>";
  const worst=(CV.worst||[]).slice(0,4).map(p=>L[p.x]+(p.y+1)+"("+(p.loss*100).toFixed(0)+"%)").join("、");
  $("cvInfo").textContent="模式 "+CV.mode+"　终局黑方胜率 "+(CV.final_black_win*100).toFixed(1)+"%　疑似坏棋 "+
    (CV.worst||[]).length+" 手"+(worst?"："+worst:"")+"　（红点=单步跌幅>"+(CV.threshold*100).toFixed(0)+"%）";
}
// ---- 网络层：所有请求都不许抛异常 ----
// 以前 post() 直接 fetch，网络一抖就 reject；newGame/undo/hint 里 `S=await post(...)`
// 一抛就把 busy 永久钉在 true，之后每次点棋盘都被 `if(busy) return` 吞掉 ——
// 这就是「点了几次，子还没出现在棋盘上」的主因。
function banner(on, text){
  let el=document.getElementById("netbar");
  if(!el){ el=document.createElement("div"); el.id="netbar"; document.body.appendChild(el); }
  el.textContent=text||"";
  el.style.display = on? "block":"none";
}
let _netRetry=null;
function netFail(u, e){
  const why = (e && e.name==="AbortError") ? "请求超时" : "连不上服务端";
  banner(true, why+"（"+u+"）——正在自动重连；确认「Flash 分析台」窗口没关。");
  busy=false; stopPoll();
  if(_netRetry) clearTimeout(_netRetry);
  _netRetry=setTimeout(async()=>{ _netRetry=null;
    try{ const p=await fetch("/api/ping"); if(!p.ok) throw new Error("HTTP "+p.status);
         banner(false,""); await refresh(); }
    catch(err){ netFail(u, err); } }, 1200);
}
async function post(u,b,ms){
  const ac = (typeof AbortController!=="undefined") ? new AbortController() : null;
  const t  = ac ? setTimeout(()=>ac.abort(), ms||20000) : null;
  try{
    const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},
                           body:JSON.stringify(b||{}),signal:ac?ac.signal:undefined});
    if(!r.ok) throw new Error("HTTP "+r.status);
    return await r.json();
  }catch(e){ netFail(u,e); return {__net:true}; }
  finally{ if(t) clearTimeout(t); }
}
async function refresh(){
  try{ const r=await fetch("/api/state"); if(!r.ok) throw new Error("HTTP "+r.status);
       S=await r.json(); banner(false,""); draw(); }
  catch(e){ netFail("/api/state", e); }
}
async function analyze(){
  if(!S||busy) return;
  stopPoll();
  if(anCont()){                                   // ---- 持续搜索 ----
    // 注意：分析不置 busy —— 它不改棋局，没必要把棋盘锁死；
    // 以前 busy=true 顶到返回为止，用户在这 0.3~1 s 里点棋盘会被 `if(busy) return` 吞掉。
    const r=await post("/api/analyze/start",{batch:+$("anSims").value,
        full_root:$("anFull").checked, max_sims:+$("anCap").value,
        engine:$("engine").value, cpp_net:$("cppnet").value});
    if(!r||r.__net||r.error){ AN=null; $("anInfo").textContent=(r&&r.error)||"分析请求失败（服务端没响应）"; draw(); return; }
    if(r&&r.points&&r.points.length){ AN=r; }
    draw(); if(AN) anInfoText(AN);
    startPoll();                                  // 之后数字由轮询刷新
    return;
  }
  // ---- 一次性 ----
  $("anInfo").textContent="分析中…（"+$("anSims").value+" 模拟 + 根全展开）";
  const r=await post("/api/analyze",{n_sim:+$("anSims").value, full_root:$("anFull").checked,
        engine:$("engine").value, cpp_net:$("cppnet").value});
  if(!r||r.__net||r.error){ AN=null; $("anInfo").textContent=(r&&r.error)||"分析请求失败（服务端没响应）"; draw(); return; }
  if(r&&r.stale){ return; }        // 算的过程中又落子了 ⇒ 这份结果已经错位，丢掉
  AN=r; draw(); anInfoText(r);
}
function anToggle(){ if(anOn()){ analyze(); } else { stopAnalyze(); AN=null; CV=null; draw(); } }
function anCont(){ return $("anMode").value==="cont"; }          // 持续搜索（KataGo 式）
function anHintText(){ $("anHint").textContent = anCont()
  ? ("每批 "+$("anSims").value+" 模拟") : ("一次搜 "+$("anSims").value+" 模拟"); }
function anModeChange(){ anHintText(); if(anOn()) analyze(); }
function anShowOnly(){ anHintText(); if(anOn()) analyze(); }     // 换批大小＝重算
// ---- 持续搜索：轮询 /api/analyze/state，数字一直在爬 ----
let anTimer=null;
function stopPoll(){ if(anTimer){ clearInterval(anTimer); anTimer=null; } }
function startPoll(){ stopPoll(); anTimer=setInterval(pollAnalyze, 280); }
async function pollAnalyze(){
  if(!S||busy) return;
  let r=null;
  try{ r=await (await fetch("/api/analyze/state")).json(); }catch(e){ return; }
  if(!r) return;
  if(r.points && r.points.length){
    AN=r;                                  // paintOverlay 里会核对 ply，局面变了就不画
    paintOverlay(); paintTop(); anInfoText(r);
  }
  if(!r.running) stopPoll();
}
function anInfoText(r){
  const head = r.running
    ? ("🔴 持续搜索中：已 "+(r.total||0)+" 次访问，约 "+(r.tps||0)+" 次/秒")
    : (r.total ? ("已停止（共 "+(r.total||0)+" 次访问）") : "未分析");
  $("anInfo").textContent = head+"　|　第 "+(r.ply||0)+" 手后轮到 "+(r.mover===1?"黑":"白")+
    "　搜索胜率 "+(r.root_win==null?"-":(r.root_win*100).toFixed(1))+"%"+
    "　网络价值 "+(r.net_win==null?"-":(r.net_win*100).toFixed(1))+"%"+
    "　考虑 "+(r.legal_count||0)+" 点"+(r.elapsed?("　用时 "+r.elapsed+"s"):"");
}
async function stopAnalyze(){
  stopPoll();
  const r = await post("/api/analyze/stop",{});   // 服务端会回最后一份快照
  if(r && r.points && r.points.length){ AN=r; }
  else if(AN){ AN.running=false; }
  if(AN){ anInfoText(AN); paintTop(); paintOverlay(); }
}
async function curve(){
  if(!S) return; const m=+$("cvMode").value;
  // 也不置 busy：曲线是只读计算，服务端拿快照算，落子照常
  $("cvInfo").textContent = m>0? ("计算中…每手 "+m+" 模拟，请稍等") : "计算中…（网络价值头）";
  const r=await post("/api/curve",{n_sim:m});
  if(r&&r.error){ CV=null; $("cvInfo").textContent=r.error; paintCurve(); return; }
  CV=r; draw();
}
async function gotoPly(n){
  if(busy||!S) return; busy=true; stopPoll();
  S=await post("/api/goto",{ply:n}); busy=false; hintCell=null; draw();
  if(anOn()) analyze();
}
// ---- 摆棋：自动交替（放完换色）让"连点两下就是一黑一白" ----
function edAuto(){ return $("edAuto").checked; }
function edHint(){
  if(!S){ $("edInfo").textContent=""; return; }
  const sel=$("edColor");
  if(!$("anEdit").checked){
    $("edInfo").textContent="勾上「摆棋模式」后：点空点放子（默认自动黑白交替）、点自己的子拿掉。";
    return;
  }
  if(edAuto()){
    const t = S.turn;                        // 盘面轮到谁 ⇒ 下一颗就是那个颜色
    sel.value = String(t);
    $("edInfo").textContent="下一手自动放 "+(t===1?"黑 ●":"白 ○")+"（放完换色）　|　再点自己放的子 = 拿掉　|　当前 "+
      S.moves.length+" 颗";
  }else{
    $("edInfo").textContent="固定放 "+(sel.value==="1"?"黑 ●":"白 ○")+"（不自动换色）　|　当前 "+S.moves.length+" 颗";
  }
}
async function editUndo(){
  if(busy||!S) return; busy=true; stopPoll();
  S=await post("/api/edit/undo",{}); busy=false; AN=null; CV=null; draw();
  if(anOn()) analyze();
}
async function clearBoard(){
  if(busy) return; busy=true; stopPoll();
  S=await post("/api/clear",{}); busy=false; AN=null; CV=null; draw();
  if(anOn()) analyze();
}
// 点棋盘时如果服务端正忙（上一步还没回/曲线在算），把这一手**记下来**，
// 腾出来立刻补下——以前是 `if(busy) return` 直接吞掉，用户点了几下都没反应。
let pendClick=null, pendTimer=null;
function flushClick(){
  pendTimer=null;
  if(!pendClick) return;
  if(busy){ pendTimer=setTimeout(flushClick,60); return; }
  const c=pendClick; pendClick=null;
  if(!S||S.__net||!S.moves) return;
  if(S.moves.some(m=>m[0]===c[0]&&m[1]===c[1])) return;   // 那点已经有人了
  play(c[0],c[1]);
}
async function play(x,y){
  if(!S){ $("status").textContent="还没连上服务端——请刷新页面（或先启动「Flash 分析台」）"; return; }
  if(S.__net){ $("status").textContent="服务端连接中断，正在自动重连…"; return; }
  if(busy){ pendClick=[x,y]; $("status").textContent="上一个操作还没回完，已排队…"; if(!pendTimer) pendTimer=setTimeout(flushClick,60); return; }
  if($("anEdit").checked){                       // 摆棋模式：点击棋盘放/拿子
    busy=true; stopPoll();
    S=await post("/api/edit",{x:x,y:y,player: edAuto()?0:+$("edColor").value}); busy=false;
    AN=null; CV=null; draw(); if(anOn()) analyze(); return;
  }
  if(S.over){ $("status").textContent="这局已经结束，点「新开一局」再下。"; return; }
  // 以前这里是静默 return —— 用户点了看不到任何反应。现在必须给出提示。
  if(S.turn!==S.human){ $("status").textContent="现在是 AI 的回合（"+(S.turn===1?"黑":"白")+"），等它落子。"; return; }
  busy=true; hintCell=null; stopPoll();
  document.getElementById("status").textContent="AI 思考中…";
  // 先把自己的子画上去：服务端一般 40 ms 回包，但 C++ 进程冷启动/曲线在算时
  // 要等 0.3~1 s ——「点了没反应」多半就是这 1 s 的空白。
  if(!S.moves.some(m=>m[0]===x&&m[1]===y)){
    S.moves.push([x,y,S.human]); S.turn=3-S.human; S.last_human=[x,y]; draw();
  }
  S=await post("/api/move",{x:x,y:y});
  busy=false;
  if(S.__net){ draw(); return; }     // 顶部红条+自动重连；棋子先留在盘上
  draw();
  if(S&&anOn()&&!S.over) analyze();
  if(pendClick) flushClick();
}
async function newGame(){ busy=true; stopPoll();
  S=await post("/api/new",{model:document.getElementById("model").value,
                           cpp_net:document.getElementById("cppnet").value,
                           engine:document.getElementById("engine").value,
                           human:+document.getElementById("side").value,
                           n_sim:+document.getElementById("sims").value});
  busy=false; hintCell=null; AN=null; CV=null; draw();
  if(anOn()) analyze(); }
// 引擎下拉：Python 用 .pt 列表，C++ 用 .bin 列表；切换时**立刻告诉服务端**
// （以前只换显示，服务端要等「新开一局」才知道 —— 于是选了 C++ 点分析还是走 Python）。
function engineChanged(){
  const el = document.getElementById("engine");
  const cpp = el.value === "cpp";
  document.getElementById("model").style.display = cpp ? "none" : "";
  document.getElementById("cppnet").style.display = cpp ? "" : "none";
  try{ localStorage.setItem("flashEngine", el.value); }catch(e){}
  post("/api/engine",{engine:el.value, cpp_net:document.getElementById("cppnet").value});
}
async function undo(){ if(busy) return; stopPoll(); S=await post("/api/undo",{}); hintCell=null; AN=null; CV=null; draw();
  if(anOn()) analyze(); }
async function hint(){ if(busy||!S||S.over||S.turn!==S.human) return; busy=true;
  const r=await post("/api/hint",{}); busy=false;
  if(r&&r.move){ hintCell=[r.move.x,r.move.y]; draw(); } }
async function models(){ const m=await (await fetch("/api/models")).json();
  const sel=document.getElementById("model"); sel.innerHTML="";
  m.models.forEach(p=>{ const o=document.createElement("option"); o.value=p; o.textContent=p; sel.appendChild(o); });
  const pick = m.default && m.models.includes(m.default) ? m.default : m.models[0];
  if(pick) sel.value=pick;
  const cs=document.getElementById("cppnet"); cs.innerHTML="";
  (m.cpp_models||[]).forEach(p=>{ const o=document.createElement("option"); o.value=p; o.textContent="C++ · "+p; cs.appendChild(o); });
  if((m.cpp_models||[]).length) cs.value=m.cpp_default||m.cpp_models[0];
  try{ const e=localStorage.getItem("flashEngine");
       if(e==="cpp"||e==="python") document.getElementById("engine").value=e; }catch(err){}
  engineChanged(); }
// 启动：连不上就每秒重试。以前这里 `await models()` 一抛，整个脚本死掉 ——
// 页面看起来正常、点棋盘却永远没反应（连个错都不报），就是这种状态。
async function boot(){
  try{
    await models();
    await refresh();
    if(!S) throw new Error("no state");
    banner(false,""); startHeartbeat();
  }catch(e){
    banner(true,"连不上服务端，正在自动重试…（确认「Flash 分析台」窗口没被关掉）");
    setTimeout(boot, 1000);
  }
}
let hbTimer=null;
function startHeartbeat(){
  if(hbTimer) clearInterval(hbTimer);
  hbTimer=setInterval(async()=>{
    try{ const r=await fetch("/api/ping"); if(!r.ok) throw new Error("HTTP "+r.status);
         if(!S||S.__net) await refresh();
         if(S&&!S.__net) banner(false,""); }
    catch(e){ banner(true,"服务端已断开，正在重试…（确认「Flash 分析台」还开着）"); }
  }, 3000);
}
(async function(){ build(); anHintText(); boot(); })();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):      # 安静一点：只记 API 调用
        if "/api/" in (self.path or ""):
            sys.stdout.write(f"[web] {self.path} {args[1] if len(args) > 1 else ''}\n")
            sys.stdout.flush()

    def _net_outside(self, model_path):
        """在 _LOCK 外面加载/取网络（第一次 load .pt 要 1~3 s，别让轮询陪着等）。"""
        net = _NET_CACHE.get(model_path)
        if net is None:
            net = load_model(model_path)
            _NET_CACHE[model_path] = net
        return net

    # ---- 两个"重计算"接口：算的时候不拿 _LOCK ----
    # _LOCK 保护的是棋局状态（moves/board/over）。以前这两支整段都在 _LOCK 里算，
    # 于是：Python 4000 模拟 ≈ 1 s、C++ 8000 ≈ 0.8 s、曲线 n_sim=2000 ≈ 20 s ——
    # 期间点棋盘的那条 /api/move 只能排在后面，用户看到的就是「点了，过几秒才落子」。
    # 现在：在 _LOCK 里只取一份快照（手顺 / 棋盘深拷贝），算完再回包；_CALC 只用来
    # 让两个重计算彼此排队（不挡住落子）。
    def _one_shot_analyze(self, b):
        with _CALC:
            with _LOCK:
                stop_analyzer()
                if b.get("engine"):
                    switch_engine(b.get("engine"), b.get("cpp_net"))
                ply0 = len(GAME.moves)
                moves = [(m[0], m[1]) for m in GAME.moves]
                engine = GAME.engine
                hw, pw = GAME.hw, GAME.pw
                cpp_net, model_path = GAME.cpp_net, GAME.model_path
                # Python 那支直接吃棋盘对象 ⇒ 必须深拷贝，否则算到一半落子会读到半边
                board = copy.deepcopy(GAME.board) if engine != "cpp" else None
            net = None if engine == "cpp" else self._net_outside(model_path)
            try:
                t_an = time.time()
                if engine == "cpp":
                    r = cpp_analyzer(cpp_net, hw, pw).analyze_once(
                        moves, int(b.get("n_sim") or ANALYZE_SIMS_DEFAULT),
                        bool(b.get("full_root", True)))
                else:
                    r = analysis.analyze_board(
                        board, net, n_sim=int(b.get("n_sim") or ANALYZE_SIMS_DEFAULT),
                        heuristic_weight=hw, policy_weight=pw,
                        full_root=bool(b.get("full_root", True)))
            except Exception as e:                      # noqa: BLE001
                return self._send(200, {"error": f"分析失败: {e}"})
            # Python 那支的返回里没有 total/tps（字段叫 n_sim），前端 anInfoText
            # 读的是 total/tps ⇒ 会显示"未分析"。这里补齐，两条引擎口径一致。
            r.setdefault("total", int(r.get("n_sim") or 0))
            if not r.get("tps"):
                r["tps"] = round(int(r.get("total") or 0)
                                 / max(1e-6, time.time() - t_an), 1)
            # 算完发现局面已经变了（用户趁这 1 s 又落了一子）⇒ 前端丢弃，别画旧盘
            r["ply"] = ply0
            r["stale"] = len(GAME.moves) != ply0
            return self._send(200, r)

    def _curve(self, b):
        with _CALC:
            with _LOCK:
                stop_analyzer()
                moves = list(GAME.moves)
                hw, pw = GAME.hw, GAME.pw
                model_path = GAME.model_path
            net = self._net_outside(model_path)
            try:
                r = analysis.winrate_curve(moves, net, n_sim=int(b.get("n_sim") or 0),
                                           heuristic_weight=hw, policy_weight=pw)
            except Exception as e:                      # noqa: BLE001
                return self._send(200, {"error": f"曲线计算失败: {e}"})
            r["ply"] = len(GAME.moves)
            return self._send(200, r)

    def do_GET(self):
        global GAME
        if self.path in ("/", "/index.html"):
            return self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        if self.path == "/api/state":
            with _LOCK:
                return self._send(200, GAME.snapshot())
        if self.path == "/api/ping":
            return self._send(200, {"ok": 1, "pid": os.getpid()})
        if self.path == "/api/models":
            ms = discover_models()
            return self._send(200, {"models": ms, "default": DEFAULT_MODEL,
                                    "cpp_models": discover_cpp_models(),
                                    "cpp_default": (discover_cpp_models() or [""])[0]})
        if self.path == "/api/analyze/state":
            # 持续搜索的最新快照（前端每 ~0.3 s 拉一次，看到数字一直在爬）
            with _LOCK:
                if ANALYZER is None:
                    return self._send(200, {"running": False, "total": 0, "tps": 0.0,
                                            "points": [], "cands": []})
                return self._send(200, ANALYZER.snapshot())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        global GAME, ANALYZER
        b = self._body()
        path = self.path
        # 两支重计算先分流出去：它们在 _LOCK 外面算（见 _one_shot_analyze / _curve）
        if path == "/api/analyze":
            return self._one_shot_analyze(b)
        if path == "/api/curve":
            return self._curve(b)
        with _LOCK:
            # 除分析族接口（/api/analyze*，它们自己管理搜索）以外，
            # 任何 POST 都先把持续搜索停掉：局面要变 / 要重算，一个局面只能有一个搜索
            if not path.startswith("/api/analyze"):
                stop_analyzer()
            if path == "/api/new":
                try:
                    eng = (b.get("engine") or "python").strip()
                    GAME = Game(b.get("model") or DEFAULT_MODEL,
                                int(b.get("human") or 1),
                                int(b.get("n_sim") or 400),
                                engine=eng,
                                cpp_net=(b.get("cpp_net") or "").strip())
                except Exception as e:      # noqa: BLE001
                    return self._send(200, {"error": f"加载模型失败: {e}"})
                if GAME.engine != "cpp":
                    cpp_analyzers_close()   # 切回 Python 引擎：收掉常驻的 C++ 分析进程
                else:
                    # 选了 C++ 就把分析进程先拉起来（后台线程，不挡 /api/new）：
                    # 用户点「分析」时省掉 ~0.9 s 的加载+CUDA 上下文
                    cpp_analyzer_warm(GAME.cpp_net, GAME.hw, GAME.pw)
                    cpp_engine_warm(GAME)
                if GAME.board.current_player != GAME.human:
                    GAME.ai_move()
                    if GAME.over and not GAME.moves:
                        return self._send(200, {**GAME.snapshot(),
                                                "error": (getattr(GAME, "note_cpp", "")
                                                          or "C++ 引擎没有返回落点")})
                return self._send(200, GAME.snapshot())
            if path == "/api/engine":
                # 只换引擎、不动棋局（下拉框 onchange 就调这里）
                switch_engine(b.get("engine"), b.get("cpp_net"))
                return self._send(200, {**GAME.snapshot(),
                                        "note": "引擎已切到 " + GAME.engine})
            if self.path == "/api/move":
                ok, msg = GAME.human_move(int(b.get("x", -1)), int(b.get("y", -1)))
                if not ok:
                    return self._send(200, {**GAME.snapshot(), "note": msg})
                if not GAME.over:
                    GAME.ai_move()
                return self._send(200, GAME.snapshot())
            if self.path == "/api/undo":
                GAME.undo()
                return self._send(200, GAME.snapshot())
            if self.path == "/api/hint":
                return self._send(200, {"move": GAME.hint()})
            # ---------------- 分析模式（KataGo Analysis 风格）----------------
            if self.path == "/api/analyze/start":
                # 持续搜索：后台线程在同一棵树上不停加模拟，前端轮询 /api/analyze/state
                if b.get("engine"):
                    switch_engine(b.get("engine"), b.get("cpp_net"))
                try:
                    if GAME.engine == "cpp":
                        # C++ 引擎：常驻 gk.exe analyze（同一棵树一直加深，~6.5k 次/秒）
                        ANALYZER = cpp_analyzer(GAME.cpp_net, GAME.hw, GAME.pw)
                    else:
                        ANALYZER = analysis.ContinuousAnalyzer(
                            GAME._net(), hw=GAME.hw, pw=GAME.pw)
                    snap = ANALYZER.start(
                        GAME.board,
                        batch=int(b.get("batch") or 400),
                        full_root=bool(b.get("full_root", True)),
                        max_sims=int(b.get("max_sims") or 0),
                        # C++ 那批是独立进程里的一次阻塞往返，不需要靠 sleep 让路；
                        # 留 0 才不会被 120 ms 空转吃掉 ~1/3 吞吐（停止延迟仍 ≤1 批）。
                        pause=0.0 if GAME.engine == "cpp" else float(b.get("pause", 0.12)))
                except Exception as e:                      # noqa: BLE001
                    stop_analyzer()
                    return self._send(200, {"error": f"持续搜索启动失败: {e}"})
                return self._send(200, snap)
            if self.path == "/api/analyze/stop":
                stop_analyzer(keep=True)             # 保留最后结果，前端要继续显示
                return self._send(200, ANALYZER.snapshot() if ANALYZER else
                                  {"running": False})
            if self.path == "/api/goto":
                ok, msg = GAME.goto(int(b.get("ply", 0)))
                return self._send(200, {**GAME.snapshot(), "note": "" if ok else msg})
            if self.path == "/api/edit":
                # player=0（或没给）⇒ 自动：放"轮到的那一方"，于是连点即黑白交替
                p = int(b.get("player") or 0)
                if p not in (1, 2):
                    p = int(GAME.board.current_player)
                ok, msg = GAME.edit(int(b.get("x", -1)), int(b.get("y", -1)), p)
                return self._send(200, {**GAME.snapshot(), "note": "" if ok else msg})
            if self.path == "/api/edit/undo":
                ok, msg = GAME.edit_undo()
                return self._send(200, {**GAME.snapshot(), "note": "" if ok else msg})
            if self.path == "/api/clear":
                ok, msg = GAME._replay([])
                return self._send(200, {**GAME.snapshot(), "note": "" if ok else msg})
        return self._send(404, {"error": "not found"})


def selftest(model=DEFAULT_MODEL):
    """自检：棋盘朝向 + 一手真对战。**发现 bug 就返回非 0**，别靠肉眼看棋盘。

    【为什么要有它】2026-10-12 修掉的 P0 是"棋盘沿主对角线镜像"：
    `Board.board` 是 numpy 数组、下标 `[行=y, 列=x]`，而 snapshot 写成了
    `board[x][y]` ⇒ 引擎完全正确、只有显示是转置的。它**只让 x≠y 的点错**，
    所以天元/对角线上看着一切正常，极易被当成"偶尔点不准"糊过去。
    人眼看棋盘抓不到这种错；固定坐标探针一抓就中。
    """
    from gomoku_env import Board

    fails = []

    # ---- ① 朝向：落子 (x,y) 必须出现在 (x,y)，且不能在转置点 (y,x) ----
    g = Game.__new__(Game)                      # 绕过 __init__，不加载模型
    g.board = Board()
    g.moves, g.over, g.winner = [], False, None
    g.last_ai = g.last_human = None
    g.human, g.n_sim, g.model_path = 1, 1, "-"
    g.hw, g.pw = PROD_HW, PROD_PW
    g.mcts = None
    g.ai_time, g.root_stats = 0.0, []
    g.engine, g.cpp_net, g.cpp_win = "python", "", None     # 自检不碰 C++
    g.value_estimate = lambda: None             # 不碰网络
    # 故意挑两个**不对称**的点：转置 bug 在这里必然暴露
    probes = [(3, 10, 1), (14, 2, 2)]
    for x, y, p in probes:
        if not g.board.move(x, y, p):
            fails.append(f"探针 ({x},{y}) 落子失败——测试本身有问题")
        g.moves.append((x, y, p))
    snap = g.snapshot()
    for x, y, p in probes:
        got = snap["rows"][18 - y][x]
        if got != p:
            fails.append(f"({x},{y}) 显示为 {got}，应为 {p}")
        if x != y and snap["rows"][18 - x][y] == p:
            fails.append(f"({x},{y}) 的棋子被画到了转置点 ({y},{x})")
    print(f"  [{'PASS' if not fails else 'FAIL'}] ① 棋盘朝向（探针 {[(a, b) for a, b, _ in probes]}）")
    if not fails:
        print(f"        (3,10)→rows[8][3]={snap['rows'][8][3]}　"
              f"(14,2)→rows[16][14]={snap['rows'][16][14]}")

    # ---- ② 真打一手：人类的子必须真在盘上，AI 必须回一手 ----
    n_orient = len(fails)
    try:
        game = Game(model, human=1, n_sim=200)
        ok, msg = game.human_move(9, 9)
        if not ok:
            fails.append(f"人类落子 (9,9) 被拒：{msg}")
        else:
            ai = game.ai_move()
            if ai is None:
                fails.append("AI 没有应答")
            else:
                s = game.snapshot()
                if s["rows"][18 - 9][9] != 1:
                    fails.append("人类的 (9,9) 没出现在快照里（朝向又错了？）")
                if s["rows"][18 - ai[1]][ai[0]] != 2:
                    fails.append(f"AI 的 {ai} 没出现在快照里")
                print(f"  [{'PASS' if len(fails) == n_orient else 'FAIL'}] "
                      f"② 真对战一手：人类 (9,9) → AI {ai}（{game.ai_time:.2f}s）")
    except Exception as exc:                    # noqa: BLE001
        fails.append(f"真对战一手抛异常：{exc}")

    # ---- ③ 分析层自检（朝向 / 视角 / 曲线，自带反向验证）----
    print()
    rc = analysis.self_test()
    if rc:
        fails.append(f"分析层自检有 {rc} 项未过（见上面输出）")

    # ---- ④ 分析接口端到端（真模型 + 真 _replay 路径）----
    # 用户看到的浮层就是这条链路：_replay 造局面 → analyze_board 出 361 点
    # → 前端按 (x,y) 贴到格子上。这里断言"必胜手落在正确坐标、且不在转置点"。
    try:
        game = Game(model, human=1, n_sim=200)
        probe = [(0, 5, 1), (1, 5, 1), (2, 5, 1), (3, 5, 1),
                 (0, 10, 2), (1, 10, 2), (2, 10, 2)]
        ok, msg = game._replay(probe)          # 黑 (0..3,5) 四连 ⇒ (4,5) 成五
        if not ok:
            fails.append(f"_replay 造局面失败：{msg}")
        else:
            res = game.analyze(n_sim=200)
            pts = {p["x"] * 100 + p["y"]: p for p in res["points"]}
            w5 = pts.get(4 * 100 + 5)
            tr = pts.get(5 * 100 + 4)
            probs = []
            if len(res["points"]) != 361:
                probs.append(f"points 应有 361 格，实际 {len(res['points'])}")
            if not w5 or not w5["w5"]:
                probs.append("(4,5) 没被标成『一步成五』")
            if tr and tr["w5"]:
                probs.append("转置点 (5,4) 被误标成成五点（朝向反了）")
            if not res["best"] or (res["best"]["x"], res["best"]["y"]) != (4, 5):
                probs.append(f"最佳点应为 (4,5)，实际 {res['best']}")
            if w5 and (w5["win"] is None or w5["win"] < 0.9):
                probs.append(f"(4,5) 必胜手的胜率只有 {w5['win']}（视角取反了？）")
            fails.extend(probs)
            print(f"  [{'PASS' if not probs else 'FAIL'}] "
                  f"④ 分析接口端到端：361 点 + 必胜手 (4,5)（{game.model_path}，"
                  f"{res['elapsed']}s，根访问 {res['root_n']}）")
    except Exception as exc:                    # noqa: BLE001
        fails.append(f"分析接口端到端抛异常：{exc}")

    if fails:
        print("\n>>> 【自检未过】")
        for f in fails:
            print(f"    · {f}")
        return 1
    print("\n>>> 【自检通过】棋盘朝向、落子、分析层（含反向验证）全部一致")
    return 0


def main():
    ap = argparse.ArgumentParser(description="网页版人机对战")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--human-white", action="store_true")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--hw", type=float, default=PROD_HW, help="规则先验权重（生产配置 0.5）")
    ap.add_argument("--pw", type=float, default=PROD_PW, help="策略先验权重（生产配置 1.0）")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ap.add_argument("--selftest", action="store_true",
                    help="只跑自检（棋盘朝向+一手对战）然后退出，不启动服务")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 70)
        print(f"web_play 自检　模型 {args.model}")
        print("=" * 70)
        return selftest(args.model)

    global GAME
    GAME = Game(args.model, 2 if args.human_white else 1, args.n_sim,
                hw=args.hw, pw=args.pw)
    url = f"http://{args.host}:{args.port}/"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"网页版已启动：{url}")
    print(f"模型 {args.model}　n_sim={args.n_sim}　"
          f"规则权重 hw={args.hw}/pw={args.pw}　"
          f"你执{'白' if args.human_white else '黑'}　（Ctrl+C 停止）")
    sys.stdout.flush()
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
