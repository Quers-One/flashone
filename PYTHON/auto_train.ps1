# auto_train.ps1 —— 无人值守训练监督器（P0）
#
# 它干四件事：
#   ① 有 resume.pt 就续跑，没有就从初值开始
#   ② 跑满当前代数上限就**自动 +GensPerRound 代继续**（代数上限递增，永不"跑完就停"）
#      例外：给了 -StopGen N 就跑到 gen N 收线（标尺照做，做完退出）。
#   ③ 进程崩了 / 被 kill 了 → 等 RetryDelay 秒自动重启，并从上一代继续（靠每代的 resume.pt）
#   ④ 每轮写独立日志 + 事件流（供外部（agent 会话/看门狗）推微信）
#
# 用法（单入口）：
#   pwsh -NoProfile -File auto_train.ps1 -RunName big6x128 -SaveDir checkpoints/pilot/big `
#        -Workers 8 -Blocks 6 -Channels 128 -GensPerRound 15
#
# 小规模自测（验证"自增代数 + 崩溃恢复"，1 分钟级）：
#   pwsh -NoProfile -File auto_train.ps1 -RunName autotest -SaveDir checkpoints/_autotest `
#        -Workers 1 -GensPerRound 1 -RetryDelay 20 `
#        -ExtraArgs '--games 2','--n-sim 20','--min-steps 5','--eval-games 2','--eval-every 1','--blocks 1','--channels 8'
#
# 事件流：<SaveDir>/auto_events.log（每行 `<时间> <事件> <详情>`，外部看门狗读它推送）
# 锁：<SaveDir>/auto_train.lock（写自己的 PID；发现锁里 PID 已死则自动接管）

param(
    [string]$RunName = 'big6x128',
    [string]$SaveDir = 'checkpoints/pilot/big',
    [int]$Workers = 8,
    [int]$Blocks = 6,
    [int]$Channels = 128,
    [int]$GensPerRound = 15,
    [int]$StartTarget = 0,            # 第一轮的代数上限（0 = 用 GensPerRound）。
                                      # **续训到新目录时必须给**：resume 已经在 gen 75 时，
                                      # 若第一轮 target=15，就是"目标已完成的空轮"，而空轮
                                      # 不写 train_log → 旧的 $reached 判定会判成 CRASH 并无限重启。
    [int]$MaxRounds = 0,              # 0 = 无限轮
    [int]$RetryDelay = 60,            # 崩溃后等待秒数
    [int]$Games = 40,
    [int]$NSim = 400,
    [int]$MinSteps = 1200,
    [int]$EvalGames = 20,
    [int]$EvalEvery = 2,
    [int]$EvalWorkers = 8,            # 门控对局并行进程数（2026-10-11 实测：4 → 8，门控阶段
                                      # CPU 仅 15%、GPU 47%，两头都闲；8 可让门控代接近减半）
    [int]$MaxGenCap = 2000,           # 代数上限的安全阀
    [int]$StopGen = 0,                # >0 = 跑满这一代就 STOP（不再 +GensPerRound）；
                                      # 0 = 老行为（永不自己停）。2026-10-11 加：V1 被判定
                                      # 到平台期后，需要"跑到 gen 75 就收线"这种能力。
    [string]$Seed = '12345',
    [int]$StartGen = 1,               # 本 run 第 1 代对外的代数（接着上一轮编号，如 61）
    [switch]$AutoEval,                # 每轮结束自动跑 auto_eval.ps1（标尺测量 + 硬门槛 + 晋升）
    [int]$RulerGames = 200,           # 标尺测量局数（200 局 ≈ ±50 Elo）
    [int]$RulerEveryRounds = 1,       # 每几轮测一次
    # ---- 【2026-10-12 加，V3 用】数据量/回放池/先验权重 ----
    # 依据：项目唯一被证明有效的改动就是"数据×2 + min-steps 重配"（gen80 打平 → gen120 +103.7）；
    # 而容量（更大网络）是负结果 ⇒ 数据受限，不是容量受限。这几个开关就是"放大数据"的实现。
    [int]$BufferGens = 0,             # >0 = 覆盖回放池保留代数（默认 0 = 不传，沿用 train.py 的 30）。
                                      # 数据×2 后必须一起放大（如 60），否则新数据边加边扔。
    [double]$FreshRatio = -1,         # >=0 = 覆盖"每 batch 里当代新数据的比例"（默认 -1 = 不传）
    [double]$SamplesPerGen = -1,      # >=0 = 覆盖"训练步数 = 新数据量 × 该系数"（默认 -1 = 不传）
    [double]$HeuristicWeight = -1,    # >=0 = 固定规则先验权重到这个值；配 -NoAnneal 用
                                      # （V3 建议 0：规则层占每手 CPU ~19%，而纯网络先验与融合先验实测打平）
    [switch]$NoAnneal,                # 关掉退火：全程用固定的 -HeuristicWeight/-PolicyWeight
    [double]$PolicyWeight = -1,       # >=0 = 固定策略先验权重（配 -NoAnneal）
    # 【2026-10-12 加】冷启动必须显式允许覆盖：train.py 的安全闸会拒绝往
    # "已有训练产物（best.pt/resume.pt/train_log.jsonl/gen_*.pt）"的目录里开新训练
    # （防止误覆盖历史数据）。V3 这类"人工准备好 best.pt 当初始权重"的冷启动**一定**会撞上它：
    # 现场症状是 train.py **exit=3 秒退**、监督器每 60 s 重启一次、什么也不训。
    [switch]$AllowOverwrite,
    [string]$ExtraArgs = ''           # 额外训练参数（空格分隔的字符串；自测/试验用）
)

$ErrorActionPreference = 'Continue'
# ============================================================================
# 【安全闸 · 2026-10-11 22:2x 加】
# 本脚本的 -SaveDir 默认值是 checkpoints/pilot/big —— 而它**不是普通目录**：
#   models\V1 是指向 checkpoints\pilot\big 的 Junction（fsutil 已核实：
#   同一个文件 ID、Reparse Tag 0xa0000003），两者是**同一个目录**。
# 所以"不带 -SaveDir 直接跑"会把新训练的 gen_*.pt / resume.pt / data / logs
# **直接写进 V1 那条线里**，把两条线搅在一起（而且是不可逆的覆盖）。
# 因此这里强制要求显式指定 -SaveDir。
if (-not $PSBoundParameters.ContainsKey('SaveDir')) {
    Write-Output "[ABORT] 必须显式指定 -SaveDir。"
    Write-Output "        旧默认值 'checkpoints/pilot/big' 与 'models/V1' 是同一个目录（junction），"
    Write-Output "        不带参数运行会把结果写进 V1 那条线。例：-SaveDir models/V2"
    exit 4
}
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
# 【必须设】否则下面 `*> $logFile` 重定向 python 的 UTF-8 输出时，PowerShell 会按系统
# 代码页(GBK)解码再存成 UTF-8 ⇒ 日志正文变双重编码（"数据:" 变成 "鏁版嵁:"）。
# 后果：日志中文不可读，全项目只能靠 ASCII 片段解析。2026-10-11 补上。
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SaveDir 'logs') | Out-Null
$events = Join-Path $SaveDir 'auto_events.log'
$lock = Join-Path $SaveDir 'auto_train.lock'

function Log([string]$ev, [string]$msg) {
    $line = "{0} {1} {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $ev, $msg
    Add-Content -Path $events -Value $line -Encoding utf8
    Write-Output $line
}

# ---- 单实例锁：锁里那个监督器**真的**还在跑才退出 ----
# 【2026-10-12 加固】旧写法只 `Get-Process -Id` 判 PID 是否存在，两个坑：
#   ① Windows 会复用 PID ⇒ 一个不相干的新进程可能"证明"锁还活着；
#   ② 监督器被强杀时 finally 跑不到、锁会留下 ⇒ 下次启动被自己的死锁拒掉
#      （用户 2026-10-12 遇到的「点开始训练没用」就是这个）。
# 现在锁里记 `PID|进程创建时间`，只有**两者都对上**才算"同一个监督器还在跑"。
function Get-LockInfo([string]$line) {
    $parts = "$line".Trim() -split '\|'
    return [pscustomobject]@{ Pid = $parts[0]; Created = if ($parts.Count -gt 1) { $parts[1] } else { '' } }
}
function Test-SupervisorAlive([string]$pidTxt, [string]$created) {
    if ($pidTxt -notmatch '^\d+$') { return $false }
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$pidTxt" -ErrorAction SilentlyContinue
    if (-not $p) { return $false }
    if ("$($p.CommandLine)" -notmatch 'auto_train\.ps1') { return $false }   # PID 被复用（别的进程）
    if ($created -and "$($p.CreationDate)" -ne $created) { return $false }   # 同一个 PID 但是新进程
    return $true
}
if (Test-Path $lock) {
    $oldLine = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    $li = Get-LockInfo $oldLine
    if (Test-SupervisorAlive $li.Pid $li.Created) {
        Log 'ABORT' "已有监督器在跑（PID $($li.Pid)），退出"
        exit 3
    }
    Log 'LOCK' "旧锁 PID=$($li.Pid) 已不是监督器（残留锁），接管"
}
$me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue
$stamp = if ($me) { "$($me.CreationDate)" } else { '' }
("$PID|$stamp") | Set-Content -Path $lock -Encoding utf8 -NoNewline
Log 'START' "run=$RunName workers=$Workers gens/round=$GensPerRound save=$SaveDir"

$target = if ($StartTarget -gt 0) { $StartTarget } else { $GensPerRound }
$round = 0
try {
    while ($true) {
        $round++
        if ($MaxRounds -gt 0 -and $round -gt $MaxRounds) {
            Log 'STOP' "达到 MaxRounds=$MaxRounds，退出"; break
        }
        if ($target -gt $MaxGenCap) { Log 'STOP' "代数上限 $MaxGenCap 安全阀触发"; break }

        # ---- 磁盘体检（长跑前必须）----
        $drive = (Resolve-Path $SaveDir).Path.Substring(0, 1)
        $freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
        if ($freeGB -lt 15) { Log 'WARN' "磁盘 $drive 只剩 ${freeGB}GB，暂停 10 分钟后再看"; Start-Sleep -Seconds 600; continue }

        $resume = Join-Path $SaveDir 'resume.pt'
        $logFile = Join-Path $SaveDir ("logs/round{0}_gen{1}.log" -f $round, $target)
        $common = @('train.py', '--games', "$Games", '--n-sim', "$NSim",
                    '--min-steps', "$MinSteps", '--eval-games', "$EvalGames",
                    '--eval-every', "$EvalEvery", '--eval-workers', "$EvalWorkers",
                    '--workers', "$Workers", '--max-gen', "$target",
                    '--anneal-start', '5', '--anneal-mid', '25', '--anneal-end', '45',
                    '--save-dir', $SaveDir, '--seed', $Seed,
                    '--blocks', "$Blocks", '--channels', "$Channels",
                    '--start-gen', "$StartGen")
        # ---- 【2026-10-12 加】放大数据 / 回放池 / 先验权重（只在显式给了参数时才追加）----
        # 不传就完全保持旧行为（V2 那些轮次的可复现性不受影响）。
        $tuneList = @()
        if ($BufferGens -gt 0) { $tuneList += @('--buffer-gens', "$BufferGens") }
        if ($FreshRatio -ge 0) { $tuneList += @('--fresh-ratio', "$FreshRatio") }
        if ($SamplesPerGen -ge 0) { $tuneList += @('--samples-per-gen', "$SamplesPerGen") }
        if ($NoAnneal) {
            $tuneList += @('--no-anneal')
            if ($HeuristicWeight -ge 0) { $tuneList += @('--heuristic-weight', "$HeuristicWeight") }
            if ($PolicyWeight -ge 0) { $tuneList += @('--policy-weight', "$PolicyWeight") }
        }
        if ($AllowOverwrite) { $tuneList += '--allow-overwrite' }
        $extraList = @()
        if ($ExtraArgs -and $ExtraArgs.Trim()) {
            $extraList = ($ExtraArgs.Trim() -split '\s+') | Where-Object { $_ }
        }
        $common = $common + $tuneList
        if (Test-Path $resume) {
            Log 'ROUND' "第 $round 轮：续训到 gen $target（日志 $logFile）"
            $cmdArgs = $common + @('--resume-from', $resume) + $extraList
        } else {
            Log 'ROUND' "第 $round 轮：从初值训到 gen $target（日志 $logFile）"
            $cmdArgs = $common + @('--fresh') + $extraList
        }
        Log 'CMDLINE' ("{0} gen{1}: {2}" -f $RunName, $target, ($cmdArgs -join ' '))

        $t0 = Get-Date
        python -u @cmdArgs *> $logFile
        $code = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

        # ---- 判定：真的跑到 target 了吗（只看日志，不信任退出码）----
        # 【2026-10-11 改】旧写法是字符串匹配 `"local_gen": $target,`，导致
        # "目标已完成的空轮"（比如续训目录里 resume 已经在 gen 75、而 target=15）
        # 被判成 CRASH → 无限重启。现在改成"日志里的**最大** local_gen >= target"，
        # 空轮也能正确判为 reached（然后 EXTEND 到下一个目标）。
        $reached = $false
        $maxGen = -1
        $logPath = Join-Path $SaveDir 'train_log.jsonl'
        if (Test-Path $logPath) {
            foreach ($line in (Get-Content $logPath -Encoding utf8)) {
                if (-not $line.Trim()) { continue }
                $m = [regex]::Match($line, '"local_gen"\s*:\s*(\d+)')
                if (-not $m.Success) { $m = [regex]::Match($line, '"gen"\s*:\s*(\d+)') }
                if ($m.Success) {
                    $v = [int]$m.Groups[1].Value
                    if ($v -gt $maxGen) { $maxGen = $v }
                }
            }
            $reached = ($maxGen -ge $target)
        }
        if ($reached) {
            Log 'DONE' ("第 $round 轮完成：gen $target（{0} 分钟，exit={1}）" -f $mins, $code)
            # ---- StopGen：跑到指定代数就收线（标尺测量照做，做完再退出）----
            $stopNow = ($StopGen -gt 0 -and $target -ge $StopGen)
            if ($stopNow) {
                Log 'STOP' "已到 StopGen=$StopGen：做完本轮标尺测量后停止（不再 +$GensPerRound）"
            } else {
                $target += $GensPerRound
                Log 'EXTEND' "代数上限 +$GensPerRound -> $target"
            }

            # ---- P1：轮边界自动测量（此时 GPU 空闲，测量不会被训练踩）----
            if ($AutoEval -and ($round % [math]::Max(1, $RulerEveryRounds) -eq 0)) {
                Log 'EVAL_TRIGGER' "轮边界触发标尺测量（$RulerGames 局）"
                # 【必须带 -NoPromote】项目现行政策是"晋升由人工决定"（watch_ruler 也传这个）。
                # 不带的话，ΔElo ≥ MinGain 时会自动把 champion 换成新模型 → 曲线基准漂移，
                # 后面所有点都不可比。
                pwsh -NoProfile -File (Join-Path $PSScriptRoot 'auto_eval.ps1') `
                    -RunName $RunName -SaveDir $SaveDir -Games $RulerGames -NoPromote *>> $logFile
                $ec = $LASTEXITCODE
                Log 'EVAL_RESULT' "auto_eval exit=$ec"
                if ($ec -eq 2) {
                    Log 'PAUSE' "硬门槛失败 -> 暂停训练等人工确认（删除 checkpoints/history/HARDGATE_FAIL.txt 即继续）"
                    while (Test-Path 'checkpoints/history/HARDGATE_FAIL.txt') { Start-Sleep -Seconds 300 }
                    Log 'RESUME' "HARDGATE_FAIL.txt 已删除，继续训练"
                }
            }
            if ($stopNow) { break }   # 标尺已做完 -> 结束监督链（finally 会清锁并写 EXIT）
        } else {
            Log 'CRASH' ("第 $round 轮未到 gen $target（{0} 分钟，exit={1}），{2}s 后重启续训" -f $mins, $code, $RetryDelay)
            Start-Sleep -Seconds $RetryDelay
        }
    }
} finally {
    Remove-Item -Force $lock -ErrorAction SilentlyContinue
    Log 'EXIT' "监督器退出（round=$round target=$target）"
}
