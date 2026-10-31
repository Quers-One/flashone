# stop_train.ps1 —— 停止训练（供看板的「停止训练」按钮调用，也可手工跑）
#
# 【顺序至关重要】必须先杀监督器（auto_train.ps1），再杀 train.py。
# 顺序反了：监督器发现 train.py 没到 target 就退出 ⇒ 判定成 CRASH ⇒ 立刻重启训练。
#
# 输出：末尾打一行 JSON，便于调用方（serve_dashboard.py）原样返回给网页。
param(
    [switch]$KeepEval,      # 保留正在跑的评测（默认也一起停，符合"停止"的直觉）
    [switch]$Quiet
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$mine = $PID
$killed = [ordered]@{ 监督器 = 0; 训练 = 0; 训练子进程 = 0; 评测 = 0 }
$errs = @()

function Kill-List($procs, [string]$kind) {
    foreach ($p in $procs) {
        if ($p.ProcessId -eq $mine) { continue }
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            $killed[$kind] = $killed[$kind] + 1
            if (-not $Quiet) { Write-Output ("  已停 {0} PID {1}" -f $kind, $p.ProcessId) }
        } catch { $script:errs += ("{0} PID {1}: {2}" -f $kind, $p.ProcessId, $_.Exception.Message) }
    }
}

function Find-Procs([string]$namePattern, [string]$cmdPattern) {
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessId -ne $mine -and $_.Name -like $namePattern -and $_.CommandLine -and $_.CommandLine -match $cmdPattern
    })
}

function Find-Supervisors {
    # 【2026-10-13 加固】必须同时认两种真实调用形状 —— watch_progress.ps1 早就认两种了，这里只认一种。
    #   A) pwsh -File    ...\auto_train.ps1 ...
    #   B) pwsh -Command "& '...\auto_train.ps1' ..."（start_train.ps1 的 WMI 脱离进程方式）
    # 只认 A 的后果：形状 B 的监督器杀不掉 ⇒ train.py 被杀后它判定 CRASH，
    # **60 秒后把训练重新拉起来**，用户看到的是"点了停止训练、训练又自己回来了"。
    # 为什么不能搜裸子串 'auto_train.ps1'：本工具自己和 DSH shell 的命令行里就含这段文本。
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.ProcessId -ne $mine -and $_.Name -like 'pwsh*' -and $_.CommandLine -and (
                $_.CommandLine -match '-File\s+\S*auto_train\.ps1' -or
                $_.CommandLine -match '-Command\s+"&\s*''[^'']*auto_train\.ps1''')
        })
}

$runDirs = @()      # 从监督器命令行里解析出的 run 目录，供 ⑥ 把审计事件写进对的那个 run

# ---- ① 监督器（必须最先）----
$supProcs = Find-Supervisors
foreach ($sp in $supProcs) {
    if ($sp.CommandLine -match '-SaveDir\s+[''"]?([^''"\s]+)') { $runDirs += $Matches[1].TrimEnd('/') }
}
Kill-List $supProcs '监督器'
Start-Sleep -Milliseconds 500

# ---- ② 评测类（auto_eval / ladder / 硬门槛测试）----
if (-not $KeepEval) {
    $evalProcs = @()
    $evalProcs += Find-Procs 'pwsh*' '-File\s+\S*auto_eval\.ps1'
    $evalProcs += Find-Procs 'python*' '-u\s+ladder\.py'
    $evalProcs += Find-Procs 'python*' '-u\s+compare\.py'
    $evalProcs += Find-Procs 'python*' '-u\s+test_(tactics|selfplay)\.py'
    Kill-List $evalProcs '评测'
    Start-Sleep -Milliseconds 500
}

# ---- ③ train.py 主进程 + 它的 spawn 子进程 ----
Kill-List (Find-Procs 'python*' '-u\s+train\.py') '训练'
Kill-List (Find-Procs 'python*' '(from multiprocessing|multiprocessing\.spawn)') '训练子进程'
Start-Sleep -Seconds 2

# ---- ④ 清残留锁（force kill 会让监督器的 finally 跑不到）----
$lockCleared = $false
# 【2026-10-13 修】原来只列了 V2/V1/run3 —— V3 的锁永远清不掉。
# 实测：停掉 V3 后报 lock_cleared=False，锁里躺着已经死掉的 PID 21292。
# 改成"扫 models\* 下的所有 run"，以后新增 run 不必再改这里。
$lockDirs = @(Get-ChildItem (Join-Path $PSScriptRoot 'models') -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path 'models' $_.Name })
$lockDirs += @('models\V2', 'models\V1', 'models\run3')
foreach ($d in ($lockDirs | Select-Object -Unique)) {
    $lock = Join-Path $d 'auto_train.lock'
    if (Test-Path $lock) {
        $pidTxt = (Get-Content $lock -Raw -ErrorAction SilentlyContinue).Trim()
        $alive = $false
        if ($pidTxt -match '^\d+$') { $alive = [bool](Get-Process -Id ([int]$pidTxt) -ErrorAction SilentlyContinue) }
        if (-not $alive) { Remove-Item $lock -Force -ErrorAction SilentlyContinue; $lockCleared = $true }
    }
}

# ---- ⑤ 复查 ----
$leftTrain = @(Find-Procs 'python*' '-u\s+train\.py').Count
# 必须和 ① 用同一套形状匹配，否则形状 B 的监督器会被漏掉 ⇒ 误报"已停干净"、ok=true
$leftSup = @(Find-Supervisors).Count

# ---- ⑥ 审计事件 ----
$summary = ("STOP_MANUAL 停止训练：监督器={0} 训练={1} 子进程={2} 评测={3}；复查剩余 train.py={4} auto_train={5}；锁清理={6}" -f `
        $killed['监督器'], $killed['训练'], $killed['训练子进程'], $killed['评测'], $leftTrain, $leftSup, $lockCleared)
# 【2026-10-13 修】原来硬编码写进 models\V2\auto_events.log —— 停 V3 的审计事件
# 记到了 V2 的日志里，排查时会被彻底带偏。改为写进"监督器命令行里那个 -SaveDir"。
$evTargets = if ($runDirs.Count) { $runDirs } else { @('models\V2') }
foreach ($rd in ($evTargets | Select-Object -Unique)) {
    $evLog = Join-Path $PSScriptRoot (Join-Path $rd 'auto_events.log')
    try { Add-Content -Path $evLog -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $summary) -Encoding utf8 } catch { }
}

$result = [ordered]@{
    ok = ($leftTrain -eq 0 -and $leftSup -eq 0)
    killed = $killed
    lock_cleared = $lockCleared
    left_train = $leftTrain
    left_supervisor = $leftSup
    errors = $errs
    summary = $summary
}
if (-not $Quiet) { Write-Output $summary }
Write-Output ("FLASHJSON " + ($result | ConvertTo-Json -Compress))
