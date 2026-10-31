# train_task.ps1 —— 用 **Windows 计划任务**启动训练（真正脱离 DSH / 终端 / 看板）
#
# 【为什么不用 WMI/Win32_Process.Create】
# 2026-10-12 实测（两次，日志都是"腰斩"：无 traceback、无 STOP 事件、锁残留）：
# DSH 会把**每条命令**放进一个 Windows **Job 对象**里跑，命令结束时收掉整个 Job。
# `Win32_Process.Create` 建出来的进程**仍然是那个 Job 的成员**（WMI 没有用
# CREATE_BREAKAWAY_FROM_JOB，普通权限下也用不了 breakaway），所以父进程虽然显示
# `WmiPrvSE.exe`，人一关终端照样被杀。唯一的可靠办法是**换一个不属于该 Job 的根**：
# 计划任务由「任务计划服务」拉起，天然独立于任何终端/应用。
#
# 用法：
#   pwsh -NoProfile -File train_task.ps1                      # 默认口径：续训、目标自动
#   pwsh -NoProfile -File train_task.ps1 -GensPerRound 12
#   pwsh -NoProfile -File train_task.ps1 -StartTarget 180
#   pwsh -NoProfile -File train_task.ps1 -Status               # 看任务与训练状态
#   pwsh -NoProfile -File train_task.ps1 -Stop                 # 停训练（含任务）
#   pwsh -NoProfile -File train_task.ps1 -Remove               # 删任务
param(
    [string]$RunName = 'V2',
    [string]$SaveDir = 'models/V2',
    [int]$Workers = 16,
    [int]$Games = 80,
    [int]$MinSteps = 160,
    [int]$EvalGames = 20,
    [int]$EvalEvery = 2,
    [int]$EvalWorkers = 8,
    [int]$GensPerRound = 12,
    [int]$StartTarget = 0,
    [int]$RulerGames = 200,
    [int]$Seed = 12345,
    [switch]$NoAutoEval,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Remove
)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

$TASK = 'FlashGomokuTrain'
$logDir = Join-Path $PSScriptRoot 'checkpoints\history'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$taskOut = Join-Path $logDir 'train_task.out.log'

function Say([string]$m) { Write-Output ("[计划任务] " + $m) }

function Show-Status {
    $t = Get-ScheduledTask -TaskName $TASK -ErrorAction SilentlyContinue
    if ($t) {
        $i = Get-ScheduledTaskInfo -TaskName $TASK -ErrorAction SilentlyContinue
        Say ("任务存在：State=$($t.State) 上次运行=$($i.LastRunTime) 上次结果=$($i.LastTaskResult)")
    } else { Say "任务不存在（$TASK）" }
    $sup = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-File\s+\S*auto_train\.ps1' })
    $tr = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match '\s-u\s+train\.py' })
    Say ("训练：监督器=$($sup.Count) train.py=$($tr.Count)")
    foreach ($s in $sup) {
        $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$($s.ParentProcessId)" -ErrorAction SilentlyContinue).Name
        Say ("  监督器 PID $($s.ProcessId) 父进程=$par（计划任务拉起时应是 svchost/taskeng）")
    }
    $d = Join-Path $PSScriptRoot $SaveDir
    $newest = Get-ChildItem (Join-Path $d 'gen_*.pt') -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace 'gen_', '') } | Select-Object -Last 1
    if ($newest) { Say ("最新模型：$($newest.Name)（$($newest.LastWriteTime)）") }
}

if ($Status) { Show-Status; exit 0 }

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TASK -Confirm:$false -ErrorAction SilentlyContinue
    Say "任务已删除"
    Show-Status
    exit 0
}

if ($Stop) {
    Unregister-ScheduledTask -TaskName $TASK -Confirm:$false -ErrorAction SilentlyContinue
    Say "任务已删除；接下来停训练"
    & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'stop_train.ps1')
    Show-Status
    exit 0
}

# ---- ① 已在跑/残留锁：先看有没有活着的训练 ----
$live = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and ($_.CommandLine -match '-File\s+\S*auto_train\.ps1' -or
            ($_.Name -like 'python*' -and $_.CommandLine -match '\s-u\s+train\.py'))
    })
if ($live.Count -gt 0) {
    Say ("已有训练在跑（$($live.Count) 个进程），不重复启动。要看状态用 -Status")
    exit 0
}

# ---- ② 组命令行（与看板/start_train.ps1 同源参数）----
$a = @('-NoLogo', '-NoProfile', '-File', (Join-Path $PSScriptRoot 'start_train.ps1'),
    '-RunName', $RunName, '-SaveDir', $SaveDir,
    '-Workers', "$Workers", '-Games', "$Games", '-MinSteps', "$MinSteps",
    '-EvalGames', "$EvalGames", '-EvalEvery', "$EvalEvery", '-EvalWorkers', "$EvalWorkers",
    '-GensPerRound', "$GensPerRound", '-StartTarget', "$StartTarget",
    '-RulerGames', "$RulerGames", '-Seed', "$Seed")
if (-not $NoAutoEval) { $a += '-AutoEval' }
# 计划任务里**不要** -Detach：任务本身就是独立的根，再套一层没必要
$argText = ($a | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '

# ---- ③ 注册任务并立刻跑 ----
# 【为什么用 AtLogOn 触发器 + "过去的时间"】计划服务允许触发器时间在过去 ⇒ 立即启动；
# 不依赖当前登录会话的交互式主体（`-User <当前用户>` 走默认主体，实测可用；
# `New-ScheduledTaskPrincipal -UserId` 对 `域\用户` 格式会报"参数错误 (16,8):UserId"）。
$action = New-ScheduledTaskAction -Execute 'pwsh' -Argument $argText -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -MultipleInstances IgnoreNew

Unregister-ScheduledTask -TaskName $TASK -Confirm:$false -ErrorAction SilentlyContinue
try {
    # 【不加 -User】本机账号是微软账户（whoami 显示 luopcc\luopcc），
    # 传 -User 时注册器会报"参数错误 (16,8):UserId"；省略即默认当前用户。
    Register-ScheduledTask -TaskName $TASK -Action $action -Trigger $trigger -Settings $settings `
        -Description '五子棋 AlphaZero 训练（独立于终端/会话）' -Force | Out-Null
} catch {
    Say ("★ Register-ScheduledTask 失败：$($_.Exception.Message)")
    Say "  改用 schtasks.exe 兜底…"
    $st = (Get-Date).AddMinutes(2).ToString('HH:mm')
    $tr = ('pwsh -NoProfile -File "{0}" {1}' -f (Join-Path $PSScriptRoot 'start_train.ps1'), ($a -join ' '))
    $r = & schtasks /create /tn $TASK /tr $tr /sc once /st $st /it /f 2>&1
    Say ("  schtasks: " + ($r -join ' | '))
    if ($LASTEXITCODE -ne 0) {
        Say "  ★ 两种注册方式都失败。退路：在你自己的 PowerShell 里直接跑 ——"
        Say ("  pwsh -NoProfile -File `"$(Join-Path $PSScriptRoot 'train_detached.ps1')`" -GensPerRound $GensPerRound")
        exit 1
    }
    & schtasks /run /tn $TASK 2>&1 | Out-String | ForEach-Object { Say ("  schtasks /run: " + $_.Trim()) }
}
Start-ScheduledTask -TaskName $TASK -ErrorAction SilentlyContinue
Say "任务已注册并启动：$TASK"

# 任务跑完（curl 完整条链）就把它删掉，避免积累
Start-Job -ScriptBlock {
    param($t)
    for ($i = 0; $i -lt 720; $i++) {          # 最多等 2 小时
        Start-Sleep -Seconds 10
        $s = (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue).State
        if (-not $s -or $s -eq 'Ready') { break }
    }
    Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue
} -ArgumentList $TASK | Out-Null

# ---- ④ 等它真起来（最多 40 s），并核对父进程不是本会话 ----
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    $sup = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-File\s+\S*auto_train\.ps1' })
    if ($sup.Count -gt 0) { $ok = $true; break }
    Start-Sleep -Seconds 1
}
if ($ok) {
    Start-Sleep -Seconds 2
    $sup = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-File\s+\S*auto_train\.ps1' })[0]
    $par = Get-CimInstance Win32_Process -Filter "ProcessId=$($sup.ParentProcessId)" -ErrorAction SilentlyContinue
    Say ("训练已启动：监督器 PID $($sup.ProcessId)，父进程 = $($par.Name)（PID $($par.ProcessId)）")
    Say "**它的生命周期归任务计划服务，关终端/关 DSH 都不影响**"
} else {
    Say "★ 40 秒内没看到监督器起来，看 $taskOut 与 checkpoints\history\dashboard_launch.log"
}
Show-Status
