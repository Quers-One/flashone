# start_train.ps1 —— 由看板的「开始训练」按钮（经 serve_dashboard.py）调用，启动训练监督器。
#
# 【为什么不让 Python 直接 Popen】
# 实测：从 Python 服务里 DETACHED_PROCESS 直接拉 pwsh，子进程会"写了锁和第一轮日志后静默死掉"。
# 而 `Start-Process` 拉起的进程在本机一直稳定（已跑过整晚的训练）。所以把"起进程"这件事
# 交回 PowerShell，Python 只负责 HTTP + 校验。
#
# 用法（serve_dashboard.py 会这样调）：
#   pwsh -File start_train.ps1 -RunName V2 -SaveDir models/V2 -Workers 16 -Games 80 ... -AutoEval
#   pwsh -File start_train.ps1 -RunName V4 -SaveDir models/V4 ... -ExtraArgs '--buffer-decay-half-life 20'
# 输出末行：FLASHJSON {...}  便于调用方解析
param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$SaveDir,
    [int]$Workers = 16,
    [int]$Games = 80,
    [int]$MinSteps = 160,
    [int]$EvalGames = 20,
    [int]$EvalEvery = 2,
    [int]$EvalWorkers = 8,
    [int]$GensPerRound = 15,
    [int]$StartTarget = 0,
    [int]$RulerGames = 200,
    [int]$Seed = 12345,
    [switch]$AutoEval,
    [switch]$DryRun,
    # 【2026-10-12 加】脱离调用者进程树启动（WMI 创建 ⇒ 父进程是 WMI 提供程序）。
    # 为什么必须：训练常驻跑 1.7 h+，而起它的可能是"看板服务"甚至"DSH 会话本身"；
    # 只要训练还是它们的子进程，上层一退就可能被连带收走（用户实测到过）。
    [switch]$Detach,
    # 自检用：验证 Detach 链路真的能起一个脱离进程树的进程（跑 sleep，不训模型）
    [switch]$ProbeDetach,
    # ---- 【2026-10-12 加】V3 配方的高级参数（哨兵：负数 = 不传，0 是合法值）----
    [int]$BufferGens = -1,          # >0 才传：回放池保留代数
    [double]$HeuristicWeight = -1,  # >=0 才传：固定规则先验权重
    [double]$PolicyWeight = -1,     # >=0 才传：固定策略先验权重
    [switch]$NoAnneal,              # 关退火（配上面两个固定权重）
    [switch]$AllowOverwrite,        # 冷启动必须：train.py 默认拒绝往已有产物的目录开新训练
    # ---- 【2026-10-13 加】透传给 train.py 的额外参数（经 auto_train.ps1 的 -ExtraArgs 转发）----
    # 为什么必须补这一层：train.py 的新开关（`--aux-heads threat,ownership`、
    # `--buffer-decay-half-life 20`、`--buffer-entropy-weight 2`）**不在本脚本的具名参数表里**，
    # 而看板「开始训练」→ serve_dashboard.py → 本脚本 → auto_train.ps1 是**四级透传**：
    # 中间漏一级就"静默退回旧行为"（不报错、只是参数没生效）。
    [string]$ExtraArgs = '',
    [switch]$DryRunParam            # 自检用：只组装命令行并打印（不改任何状态、不启动）
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

function Fail([string]$msg) {
    Write-Output ("FLASHJSON " + ([ordered]@{ ok = $false; errors = @($msg) } | ConvertTo-Json -Compress))
    exit 1
}

# 【2026-10-12 加】判断"锁里那个 PID"是不是**真的还有监督器**。
# 旧写法只做 `Get-Process -Id`：只要系统里存在这个 PID 就判"有人在跑"。
# 但 Windows 会**复用 PID**，而"监督器被强杀、finally 没跑到"又会留下死锁
# ⇒ 症状就是用户遇到的「点开始训练没用」：每次都回"锁里 PID x 还活着"，
# 可训练进程一个都没有。现在要求 **PID 存在 且 命令行确实含 auto_train.ps1**，
# 这才是"还活着"的证据；否则一律当死锁清掉。
function Test-LockAlive([string]$pidTxt) {
    if ($pidTxt -notmatch '^\d+$') { return $false }
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$pidTxt" -ErrorAction SilentlyContinue
    if (-not $p) { return $false }
    $cl = "$($p.CommandLine)"
    if ($cl -match 'auto_train\.ps1') { return $true }
    # PID 被别的进程复用了 ⇒ 不是我们的监督器
    Write-Output ("[提示] 锁里的 PID {0} 现在是 {1}（不是监督器，判定为残留锁）" -f $pidTxt, $p.Name)
    return $false
}

# ---- ① 参数校验（与 serve_dashboard.py 同源的第二道防线）----
# SaveDir 只允许 models/<名字>：绝对路径、`..`、以及 checkpoints/pilot/big
#（它是 models/V1 的 junction 目标，写进去会污染 V1 那条线）全部拒绝。
if ($SaveDir -notmatch '^models/[A-Za-z0-9_.\-]{1,40}$') {
    Fail "SaveDir 非法：必须是 models/<名字> 形状（收到 '$SaveDir'）"
}
if ($SaveDir.TrimEnd('/').ToLower() -in @('models/v1', 'checkpoints/pilot/big')) {
    Fail "SaveDir 指向 V1 所在目录（junction 陷阱），拒绝启动"
}
if ($RunName -notmatch '^[A-Za-z0-9_.\-]{1,32}$') { Fail "RunName 非法：'$RunName'" }
# 【2026-10-13 加】额外参数白名单：它最终会变成 train.py 的命令行片段，
# 必须挡住引号/反引号/$( )/; | & 这类能"逃出去"的字符（同源的第二道防线）。
if ($ExtraArgs -and $ExtraArgs.Trim() -notmatch '^[A-Za-z0-9_\-=,./ ]{1,200}$') {
    Fail ("ExtraArgs 含不允许的字符（只允许字母/数字/下划线/短横/=/,/./空格）：'$ExtraArgs'")
}
foreach ($pair in @(@('Workers', $Workers, 1, 64), @('Games', $Games, 1, 2000), @('MinSteps', $MinSteps, 1, 200000),
        @('EvalGames', $EvalGames, 1, 2000), @('EvalEvery', $EvalEvery, 1, 100),
        @('EvalWorkers', $EvalWorkers, 1, 64), @('GensPerRound', $GensPerRound, 1, 500),
        @('RulerGames', $RulerGames, 1, 2000))) {
    if ($pair[1] -lt $pair[2] -or $pair[1] -gt $pair[3]) {
        Fail ("{0} 超出范围 [{1}, {2}]：{3}" -f $pair[0], $pair[2], $pair[3], $pair[1])
    }
}

# ---- ② StartTarget=0 视为自动：当前最新代数 + GensPerRound ----
$maxGen = 0
$d = Join-Path $PSScriptRoot $SaveDir
if (Test-Path $d) {
    foreach ($f in (Get-ChildItem (Join-Path $d 'gen_*.pt') -ErrorAction SilentlyContinue)) {
        if ($f.BaseName -match '^gen_(\d+)$') { $maxGen = [Math]::Max($maxGen, [int]$Matches[1]) }
    }
}
if ($StartTarget -le 0) { $StartTarget = $maxGen + $GensPerRound }
if ($StartTarget -le $maxGen) {
    Fail ("StartTarget({0}) 不大于当前最新 gen({1})，无可训的代数。把「本轮目标代数」留空（=自动取 {1}+每轮代数）或填更大的值。" -f `
            $StartTarget, $maxGen)
}

# ---- ③ 组命令（先组好，-DryRun 要原样打印它）----
$argList = @('-NoLogo', '-NoProfile', '-File', (Join-Path $PSScriptRoot 'auto_train.ps1'),
    '-RunName', $RunName, '-SaveDir', $SaveDir,
    '-Workers', "$Workers", '-Games', "$Games", '-MinSteps', "$MinSteps",
    '-EvalGames', "$EvalGames", '-EvalEvery', "$EvalEvery", '-EvalWorkers', "$EvalWorkers",
    '-GensPerRound', "$GensPerRound", '-StartTarget', "$StartTarget",
    '-RulerGames', "$RulerGames", '-Seed', "$Seed")
if ($AutoEval) { $argList += '-AutoEval' }
# ---- V3 高级参数：负数 = 不传（0 是合法值，例如 hw=0）----
if ($BufferGens -gt 0) { $argList += @('-BufferGens', "$BufferGens") }
if ($HeuristicWeight -ge 0) { $argList += @('-HeuristicWeight', "$HeuristicWeight") }
if ($PolicyWeight -ge 0) { $argList += @('-PolicyWeight', "$PolicyWeight") }
if ($NoAnneal) { $argList += '-NoAnneal' }
if ($AllowOverwrite) { $argList += '-AllowOverwrite' }
# 【2026-10-13 加】额外参数整串交给 auto_train.ps1（它按空格拆开拼到 train.py 后面）
if ($ExtraArgs) { $argList += @('-ExtraArgs', $ExtraArgs) }

$cmdText = "auto_train.ps1 -RunName $RunName -SaveDir $SaveDir -Workers $Workers -Games $Games -MinSteps $MinSteps " +
"-EvalGames $EvalGames -EvalEvery $EvalEvery -EvalWorkers $EvalWorkers -GensPerRound $GensPerRound " +
"-StartTarget $StartTarget -RulerGames $RulerGames -Seed $Seed" + $(if ($AutoEval) { ' -AutoEval' } else { '' }) +
$(if ($BufferGens -gt 0) { " -BufferGens $BufferGens" } else { '' }) +
$(if ($HeuristicWeight -ge 0) { " -HeuristicWeight $HeuristicWeight" } else { '' }) +
$(if ($PolicyWeight -ge 0) { " -PolicyWeight $PolicyWeight" } else { '' }) +
$(if ($NoAnneal) { ' -NoAnneal' } else { '' }) +
$(if ($AllowOverwrite) { ' -AllowOverwrite' } else { '' }) +
$(if ($ExtraArgs) { " -ExtraArgs '$ExtraArgs'" } else { '' })

$running = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match '-File\s+\S*auto_train\.ps1' -or $_.CommandLine -match '-u\s+train\.py')
})

# ---- ④ -DryRun：只预览，不改任何状态（必须在"已在跑/清锁"之前）----
if ($DryRun) {
    Write-Output ("FLASHJSON " + ([ordered]@{
                ok = $true; dry_run = $true; pid = $null; args = $argList; summary = $cmdText
                resume_from = (Join-Path $d 'resume.pt'); start_target = $StartTarget; current_gen = $maxGen
                detach = [bool]$Detach; training_running = [bool]$running
                extra_args = $ExtraArgs
            } | ConvertTo-Json -Compress))
    exit 0
}

# ---- ④b -DryRunParam：只验证"看板参数 → auto_train 命令行"的组装（不启动）----
# 为什么需要：V3 的 5 个高级参数（BufferGens/hw/pw/NoAnneal/AllowOverwrite）是
# serve_dashboard.py → start_train.ps1 → auto_train.ps1 三级透传，
# 中间漏一个就会**静默退回旧行为**（不报错、只是参数没生效）。这个开关让验收脚本能查出来。
if ($DryRunParam) {
    Write-Output ("FLASHJSON " + ([ordered]@{
                ok = $true; dry_run_param = $true; args = $argList; summary = $cmdText
                start_target = $StartTarget; current_gen = $maxGen
                buffer_gens = $BufferGens; heuristic_weight = $HeuristicWeight
                policy_weight = $PolicyWeight; no_anneal = [bool]$NoAnneal
                allow_overwrite = [bool]$AllowOverwrite
                extra_args = $ExtraArgs
            } | ConvertTo-Json -Compress))
    exit 0
}

# ---- ⑤ 已在跑就拒绝 / ⑥ 清残留锁 ----
# 【顺序很关键】先查"真有没有监督器在跑"，再处理锁。
# 反过来的话，锁残留时用户看到的是"锁里 PID x 还活着"（误导），
# 而真正的原因（没有训练在跑、只是锁脏了）反而看不到。
if ($running.Count -gt 0) { Fail ("已有训练在跑（{0} 个进程），先停止再启动" -f $running.Count) }

# 走到这里 = 没有任何监督器/train.py 在跑 ⇒ 锁一定是**残留**（强杀留下的），直接清。
# 仍然打印一行提示，方便事后核对（谁留下的、当时是不是活过）。
$lock = Join-Path $d 'auto_train.lock'
if (Test-Path $lock) {
    $pidTxt = (Get-Content $lock -Raw -ErrorAction SilentlyContinue).Trim()
    $alive = Test-LockAlive $pidTxt
    if ($alive) {
        # 理论到不了这里（上面已经确认没有监督器）；保守起见仍然拒绝，并把两个事实都写出来
        Fail ("锁里 PID {0} 看起来还是监督器，但进程扫描没找到；为安全拒绝启动。可手工删除 {1} 后重试" -f $pidTxt, $lock)
    }
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
    Write-Output ("[提示] 清掉残留锁（PID {0} 已不是监督器）" -f $pidTxt)
}

# ---- ⑤b 起进程：Detach ⇒ 用 WMI 创建，父进程是 WMI 提供程序（不是本进程）----
# 【为什么要绕这一下】普通 Start-Process 起出来的进程是本进程的**子进程**，
# 于是它的生命周期被绑在上层身上：看板服务/终端/DSH 会话一退，Windows 可能把整棵
# 进程树收走。用户 2026-10-12 就吃到这个亏（训练在看板重启前后被 STOP 掉）。
# 用 Win32_Process::Create 创建的进程，父进程是 WMI 提供程序 ⇒ 与调用者彻底解耦。
$procId = $null
$parentId = $null
$launchMode = 'child'

function Start-Detached([string[]]$argv, [string]$stdoutPath) {
    # 用单引号包裹每个参数（外部命令的参数是**字面量**，引号必须在字符串里）
    $parts = @()
    foreach ($a in $argv) {
        $s = "$a"
        if ($s -match '^[A-Za-z0-9_.:/\\\-]+$') { $parts += $s } else { $parts += ("'" + $s.Replace("'", "''") + "'") }
    }
    # 输出重定向到文件；PowerShell 用 *> 把 stdout+stderr 都收进去
    $cmdLine = ("& '" + (Join-Path $PSScriptRoot 'auto_train.ps1') + "' " + ($parts -join ' ') +
                " *> '" + $stdoutPath + "'")
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = ('pwsh -NoLogo -NoProfile -NonInteractive -Command "' + $cmdLine.Replace('"', '\"') + '"')
        CurrentDirectory = $PSScriptRoot
    }
    return $r
}

if ($Detach -or $ProbeDetach) {
    $detLog = Join-Path $SaveDir ("logs/detach_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Force -Path (Split-Path $detLog -Parent) | Out-Null
    if ($ProbeDetach) {
        # 自检：起一个只睡 45 秒的进程，用来验证"脱离进程树"这条链路本身是否可用
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine      = 'pwsh -NoLogo -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 45"'
            CurrentDirectory = $PSScriptRoot
        }
    } else {
        $r = Start-Detached $argList $detLog
    }
    if ($r -and $r.ReturnValue -eq 0) {
        $procId = [int]$r.ProcessId
        $launchMode = 'detached'
        Start-Sleep -Milliseconds 400
        try {
            $parentId = (Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction Stop).ParentProcessId
        } catch { $parentId = $null }
    } else {
        # 脱离失败不致命：退回旧的子进程方式，但要留下明确痕迹
        $rc = if ($r) { $r.ReturnValue } else { 'null' }
        Write-Output ("[WARN] Detach 失败（Win32_Process.Create 返回 $rc），退回 Start-Process 子进程方式")
        if ($ProbeDetach) { Fail "ProbeDetach 失败：Win32_Process.Create 返回 $rc" }
        $Detach = $false
    }
}

if ($ProbeDetach) {
    Write-Output ("FLASHJSON " + ([ordered]@{
                ok = $true; probe_detach = $true; pid = $procId; parent_pid = $parentId
                launch_mode = $launchMode; my_pid = $PID
                parent_is_me = ($parentId -eq $PID)
            } | ConvertTo-Json -Compress))
    exit 0
}

if (-not $procId) {
    $proc = Start-Process -FilePath 'pwsh' -ArgumentList $argList -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden -PassThru -ErrorAction Stop
    if (-not $proc) { Fail 'Start-Process 未返回进程对象' }
    $procId = $proc.Id
    Start-Sleep -Milliseconds 300
    try {
        $parentId = (Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction Stop).ParentProcessId
    } catch { $parentId = $null }
}

# ---- ⑥ 记下"上次参数"（网页的「沿用上轮参数」就读它）----
$paramFile = Join-Path $PSScriptRoot 'checkpoints\last_train_params.json'
New-Item -ItemType Directory -Force -Path (Split-Path $paramFile -Parent) | Out-Null
[ordered]@{
    saved_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); pid = $procId
    launch_mode = $launchMode; parent_pid = $parentId
    params = [ordered]@{
        RunName = $RunName; SaveDir = $SaveDir; Workers = $Workers; Games = $Games
        MinSteps = $MinSteps; EvalGames = $EvalGames; EvalEvery = $EvalEvery
        EvalWorkers = $EvalWorkers; GensPerRound = $GensPerRound; StartTarget = $StartTarget
        RulerGames = $RulerGames; Seed = $Seed; AutoEval = [bool]$AutoEval
        # 【2026-10-12 补】这 5 个以前没记下来 ⇒ 「沿用上轮参数」拿不到、看板也显示不出
        # 完整配方（-1 = 不传/用退火或默认）。当天就是因此看不清"在跑的 hw 是多少"。
        BufferGens = $BufferGens; HeuristicWeight = $HeuristicWeight
        PolicyWeight = $PolicyWeight
        NoAnneal = [bool]$NoAnneal; AllowOverwrite = [bool]$AllowOverwrite
        # 【2026-10-13 补】同理：不记下来「沿用上轮参数」就拿不到，看板也显示不出完整配方
        ExtraArgs = $ExtraArgs
    }
} | ConvertTo-Json -Depth 4 | Set-Content -Path $paramFile -Encoding utf8

# 日志一定要有编码头，否则英文环境下按 GBK 读会变乱码（2026-10-12 实测过）
$encNote = "=== 本行由 start_train.ps1 写于 {0}（UTF-8）===" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Add-Content -Path (Join-Path $PSScriptRoot 'checkpoints\history\dashboard_launch.log') -Value $encNote -Encoding utf8

Write-Output ("FLASHJSON " + ([ordered]@{
            ok = $true; pid = $procId; parent_pid = $parentId; launch_mode = $launchMode
            detached = ($launchMode -eq 'detached')
            args = $argList; summary = $cmdText
            start_target = $StartTarget; current_gen = $maxGen
            resume_from = (Join-Path $d 'resume.pt')
        } | ConvertTo-Json -Compress))
