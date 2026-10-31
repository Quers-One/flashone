# train_detached.ps1 —— 从**用户自己的终端**一条命令启动训练（真正脱离 DSH 的 Job）
#
# ⚠️ 2026-10-12 血泪教训：**必须在你自己的 PowerShell 窗口里跑这个脚本**。
#    DSH 会把每条命令放在一个 Windows Job 对象里，Job 一收尾就连带杀掉里面的进程
#    （WMI `Win32_Process.Create` 也一样，父进程显示 WmiPrvSE 也没用）。
#    由 DSH 会话发起的任何启动方式都活不过会话收尾；`explorer.exe` 起的 shell 不在该 Job 内。
#
# 用法（默认 = V2 续训，口径与看板/start_train.ps1 一致）：
#   pwsh -NoProfile -File train_detached.ps1
#   pwsh -NoProfile -File train_detached.ps1 -GensPerRound 12
#   pwsh -NoProfile -File train_detached.ps1 -StartTarget 180        # 显式目标
#   pwsh -NoProfile -File train_detached.ps1 -DryRun                 # 只看它会跑什么（不启动）
#
# 用法（V3：同一个模型 + 放大数据；见 NOTES「V3 设计」）：
#   pwsh -NoProfile -File train_detached.ps1 -RunName V3 -SaveDir models/V3 `
#        -GensPerRound 30 -Games 160 -MinSteps 320 -BufferGens 60 `
#        -HeuristicWeight 0 -PolicyWeight 1.0 -NoAnneal
#
# 说明：内部用 WMI 脱离创建进程（父进程 = WmiPrvSE），并直接调用 `auto_train.ps1`
#       （因此支持 V3 的新参数：数据量 / 回放池 / 固定先验权重）。
#
# 【2026-10-13 加】`-ExtraArgs` = 透传给 train.py 的额外开关（本脚本与看板表单都没具名的那些）：
#   pwsh -NoProfile -File train_detached.ps1 -RunName V4 -SaveDir models/V4 -GensPerRound 30 `
#        -Games 160 -MinSteps 320 -BufferGens 60 -HeuristicWeight 0.5 -PolicyWeight 1 -NoAnneal `
#        -ExtraArgs '--buffer-decay-half-life 20'
#   `auto_train.ps1` 本来就支持 -ExtraArgs，这里只是把这一层补上（否则双击启动永远用不上新开关）。
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
    [int]$StartTarget = 0,          # 0 = 自动（当前最新代数 + GensPerRound）
    [int]$RulerGames = 200,
    [int]$Seed = 12345,
    # ---- V3 用：放大数据 / 回放池 / 先验权重（不传 = 旧行为）----
    [int]$BufferGens = 0,
    [double]$FreshRatio = -1,
    [double]$SamplesPerGen = -1,
    [double]$HeuristicWeight = -1,
    [double]$PolicyWeight = -1,
    [switch]$NoAnneal,
    [switch]$AllowOverwrite,        # 冷启动必须给：train.py 默认拒绝往已有产物的目录里开新训练
    [int]$NSim = 400,
    [int]$Blocks = 6,
    [int]$Channels = 128,
    # ---- 【2026-10-13 加】透传给 train.py 的额外参数 ----
    # 为什么要它：train.py 的新开关（如 `--aux-heads threat,ownership`、`--buffer-decay-half-life 20`）
    # 不在本脚本的具名参数表里，而 `auto_train.ps1` 本来就支持 `-ExtraArgs`（原样拼进 train.py 命令行）。
    # 不补这一层，**双击 `Flash V3续训.cmd` 就永远用不上新开关**（静默退回旧行为，不报错）。
    # 例：-ExtraArgs '--buffer-decay-half-life 20 --aux-heads threat,ownership'
    [string]$ExtraArgs = '',
    [switch]$NoAutoEval,
    [switch]$DryRun
)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

# ---- 与 start_train.ps1 同源的安全闸 ----
if ($SaveDir -notmatch '^models/[A-Za-z0-9_.\-]{1,40}$') {
    Write-Output ("FLASHJSON " + ([ordered]@{ ok = $false; errors = @("SaveDir 非法（须 models/<名字>）：$SaveDir") } | ConvertTo-Json -Compress)); exit 1
}
if ($SaveDir.TrimEnd('/').ToLower() -in @('models/v1', 'checkpoints/pilot/big')) {
    Write-Output ("FLASHJSON " + ([ordered]@{ ok = $false; errors = @('SaveDir 指向 V1（junction 陷阱），拒绝') } | ConvertTo-Json -Compress)); exit 1
}
# 【2026-10-13 加】额外参数白名单：它最终会被拼进 WMI 命令行，必须挡住引号/反引号/$()/; | & 这类能"逃出去"的字符。
if ($ExtraArgs -and $ExtraArgs.Trim() -notmatch '^[A-Za-z0-9_\-=,./ ]{1,200}$') {
    Write-Output ("FLASHJSON " + ([ordered]@{ ok = $false; errors = @("ExtraArgs 含不允许的字符（只允许字母/数字/下划线/短横/=/,/./空格）：'$ExtraArgs'") } | ConvertTo-Json -Compress)); exit 1
}

$a = @('-NoLogo', '-NoProfile', '-File', (Join-Path $PSScriptRoot 'auto_train.ps1'),
    '-RunName', $RunName, '-SaveDir', $SaveDir,
    '-Workers', "$Workers", '-Games', "$Games", '-MinSteps', "$MinSteps",
    '-EvalGames', "$EvalGames", '-EvalEvery', "$EvalEvery", '-EvalWorkers', "$EvalWorkers",
    '-GensPerRound', "$GensPerRound", '-StartTarget', "$StartTarget",
    '-RulerGames', "$RulerGames", '-Seed', "$Seed",
    '-NSim', "$NSim", '-Blocks', "$Blocks", '-Channels', "$Channels")
if (-not $NoAutoEval) { $a += '-AutoEval' }
if ($BufferGens -gt 0) { $a += @('-BufferGens', "$BufferGens") }
if ($FreshRatio -ge 0) { $a += @('-FreshRatio', "$FreshRatio") }
if ($SamplesPerGen -ge 0) { $a += @('-SamplesPerGen', "$SamplesPerGen") }
if ($NoAnneal) { $a += '-NoAnneal' }
if ($AllowOverwrite) { $a += '-AllowOverwrite' }
if ($HeuristicWeight -ge 0) { $a += @('-HeuristicWeight', "$HeuristicWeight") }
if ($PolicyWeight -ge 0) { $a += @('-PolicyWeight', "$PolicyWeight") }
# 【2026-10-13 加】额外参数整串交给 auto_train.ps1（它会按空格拆开拼到 train.py 后面）
if ($ExtraArgs) { $a += @('-ExtraArgs', $ExtraArgs) }
$a += '-Detach'

$d = Join-Path $PSScriptRoot $SaveDir
$maxGen = 0
if (Test-Path $d) {
    foreach ($f in (Get-ChildItem (Join-Path $d 'gen_*.pt') -ErrorAction SilentlyContinue)) {
        if ($f.BaseName -match '^gen_(\d+)$') { $maxGen = [Math]::Max($maxGen, [int]$Matches[1]) }
    }
}
$tgt = if ($StartTarget -gt 0) { $StartTarget } else { $maxGen + $GensPerRound }
$hasResume = Test-Path (Join-Path $d 'resume.pt')
$hasBest = Test-Path (Join-Path $d 'best.pt')

# 【2026-10-12 修】把 -StartTarget 换成**上面算好的数字**。
# 原来原样传 0（"自动"）给 auto_train.ps1，而 auto_train 对 0 的理解与这里的预览不同：
# 它会先按"当前代数"跑一个 0.1 分钟的假轮（train.py 一看 max-gen <= 已完成代数就直接 exit=0），
# 再 EXTEND +30 —— 代价是白跑一次 200 局轮边界测量（约 13 分钟），
# 而且看板参数卡上"本轮目标代数"显示成 0。传具体数字后预览 = 实际。
$i = $a.IndexOf('-StartTarget')
if ($i -ge 0 -and ($i + 1) -lt $a.Count) { $a[$i + 1] = "$tgt" }

if ($DryRun) {
    $mode = if ($hasResume) { '续训（--resume-from，回放池一起恢复）' }
    elseif ($hasBest) { "从初值（--fresh，用 $SaveDir/best.pt 初始化权重）" }
    else { '从初值（--fresh，**随机权重**！该目录下没有 best.pt）' }
    Write-Output ("[预览] 目录=$SaveDir  当前最新 gen=$maxGen  本轮目标 gen=$tgt  模式=$mode")
    Write-Output ("[预览] 每代对局=$Games  每代训练步数=$MinSteps  workers=$Workers  评测=$EvalGames/$EvalEvery  " +
        "回放池=$(if ($BufferGens -gt 0) { "$BufferGens 代" } else { '沿用 train.py 默认 30 代' })  " +
        "退火=$(if ($NoAnneal) { "关闭（固定 hw=$(if ($HeuristicWeight -ge 0) { $HeuristicWeight } else { '默认' })/pw=$(if ($PolicyWeight -ge 0) { $PolicyWeight } else { '默认' })）" } else { '开启（5/25/45）' })")
    if ($ExtraArgs) { Write-Output ("[预览] 额外参数（原样交给 train.py）: " + $ExtraArgs) }
    Write-Output ("[预览] 将执行: pwsh " + ($a -join ' '))
    exit 0
}

# ---- 脱离创建（父进程 = WmiPrvSE，不随本终端/Session 走）----
$logDir = Join-Path $PSScriptRoot 'checkpoints\history'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir ("launch_{0}_{1}.log" -f $RunName, (Get-Date -Format 'yyyyMMdd_HHmmss'))
$parts = @()
foreach ($x in $a) {
    $s = "$x"
    if ($s -match '^[A-Za-z0-9_.:/\\\-]+$') { $parts += $s } else { $parts += ("'" + $s.Replace("'", "''") + "'") }
}
# $a 的前 3 项是 -NoLogo/-NoProfile/-File，跳过它们（-File 的值单独拼）
$inner = ("& '" + (Join-Path $PSScriptRoot 'auto_train.ps1') + "' " + (($parts | Select-Object -Skip 4) -join ' ') +
    " *> '" + $outLog + "'")
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = ('pwsh -NoLogo -NoProfile -NonInteractive -Command "' + $inner.Replace('"', '\"') + '"')
    CurrentDirectory = $PSScriptRoot
}
if (-not $r -or $r.ReturnValue -ne 0) {
    Write-Output ("FLASHJSON " + ([ordered]@{ ok = $false; errors = @("脱离创建失败，Win32_Process.Create 返回 $($r.ReturnValue)") } | ConvertTo-Json -Compress))
    exit 1
}
$procId = [int]$r.ProcessId
Start-Sleep -Seconds 6
$par = $null
try { $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$procId").ParentProcessId } catch { }
$parName = ''
if ($par) { $parName = (Get-CimInstance Win32_Process -Filter "ProcessId=$par" -ErrorAction SilentlyContinue).Name }
Write-Output ("FLASHJSON " + ([ordered]@{
            ok = $true; pid = $procId; parent_pid = $par; parent_name = $parName
            save_dir = $SaveDir; run = $RunName; games = $Games; min_steps = $MinSteps
            buffer_gens = $BufferGens; no_anneal = [bool]$NoAnneal
            heuristic_weight = $HeuristicWeight; policy_weight = $PolicyWeight
            extra_args = $ExtraArgs
            target_gen = $tgt; mode = $(if ($hasResume) { 'resume' } else { 'fresh' })
            out_log = $outLog
        } | ConvertTo-Json -Compress))
Write-Output ""
Write-Output "已脱离启动（父进程 $parName，PID $par）。日志："
Write-Output ("  " + (Join-Path $PSScriptRoot (Join-Path $SaveDir 'auto_events.log')))
Write-Output ("  " + (Join-Path $PSScriptRoot (Join-Path $SaveDir "logs\round1_gen$tgt.log")))
Write-Output ("  " + $outLog)
Write-Output "**这个终端窗口现在就可以关掉了。**"
