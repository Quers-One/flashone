# rhine-lab.ps1 —— 一键把「莱茵生命终端风格 + 五子棋看板」跑起来。
#
# 它做四件事，缺一不可：
#   ① 看板 HTTP 服务（serve_dashboard.py，默认 8791）—— 提供 /api/state 与 /api/last-params
#   ② 看板写入进程（watch_progress.ps1 -Json）—— 每 N 秒把算好的快照落盘成 JSON
#   ③ RhineLabUI 开发服务（vite，5173）—— 三维界面本体，并把 /api 代理到 ①
#   ④ 打开浏览器
#
# 【为什么必须是「写入进程 + JSON 文件」这条链路】
# 不能让 3D 前端直接去算「第几代 / 还剩多久」：那是第二套口径，
# 迟早和看板页面漂移（2026-10-13 就因为「每代耗时跨 run 混排」报错过 ETA）。
# 所以由 watch_progress.ps1 把它**已经算好的那个对象**顺手序列化，
# Python 只负责读文件发出去，前端只负责显示。全链路只有一套口径。
#
# 【为什么 ② 不能省】没有它，/api/state 就没文件可读，
# 前端会走离线兜底、退回项目自带的 40 条档案 —— 界面照常好看，
# 但全是假数据。这是最容易误判「好像也还行」的地方，所以下面会显式检查。
#
# 用法（双击「启动莱茵终端.cmd」即可，不需要手敲）：
#   pwsh -NoProfile -File rhine-lab.ps1 [-Port 8791] [-Interval 3] [-NoOpen] [-NoWriter] [-Restart]
param(
    [int]$Port = 8791,
    [int]$Interval = 3,
    [string]$UiRoot = '',
    [switch]$NoOpen,
    [switch]$NoWriter,
    [switch]$Restart
)

$ErrorActionPreference = 'Continue'

# 【必须】$PSScriptRoot 在函数/脚本块里会为空（2026-10-12 踩过），
# 这里只在顶层读一次，之后一律用 $script:RunRoot。
$script:RunRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($script:RunRoot)) {
    $script:RunRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# RhineLabUI 与 Flashone 是**两棵并列的树**（D:\RhineLabUI 与 D:\DHSGZQ\Flashone），
# 所以不能从 $PSScriptRoot 推父目录 —— 那样只会得到 D:\DHSGZQ。
#
# 【判据用 vite.config.ts 而不是 package.json】
# 本脚本自己也被复制了一份到 D:\RhineLabUI 下，而 Flashone 侧**没有** package.json，
# 所以单看 package.json 也能区分。但真正要启动的是 vite 项目，
# 用 vite.config.ts 当判据更贴近意图，也避免将来 Flashone 加了 package.json 后误判。
function Test-UiRoot([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if (-not (Test-Path $Path)) { return $false }
    return (Test-Path (Join-Path $Path 'vite.config.ts'))
}

function Resolve-UiRoot([string]$Hint) {
    $cands = @()
    if (-not [string]::IsNullOrWhiteSpace($Hint)) { $cands += $Hint }
    if ($env:RHINE_UI_ROOT) { $cands += $env:RHINE_UI_ROOT }
    $cands += @(
        # 本脚本自己就在 UI 根目录里（被复制过去的那份）
        $script:RunRoot,
        'D:\RhineLabUI',
        # 两者互为兄弟目录的常见布局
        (Join-Path (Split-Path -Parent $script:RunRoot) 'RhineLabUI'),
        (Join-Path $script:RunRoot 'RhineLabUI'),
        (Join-Path $env:USERPROFILE 'RhineLabUI')
    )
    foreach ($c in $cands) {
        if (Test-UiRoot $c) { return (Resolve-Path $c).Path }
    }
    return $null
}
$UI_ROOT = Resolve-UiRoot $UiRoot

function Say([string]$m) { Write-Host ("[莱茵] " + $m) }
function Warn([string]$m) { Write-Host ("[莱茵] " + $m) -ForegroundColor Yellow }
function Bad([string]$m) { Write-Host ("[莱茵] " + $m) -ForegroundColor Red }

function Test-Port([int]$p) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $p)
        $c.Close()
        return $true
    } catch { return $false }
}

# 等端口起来，最多等 $Seconds 秒
function Wait-Port([int]$p, [int]$Seconds) {
    for ($i = 0; $i -lt ($Seconds * 2); $i++) {
        if (Test-Port $p) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# 精确找某个端口的监听进程 PID（用于 -Restart）
function Get-PortOwner([int]$p) {
    $owners = @()
    try {
        $owners = @(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty OwningProcess -Unique)
    } catch { }
    return $owners
}

Say "-----------------------------------------------------"
Say "莱茵生命终端 × 五子棋实时看板"
Say "-----------------------------------------------------"

# ---------- 前置检查 ----------
if (-not $UI_ROOT) {
    Bad "找不到 RhineLabUI 项目（找过：-UiRoot 参数、`$env:RHINE_UI_ROOT、D:\RhineLabUI、$script:RunRoot\RhineLabUI、$env:USERPROFILE\RhineLabUI）"
    Bad "请把 RhineLabUI 放到 D:\RhineLabUI，或用 -UiRoot <路径> 指定。"
    exit 1
}
Say "前端项目    $UI_ROOT"
if (-not (Test-Path (Join-Path $UI_ROOT 'node_modules'))) {
    Warn "node_modules 不存在，需要先装依赖（约十几秒）…"
    Push-Location $UI_ROOT
    & npm ci
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Bad "npm ci 失败（退出码 $rc）"; exit 1 }
}
if (-not (Test-Path (Join-Path $script:RunRoot 'serve_dashboard.py'))) {
    Bad "找不到 serve_dashboard.py（应在 $script:RunRoot）"
    exit 1
}

# ---------- -Restart：先拆掉旧服务 ----------
if ($Restart) {
    Warn "-Restart：正在停止 $Port 与 5173 上的旧进程…"
    foreach ($p in @($Port, 5173)) {
        foreach ($owner in (Get-PortOwner $p)) {
            try { Stop-Process -Id $owner -Force -ErrorAction Stop; Say "已停止 PID $owner（端口 $p）" }
            catch { Warn "无法停止 PID $owner（端口 $p）：$($_.Exception.Message)" }
        }
    }
    Start-Sleep -Seconds 2
}

# ---------- ① 看板 HTTP 服务 ----------
$dashUrl = "http://127.0.0.1:$Port/"
if (Test-Port $Port) {
    Say "① 看板服务已在运行（$dashUrl）"
} else {
    Say "① 正在启动看板服务（端口 $Port）…"
    Start-Process -FilePath 'python' `
        -ArgumentList '-u', (Join-Path $script:RunRoot 'serve_dashboard.py'), "$Port" `
        -WorkingDirectory $script:RunRoot -WindowStyle Hidden -ErrorAction SilentlyContinue
    if (Wait-Port $Port 10) { Say "   已启动" }
    else {
        Bad "   启动失败：$Port 端口没起来。请确认已安装 python，且端口没被占用。"
        exit 1
    }
}

# ---------- ② 看板写入进程 ----------
# 判据要和 dashboard.ps1 保持一致，否则会把自己（命令行含脚本名的 shell）误判成已在运行：
#   含 watch_progress.ps1 + 含 -Html + 不是 -NonInteractive 的工具 shell + 不是一次性 -Once
$writer = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match 'watch_progress\.ps1' -and
    $_.CommandLine -match '-Html' -and
    $_.CommandLine -notmatch '-NonInteractive' -and
    $_.CommandLine -notmatch '-Once'
})
$statePath = Join-Path $script:RunRoot 'checkpoints\dashboard_state.json'

if ($NoWriter) {
    Warn "② -NoWriter：跳过写入进程（前端会走离线兜底，显示自带的 40 条档案）"
} elseif ($writer.Count -gt 0) {
    Say "② 写入进程已在运行（PID $(($writer | ForEach-Object { $_.ProcessId }) -join ',')）"
} else {
    Say "② 正在启动写入进程（每 $Interval 秒写一次快照）…"
    # 关键：必须带 -Json，否则 /api/state 没文件可读，前端默默退回假数据。
    Start-Process -FilePath 'pwsh' `
        -ArgumentList '-NoLogo', '-NoProfile', '-File', (Join-Path $script:RunRoot 'watch_progress.ps1'),
                      '-Html', (Join-Path $script:RunRoot 'progress_live.html'),
                      '-Json', $statePath,
                      '-Interval', "$Interval", '-MaxMinutes', '1440' `
        -WorkingDirectory $script:RunRoot -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Say "   已启动"
}

# ---------- ③ 前端开发服务（vite） ----------
$uiUrl = 'http://127.0.0.1:5173/'
if (Test-Port 5173) {
    Say "③ 前端服务已在运行（$uiUrl）"
} else {
    Say "③ 正在启动前端服务（首次编译约几秒）…"
    # 【为什么走一个 .cmd 而不是直接 start cmd /k npm run dev】
    # 实测 `cmd /c start "标题" cmd /k npm run dev` 会留下一个僵着的 cmd、
    # 而嵌套窗口根本不出现（`start` 把带引号的首个词当窗口标题，
    # 中文标题 + 嵌套 cmd 参数组合起来不可靠）。
    # 走项目里那个 ASCII 的 dev-server.cmd 稳定得多，而且它会自己 `cd` 到项目根，
    # 不依赖本脚本的工作目录。
    $devCmd = Join-Path $UI_ROOT 'dev-server.cmd'
    if (Test-Path $devCmd) {
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'start', '""', '/min', $devCmd `
            -WorkingDirectory $UI_ROOT -ErrorAction SilentlyContinue
    } else {
        # 兜底：没那个 .cmd 就退回直接起 npm（也能用，只是窗口标题会朴素些）
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'start', '""', '/min', 'cmd', '/k', 'npm run dev' `
            -WorkingDirectory $UI_ROOT -ErrorAction SilentlyContinue
    }
    if (-not (Wait-Port 5173 60)) {
        Bad "   启动失败：5173 没起来。请到那个最小化的 dev server 窗口看报错。"
        exit 1
    }
    Say "   已启动（dev server 窗口已最小化，关掉它即停止前端）"
}

# ---------- ④ 自检：数据是不是真的通的 ----------
# 【为什么要自检】离线兜底**不会报错**：界面照样能开、能翻、能解密，
# 只是全在显示项目自带的 40 条假档案。不显式检查的话，
# 「好像也还行」会一直骗到用户点开详情页才发现数字不对。
Say "④ 自检中…"
Start-Sleep -Seconds 2

$live = $false
$note = ''
try {
    $r = Invoke-WebRequest -Uri ($uiUrl + 'api/state') -UseBasicParsing -TimeoutSec 6
    $j = $r.Content | ConvertFrom-Json
    if ($j.ok) {
        $age = [math]::Round([double]$j.age_sec, 1)
        if ($j.stale) { $note = "快照已过期（age=$age 秒）——写入进程可能停了"; }
        else { $live = $true; $note = "快照 age=$age 秒" }
    } else {
        $note = "接口返回 ok=false：" + $j.reason
    }
} catch {
    $note = "取 /api/state 失败：" + $_.Exception.Message
}

if ($live) {
    Write-Host "[莱茵]    ✓ 实时数据已接通（$note）" -ForegroundColor Green
    if (Test-Path $statePath) {
        $kb = [math]::Round((Get-Item $statePath).Length / 1KB, 1)
        Say "       快照：$statePath（$kb KB）"
    }
} else {
    Warn "    ✗ 未接通实时数据：$note"
    if (-not $NoWriter) {
        Warn "      前端会显示项目自带的 40 条档案（界面正常但数字是假的）。"
        Warn "      排查：1) 跑 .\watch_progress.ps1 -Once 1 -Json .\checkpoints\dashboard_state.json 看有没有报错"
        Warn "            2) 确认上面第 ② 步没有报启动失败"
    }
}

# ---------- 打开浏览器 ----------
if ($NoOpen) {
    Say "（-NoOpen：不打开浏览器）"
} else {
    Say "正在打开 $uiUrl"
    Start-Process $uiUrl
}

Say "-----------------------------------------------------"
Say "界面        $uiUrl"
Say "原版看板    $dashUrl"
Say "停止前端    关掉「莱茵终端 dev server」那个最小化窗口"
Say "全部重启    再跑一次本脚本并加 -Restart"
Say "可以重复双击，已在跑的服务不会被重复启动。"
Say "-----------------------------------------------------"
