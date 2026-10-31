# dashboard.ps1 —— 一键启动看板：确保 HTTP 服务与写入进程在跑，然后用浏览器打开 HTTP 地址。
#
# 为什么需要它：双击 progress_live.html（file://）时浏览器**禁止 fetch**，
# 页面只能靠 meta refresh 每 3 秒整页重载 ⇒ 必然闪、且每次跳回页首。
# 走 http://127.0.0.1:8791/ 才能用"原地替换变化的卡"的顺滑模式。
#
# 用法：双击「Flash UI.cmd」（项目目录和桌面各有一份，内容相同、都能用），或
#   pwsh -NoProfile -File dashboard.ps1 [-Port 8791] [-NoOpen] [-Interval 3]
param(
    [int]$Port = 8791,
    [switch]$NoOpen,
    [int]$Interval = 3
)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$url = "http://127.0.0.1:$Port/"

function Say([string]$m) { Write-Output ("[看板] " + $m) }

# ---- ① HTTP 服务 ----
$up = $false
try {
    $null = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
    $up = $true
} catch { $up = $false }

if ($up) {
    Say "HTTP 服务已在运行：$url"
} else {
    Say "HTTP 服务未运行，正在启动（端口 $Port）…"
    Start-Process -FilePath 'python' `
        -ArgumentList '-u', (Join-Path $PSScriptRoot 'serve_dashboard.py'), "$Port" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    try { $null = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3; Say "已启动" }
    catch { Say "★启动失败：请确认已安装 python 且 8791 端口没被占用" }
}

# ---- ② 看板写入进程（每 Interval 秒重写 progress_live.html）----
# 判据要精确，否则会把自己（命令行里含脚本名的 shell）误判成"已在运行"：
#   含 watch_progress.ps1 + 含 -Html + 不是 -NonInteractive 的工具 shell + 不是一次性 -Once
$writer = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match 'watch_progress\.ps1' -and
    $_.CommandLine -match '-Html' -and
    $_.CommandLine -notmatch '-NonInteractive' -and
    $_.CommandLine -notmatch '-Once'
})
if ($writer.Count -gt 0) {
    Say ("写入进程已在运行（PID " + (($writer | ForEach-Object { $_.ProcessId }) -join ',') + "）")
} else {
    Say "写入进程未运行，正在启动…"
    Start-Process -FilePath 'pwsh' `
        -ArgumentList '-NoLogo', '-NoProfile', '-File', (Join-Path $PSScriptRoot 'watch_progress.ps1'),
                      '-Html', (Join-Path $PSScriptRoot 'progress_live.html'),
                      '-Interval', "$Interval", '-MaxMinutes', '1440' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Say "已启动"
}

# ---- ③ 打开浏览器 ----
if ($NoOpen) {
    Say "（-NoOpen：不打开浏览器）地址 $url"
} else {
    Say "正在用默认浏览器打开 $url"
    Start-Process $url
}
Say "可以重复双击，已启动的服务不会被重复启动。"
