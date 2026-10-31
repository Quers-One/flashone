# start_services_detached.ps1 —— 把看板两个常驻服务**脱离当前进程树**拉起来
#
# 【为什么需要它】2026-10-12 的教训：训练/服务如果是"当前 shell / DSH 会话 / 终端"的
# 子进程，上层一退就可能被连带收走（用户实测到训练在看板重启前后被 STOP 掉）。
# 这里用 WMI（Win32_Process.Create）创建进程 ⇒ 父进程是 WmiPrvSE.exe 而不是调用者，
# 生命周期与调用者彻底解耦。
#
# 两个服务：
#   serve_dashboard.py  <Port>      HTTP + JSON API + 启停训练（日志 logs/dashboard_http.log）
#   watch_progress.ps1  -Html ...   每 Interval 秒重写 progress_live.html（日志 logs/dashboard_writer.log）
#
# 用法：
#   pwsh -NoProfile -File start_services_detached.ps1              # 缺啥补啥（默认）
#   pwsh -NoProfile -File start_services_detached.ps1 -Restart     # 先杀掉旧的再起
#   pwsh -NoProfile -File start_services_detached.ps1 -Stop        # 只停
#   pwsh -NoProfile -File start_services_detached.ps1 -Status      # 只看状态
param(
    [int]$Port = 8791,
    [int]$Interval = 3,
    [switch]$Restart,
    [switch]$Stop,
    [switch]$Status
)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Say([string]$m) { Write-Output ("[服务] " + $m) }

$patHttp = '-u\s+\S*serve_dashboard\.py'
$patWriter = '-File\s+\S*watch_progress\.ps1'

# 上面两条还会命中"探测用的一次性 shell"（命令行里含同样的脚本字样，
# 例如 `-NonInteractive -Command "Get-CimInstance ... serve_dashboard.py"`）。
# 这不只是好看问题：dashboard.ps1 用同样的模糊判据，可能在**服务其实已死**时报
# "已在运行"，于是永远不再拉起真正的服务。这里加一条硬过滤：一次性探测 shell 不算服务。
function Find-Svc([string]$pattern) {
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -and $_.CommandLine -match $pattern -and
            $_.CommandLine -notmatch '-NonInteractive' -and
            $_.CommandLine -notmatch 'Get-CimInstance'
        })
}

function Show-Status {
    $h = Find-Svc $patHttp
    $w = Find-Svc $patWriter
    Say ("HTTP 服务: " + $(if ($h) { "在跑 PID " + (($h | ForEach-Object { $_.ProcessId }) -join ',') } else { "未运行" }))
    Say ("看板写入: " + $(if ($w) { "在跑 PID " + (($w | ForEach-Object { $_.ProcessId }) -join ',') } else { "未运行" }))
}

function Start-Detached([string]$commandLine, [string]$outLog) {
    # 单引号包裹参数（外部命令的参数是字面量），输出重定向到文件
    $full = $commandLine + " *> '" + $outLog + "'"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = ('pwsh -NoLogo -NoProfile -NonInteractive -Command "' + $full.Replace('"', '\"') + '"')
        CurrentDirectory = $PSScriptRoot
    }
    if (-not $r -or $r.ReturnValue -ne 0) {
        Say ("★ 脱离启动失败（Win32_Process.Create 返回 " + $(if ($r) { $r.ReturnValue } else { 'null' }) + "）")
        return $null
    }
    Start-Sleep -Milliseconds 300
    $parent = $null
    try { $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)").ParentProcessId } catch { }
    Say ("已脱离启动 PID $($r.ProcessId)（父进程 $parent）")
    return [int]$r.ProcessId
}

if ($Stop -or $Restart) {
    foreach ($p in (Find-Svc $patWriter)) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Say "已停 写入进程 $($p.ProcessId)" }
    foreach ($p in (Find-Svc $patHttp)) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Say "已停 HTTP 服务 $($p.ProcessId)" }
    Start-Sleep -Seconds 1
}
if ($Stop) { Show-Status; exit 0 }

# ---- ① HTTP 服务 ----
$needHttp = -not (Find-Svc $patHttp)
$up = $false
try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 3; $up = $true } catch { $up = $false }
if ($needHttp -and -not $up) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = 'python' }
    $null = Start-Detached ("& '" + $py + "' -u '" + (Join-Path $PSScriptRoot 'serve_dashboard.py') + "' $Port") (Join-Path $logDir 'dashboard_http.log')
    Start-Sleep -Seconds 2
} elseif ($up) {
    Say "HTTP 服务已在运行（$Port）"
}

# ---- ② 看板写入进程 ----
if (-not (Find-Svc $patWriter)) {
    $null = Start-Detached ("& 'pwsh' -NoLogo -NoProfile -File '" + (Join-Path $PSScriptRoot 'watch_progress.ps1') +
        "' -Html '" + (Join-Path $PSScriptRoot 'progress_live.html') + "' -Interval $Interval -MaxMinutes 1440") `
        (Join-Path $logDir 'dashboard_writer.log')
} else {
    Say "写入进程已在运行"
}

Start-Sleep -Seconds 2
Show-Status
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 5
    Say ("页面可访问：http://127.0.0.1:$Port/ （" + $r.RawContentLength + " 字节）")
} catch {
    Say ("★ 页面不可访问：http://127.0.0.1:$Port/ —— 见 logs/dashboard_http.log")
}
