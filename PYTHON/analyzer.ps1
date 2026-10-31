# analyzer.ps1 —— 一键启动「五子棋分析台」（双击 Flash 分析台.cmd 即可）
#
# 【为什么需要它】分析台的地址是 http://127.0.0.1:8770/ —— 手输麻烦、也记不住。
# 这个脚本做三件事，**幂等、可以反复双击**：
#   ① 8770 已经在服务 → 什么都不启动，直接用浏览器打开
#   ② 没在服务 → 用**隐藏窗口**把 web_play.py 拉起来（日志写 web_play.log），等它就绪
#   ③ 用默认浏览器打开 8770
#
# 【能不能做成 exe】不建议：web_play.py 依赖 PyTorch，打包出来 2~3 GB、启动还更慢，
# 而且每次改代码都要重新打包。这个 .cmd + .ps1 的组合是"双击即用"又"改代码立刻生效"。
#
# 【生命周期】由**你自己双击**启动时，进程挂在 explorer.exe 下 ⇒ 关掉脚本窗口、
# 关掉浏览器都不影响服务，它会一直在后台跑（下次双击只会开浏览器，不会重复启动）。
# ⚠️ 反过来：如果是从 DSH 会话里调用，它会随会话结束被收走 —— 这是本机的已知运行学
# （见 HANDOFF.md §3「DSH 的 Job 对象」）。
#
# 用法：
#   pwsh -NoProfile -File analyzer.ps1              # 起服务并打开浏览器
#   pwsh -NoProfile -File analyzer.ps1 -NoOpen      # 只起服务
#   pwsh -NoProfile -File analyzer.ps1 -Stop        # 关掉分析台服务
#   pwsh -NoProfile -File analyzer.ps1 -Model models/V2/gen_175.pt
param(
    [int]$Port = 8770,
    [string]$Model = "",
    [switch]$NoOpen,
    [switch]$Stop,
    [int]$WaitSeconds = 90
)
$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
if (-not $Root) { $Root = 'D:\DHSGZQ\Flashone' }      # 从别处调用时的回落
Set-Location $Root
$url = "http://127.0.0.1:$Port/"
$log = Join-Path $Root 'web_play.log'

function Say([string]$m) { Write-Output ("[分析台] " + $m) }
# 【踩过的坑】Read-Host 在"被脚本/CI 调用"时会一直等输入 ⇒ 整个调用挂死。
# 只在真正的交互式控制台（双击那种）里等按键，其它情况直接返回。
function Pause-End {
    try {
        if ([Console]::IsInputRedirected) { return }
        if (-not [Environment]::UserInteractive) { return }
        Read-Host "按回车关闭" | Out-Null
    } catch { }
}
function Test-Up {
    try { $null = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3; return $true }
    catch { return $false }
}
# 只认"就是我们这个 web_play，而且是这个端口"的进程，避免误杀别的 python
function Get-Server {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match 'web_play\.py' -and
        $_.CommandLine -match ("--port\s+" + $Port + "(\s|$)")
    })
}

# ---- 0) -Stop：关掉服务 ----
if ($Stop) {
    $sv = Get-Server
    if ($sv.Count -eq 0) { Say "没有在跑的分析台服务（端口 $Port）" }
    else {
        foreach ($p in $sv) { Say ("停止 PID " + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 1
        Say "已停止"
    }
    exit 0
}

# ---- 1) 已经在跑？ ----
if (Test-Up) {
    Say "服务已在运行：$url"
} else {
    # 找 python（PATH 优先，再回落本机已知安装位置）
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) {
        foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
            if (Test-Path $c) { $py = $c; break }
        }
    }
    if (-not $py) {
        Say "★找不到 python —— 请先装 Python，或把它加进 PATH"
        Pause-End
        exit 1
    }
    $script = Join-Path $Root 'web_play.py'
    if (-not (Test-Path $script)) {
        Say "★找不到 web_play.py（应该在 $Root）"
        Pause-End
        exit 1
    }
    $pyArgs = @('-u', $script, '--port', "$Port")
    if ($Model) { $pyArgs += @('--model', $Model) }
    Say "正在启动分析台（端口 $Port，模型 $(if ($Model) { $Model } else { '默认 = 当前 champion' })）…"
    # 日志轮转：刚被强杀的进程可能还短暂占着 web_play.log，此时直接重定向会失败
    # ⇒ 换一个带时间戳的名字，绝不因为"日志文件被占"而起不来。
    try {
        if (Test-Path $log) { Remove-Item $log -Force -ErrorAction Stop }
        if (Test-Path "$log.err") { Remove-Item "$log.err" -Force -ErrorAction Stop }
    } catch {
        $log = Join-Path $Root ("web_play_" + (Get-Date -Format 'HHmmss') + ".log")
        Say "旧日志被占用，本轮改写 $log"
    }
    # 隐藏窗口 + 日志落盘：出错时用户能看到原因，而不是"点了没反应"。
    # ⚠️ 这里**不能**用 -ErrorAction SilentlyContinue：那样启动失败会被吞掉，
    # 表现成"等了 90 秒然后什么都不说"（2026-10-12 实测踩过）。
    $proc = $null
    try {
        $proc = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $Root `
            -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
            -PassThru -ErrorAction Stop
    } catch {
        Say "★无法启动 python：$($_.Exception.Message)"
        Pause-End
        exit 1
    }
    $ok = $false
    for ($i = 0; $i -lt $WaitSeconds; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Up) { $ok = $true; break }
        if ($proc -and $proc.HasExited) { break }          # 进程已经死了，不用再等
    }
    if (-not $ok) {
        Say "★启动失败（等了 $($i + 1) 秒还没起来）"
        if ($proc -and $proc.HasExited) {
            Say ("   python 已退出，ExitCode=" + $proc.ExitCode + "（多半是缺依赖或端口被占）")
        } elseif ($proc) {
            Say "   python 还活着但端口没响应（可能在加载模型，或端口被别的程序占着）"
        }
        if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Say "   $_" } }
        Say "   完整日志：$log"
        Pause-End
        exit 1
    }
    Say "已启动（模型加载完成）"
}

# ---- 3) 打开浏览器 ----
if ($NoOpen) {
    Say "（-NoOpen：不打开浏览器）地址 $url"
} else {
    Say "正在用默认浏览器打开 $url"
    Start-Process $url
}
Say "可以重复双击：已在运行的服务不会被重复启动。"
Say "想关掉服务：pwsh -NoProfile -File analyzer.ps1 -Stop"
