$ErrorActionPreference = 'Continue'
$pyDir = 'D:\DHSGZQ\Flashone\PYTHON'

# 1) 看板服务（8791）：按端口找进程，别按命令行匹配（会匹配到本脚本自己）
$conn = Get-NetTCPConnection -LocalPort 8791 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conn) {
    $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'python') {
        Write-Host ("停止看板服务 PID " + $p.Id)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Milliseconds 700

# 2) watcher：找 pwsh 进程里命令行含 watch_progress.ps1 的（排除自己：自己的命令行含 restart_dash.ps1）
$self = $PID
Get-CimInstance Win32_Process -Filter "Name='pwsh.exe' OR Name='powershell.exe'" | ForEach-Object {
    $cl = [string]$_.CommandLine
    if ($cl -match 'watch_progress\.ps1' -and $_.ProcessId -ne $self) {
        Write-Host ("停止 watcher PID " + $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Milliseconds 500

# 3) 起服务
Start-Process -FilePath 'python' -ArgumentList '-u','serve_dashboard.py','8791' -WorkingDirectory $pyDir -WindowStyle Hidden
Start-Sleep -Milliseconds 900

# 4) 起 watcher（必须带 -Html，否则只打印控制台、不写 progress_live.html）
Start-Process -FilePath 'pwsh' -ArgumentList '-NoLogo','-NoProfile','-File',"$pyDir\watch_progress.ps1",'-Html','progress_live.html','-Interval','3','-MaxMinutes','1440' -WorkingDirectory $pyDir -WindowStyle Hidden
Start-Sleep -Seconds 6

# 5) 自检
try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/cpp-state' -UseBasicParsing -TimeoutSec 25
    $j = $r.Content | ConvertFrom-Json
    Write-Host ("cpp-state: runs=" + ($j.runs -join ', '))
    foreach ($d in $j.runs_detail) {
        Write-Host ("  " + $d.dir + " role=" + $d.role + " total_gen=" + $d.total_gen + " recommended=" + $d.recommended + " scratch=" + $d.scratch)
    }
    $p = Invoke-WebRequest -Uri 'http://127.0.0.1:8791/progress_live.html' -UseBasicParsing -TimeoutSec 15
    Write-Host ("page bytes=" + $p.RawContentLength)
    foreach ($needle in @('p_CppDirSel','cppDirHint','cppDirSync','全局第')) {
        Write-Host ("  contains " + $needle + " = " + ($p.Content.Contains($needle)))
    }
} catch {
    Write-Host ("自检失败: " + $_.Exception.Message)
}
