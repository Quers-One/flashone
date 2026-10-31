$ErrorActionPreference = 'Continue'

# 1) 停掉当前 gk loop（PID 9416，160 局那轮）
$gk = Get-CimInstance Win32_Process -Filter "Name='gk.exe'" | Where-Object { $_.CommandLine -match '\bloop\b' }
foreach ($p in $gk) {
    Write-Host ("停止 gk loop PID " + $p.ProcessId)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
$left = Get-CimInstance Win32_Process -Filter "Name='gk.exe'" | Where-Object { $_.CommandLine -match '\bloop\b' }
Write-Host ("剩余 gk loop: " + @($left).Count)

# 2) dry-run 先看命令行
$body = @{ engine = 'cpp'; CppDir = 'model/run_v2'; GensPerRound = 30;
           Games = 320; MinSteps = 160; Workers = 16; EvalGames = 20; EvalEvery = 2 } | ConvertTo-Json
$r = Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/start-train' -Method Post -Body $body -ContentType 'application/json' -UseBasicParsing
$j = $r.Content | ConvertFrom-Json
Write-Host ("dry-run: " + $j.cmd)

# 3) 真启动
$r2 = Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/start-train' -Method Post -Body $body -ContentType 'application/json' -UseBasicParsing
Write-Host ("start => " + $r2.Content)
Start-Sleep -Seconds 6
Get-CimInstance Win32_Process -Filter "Name='gk.exe'" | ForEach-Object { Write-Host ("RUNNING " + $_.ProcessId + " :: " + $_.CommandLine) }
$st = (Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/cpp-state' -UseBasicParsing).Content | ConvertFrom-Json
Write-Host ("state running=" + $st.running + " dir=" + $st.dir + " phase=" + $st.phase)
