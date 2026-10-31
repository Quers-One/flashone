# watch_progress.ps1 —— 训练/测量 实时看板（浏览器版 + 终端版）
#
# 用法：
#   pwsh -NoProfile -File watch_progress.ps1                       # 终端里每 5 秒刷新
#   pwsh -NoProfile -File watch_progress.ps1 -Html progress_live.html -Interval 3
#       ^ 只写 HTML（不刷屏）；配合 serve_dashboard.py 可用 http://127.0.0.1:8791/ 打开
#
# 看板内容（"训练/测试的一切都同步到这里"）：
#   ① 训练进度：当前第几代 / 本轮目标 / 阶段（自对弈·训练步·门控·落盘）/ 阶段内步骤条 / 已用时 / 每代耗时 / 本轮 ETA
#   ② 训练历史：最近若干代的 loss_p / loss_v / steps / lr / win_rate / 每代耗时
#   ③ 正在跑的进程（训练/测量/监督器），带过滤，不会被宿主程序或自己的命令行误匹配
#   ④ 测量中：任意正在写的测量日志（不认文件名，只认"谁在写"），含局数/速率/ETA
#   ⑤ 测量结果：曲线账本 + 一次性对照实验，全部列出得分率/ΔElo/区间/判定
#   ⑥ 当前 champion（名称/哈希/相对 root 的 Elo/hops）
#   ⑦ GPU 功耗·利用率·显存·温度  ⑧ 最近事件（auto_events.log）
param(
    [int]$Interval = 5,
    [string]$Html = '',
    [string]$Json = '',         # 非空 = 每轮把同一份 $s 序列化落盘（供 serve_dashboard.py 的 /api/state 发出去）
    [int]$Once = 0,
    [int]$MaxMinutes = 0,       # >0 = 跑够这么久就自己退出（避免留下野进程）
    [switch]$Force              # 忽略"已有写入进程"检查，强行再起一个
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
# 脚本根目录存一份：**函数/脚本块里不能直接用 $PSScriptRoot**
# （Sort-Object/Where-Object 的脚本块是子作用域，那里 $PSScriptRoot 是空的 —— 2026-10-12 踩过）
$script:RunRoot = $PSScriptRoot

# 【单实例保护】两个写入进程会互相覆盖 progress_live.html，导致页面在两种版本之间跳
# （实测踩过：旧版进程把带控制面板的新页面覆盖掉，看起来像"我的改动没生效"）。
# 只在"长期写 html"的模式下检查；-Once 一次性出帧不受限制（方便自检/调试）。
if ($Html -and $Once -eq 0 -and -not $Force) {
    $mine = $PID
    $others = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        # 判据要精确：必须是"长期写 html 的那个 pwsh"
        #   · 命令行含 watch_progress.ps1 且含 -Html（一次性 -Once 帧不加 -Html 的用法不在此列）
        #   · 排除 -NonInteractive：那是工具/自动化 shell 的形状，不是我们自己起的写入进程
        $_.ProcessId -ne $mine -and $_.Name -like 'pwsh*' -and $_.CommandLine -and
        $_.CommandLine -match 'watch_progress\.ps1' -and
        $_.CommandLine -match '-Html' -and
        $_.CommandLine -notmatch '-NonInteractive' -and
        $_.CommandLine -notmatch '-Once'
    })
    if ($others.Count -gt 0) {
        Write-Output ("[看板] 已有写入进程在跑（PID " + (($others | ForEach-Object { $_.ProcessId }) -join ',') + "），本进程退出。要强起请加 -Force")
        exit 0
    }
}

$script:prevMeas = $null      # 上一次的 (文件, 时刻, 局数) 用来算速率
$started = Get-Date

function Repair-Mojibake([string]$t) {
    # 【为什么需要】auto_train.ps1 用 `*> $logFile` 重定向 python 的 stdout 时，
    # 若宿主 [Console]::OutputEncoding 不是 UTF-8，PowerShell 会把子进程的 UTF-8 字节
    # 按系统代码页(GBK)解码后再存成 UTF-8 ⇒ **日志正文变成双重编码**。
    # 校验：原始 "数据: 3709 条" 在文件里是 "鏁版嵁: 3709 鏉?"。
    # 显示前按 GBK 编回字节、再按 UTF-8 解码即可还原大部分（被有损替换成 '?' 的救不回）。
    if (-not $t) { return $t }
    if ($t -notmatch '[锛鏄鐨瀵鏁鎵缁鑰鍜浜涓鐢熷眬]') { return $t }   # 不是乱码就原样返回
    try {
        $gbk = [System.Text.Encoding]::GetEncoding(936)
        $fixed = [System.Text.Encoding]::UTF8.GetString($gbk.GetBytes($t))
        return $fixed
    } catch { return $t }
}

function Read-Lines($path) {
    # 日志可能正被别的进程写，读失败就当作空，不要把整个看板搞崩
    try { return @(Get-Content $path -Encoding utf8 -ErrorAction Stop) } catch { return @() }
}

# 【2026-10-12 加】run 目录**动态扫描**：models/<名字>
# 旧写法把 run 名单写死在多处（'models\V2','models\V1','models\run3'…），
# 于是新建的 V3 在"训练历史"卡上永远不显示（用户实测发现）。
# 判据用 train_log.jsonl（训练历史的数据源）比"目录存在"更准：
# 只有真的训过、写过后才会被列进来。排序：最近有写入的排前面。
function Get-RunDirs {
    # ⚠️ 这段**绝不能用 $PSScriptRoot**：`Sort-Object { ... }` 的脚本块是**子作用域**，
    # 里面 $PSScriptRoot 是空字符串 ⇒ Join-Path 抛异常 ⇒ 整个函数静默落到兜底
    # （只剩 models\V2），症状就是"新 run 永远不出现在看板上"（2026-10-12 实测踩到）。
    # 所以这里用 `$script:RunRoot`（在文件顶层按脚本位置赋一次值）。
    $base = Join-Path $script:RunRoot 'models'
    $cands = @(Get-ChildItem $base -Directory -ErrorAction SilentlyContinue)
    $pairs = @()
    foreach ($d in $cands) {
        $tl = Join-Path $d.FullName 'train_log.jsonl'
        if (Test-Path $tl) {
            $pairs += [pscustomobject]@{ dir = 'models\' + $d.Name; at = (Get-Item $tl).LastWriteTime }
        }
    }
    $dirs = @($pairs | Sort-Object at -Descending | ForEach-Object { $_.dir })
    if ($dirs.Count -eq 0) { $dirs = @('models\V2') }   # 兜底，别让上层拿到空列表
    return $dirs
}

function Get-Procs {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and
        ($_.CommandLine -match 'train\.py|ladder\.py|compare\.py|auto_eval|auto_train|probe_') -and
        # 【必须排除宿主程序】DSH 主进程命令行里嵌着"我自己的工具调用文本"，
        # 于是 'auto_train' 这种子串会把它匹配成一个假"监督器"（实测踩过）。
        ($_.Name -notlike 'DeepSeek*')
    } | ForEach-Object {
        $kind = '其它'
        # 【匹配必须贴住真实的调用形式】不能只搜 'auto_train' 这种裸子串：
        # 我自己的工具 shell 命令行里会**包含**这些脚本名（作为过滤文本），
        # 于是会被误报成"监督器"（实测踩过两次）。所以按实际调用形状匹配：
        #   pwsh -File ...\auto_train.ps1     python -u train.py ...
        if ($_.CommandLine -match '-File\s+\S*auto_train\.ps1') { $kind = '监督器' }
        elseif ($_.CommandLine -match '-u\s+train\.py') { $kind = '训练' }
        elseif ($_.CommandLine -match '-File\s+\S*auto_eval\.ps1') { $kind = '自动测量' }
        elseif ($_.CommandLine -match '-u\s+ladder\.py') { $kind = '标尺' }
        elseif ($_.CommandLine -match '-u\s+compare\.py') { $kind = '对照' }
        elseif ($_.CommandLine -match '-u\s+probe_(\w+)\.py') { $kind = "探针:$($Matches[1])" }
        elseif ($_.CommandLine -match '-u\s+test_(\w+)\.py') { $kind = "测试:$($Matches[1])" }
        else { return }   # 认不出真实调用形状的一律不列，避免自我匹配
        $el = if ($_.CreationDate) { [math]::Round(((Get-Date) - $_.CreationDate).TotalMinutes, 1) } else { 0 }
        [pscustomobject]@{ pid = $_.ProcessId; kind = $kind; mins = $el }
    }
}

function Get-LogProgress($path) {
    # 解析一个测量日志：已完成局数 / 总局数 / 速率 / 预计完成。
    if (-not (Test-Path $path)) { return $null }
    $txt = Read-Lines $path
    if (-not $txt) { return $null }
    # 并行是每行一个 [wK/N]，串行是"第 N 局"；两者都数，取较大者（不是相加！）
    $par = ($txt | Where-Object { $_ -match '\[w\d+/\d+\]' }).Count
    $ser = ($txt | Where-Object { $_ -match '第\s*\d+\s*局' }).Count
    $done = [math]::Max($par, $ser)
    $total = 0
    $hdr = $txt | Where-Object { $_ -match '并行(测量|对局)：' } | Select-Object -First 1
    if ($hdr -and $hdr -match '×\s*([\d/]+)\s*局') {
        foreach ($n in ($Matches[1] -split '/')) { $total += [int]$n }
    }
    if (-not $total) {
        $g = $txt | Where-Object { $_ -match '局数\s*=?\s*(\d+)' } | Select-Object -First 1
        if ($g -and $g -match '局数\s*=?\s*(\d+)') { $total = [int]$Matches[1] } else { $total = 200 }
    }
    $now = Get-Date
    $rate = $null
    if ($script:prevMeas -and $script:prevMeas.file -eq (Split-Path $path -Leaf)) {
        $dt = ($now - $script:prevMeas.t).TotalSeconds
        if ($dt -gt 0.5) { $rate = ($done - $script:prevMeas.done) / $dt * 60.0 }
    }
    $script:prevMeas = @{ file = (Split-Path $path -Leaf); t = $now; done = $done }
    $eta = $null
    if ($rate -and $rate -gt 0.05 -and $done -lt $total) {
        $eta = $now.AddMinutes(($total - $done) / $rate)
    }
    [pscustomobject]@{
        file = Split-Path $path -Leaf; done = $done; total = $total; rate = $rate; eta = $eta
        age = [math]::Round(($now - (Get-Item $path).LastWriteTime).TotalSeconds, 0)
    }
}

function Get-Measurement {
    # 两级判定：
    #   ① 有逐局进度行（[wK/N] 或 "第 N 局"）的日志 → 报局数/速率/ETA（完整进度条）
    #   ② 只是"最近在写"的其它日志（硬门槛 test_tactics/test_selfplay、画图 …）→
    #      也报出来，附最后几行输出。**不能因为格式不同就当它不存在**，
    #      否则"所有测试都上网页"这条会漏掉整个硬门槛阶段。
    $cands = @()
    foreach ($d in @('checkpoints\history', 'models\V2\eval', 'models\V1\eval', '.')) {
        $full = Join-Path $PSScriptRoot $d
        if (-not (Test-Path $full)) { continue }
        $cands += Get-ChildItem (Join-Path $full '*.log') -ErrorAction SilentlyContinue
    }
    $cands += Get-ChildItem (Join-Path $PSScriptRoot 'models\*\eval\*.log') -ErrorAction SilentlyContinue
    $cands = @($cands | Where-Object { ((Get-Date) - $_.LastWriteTime).TotalSeconds -le 180 } |
        Sort-Object LastWriteTime -Descending)
    foreach ($f in $cands) {
        $txt = Read-Lines $f.FullName
        if (-not $txt) { continue }
        if ($txt | Where-Object { $_ -match '\[w\d+/\d+\]|第\s*\d+\s*局' }) {
            return (Get-LogProgress $f.FullName)
        }
    }
    $newest = $cands | Select-Object -First 1
    if ($newest) {
        $script:prevMeas = $null      # 别让"纯文本阶段"污染下一个进度条的速率
        $tail = @(Read-Lines $newest.FullName | Where-Object { $_.Trim() } | Select-Object -Last 3)
        return [pscustomobject]@{
            file = $newest.Name; done = 0; total = 0; rate = $null; eta = $null
            plain = $true; tail = (Repair-Mojibake ($tail -join ' ⏎ '))
            age = [math]::Round(((Get-Date) - $newest.LastWriteTime).TotalSeconds, 0)
        }
    }
    return $null
}

function Get-EvalProgress {
    # 轮边界测量（auto_eval.ps1）的**步骤级**进度。
    # 步骤顺序是固定的，每步写一个日志文件：gen{N}_tactics.log → _selfplay.log → _ladder.log → _plot.log
    # 判据：**最新被写的那个文件 = 当前步骤**，它之前的都算完成、之后的算未开始。
    # 只有整个 eval 在近 3 分钟内还在写时才认为"正在测量"。
    $steps = @(
        @{ key = 'tactics';  name = '① 战术测试 test_tactics' },
        @{ key = 'selfplay'; name = '② 自战测试 test_selfplay' },
        @{ key = 'ladder';   name = '③ 标尺对局 vs champion' },
        @{ key = 'plot';     name = '④ 画曲线图 plot_progress' }
    )
    $files = @()
    # 【2026-10-12 修】不再写死 run 名单：**扫 models/*/eval**。
    # 旧写法只列 V2/V1/run3，于是新建的 V3 永远不出现在"正在测量"卡上（用户实测发现）。
    foreach ($run in Get-RunDirs) {
        $d = Join-Path $PSScriptRoot (Join-Path $run 'eval')
        if (Test-Path $d) { $files += Get-ChildItem (Join-Path $d 'gen*_*.log') -ErrorAction SilentlyContinue }
    }
    if (-not $files) { return $null }
    $newest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (((Get-Date) - $newest.LastWriteTime).TotalMinutes -gt 3) { return $null }   # 不在评测中就整卡隐藏
    $evalDir = Split-Path $newest.FullName -Parent
    # 当前评测的代数：从最近写的文件里取最大的 gen
    $curGen = $null
    foreach ($f in ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 8)) {
        if ($f.Name -match 'gen(\d+)_') { $curGen = [int]$Matches[1]; break }
    }
    if (-not $curGen) { return $null }
    # 整体已跑多久：从 auto_events.log 最后一条 EVAL_START 算
    $elapsed = $null
    $ev = Read-Lines (Join-Path $PSScriptRoot 'models\V2\auto_events.log')
    for ($i = $ev.Count - 1; $i -ge 0; $i--) {
        if ($ev[$i] -match '^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) EVAL_START') {
            try { $elapsed = [math]::Round(((Get-Date) - [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null)).TotalMinutes, 1) } catch { }
            break
        }
    }
    $newestKey = $null
    if ($newest.Name -match 'gen\d+_([a-z_]+)\.log') { $newestKey = $Matches[1] }
    # 整体是否已经收线：auto_events.log 里最后一条 EVAL_RESULT 若晚于最新 eval 日志，
    # 说明 eval 已经结束 —— 否则"画图"那一步会一直显示"进行中"（实测卡了 3 分钟）。
    $finishedAt = $null
    for ($i = $ev.Count - 1; $i -ge 0; $i--) {
        if ($ev[$i] -match '^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) EVAL_RESULT') {
            try {
                $t = [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null)
                if ($t -gt $newest.LastWriteTime) { $finishedAt = $t }
            } catch { }
            break
        }
    }
    $rows = @()
    foreach ($st in $steps) {
        $fp = Join-Path $evalDir ("gen{0}_{1}.log" -f $curGen, $st.key)
        $f = if (Test-Path $fp) { Get-Item $fp } else { $null }
        $status = if (-not $f) { '待开始' }
                  elseif ($finishedAt) { '完成' }
                  elseif ($st.key -eq $newestKey) { '进行中' }
                  else { '完成' }
        $detail = ''
        if ($f) {
            if ($status -eq '完成') {
                $detail = '完成于 ' + $f.LastWriteTime.ToString('HH:mm:ss')
            } else {
                switch ($st.key) {
                    'selfplay' {
                        # test_selfplay 跑 3 局固定局面；每局一行输出里带 "(\d+ms/…"，
                        # 用 ASCII 片段计数（正文中文是双重编码，匹配中文会失败）
                        $n = (Read-Lines $f.FullName | Where-Object { $_ -match '\d+ms/' }).Count
                        $detail = "已跑 $n/3 局　最近写入 $([math]::Round(((Get-Date) - $f.LastWriteTime).TotalSeconds,0)) 秒前"
                    }
                    'ladder' {
                        $p = Get-LogProgress $f.FullName
                        if ($p) {
                            $detail = "$($p.done)/$($p.total) 局"
                            if ($p.rate) { $detail += ("　{0:N1} 局/分" -f $p.rate) }
                            if ($p.eta) { $detail += "　预计 " + $p.eta.ToString('HH:mm:ss') }
                        } else { $detail = '刚开始' }
                    }
                    default {
                        $detail = "最近写入 $([math]::Round(((Get-Date) - $f.LastWriteTime).TotalSeconds,0)) 秒前"
                    }
                }
            }
        }
        $rows += [pscustomobject]@{ name = $st.name; status = $status; detail = $detail }
    }
    return [pscustomobject]@{ gen = $curGen; elapsed = $elapsed; rows = $rows; dir = $evalDir; finished = [bool]$finishedAt }
}

function Format-Span($mins) {
    # 把分钟格式化成"X 小时" / "X 天 Y 小时"，太小就显示分钟
    if ($null -eq $mins -or $mins -le 0) { return '—' }
    if ($mins -lt 60) { return ("{0:N0} 分钟" -f $mins) }
    $h = $mins / 60.0
    if ($h -lt 24) { return ("{0:N1} 小时" -f $h) }
    $d = [math]::Floor($h / 24); $rh = $h - $d * 24
    return ("{0} 天 {1:N1} 小时" -f $d, $rh)
}

function Get-RunTotals {
    # 每个"大轮"（run3 / V1 / V2 …）的**总局数与总训练时间**。
    # 时间有两个口径，都不完美，所以两个都给：
    #   · 实际训练时长 = 墙钟 − 停顿（单代间隔 > 20 分钟按停顿计）
    #   · 墙钟跨度     = 首尾时间戳之差（含所有停顿与评测时间）
    # 局数用 train_log.jsonl 的 n_games 求和（不是"代数×每代局数"推算）。
    #
    # 【刻意不列入 checkpoints/pilot/big】它**不是独立的一轮，也不是副本**：
    #   models\V1 是指向 checkpoints\pilot\big 的 Junction（fsutil 已核实是同一 File ID），
    #   两者是**同一个目录**，真目录就是 pilot/big。列出来会造成两个误导：
    #   （a）看起来像"多跑了一轮"；（b）更糟 —— 会让人以为那个"副本"可以删，
    #   而删它等于删掉 V1 全线（gen 1–75、resume、best、data、eval，780 MB）。
    # 所以页面上只显示 V1 一行；junction 这件事只记在 NOTES.md §0。
    $defs = @(
        @{ name = 'run3'; dir = 'models\run3'; note = '旧线（被 champion 取代）' },
        @{ name = 'V1'; dir = 'models\V1'; note = '40 局/代，1200 步' },
        @{ name = 'pilot/small'; dir = 'checkpoints\pilot\small'; note = '试跑（3×64，仅 1 代）' },
        @{ name = 'V2'; dir = 'models\V2'; note = '80 局/代，160 步' }
    )
    # 【2026-10-12 加】自动补上"没在上面列过的 models/* 轮次"（例如 V3）。
    # 不这么做的话：训练历史/测量结果里有 V3，但"各轮训练量"卡里没有它（用户会以为数据丢了）。
    $known = @($defs | ForEach-Object { $_.dir.ToLower() })
    foreach ($d in Get-RunDirs) {
        if ($known -contains $d.ToLower()) { continue }
        $nm = Split-Path $d -Leaf
        $defs += @{ name = $nm; dir = $d; note = '（自动发现）' }
    }
    $out = @()
    $seen = @{}
    foreach ($d in $defs) {
        $p = Join-Path $PSScriptRoot (Join-Path $d.dir 'train_log.jsonl')
        if (-not (Test-Path $p)) { continue }
        $rows = @()
        foreach ($ln in (Read-Lines $p)) {
            if (-not $ln.Trim()) { continue }
            try { $o = $ln | ConvertFrom-Json } catch { continue }
            if ($null -ne $o.gen) { $rows += $o }
        }
        if (-not $rows) { continue }
        $dup = $false
        $key = "{0}|{1}|{2}" -f $rows.Count, $rows[0].timestamp, $rows[-1].timestamp
        if ($seen.ContainsKey($key)) { $dup = $true } else { $seen[$key] = $d.name }
        $ts = @()
        foreach ($r in $rows) {
            try { $ts += [datetime]::ParseExact($r.timestamp, 'yyyy-MM-dd HH:mm:ss', $null) } catch { $ts += $null }
        }
        $ok = @($ts | Where-Object { $_ })
        $wall = if ($ok.Count -ge 2) { ($ok[-1] - $ok[0]).TotalMinutes } else { 0 }
        $durs = @()
        for ($i = 1; $i -lt $ok.Count; $i++) { $durs += ($ok[$i] - $ok[$i - 1]).TotalMinutes }
        # 【口径】"实际训练时长" = 墙钟 − 停顿。停顿判据：单代间隔 > 20 分钟。
        # 为什么不按"中位数×2.5"丢：轮边界那一代要跑硬门槛 + 200 局评测，本来就慢
        # （10~12 分钟），按倍数筛会把它当异常丢掉，反而少算真实工作量。
        $actual = 0.0; $pauseMin = 0.0
        foreach ($x in $durs) {
            if ($x -gt 20) { $pauseMin += $x } else { $actual += $x }
        }
        $games = 0; $steps = 0; $samples = 0
        foreach ($r in $rows) {
            if ($null -ne $r.n_games) { $games += [int]$r.n_games }
            if ($null -ne $r.train_steps) { $steps += [int]$r.train_steps }
            if ($null -ne $r.data_samples) { $samples += [int]$r.data_samples }
        }
        $g0 = ($rows | ForEach-Object { [int]$_.gen } | Measure-Object -Minimum).Minimum
        $g1 = ($rows | ForEach-Object { [int]$_.gen } | Measure-Object -Maximum).Maximum
        $out += [pscustomobject]@{
            name = $d.name; note = $d.note; dup = $dup
            g0 = $g0; g1 = $g1; gens = $rows.Count
            games = $games; steps = $steps; samples = $samples
            actualMin = $actual; wallMin = $wall; pauseMin = $pauseMin
            perGen = $(if ($rows.Count -gt 1 -and $actual -gt 0) { [math]::Round($actual / ($rows.Count - 1), 2) } else { $null })
            first = $rows[0].timestamp; last = $rows[-1].timestamp
        }
    }
    return @($out)
}

function Get-RoundLog {
    $cands = @()
    foreach ($pat in @('models\*\logs\round*_gen*.log', 'checkpoints\*\*\logs\round*_gen*.log', 'checkpoints\*\logs\round*_gen*.log')) {
        $cands += Get-ChildItem (Join-Path $PSScriptRoot $pat) -ErrorAction SilentlyContinue
    }
    return ($cands | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
}

function Get-TrainHistory($n = 10) {
    # train_log.jsonl 全是 ASCII 键，比从日志正文抠中文可靠得多
    $rows = @()
    # 【2026-10-12 修】用动态扫描（见 Get-RunDirs），否则新 run（如 V3）不显示
    foreach ($run in Get-RunDirs) {
        $p = Join-Path $PSScriptRoot (Join-Path $run 'train_log.jsonl')
        if (-not (Test-Path $p)) { continue }
        foreach ($line in (Read-Lines $p)) {
            if (-not $line.Trim()) { continue }
            try { $o = $line | ConvertFrom-Json } catch { continue }
            if ($null -eq $o.gen) { continue }
            $o | Add-Member -NotePropertyName run -NotePropertyValue $run -Force
            $rows += $o
        }
    }
    if (-not $rows) { return @() }
    # 【2026-10-12 改】按**时间**排序，而不是按代数。
    # 旧写法 `Sort-Object { [int]$_.gen }` 在多 run 混排时会把新 run 挤掉：
    # V3 刚开始只有 gen 1，代数排序排在 V2 的 gen 166–175 之前 ⇒ 取"最后 10 行"时
    # **完全看不到 V3**（用户实测发现"网页里训练历史不显示这个模型"）。
    # 时间戳是 'yyyy-MM-dd HH:mm:ss'（定长、可直接字符串比较）。
    $rows = $rows | Sort-Object { [string]$_.timestamp }
    # 每代耗时 = 与上一代的时间戳之差。⚠️ **跨 run 不能算**：V3 的第一代减 V2 的最后一代
    # 那个差值毫无意义（会显示成"46 分钟一代"这种假数字）。
    $durs = @{}
    for ($i = 1; $i -lt $rows.Count; $i++) {
        if ($rows[$i].run -ne $rows[$i - 1].run) { continue }
        try {
            $t0 = [datetime]::ParseExact($rows[$i - 1].timestamp, 'yyyy-MM-dd HH:mm:ss', $null)
            $t1 = [datetime]::ParseExact($rows[$i].timestamp, 'yyyy-MM-dd HH:mm:ss', $null)
            $durs[[int]$rows[$i].gen] = [math]::Round(($t1 - $t0).TotalMinutes, 2)
        } catch { }
    }
    $tail = $rows | Select-Object -Last $n
    return @($tail | ForEach-Object {
        [pscustomobject]@{
            gen = [int]$_.gen; run = $_.run; loss_p = $_.loss_policy; loss_v = $_.loss_value
            steps = $_.train_steps; lr = $_.lr; win_rate = $_.win_rate
            win_raw = $_.win_rate_raw; win_ema = $_.win_rate_ema
            chosen = $_.chosen; accepted = $_.accepted
            samples = $_.data_samples; games = $_.n_games
            mins = $(if ($durs.ContainsKey([int]$_.gen)) { $durs[[int]$_.gen] } else { $null })
            at = $_.timestamp
        }
    })
}

function Get-CadenceInfo($hist, $run = $null) {
    # 每代耗时：用最近几代的中位数，并丢掉异常大的间隔（比如"停训再续训"的那一段）。
    #
    # 【2026-10-13 修 · 用户报「本轮预计完成时间有问题，数据太假了」】
    # 旧写法把 `$hist`（最近 10 行，**跨 V2/V3/V4 混排**）里所有 `mins` 一起丢进中位数
    # ⇒ 拿**别的 run** 的节奏给当前 run 预报完成时间。实测 21:33 那一幕：
    #   混合中位数 = 5.8 分钟/代（6 个样本里 5 个属于 V3 的 4.3~6.4）
    #   而 V4 自己前两代 = 7.55 分钟/代
    #   ⇒ 页面报「本轮预计完成 00:06」，真实约 00:57（差 50 分钟，且看起来很精确）。
    # 现在**只认本 run 自己**的每代耗时，并把样本数一起返回，好在页面上写清依据。
    $rows = if ($null -ne $run) { @($hist | Where-Object { $_.run -eq $run }) } else { @($hist) }
    $ds = @($rows | Where-Object { $null -ne $_.mins } | ForEach-Object { [double]$_.mins })
    if ($ds.Count -eq 0) { return [pscustomobject]@{ cadence = $null; n = 0; run = $run } }
    if ($ds.Count -eq 1) {
        $one = if ($ds[0] -gt 0) { [math]::Round($ds[0], 2) } else { $null }
        return [pscustomobject]@{ cadence = $one; n = 1; run = $run }
    }
    $med = ($ds | Sort-Object)[[int][math]::Floor($ds.Count / 2)]
    $keep = @($ds | Where-Object { $_ -le $med * 2.5 -and $_ -gt 0 })
    $c = if ($keep) { [math]::Round((($keep | Measure-Object -Average).Average), 2) } else { $null }
    return [pscustomobject]@{ cadence = $c; n = $ds.Count; run = $run }
}

function Get-Cadence($hist, $run = $null) {
    return (Get-CadenceInfo $hist $run).cadence
}

function Get-CadenceByRun($hist) {
    # 逐 run 的每代耗时，供「训练历史」表**逐行**判断"这一代是否明显偏慢（*）"。
    # 用当前 run 的节奏去量别的 run 的代次会误标（和 ETA 那个假数据同源）。
    $h = @{}
    $runs = @($hist | ForEach-Object { [string]$_.run } | Sort-Object -Unique)
    foreach ($r in $runs) { $h[$r] = (Get-CadenceInfo $hist $r).cadence }
    return $h
}

function Get-CurrentRun {
    # 「当前 run」= 正在写的那个轮日志所属的 run 目录；记号与 Get-RunDirs 一致（相对路径，如 models\V4）
    $f = Get-RoundLog
    if (-not $f) { return $null }
    $saveDir = Split-Path (Split-Path $f.FullName -Parent) -Parent
    if (-not $saveDir) { return $null }
    $rel = [string]$saveDir
    $root = [string]$PSScriptRoot
    if ($root -and $rel.StartsWith($root)) { $rel = $rel.Substring($root.Length).TrimStart('\') }
    return $rel
}

function Get-TrainDetail {
    $f = Get-RoundLog
    if (-not $f) { return $null }
    $txt = Read-Lines $f.FullName
    if (-not $txt) { return $null }
    # 训练进程的启动时刻：**本轮真正开始的时刻**。
    # ⚠️ 必须是 **train.py 主进程**（命令行含 '-u train.py'）。
    # 不能写成"取最新启动的 python"：worker 池会反复重建（短命进程），
    # 那样锚点会被不断重置成"几秒前"，读数恒为 0.1 分钟（实测踩到）。
    $trainStart = $null
    $tp = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match '\s-u\s+train\.py' } |
        Select-Object -First 1
    if ($tp) {
        try { $trainStart = (Get-Process -Id $tp.ProcessId -ErrorAction Stop).StartTime } catch { $trainStart = $null }
    }
    $target = $null
    if ($f.BaseName -match 'gen(\d+)$') { $target = [int]$Matches[1] }
    # 阶段 = **最后出现**的那个 [k/4]。旧写法按 [1/4]→[4/4] 固定顺序判断，
    # 尾巴里同时存在多个标记时，最后执行的判断总是赢，于是永远显示成"落盘"。
    $idx = -1
    for ($i = $txt.Count - 1; $i -ge 0; $i--) {
        if ($txt[$i] -match '\[(\d)/4\]') { $idx = [int]$Matches[1]; break }
    }
    $names = @{ -1 = '（还没开始）'; 0 = '读配置'; 1 = '自对弈'; 2 = '训练步'; 3 = '门控'; 4 = '落盘' }
    $phase = $names[$idx]
    # 当前代数：优先取"对外 gen_N）"头部行；日志里 'gen_121.pt' 是已保存的、
    # 'gen_121）' 才是正在训的那一代。正则只认 ASCII（日志正文的中文是双重编码，
    # 匹配中文一定会失败），所以用"是否紧跟 .pt"来区分两种出现。
    $hdrGen = $null; $savedGen = $null
    foreach ($line in $txt) {
        if ($line -match 'gen_(\d+)\.pt') { $savedGen = [int]$Matches[1]; continue }
        if ($line -match 'gen_(\d+)') { $hdrGen = [int]$Matches[1] }
    }
    if (-not $hdrGen -and $savedGen) { $hdrGen = $savedGen + 1 }
    $doneThisRound = ($txt | Where-Object { $_ -match '\[4/4\]' }).Count
    # 步骤进度只在"训练步"阶段才有意义：日志里最后一条 step 行是**上一代**留下的，
    # 在自对弈阶段照显示会变成"训练步 160/160"这种假完成。
    $stepDone = $null; $stepTotal = $null
    if ($idx -eq 2) {
        for ($i = $txt.Count - 1; $i -ge 0; $i--) {
            if ($txt[$i] -match 'step\s+(\d+)/(\d+)') { $stepDone = [int]$Matches[1]; $stepTotal = [int]$Matches[2]; break }
        }
    }
    # 每代的两个耗时在日志里是可解析的 ASCII 片段：
    #   自对弈行以 "(...s)" 结尾（如 "3709 条（耗时 141.8s）"）→ 用 "([\d.]+)s)"
    #   训练行是行尾的 "14.3s"                                     → 用 "([\d.]+)s\s*$"
    # 有了这两个基准，才能给"自对弈阶段"算出真实进度（它本身没有逐步输出）。
    $spSecs = @(); $trSecs = @()
    foreach ($line in $txt) {
        if ($line -match '\(([\d.]+)s\)') { $spSecs += [double]$Matches[1] }
        elseif ($line -match '([\d.]+)s\s*$') { $trSecs += [double]$Matches[1] }
    }
    $spTypical = if ($spSecs.Count) { [math]::Round((($spSecs | Select-Object -Last 5) | Measure-Object -Average).Average, 1) } else { $null }
    $trTypical = if ($trSecs.Count) { [math]::Round((($trSecs | Select-Object -Last 5) | Measure-Object -Average).Average, 1) } else { $null }
    # 当前这一代已经跑了多久。锚点只有两个**可靠**来源，取较晚者：
    #   ① 上一代 gen_N.pt 的落盘时间（这一代就是从那之后开始的）
    #   ② train.py 进程的启动时刻（冷启动第一代没有 gen_*.pt ⇒ 用它）
    # ❌ 不用日志文件的 CreationTime/LastWriteTime：同一轮日志会被后续启动**追加**
    #    （12:31 失败那次留下的文件被 12:53 那次续写 ⇒ CreationTime 虚高成 25.8 分钟），
    #    而 LastWriteTime 又会被外部读写"触碰"，两个都会给出错误读数（实测各踩一次）。
    $saveDir = Split-Path (Split-Path $f.FullName -Parent) -Parent
    $lastPt = Get-ChildItem (Join-Path $saveDir 'gen_*.pt') -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $anchor = $null
    if ($lastPt -and $trainStart) { $anchor = if ($lastPt.LastWriteTime -gt $trainStart) { $lastPt.LastWriteTime } else { $trainStart } }
    elseif ($lastPt) { $anchor = $lastPt.LastWriteTime }
    elseif ($trainStart) { $anchor = $trainStart }
    else { $anchor = (Get-Date) }
    $genElapsed = [math]::Round(((Get-Date) - $anchor).TotalSeconds, 0)
    [pscustomobject]@{
        log = $f.Name; file = $f.FullName; target = $target; phase = $phase; phaseIdx = $idx
        gen = $hdrGen; doneThisRound = $doneThisRound
        stepDone = $stepDone; stepTotal = $stepTotal
        spTypical = $spTypical; trTypical = $trTypical; genElapsed = $genElapsed
        age = [math]::Round(((Get-Date) - $f.LastWriteTime).TotalSeconds, 0)
    }
}

function Get-Results($n = 12) {
    $out = @()
    # ① 曲线账本（跨代可比的口径）
    foreach ($csv in @('checkpoints\history\elo_history.csv', 'checkpoints\history\elo_history_prod.csv')) {
        $p = Join-Path $PSScriptRoot $csv
        if (-not (Test-Path $p)) { continue }
        $lines = Read-Lines $p
        if ($lines.Count -lt 2) { continue }
        $head = $lines[0] -split ','
        foreach ($ln in ($lines | Select-Object -Skip 1)) {
            $v = $ln -split ','
            # 【为什么门槛不是 $head.Count】本账本合法地混着两种行宽：
            #   2026-10-11 15:21 及更早的 6 行 = 21 字段（那时还没有 engine_full 列）
            #   之后的行           = 22 字段
            # 表头是 22 个名字。若拿 $head.Count 当门槛，那 6 条历史测量会被整行丢弃
            # （实测：给表头补上 engine_full 之后就会丢）。这里只要求"够读到 verdict(索引 15)"，
            # 缺尾列（engine_full）按 $null 处理即可 —— 面板本来也不读它。
            if ($v.Count -lt 16) { continue }
            $r = @{}
            $nHead = [Math]::Min($head.Count, $v.Count)
            for ($i = 0; $i -lt $nHead; $i++) { $r[$head[$i]] = $v[$i] }
            $prod = if ($csv -like '*_prod*') { '生产' } else { '尺子' }
            # 【口径必须配对，不能混】主账本只有"相对 root"的区间（ci_lo_root/ci_hi_root），
            # 没有"本场 vs 对手"的区间。把 elo_vs_opponent 和 ci_*_root 拼在一起是错的
            # （实测过一次：显示成 "38.4 [92.8, 191.4]"）。
            # 所以：主账本显示"相对 root"的 Elo+区间，并把本场 ΔElo 写进标签；生产账本显示本场的。
            if ($csv -like '*_prod*') {
                $metric = 'vs 对手'
                $eloV = $r['elo_vs_opponent']; $loV = $r['ci_lo']; $hiV = $r['ci_hi']
                $label = ("gen {0} vs {1}" -f $r['gen'], $r['measured_vs'])
            } else {
                $metric = '相对 root'
                $eloV = $r['elo_vs_root']; $loV = $r['ci_lo_root']; $hiV = $r['ci_hi_root']
                $label = ("gen {0} vs {1}（本场 {2}）" -f $r['gen'], $r['measured_vs'], $r['elo_vs_opponent'])
            }
            $out += [pscustomobject]@{
                at = $r['at']; label = $label
                src = "账本·$prod"; games = $r['games']; score = $r['score']
                elo = $eloV; lo = $loV; hi = $hiV; metric = $metric
                verdict = $r['verdict_vs_opponent']; extra = ''
            }
        }
    }
    # ② 一次性对照实验（ladder/compare 落盘的 json）
    $jsons = @()
    $jsons += Get-ChildItem (Join-Path $PSScriptRoot 'checkpoints\history\*.json') -ErrorAction SilentlyContinue
    $jsons += Get-ChildItem (Join-Path $PSScriptRoot 'models\*\eval\*.json') -ErrorAction SilentlyContinue
    foreach ($j in ($jsons | Sort-Object LastWriteTime -Descending | Select-Object -First 40)) {
        if ($j.BaseName -like '_*') { continue }   # verify_tools / 自测产物（_v_*、_par_*、_cmp_* …）
        try { $d = Get-Content $j.FullName -Encoding utf8 -Raw | ConvertFrom-Json } catch { continue }
        # compare.py 的字段名不一样（a_wins/b_wins/a_rate），只收有明确比分口径的
        if ($null -eq $d.score -and $null -eq $d.a_rate) { continue }
        $score = if ($null -ne $d.score) { [double]$d.score } else { [double]$d.a_rate }
        $games = if ($null -ne $d.games) { $d.games } else { $d.decided }
        $elo = if ($null -ne $d.elo) { $d.elo } else { $null }
        $lo = if ($null -ne $d.elo_lo) { $d.elo_lo } else { $null }
        $hi = if ($null -ne $d.elo_hi) { $d.elo_hi } else { $null }
        $verdict = if ($null -ne $d.verdict) { $d.verdict } elseif ($null -ne $d.p_value) { "p=$($d.p_value)" } else { '' }
        $name = $j.BaseName
        if ($name.Length -gt 30) { $name = $name.Substring(0, 30) }
        $out += [pscustomobject]@{
            at = $(if ($d.at) { $d.at } else { (Get-Item $j.FullName).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') })
            label = $name; src = '一次性'; games = $games; score = ('{0:N4}' -f $score)
            elo = $elo; lo = $lo; hi = $hi; metric = 'vs 对手'; verdict = $verdict; extra = ''
        }
    }
    if (-not $out) { return @() }
    # 去重：同一次测量会同时出现在"曲线账本"和"它自己的 json"里。
    # 账本行是先加的，所以"保留第一个"就等于优先保留账本口径。
    $seen = @{}
    $dedup = @()
    # 去重时"账本口径优先"，但**展示顺序**必须仍按时间倒序
    # （把两个目的混在一个排序里，结果表会出现"20:17 排在 19:50 后面"这种乱序）
    $sorted = $out | Sort-Object @{ Expression = { if ($_.src -like '账本*') { 0 } else { 1 } } }, `
                                   @{ Expression = { [string]$_.at }; Descending = $true }
    foreach ($r in $sorted) {
        $key = "{0}|{1}" -f $r.at, $r.score
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = 1
        $dedup += $r
    }
    return @($dedup | Sort-Object { [string]$_.at } -Descending | Select-Object -First $n)
}

function Get-Artifacts($n = 8) {
    # 【兜底】探针类实验（probe_*.py）落盘的 json 字段各不相同，
    # 没法塞进上面那张统一格式的结果表。这里把所有最近产出的结果文件都列出来，
    # 保证"任何一次测试都不会在看板上完全看不见"。
    $files = @()
    foreach ($pat in @('checkpoints\history\*.json', 'models\*\eval\*.json', 'checkpoints\history\*.txt')) {
        $files += Get-ChildItem (Join-Path $PSScriptRoot $pat) -ErrorAction SilentlyContinue
    }
    $out = @()
    foreach ($f in ($files | Sort-Object LastWriteTime -Descending | Select-Object -First ($n + 8))) {
        if ($f.BaseName -like '_*') { continue }   # verify_tools 等自测产物，不当成"实验结果"
        if ($out.Count -ge $n) { break }
        $sum = ''
        if ($f.Extension -eq '.json') {
            try {
                $d = Get-Content $f.FullName -Encoding utf8 -Raw | ConvertFrom-Json
                $bits = @()
                foreach ($k in @('score', 'a_rate', 'elo', 'verdict', 'verdict_vs_opponent', 'best_ce', 'final_ce',
                                 'top1', 'vmse', 'speedup', 'ratio', 'n_sim', 'hw', 'pw')) {
                    if ($null -ne $d.$k) { $bits += ("{0}={1}" -f $k, $d.$k) }
                    if ($bits.Count -ge 5) { break }
                }
                $sum = ($bits -join '  ')
            } catch { $sum = '(json 解析失败)' }
        }
        $out += [pscustomobject]@{
            at = $f.LastWriteTime.ToString('HH:mm:ss'); name = $f.Name
            kb = [math]::Round($f.Length / 1KB, 1); sum = $sum
        }
    }
    return @($out)
}

function Get-CurveImg {
    # 训练曲线图（plot_progress.py 的产物）。取最新的一张 —— 也就是当前那条线的。
    # 返回相对路径（相对 progress_live.html 所在目录），这样 **file:// 直接打开也能显示**；
    # 走 HTTP 时由 serve_dashboard.py 按 /models/<run>/progress.png 白名单发出来。
    $c = Get-ChildItem (Join-Path $PSScriptRoot 'models\*\progress.png') -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $c) { return $null }
    $rel = 'models/' + (Split-Path (Split-Path $c.FullName -Parent) -Leaf) + '/progress.png'
    [pscustomobject]@{
        rel = $rel; path = $c.FullName; kb = [math]::Round($c.Length / 1KB, 1)
        at = $c.LastWriteTime
        # 版本号 = 图片修改时间。图没变 ⇒ URL 不变 ⇒ 浏览器用强缓存，不会每 3 秒白拉一次
        ver = $c.LastWriteTime.Ticks
    }
}

function Get-Champion {
    $p = Join-Path $PSScriptRoot 'checkpoints\champions\current.json'
    if (-not (Test-Path $p)) { return $null }
    try { return (Get-Content $p -Encoding utf8 -Raw | ConvertFrom-Json) } catch { return $null }
}

function Get-Gpu {
    $line = (nvidia-smi 2>$null | Select-String -Pattern 'W /.*Default' | Select-Object -First 1)
    if (-not $line) { return $null }
    $s = $line.ToString()
    [pscustomobject]@{
        util = [regex]::Match($s, '(\d+)%\s+Default').Groups[1].Value
        pwr = [regex]::Match($s, '(\d+)W /').Groups[1].Value
        mem = [regex]::Match($s, '(\d+)MiB /').Groups[1].Value
        temp = [regex]::Match($s, '(\d+)C').Groups[1].Value
    }
}

function Get-Events($n = 5) {
    $e = Join-Path $PSScriptRoot 'models\V2\auto_events.log'
    if (-not (Test-Path $e)) { return @() }
    return @(Read-Lines $e | Select-Object -Last $n)
}

function Get-TrainRecipe {
    # 「现在训练的模型是什么参数」（用户 2026-10-12 要求）。
    #
    # 【数据来源的优先级】这是关键 —— 三个来源的可信度完全不同：
    #   ① 正在跑的 `train.py` 命令行：**最权威**，那是它真正收到的参数
    #      （supervisor 会从 auto_train.ps1 拼出完整命令行，含 --games/--n-sim/--buffer-gens/...）
    #   ② `checkpoints/last_train_params.json`：上次启动时表单里的值（"打算怎么跑"）
    #   ③ `models/<run>/train_log.jsonl` 最新一行：这一代实际产生的 lr / 代数 / 数据量
    # 页面上必须**同时写明用的是哪一路**，否则"参数"和"实际在跑的东西"会被混淆
    # （本项目已经踩过：表单里的目标代数过期、与监督器实际目标不一致）。
    $src = '（没有训练在跑，也没有上次启动参数）'
    $run = ''
    $live = $null
    foreach ($p in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
        if (-not $p.CommandLine) { continue }
        if ($p.Name -like 'DeepSeek*') { continue }          # 别匹配到我自己的工具 shell
        if ($p.CommandLine -match '-u\s+\S*train\.py') { $live = $p; break }
    }

    $rows = [ordered]@{}      # 中文名 -> 值
    $flags = [ordered]@{}
    # 监督器（两种真实调用形状：train_detached 的 -File、start_train 的 -Command "& '…'"）
    # ⚠️ 不能只搜裸子串 'auto_train.ps1'：我自己的工具 shell 命令行里会**包含**这个文本，
    # 会被误判成"监督器"（本项目踩过两次）。所以只认这两种调用形状。
    $sup = $null
    foreach ($p in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
        if (-not $p.CommandLine) { continue }
        if ($p.Name -like 'DeepSeek*') { continue }
        if (($p.CommandLine -match '-File\s+\S*auto_train\.ps1') -or
            ($p.CommandLine -match '-Command\s+"&\s*''[^'']*auto_train\.ps1''')) { $sup = $p; break }
    }
    if ($live -or $sup) {
        # 【关键】测量阶段没有 train.py，但监督器活着且下一代就要开跑 —— 这时参数必须
        # 取自**监督器的命令行**（它才是"打算怎么跑"），不能报"训练未在跑"。
        $srcProc = if ($live) { $live } else { $sup }
        $src = if ($live) { '① 正在跑的 train.py 命令行（最权威）' }
               else { '①b 监督器命令行（当前在轮边界测量/准备下一代，train.py 未在跑）' }
        $cl = $srcProc.CommandLine
        # 【必须归一化】同一个参数有两种写法：
        #   train.py     `--heuristic-weight 0.5`（双横线 + 连字符）
        #   auto_train.ps1 `-HeuristicWeight 0.5`（单横线 + 驼峰）
        # 早期版本只按前者建表 ⇒ 监督器在跑时 hw/pw 全部不显示（还误报"退火开"）。
        # 统一去掉所有连字符再转小写，两边就都落在同一个键上。
        $valMap = @{
            'games' = '每代自对弈局数'; 'nsim' = 'MCTS 模拟次数/手'; 'minsteps' = '每代训练步数'
            'workers' = '自对弈进程数'; 'evalgames' = '门控局数'; 'evalevery' = '门控间隔(代)'
            'evalworkers' = '门控进程数'; 'buffergens' = '回放池保留代数'
            'heuristicweight' = '规则先验 hw'; 'policyweight' = '策略先验 pw'
            'blocks' = '残差块数'; 'channels' = '通道数'; 'lr' = '学习率'
            'seed' = '随机种子'; 'maxgen' = '本轮目标代数'; 'starttarget' = '本轮目标代数'
            'samplespergen' = '每代新样本目标'; 'freshratio' = '新数据比例'
            'rulergames' = '晋升测量局数'; 'gensperround' = '每轮代数'
            # 【2026-10-13 加】任务 4/6 的新开关（它们会作为独立 token 出现在 train.py 命令行里）
            'bufferdecayhalflife' = '采样半衰期(代)'; 'bufferentropyweight' = '采样新颖度 β'
            'auxheads' = '辅助头'; 'auxch' = '辅助头通道数'; 'auxweightthreat' = '威胁头权重'
            'auxweightownership' = '归属头权重'
        }
        $strMap = @{ 'runname' = '轮次名'; 'savedir' = '模型目录'; 'resumefrom' = '续训起点'; 'extraargs' = '额外参数(train.py)' }
        $flagMap = @{
            'noanneal' = '退火（--no-anneal）'; 'allowoverwrite' = '允许写入已有目录'
            'nolrratchet' = '关闭 lr 棘轮'; 'fresh' = '从初值冷启动'; 'autoeval' = '轮边界自动测量'
        }
        $toks = @($cl -split '\s+')
        for ($i = 0; $i -lt $toks.Count; $i++) {
            $t = $toks[$i]
            if (-not $t.StartsWith('-')) { continue }
            $key = ($t -replace '-', '').ToLower()
            if ($flagMap.ContainsKey($key)) {
                $flags[$flagMap[$key]] = if ($key -eq 'noanneal') { '是（全程用下面固定的 hw/pw）' } else { '是' }
                continue
            }
            if ($strMap.ContainsKey($key)) {
                $flags[$strMap[$key]] = $toks[$i + 1]
                if ($key -eq 'runname') { $run = $toks[$i + 1] }
                continue
            }
            if (-not $valMap.ContainsKey($key)) { continue }
            $v = if (($i + 1) -lt $toks.Count -and $toks[$i + 1] -notmatch '^-') { $toks[$i + 1] } else { '是' }
            if ($valMap[$key] -eq '本轮目标代数' -and $v -eq '0') { $v = '0（自动 = 当前代数 + 每轮代数）' }
            $rows[$valMap[$key]] = $v
        }
        if (-not $flags.Contains('退火（--no-anneal）')) { $flags['退火'] = '开：hw/pw 随代数退火' }
        # 【额外参数的**原文**必须单独显示】train.py 的命令行里 `-ExtraArgs` 已经被**展开**成
        # 各自的开关（`--buffer-decay-half-life 20` 变成两个独立 token），所以从 train.py 那行
        # 拿不到原文；而**监督器**的命令行里有原文。用户 2026-10-13 实测报"参数卡里没有额外参数"
        # 就是这种情形（他看到的只有展开后的"采样半衰期(代)"那一行）。
        # 值本身在上面已按展开后的开关逐项列出，这里补的是"当初到底传了什么"。
        if ($sup -and $sup.CommandLine) {
            $me = [regex]::Match($sup.CommandLine, "-ExtraArgs\s+(.+?)(?:\s+\*?>|\s*$)")
            if ($me.Success) {
                $rows['额外参数(train.py)'] = ($me.Groups[1].Value.Trim() -replace "^'", '' -replace "'$", '')
            }
        }
    }
    # 【2026-10-27 加】C++ 侧配方：上面只认 train.py / auto_train.ps1，C++ 在跑时两个都没有，
    # 于是配方卡掉进 ② 分支，去读 checkpoints/last_train_params.json —— 那里面是 2026-10-15 的
    # Python V4 记录（**160 局 / 320 步**）。用户 18:28 报「怎么还是 160 局」就是这个：
    # 跑的明明是 320 局（gen 11 的 train_log 里 games=320），卡上印的却是那张过期 Python 表。
    # C++ 唯一权威来源是 gk.exe 自己的命令行。
    $gk = $null
    if (-not ($live -or $sup)) {
        foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='gk.exe'" -ErrorAction SilentlyContinue)) {
            if ($p.CommandLine -match 'gk\.exe"?\s+loop') { $gk = $p; break }
        }
        if (-not $gk) {
            # 兜底：进程名拿不到（权限/克隆进程）时按命令行找，但不能匹配到我自己的工具 shell
            foreach ($p in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
                if (-not $p.CommandLine) { continue }
                if ($p.Name -like 'DeepSeek*') { continue }
                if ($p.CommandLine -match '\\gk\.exe"?\s+loop\s') { $gk = $p; break }
            }
        }
    }
    if ($gk) {
        $src = '① 正在跑的 gk.exe loop 命令行（C++ 侧最权威）'
        $tokG = @($gk.CommandLine -split '\s+')
        $cppVal = [ordered]@{
            'games' = '每代自对弈局数'; 'workers' = '自对弈线程数'; 'minsteps' = '每代训练步数'
            'sims' = 'MCTS 模拟次数/手'; 'batchmcts' = '每请求叶子数'; 'evalbatch' = '评估批大小'
            'evalthreads' = '评估线程数'; 'evalgames' = '门控局数'; 'evalevery' = '门控间隔(代)'
            'batch' = '训练批大小'; 'lr' = '学习率'; 'freshratio' = '新数据比例'
            'buffergens' = '回放池保留代数'; 'halflife' = '采样半衰期(代)'; 'seed' = '随机种子'
            'gens' = '本轮目标代数'; 'procs' = '自对弈进程数'; 'keepgens' = '保留历史代权重'
            'keepbuffergens' = '保留回放分片'; 'hw' = '规则先验 hw'; 'pw' = '策略先验 pw'
        }
        for ($i = 0; $i -lt $tokG.Count; $i++) {
            $t = $tokG[$i]
            if (-not $t.StartsWith('--')) { continue }
            $key = ($t -replace '-', '').ToLower()
            if ($key -eq 'dir') { $rows['模型目录'] = $tokG[$i + 1]; $run = $tokG[$i + 1]; continue }
            # ⚠️ [ordered]@{} 是 OrderedDictionary，只有 Contains()，没有 ContainsKey()
            # （用了会抛「方法调用失败」，而且是非终止错误 → 继续跑、索引到 $null，报"数组索引为 null"）
            if (-not $cppVal.Contains($key)) { continue }
            $v = if (($i + 1) -lt $tokG.Count -and $tokG[$i + 1] -notmatch '^--') { $tokG[$i + 1] } else { '是' }
            $rows[$cppVal[$key]] = $v
        }
    }
    $lpf = Join-Path $PSScriptRoot 'checkpoints\last_train_params.json'
    if (-not ($live -or $sup -or $gk) -and (Test-Path $lpf)) {
        try {
            $lpj = Get-Content $lpf -Encoding utf8 -Raw | ConvertFrom-Json
            $pp = $lpj.params
            $src = "② 上次启动参数（{0}；训练当前没在跑）" -f $(if ($lpj.at) { $lpj.at } else { '时间未知' })
            $nameMap = [ordered]@{
                'Games' = '每代自对弈局数'; 'MinSteps' = '每代训练步数'; 'Workers' = '自对弈进程数'
                'EvalGames' = '门控局数'; 'EvalEvery' = '门控间隔(代)'; 'EvalWorkers' = '门控进程数'
                'BufferGens' = '回放池保留代数'; 'HeuristicWeight' = '规则先验 hw'; 'PolicyWeight' = '策略先验 pw'
                'Seed' = '随机种子'; 'GensPerRound' = '每轮代数'; 'StartTarget' = '本轮目标代数'
                'RulerGames' = '晋升测量局数'; 'AutoEval' = '轮边界自动测量'
                # 【2026-10-13 加】任务 4/6 的新开关走这条透传，配方卡必须显示出来，否则看不出跑的是哪个配方
                'ExtraArgs' = '额外参数(train.py)'
            }
            foreach ($k in $nameMap.Keys) {
                if ($null -eq $pp.$k) { continue }
                $v = $pp.$k
                if (($k -eq 'BufferGens' -or $k -eq 'HeuristicWeight' -or $k -eq 'PolicyWeight') -and [int]$v -lt 0) {
                    $v = '（不传 = 用 train.py 默认/退火）'
                }
                $rows[$nameMap[$k]] = $v
            }
            $ifAny = @()
            foreach ($b in @(@('NoAnneal', '关闭退火'), @('AllowOverwrite', '允许写入已有目录'))) {
                if ($pp.($b[0]) -eq $true) { $ifAny += $b[1] }
            }
            if ($ifAny.Count) { $flags['开关'] = ($ifAny -join '、') }
            $run = [string]$pp.RunName
            if ($pp.SaveDir) { $flags['模型目录'] = [string]$pp.SaveDir }
        } catch { }
    }

    # ③ 这一代实际产生的数字（lr / 代数 / 数据量），跨三路来源都能补
    # 【2026-10-27 加】C++ 在跑时必须读 C++\model\<dir>\train_log.jsonl —— 否则这一行会去扫
    # 代码的 models\<run>，显示的是 **Python V4 的 gen 37 / lr 0.001**，而屏幕上跑的是
    # C++ 的 gen 12 / lr 3e-4。配方卡里"局数"和"最新一代"必须同源，不能一半 C++ 一半 Python。
    $hist = @()
    if ($gk) {
        $md = [regex]::Match($gk.CommandLine, '--dir\s+(\S+)')
        if ($md.Success) {
            $cpDir = $md.Groups[1].Value
            $cpLog = Join-Path (Split-Path $PSScriptRoot -Parent) ('C++\' + ($cpDir -replace '/', '\') + '\train_log.jsonl')
            if (Test-Path $cpLog) {
                $lastLine = Read-Lines $cpLog | Where-Object { $_.Trim() } | Select-Object -Last 1
                if ($lastLine) {
                    try {
                        $o = $lastLine | ConvertFrom-Json
                        $stamp = (Get-Item $cpLog).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
                        $hist = @([pscustomobject]@{ gen = $o.gen; at = $stamp; lr = $o.lr
                                                     samples = $o.samples; games = $o.games; run = $cpDir })
                    } catch { }
                }
            }
        }
    }
    if (-not $hist) { $hist = @(Get-TrainHistory 1) }
    $live2 = @{}
    if ($hist.Count) {
        $h0 = $hist[-1]
        $live2['最新一代'] = "gen $($h0.gen)（$($h0.at)）"
        if ($null -ne $h0.lr) { $live2['实际 lr'] = $h0.lr }
        if ($null -ne $h0.samples) { $live2['该代数据条数'] = $h0.samples }
        if (-not $run) { $run = [string]$h0.run }
    }
    # ---- 与"生产配方"对齐检查 ----
    # 【为什么要这个】hw/pw 是**引擎口径**：hw=0.5/pw=1.0 是 train.py 退火终点 = 已验证的生产配方；
    # 偏离它就是另一个引擎，历史强度读数不可直接比（NOTES.md §12）。
    # 2026-10-12 实测踩过：看板表单的 hw 来自**浏览器草稿**（旧失败配方的 hw=0），
    # 点「开始训练」就把 hw=0 送进去了 —— 页面上必须能一眼看出这种偏差。
    $warn = @()
    foreach ($pair in @(@('规则先验 hw', '0.5'), @('策略先验 pw', '1'))) {
        if (-not $rows.Contains($pair[0])) { continue }
        $v = [string]$rows[$pair[0]]
        if ($v -like '*不传*') { continue }
        if ($v -ne $pair[1]) { $warn += ("$($pair[0])=$v（生产配方是 $($pair[1]))") }
    }
    # 【2026-10-13 加】「额外参数」这一行**必须常显**（没有就写“（无）”）。
    # 三个新开关（`--buffer-decay-half-life` / `--aux-heads`）只能靠 `-ExtraArgs` 透传，而
    # “这一行不显示”有**两种完全不同的含义**：① 真没传 ② 卡片根本不支持这个字段。
    # 用户实测就撞上了这个歧义（V3 的上次启动记录里本来就没有 ExtraArgs ⇒ 整行消失）。
    if (-not $rows.Contains('额外参数(train.py)')) { $rows['额外参数(train.py)'] = '（无）' }
    [pscustomobject]@{
        src = $src; run = $run; rows = $rows; flags = $flags; live = $live2
        alive = [bool]($live -or $sup)
        phase = if ($live) { '训练中' } elseif ($sup) { '监督器在跑（轮边界测量/准备下一代）' } else { '未在跑' }
        warn = $warn
    }
}

function Get-State {
    $hist = Get-TrainHistory 10
    $procs = @(Get-Procs)
    $trainAlive = @($procs | Where-Object { $_.kind -eq '训练' }).Count -gt 0
    # 【2026-10-13 修】每代耗时/ETA 一律**只看当前 run 自己**的几代（见 Get-CadenceInfo 的长注释）
    $curRun = Get-CurrentRun
    $cad = Get-CadenceInfo $hist $curRun
    [pscustomobject]@{
        at = Get-Date
        procs = $procs
        trainAlive = $trainAlive
        meas = Get-Measurement
        eval = Get-EvalProgress
        train = Get-TrainDetail
        recipe = Get-TrainRecipe
        hist = $hist
        cadence = $cad.cadence
        cadenceRun = $curRun
        cadenceN = $cad.n
        cadenceByRun = Get-CadenceByRun $hist
        results = Get-Results 12
        artifacts = Get-Artifacts 8
        runs = Get-RunTotals
        curve = Get-CurveImg
        champ = Get-Champion
        gpu = Get-Gpu
        events = @(Get-Events)
    }
}

function Get-TrainLines($s) {
    $o = @()
    $t = $s.train
    if (-not $t) { $o += '  （找不到训练轮日志）'; return $o }
    $last = if ($s.hist.Count) { $s.hist[-1] } else { $null }
    $o += ("  当前：gen {0} 的【{1}】阶段" -f $t.gen, $t.phase)
    if ($t.stepTotal) { $o += ("    训练步 {0}/{1}" -f $t.stepDone, $t.stepTotal) }
    if ($t.target -and $t.gen) { $o += ("    本轮 目标 gen {0}，本轮已完成 {1} 代，还剩 {2} 代" -f $t.target, $t.doneThisRound, ([math]::Max(0, $t.target - $t.gen))) }
    if ($s.cadence) { $o += ("    每代约 {0} 分钟（{1}）（按本 run 已完成的 {2} 代）" -f $s.cadence, $s.cadenceRun, $s.cadenceN) }
    if ($last) {
        # 耗时明显超出常态 = 中间停过训练/被测量占过机器，必须标出来，
        # 否则"上一代 45.88 分钟"会被当成训练真的变慢了
        $note = ''
        $lastCad = if ($s.cadenceByRun -and $s.cadenceByRun.ContainsKey([string]$last.run)) { $s.cadenceByRun[[string]$last.run] } else { $null }
        if ($lastCad -and $last.mins -and $last.mins -gt $lastCad * 2.5) { $note = '（含停训/被占机器的时间，不是训练本身慢）' }
        $o += ("    上一代 gen {0}: loss_p={1} loss_v={2} lr={3} 训练步 {4} 数据 {5} 条（{6} 分钟）{7}" -f $last.gen, $last.loss_p, $last.loss_v, $last.lr, $last.steps, $last.samples, $last.mins, $note)
    }
    return $o
}

function Format-Console($s) {
    $o = @()
    $o += "==================== 实时看板  $($s.at.ToString('HH:mm:ss')) ===================="
    # 训练
    $o += ""
    $o += "  ── 训练"
    if ($s.train -and -not $s.trainAlive) {
        $lg = if ($s.hist.Count) { $s.hist[-1] } else { $null }
        $lastTxt = if ($lg) { "最后完成 gen $($lg.gen)（$($lg.at)）" } else { '' }
        $o += "  ⏹ 已停止（没有 train.py 进程在跑）  $lastTxt"
    } else {
        $o += Get-TrainLines $s
    }
    if ($s.hist.Count) {
        $o += "     最近几代（胜率 — = 该代未门控）:"
        foreach ($h in ($s.hist | Select-Object -Last 10)) {
            $wr = if ($null -ne $h.win_rate) { [string]$h.win_rate } else { '—' }
            $acc = switch ($h.accepted) { $true { '采纳' } $false { '拒绝' } default { '—' } }
            $o += ("       gen {0,4}  数据 {1,-5} loss_p={2,-7} loss_v={3,-7} 胜率={4,-5} {5,-4} {6} 分钟" -f `
                    $h.gen, $h.samples, $h.loss_p, $h.loss_v, $wr, $acc, $h.mins)
        }
    }
    # 进程
    $o += ""
    $o += "  ── 进程"
    if ($s.procs.Count -eq 0) { $o += "     ⚠ 没有任何训练/测量进程在跑" }
    else { foreach ($p in $s.procs) { $o += ("     ● {0,-8} PID {1,-7} 已跑 {2,5} 分钟" -f $p.kind, $p.pid, $p.mins) } }
    $o += "     （loss_p/loss_v 越小越好，但都不能判断棋力：自对弈目标非平稳、网络变强目标变难。"
    $o += "      基准：loss_p 完全没学=5.89；loss_v 一律猜平局≈0.8~1.0。它们只是「训练有没有炸」的哨兵，"
    $o += "      判断棋力看门控胜率 + 测量结果里的 Elo。）"
    # 轮边界测量（步骤级）
    if ($s.eval) {
        $el = if ($null -ne $s.eval.elapsed) { "已跑 $($s.eval.elapsed) 分钟" } else { '' }
        $o += ""
        $o += "  ── 轮边界测量：gen $($s.eval.gen)  $el"
        foreach ($r in $s.eval.rows) {
            $icon = switch ($r.status) { '完成' { '[OK]' } '进行中' { '[>>]' } default { '[  ]' } }
            $o += ("     {0} {1,-26} {2}" -f $icon, $r.name, $r.detail)
        }
    }
    # 各轮训练量
    if ($s.runs.Count) {
        $o += ""
        $o += "  ── 各轮训练量（自对弈局数 / 总时长）"
        foreach ($x in $s.runs) {
            $tag = if ($x.dup) { '（副本，不计入合计）' } else { '' }
            $pg = if ($null -ne $x.perGen) { ("{0:N2} 分/代" -f $x.perGen) } else { '—' }
            $o += ("     {0,-18} gen {1,3}-{2,-3}  {3,6:N0} 局  {4,7:N0} 步  实际 {5,-14} 墙钟 {6,-14} {7,-13} {8} {9}" -f `
                    $x.name, $x.g0, $x.g1, $x.games, $x.steps, (Format-Span $x.actualMin), (Format-Span $x.wallMin), $pg, $x.note, $tag)
        }
    }
    # 测量中
    if ($s.meas -and $s.meas.plain) {
        $o += ""
        $o += "  ── 正在写: $($s.meas.file)（$($s.meas.age) 秒前；此阶段无逐局进度行）"
        $o += "     " + $s.meas.tail
    } elseif ($s.meas) {
        $frac = if ($s.meas.total) { $s.meas.done / $s.meas.total } else { 0 }
        $filled = [int]([math]::Round($frac * 30))
        $bar = ('#' * $filled) + ('.' * (30 - $filled))
        $o += ""
        $o += "  ── 测量中: $($s.meas.file)"
        $o += ("     [{0}] {1}/{2} 局  ({3:P1})" -f $bar, $s.meas.done, $s.meas.total, $frac)
        if ($s.meas.rate) { $o += ("     速率 {0:N1} 局/分钟" -f $s.meas.rate) }
        if ($s.meas.eta) { $o += ("     预计完成 {0}" -f $s.meas.eta.ToString('HH:mm:ss')) }
    }
    # 测量结果
    $o += ""
    $o += "  ── 测量结果（最近 $($s.results.Count) 条）"
    foreach ($r in $s.results) {
        $elo = if ($null -ne $r.elo) { "ΔElo {0,7:N1} [{1,6:N1},{2,6:N1}] {3,-8}" -f [double]$r.elo, [double]$r.lo, [double]$r.hi, $r.metric } else { '' }
        $o += ("     {0}  {1,-40} {2,4} 局  s={3,-7} {4}  {5}" -f $r.at, $r.label, $r.games, $r.score, $elo, $r.verdict)
    }
    # champion
    if ($s.champ) {
        $o += ""
        $o += ("  ── champion: {0}  sha1_8={1}  相对 root {2} [{3},{4}]  hops={5}  （{6}）" -f `
                $s.champ.name, $s.champ.sha1_8, $s.champ.elo_vs_root, $s.champ.ci_lo, $s.champ.ci_hi, $s.champ.hops, $s.champ.updated_at)
    }
    if ($s.artifacts.Count) {
        $o += ""
        $o += "  ── 最近产出的结果文件"
        foreach ($a in $s.artifacts) { $o += ("     {0}  {1,-34} {2,7}KB  {3}" -f $a.at, $a.name, $a.kb, $a.sum) }
    }
    if ($s.gpu) {
        $o += ""
        $o += ("  ── GPU: {0}W  利用率 {1}%  显存 {2}MiB  温度 {3}C" -f $s.gpu.pwr, $s.gpu.util, $s.gpu.mem, $s.gpu.temp)
    }
    $o += ""
    $o += "  ── 最近事件"
    foreach ($e in $s.events) { $o += "     " + $e }
    $o += "================================================================================"
    return $o
}

function Format-Html($s) {
    # ---- 训练卡：KPI 数字块 + 进度条 + 轮次信息 ----
    $resumeCmd = "pwsh -NoProfile -File auto_train.ps1 -RunName V2 -SaveDir models/V2 -Workers 16 -Games 80 -MinSteps 160 -EvalGames 20 -EvalEvery 2 -EvalWorkers 8 -GensPerRound 15 -StartTarget 165 -AutoEval -RulerGames 200 -Seed 12345"
    $cls = if ($s.trainAlive) { 'ok' } else { 'mut' }
    $stateTxt = if ($s.trainAlive) { '训练中' } else { '已停止' }
    $lg = if ($s.hist.Count) { $s.hist[-1] } else { $null }
    $curGen = if ($lg) { [string]$lg.gen } else { '—' }
    $phaseTxt = if ($s.train -and $s.trainAlive) { $s.train.phase } elseif ($lg) { '已停止' } else { '—' }
    $cadTxt = if ($s.cadence) { [string]$s.cadence } else { '—' }
    $cadTitle = if ($s.cadence) { "按 $($s.cadenceRun) 已完成的 $($s.cadenceN) 代算的中位数（不含其它 run 的代次）" } else { '本 run 还没有可用的每代耗时样本' }
    $roundTxt = '—'
    if ($s.train -and $s.train.target -and $s.train.gen) {
        $roundTxt = ("{0} / gen {1}" -f $s.train.doneThisRound, $s.train.target)
    }
    $champTxt = '—'; $champSub = ''
    if ($s.champ) {
        $champTxt = [string]$s.champ.name
        $champSub = ("相对 root {0} [{1}, {2}]" -f $s.champ.elo_vs_root, $s.champ.ci_lo, $s.champ.ci_hi)
    }
    $gpuTxt = if ($s.gpu) { "$($s.gpu.util)" } else { '—' }
    # 【2026-10-13】GPU 明细并入"训练概览"的 KPI（用户要求），于是独立的 GPU 卡片撤掉。
    # 排版跟"当前 champion"那格一致：大数字 = 利用率，下面一行小字 = 功耗/显存/温度。
    $gpuSub = if ($s.gpu) { "{0}W · 显存{1}MiB · {2}℃" -f $s.gpu.pwr, $s.gpu.mem, $s.gpu.temp } else { '' }

    $trainHtml = @"
<div class="kpis">
  <div class="kpi $cls"><div class="k">状态</div><div class="v">$stateTxt</div></div>
  <div class="kpi acc"><div class="k">当前代数</div><div class="v">$curGen</div></div>
  <div class="kpi"><div class="k">阶段</div><div class="v" style="font-size:16px">$phaseTxt</div></div>
  <div class="kpi"><div class="k">每代耗时<span class="u">分钟</span></div><div class="v" title="$cadTitle">$cadTxt</div></div>
  <div class="kpi"><div class="k">本轮进度<span class="u">代</span></div><div class="v" style="font-size:16px">$roundTxt</div></div>
  <div class="kpi pur"><div class="k">当前 champion</div><div class="v" style="font-size:16px">$champTxt</div>
    <div class="k" style="margin-top:2px;letter-spacing:0">$champSub</div></div>
  <div class="kpi"><div class="k">GPU 利用率<span class="u">%</span></div><div class="v">$gpuTxt</div>
    <div class="k" style="margin-top:2px;letter-spacing:0">$gpuSub</div></div>
</div>
"@

    $barHtml = ''
    if ($s.train -and $s.trainAlive) {
        $t = $s.train
        if ($t.stepTotal -and $t.stepTotal -gt 0) {
            $pct = [int]([math]::Round($t.stepDone / $t.stepTotal * 100))
            $barHtml = "<div class='bar'><div class='fill' style='width:$pct%'></div></div>" +
                       "<div class='plabel'><span>训练步 $($t.stepDone)/$($t.stepTotal)</span><span>$pct%</span></div>"
        } elseif ($t.phaseIdx -eq 1) {
            $el = if ($t.genElapsed) { [double]$t.genElapsed } else { 0 }
            $typ = if ($t.spTypical) { [double]$t.spTypical } else { 0 }
            $pct = if ($typ -gt 0) { [math]::Min(96, [int]([math]::Round($el / $typ * 100))) } else { 50 }
            $barHtml = "<div class='bar'><div class='fill' style='width:$pct%'></div></div>" +
                       ("<div class='plabel'><span>自对弈中 · 已跑 {0:N1} 分钟</span><span>{1}</span></div>" -f ($el / 60), $(if ($typ -gt 0) { "通常 {0:N1} 分钟" -f ($typ / 60) } else { '' }))
        }
    }
    $roundLine = ''
    if ($s.train) {
        $t = $s.train
        if (-not $s.trainAlive) {
            $lastTxt = if ($lg) { "最后完成 <b>gen $($lg.gen)</b>（$($lg.at)）" } else { '' }
            $roundLine = "<p class='small muted' style='margin:12px 0 0'>没有 <code>train.py</code> 进程在跑。$lastTxt</p>" +
                         "<p class='small muted' style='margin:8px 0 0'>恢复命令：<br><code>$resumeCmd</code></p>"
        } elseif ($t.target -and $t.gen) {
            $left = [math]::Max(0, $t.target - $t.gen)
            if ($left -eq 0 -and $t.phaseIdx -eq 4) {
                $roundLine = "<p class='small muted' style='margin:12px 0 0'>本轮 15 代已全部保存　·　现在应该在跑<b>轮边界测量</b>（硬门槛 → 200 局 ladder → 画图），测完自动开下一轮</p>"
            } elseif ($left -eq 0) {
                $roundLine = "<p class='small muted' style='margin:12px 0 0'>正在跑<b>本轮最后一代</b>（gen $($t.target)）</p>"
            } else {
                # 【2026-10-13 修】必须写清这个钟点是"按什么"算的。用户报「预计完成时间有问题，
                # 数据太假了」，正是因为旧版拿别的 run 的每代耗时算出了一个很像样、但错的钟点。
                $eta = if ($s.cadence) { (Get-Date).AddMinutes($left * $s.cadence).ToString('HH:mm') } else { '—' }
                $cadTxtInline = if ($s.cadence) { "每代约 $($s.cadence) 分钟" } else { '每代耗时未知' }
                $etaWhy = if ($s.cadenceN -ge 2) { "按本 run 已完成的 $($s.cadenceN) 代" }
                          elseif ($s.cadenceN -eq 1) { '只有本 run 的 1 代样本，仅供参考' }
                          else { '本 run 还没跑完一代，给不了估计' }
                $roundLine = "<p class='small muted' style='margin:12px 0 0'>还剩 <b>$left</b> 代到本轮目标 gen $($t.target)　·　本轮预计完成 <b>$eta</b> <span class='muted'>（$etaWhy，$cadTxtInline）</span></p>"
            }
        }
    }
    # ---- 训练历史表（最近 10 代）----
    # 胜率为空的代**不是缺数据**：门控每 eval_every(2) 代才跑一次，没跑的代 accepted=None。
    # 所以空显示成 —，并单独给出"采纳"列（是否通过阈值替换 best_net）。
    # 【2026-10-12 改】加一列「轮次」：现在多个 run（V2/V3…）混在同一张表里，
    # 不标 run 名就分不清哪一行是哪条线（用户报"网页里训练历史不显示这个模型"时发现的）。
    $histRows = ''
    $abn = $false
    $gatedN = 0; $accN = 0
    foreach ($h in ($s.hist | Select-Object -Last 10)) {
        $m = if ($null -ne $h.mins) { [string]$h.mins } else { '—' }
        # 逐行用**该行自己 run** 的节奏判"明显偏慢（*）"：拿当前 run 的节奏去量别的 run 的代次会误标
        $rowCad = if ($s.cadenceByRun -and $s.cadenceByRun.ContainsKey([string]$h.run)) { $s.cadenceByRun[[string]$h.run] } else { $null }
        if ($rowCad -and $h.mins -and $h.mins -gt $rowCad * 2.5) { $m = "$m *"; $abn = $true }
        $wr = '—'
        if ($null -ne $h.win_rate) {
            $gatedN++
            $wr = ("<span title='原始={0} EMA={1} 采用={2}'>{3}</span>" -f $h.win_raw, $h.win_ema, $h.chosen, $h.win_rate)
        }
        $acc = '—'
        if ($h.accepted -eq $true) { $acc = "<span style='color:var(--ok)'>✅ 采纳</span>"; $accN++ }
        elseif ($h.accepted -eq $false) { $acc = "<span style='color:var(--bad)'>❌ 拒绝</span>" }
        $runName = if ($h.run) { ($h.run -replace '^models\\', '') } else { '?' }
        $histRows += ("<tr><td>{0}</td><td>gen {1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td>{9}</td></tr>" -f `
                $runName, $h.gen, $h.samples, $h.games, $h.steps, $h.loss_p, $h.loss_v, $wr, $acc, $m)
    }
    if (-not $histRows) { $histRows = "<tr><td colspan=9 style='color:var(--mut)'>暂无</td></tr>" }
    $histNotes = @()
    $histNotes += "胜率 <b>—</b> = 这一代没门控（门控每 2 代跑一次，20 局 vs best_net）；悬停可看 原始/EMA 两个值"
    $histNotes += "「采纳」= 该代胜率是否超过阈值、替换 best_net —— 本 10 代里门控 <b>$gatedN</b> 次、采纳 <b>$accN</b> 次"
    if ($abn) { $histNotes += "耗时带 <b>*</b> = 这一代中间停过训练/机器被测量占过，不代表训练本身变慢" }
    $histNote = "<div class='small muted' style='line-height:1.8'>" + (($histNotes | ForEach-Object { "· $_" }) -join '<br>') + "</div>"

    # ---- loss 说明（用户问过："这两个有什么用、越高越好吗"）----
    # 结论与实测都来自 train.py:1626-1640 已经写下的教训，这里只是搬到页面上。
    $lossNote = @"
<div class="note">
<b>loss_p / loss_v 是什么、越大越好吗</b><br>
· <b>loss_p（策略损失）</b>＝ 网络给出的<b>走子概率</b> 与 <b>MCTS 搜索出来的走子分布</b> 的交叉熵。
　<b>越小越好</b> —— 衡量"网络有多像它自己的搜索"。参考：<b>完全没学 = ln(361) ≈ 5.89</b>；我们目前在 <b>≈ 1.24</b><br>
· <b>loss_v（价值损失）</b>＝ 网络对<b>胜负</b>的预测 与 <b>实际胜负</b> 的均方误差。
　<b>越小越好</b> —— 衡量价值头的预测准不准。参考：<b>一律猜平局 ≈ 0.8~1.0</b>；我们目前在 <b>≈ 0.29</b><br>
· <b style='color:var(--warn)'>但这两个都不能用来判断棋力！</b>自对弈的<b>训练目标是非平稳的</b>：
　策略目标永远是"当前网络自己这轮搜索的输出"，网络变强 ⇒ 目标跟着变难。
　所以 loss 持平的真实含义是"在追一个不断变难的目标"，不是"没在学"<br>
· <b>本 30 代（gen 121–150）实测就印证了这点</b>：loss_p 与代数的相关系数
　<b style='color:var(--warn)'>+0.357（在升）</b>、loss_v <b style='color:var(--warn)'>+0.119（无趋势）</b>，
　而同期 gen_135 比 gen_120 强 <b>+38.4 Elo</b> —— <b>损失不降、棋力照涨</b><br>
· 它们的<b>真正用途是哨兵</b>：突然 NaN / 暴涨 = 训练炸了；长期贴在 5.89 / 0.8 以上 = 根本没学。<br>
　<b style='color:var(--ok)'>判断棋力只看「门控胜率」+「测量结果」里的 Elo 尺子（独立对战）。</b>
</div>
"@
    # ---- 轮边界测量（步骤级）----
    $evalRows = ''
    if ($s.eval) {
        foreach ($r in $s.eval.rows) {
            $icon = switch ($r.status) { '完成' { '✅' } '进行中' { '🔄' } default { '⏳' } }
            $pc = switch ($r.status) { '完成' { 'ok' } '进行中' { 'warn' } default { 'mut' } }
            $evalRows += ("<tr><td>{0}</td><td><span class='pill {1}'>{2} {3}</span></td><td class='muted'>{4}</td></tr>" -f `
                    $r.name, $pc, $icon, $r.status, $r.detail)
        }
    }
    $evalHtml = ''
    if ($evalRows) {
        $el = if ($null -ne $s.eval.elapsed) { "　已跑 <b>$($s.eval.elapsed)</b> 分钟" } else { '' }
        $title = if ($s.eval.finished) { "轮边界测量：gen $($s.eval.gen)<span style='color:var(--ok)'>　✅ 已全部完成</span>$el" } else { "轮边界测量：gen $($s.eval.gen)$el" }
        $evalHtml = "<div class='card' id='c-eval'><h3>$title</h3><table class='data'>$evalRows</table></div>"
    }
    # ---- 训练曲线图 ----
    # 【为什么用固定尺寸的 CSS 背景，而不是 <img>】
    # 整页重载时 <img> 会先占位、再解码、有时先闪一个白块（图本身是白底 1080×960），
    # 用户反馈"隐约看到一张大图"就是这个。改成"固定宽高的 div + 背景图"后：
    #   · 尺寸在 CSS 里就先定好 ⇒ 不产生布局跳动
    #   · 背景图走缓存 ⇒ 通常直接绘制，不会先白块
    $curveHtml = ''
    # 【2026-10-12 修】原图链接原来写死 models/V2/progress.png —— 换成"当前最新的那张"，
# 否则 V3 跑起来后这个链接仍指向 V2（用户会以为 V3 的曲线没生成）。
$curveRel = if ($s.curve) { $s.curve.rel } else { 'models/V2/progress.png' }
# 【2026-10-12 修】标题栏的「打开曲线原图」原来用的是**不带版本号**的 $curveRel，
# 而服务端给这张图发的是 `Cache-Control: public, max-age=86400`（强缓存 1 天）
# ⇒ 图更新了、URL 没变 ⇒ 浏览器给你**旧图**（用户实测："打开原图还是旧图"）。
# 统一改成带版本号（= 图片 mtime 的 Ticks）的 URL，与卡片里那个链接、内联背景图一致。
$curveUrl = if ($s.curve) { $s.curve.rel + '?v=' + $s.curve.ver } else { $curveRel }
$curveWhich = if ($s.curve) { '（当前：' + (Split-Path (Split-Path $s.curve.rel -Parent) -Leaf) + '）' } else { '' }
if ($s.curve) {
        $w = 620; $h = 551   # 1080×960 等比缩到宽 620 ⇒ 高 551
        $curveHtml = @"
<div class="card" id="c-curve"><h3>训练进步图（Elo 相对固定基准 root + 留出集指标）</h3>
<div class='small muted' style='margin-bottom:8px'>
  $($s.curve.rel)　·　$($s.curve.kb) KB　·　画于 $($s.curve.at.ToString('MM-dd HH:mm:ss'))
  　·　只在轮边界重画　·　<a href="$($s.curve.rel)?v=$($s.curve.ver)" target="_blank" style="color:var(--acc2)">在新标签打开原图</a>
</div>
<div style="width:${w}px;max-width:100%;height:${h}px;max-height:70vh;
            background:url('$($s.curve.rel)?v=$($s.curve.ver)') center/contain no-repeat #fff;
            border:1px solid var(--line)"></div>
</div>
"@
    }

    # ---- 训练控制面板：字段在服务端就渲染好 ----
    # 【为什么不能靠 JS 生成字段】整页重载（file:// 那条路）时，JS 生成字段是**异步**的：
    # 先画出空的面板（矮）→ 拿到 /api/last-params 后才出现输入框（高）
    # ⇒ 用户看到"参数消失再出现"、而且**下面的内容跟着跳**（用户实测反馈过）。
    # 现在字段结构（含中文名/说明/基线值）由 PowerShell 直接写进 HTML，
    # JS 只负责"填值"（不改变布局）⇒ 首帧高度与内容都是最终状态，不会跳。
    $ctlSpec = @(
        @{ n = 'RunName'; k = 'text'; cn = '轮次名'; tip = '只用于记录与曲线标签，随便起（如 V2）'; py = $true },
        @{ n = 'SaveDir'; k = 'text'; cn = '模型目录'; tip = '必须 models/名字 形状；写别处会被安全白名单拒绝'; py = $true },
        @{ n = 'Workers'; k = 'int'; cn = '自对弈进程数'; tip = '实测 16 最优（12→1.21×，20 已饱和）' },
        @{ n = 'Games'; k = 'int'; cn = '每代对局数'; tip = '每代自对弈多少局' },
        @{ n = 'MinSteps'; k = 'int'; cn = '每代训练步数'; tip = '160 步约 6.9 遍新数据（1200 步反而更差）' },
        @{ n = 'EvalGames'; k = 'int'; cn = '门控局数'; tip = '内部门控每场多少局' },
        @{ n = 'EvalEvery'; k = 'int'; cn = '门控间隔'; tip = '每几代跑一次内部门控' },
        @{ n = 'EvalWorkers'; k = 'int'; cn = '门控进程数'; tip = '门控对局的并行进程数'; py = $true },
        @{ n = 'GensPerRound'; k = 'int'; cn = '每轮代数'; tip = '每轮训多少代（=1 就是只跑一代）' },
        @{ n = 'StartTarget'; k = 'int'; cn = '本轮目标代数'; tip = '留 0 或空 = 自动取 当前最新代数 + 每轮代数'; py = $true },
        @{ n = 'RulerGames'; k = 'int'; cn = '标尺对局数'; tip = '轮边界标尺多少局（200 局约 ±34 Elo）'; py = $true },
        @{ n = 'Seed'; k = 'int'; cn = '随机种子'; tip = '复现用' },
        @{ n = 'AutoEval'; k = 'bool'; cn = '轮边界自动测量'; tip = '勾上则每轮结束自动跑硬门槛 + 标尺 + 画图'; py = $true },
        # ---- 【2026-10-12 加】V3 配方用的高级参数 ----
        # 哨兵：留 -1 = 不传该参数（保持旧行为）；0 是合法值（例如 hw=0 = 纯网络先验）。
        @{ n = 'BufferGens'; k = 'int'; cn = '回放池保留代数'; tip = '留 -1 = 用 train.py 默认（30 代）；数据×2 时要一起放大（如 60），否则新数据边加边扔' },
        @{ n = 'HeuristicWeight'; k = 'num'; cn = '规则先验权重(hw)'; tip = '留 -1 = 用退火；填 0 = 纯网络先验（规则层占每手 CPU ~19%，实测与融合先验打平）' },
        @{ n = 'PolicyWeight'; k = 'num'; cn = '策略先验权重(pw)'; tip = '留 -1 = 用退火；配 hw=0 时填 1' },
        @{ n = 'NoAnneal'; k = 'bool'; cn = '关闭退火（固定 hw/pw）'; tip = '勾上 = 全程用上面两个固定权重（不随代数变化）'; py = $true },
        @{ n = 'AllowOverwrite'; k = 'bool'; cn = '允许写入已有产物的目录'; tip = '冷启动到已有 best.pt 的目录（如 V3 首次开跑）必须勾：train.py 默认拒绝'; py = $true },
        # ---- 【2026-10-13 加】透传给 train.py 的额外开关 ----
        # 为什么需要：train.py 的新开关（采样整形 --buffer-decay-half-life / --buffer-entropy-weight、
        # 辅助头 --aux-heads / --aux-ch）在本面板里没有具名字段；没有这一格，
        # 点「开始训练」就永远用不上新开关（**静默退回旧行为**，不报错）。
        # 字符白名单：字母/数字/下划线/短横/=/,/./空格（挡住引号、反引号、$()、; | & 这类注入字符）。
        @{ n = 'ExtraArgs'; k = 'text'; cn = '额外参数(透传)'; tip = '留空 = 不传。例：--buffer-decay-half-life 20 或 --aux-heads threat,ownership。只允许字母数字与 _ - = , . / 空格' }
    )

    # ---- 【2026-10-26 加】C++ 引擎的默认配方 ----
    # 唯一权威来源是 C++\src\config.h：直接问 `gk.exe defaults`（它打印 TrainConfig 的默认构造）。
    # 页面每 3 秒重新生成一次，所以这里只在看板启动时取一遍 —— 改了 config.h 要重启看板才会看到新默认值。
    # 【2026-10-27 修】这里必须真的只取一遍。原先只是注释写着"只在看板启动时取一遍"，
    # 实际每轮 Format-Html 都会重跑：于是每 3 秒 `$env:PATH = "$torchLib;$env:PATH"` 加长 ~90 字符，
    # 约 20 分钟后撞上 Win32 环境块 32KB 上限 ⇒ 之后**每个子进程启动都报
    # "应用程序无法正常启动(0xc0000017)"**（python.exe / gk.exe 各弹一次，刷屏），
    # 同时 `gk.exe defaults` 再也拿不到配方。用 script 作用域缓存 + PATH 幂等双重兜住。
    if (-not $script:CppDefaultsTaken) {
        $script:CppDefaultsTaken = $true
        $script:CppDefaults = $null
        $gkExe = Join-Path (Split-Path $PSScriptRoot -Parent) 'C++\build\gk.exe'
        if (Test-Path $gkExe) {
            try {
                # gk.exe 链接了 libtorch，启动时要能找到 torch 的 DLL（否则静默 0xC0000135）
                $torchLib = ''
                try { $torchLib = (& python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>$null) } catch { }
                if ($torchLib -and (Test-Path $torchLib) -and ($env:PATH -notlike "*$torchLib*")) {
                    $env:PATH = "$torchLib;$env:PATH"
                }
                $json = & $gkExe defaults 2>$null
                if ($json) { $script:CppDefaults = ($json | Out-String | ConvertFrom-Json) }
            } catch { $script:CppDefaults = $null }
        }
    }
    $cppDefaults = $script:CppDefaults
    # C++ 专属参数（Python 面板里没有对应概念，所以单独一块，只在引擎=C++ 时显示）
    $cppSpec = @(
        @{ n = 'Recipe'; key = 'recipe'; k = 'int'; cn = '配方编号'; tip = '--recipe；写进 train_log 的 recipe_id，看板据此在曲线上标「配方N」断点' },
        @{ n = 'Sims'; key = 'sims'; k = 'int'; cn = '每手模拟数'; tip = '--sims；600（400 时策略目标偏噪）' },
        @{ n = 'BatchMcts'; key = 'batch_mcts'; k = 'int'; cn = '搜索批大小'; tip = '--batch-mcts；叶子批 16（调大实测更弱）' },
        @{ n = 'Batch'; key = 'batch'; k = 'int'; cn = '训练批大小'; tip = '--batch；256' },
        @{ n = 'LR'; key = 'lr'; k = 'num'; cn = '学习率'; tip = '--lr；3e-4（1e-3 会把好权重拽走）' },
        @{ n = 'LrGenDecay'; key = 'lr_gen_decay'; k = 'num'; cn = '学习率按代退火'; tip = '--lr-gen-decay；0.97 = 每代乘 0.97；1 = 不降（旧行为）' },
        @{ n = 'LrFloor'; key = 'lr_floor'; k = 'num'; cn = '学习率地板'; tip = '--lr-floor；退火到此为止，1e-4' },
        @{ n = 'Fresh'; key = 'fresh_ratio'; k = 'num'; cn = '新数据比例'; tip = '--fresh-ratio；0.5（0.7 会随单代噪声来回摆）' },
        @{ n = 'EvalBatch'; key = 'eval_batch'; k = 'int'; cn = '评估合并上限'; tip = '--eval-batch；32（GPU 是延迟瓶颈，凑大反而慢）' },
        @{ n = 'EvalThreads'; key = 'eval_threads'; k = 'int'; cn = '评估线程数'; tip = '--eval-threads；3（各自独立 CUDA stream）' },
        @{ n = 'OpeningPlies'; key = 'opening_plies'; k = 'int'; cn = '门控配对开局'; tip = '--opening-plies；2 = 正反两局用同一盘面前两手，只换黑白' },
        @{ n = 'GateThreshold'; key = 'gate_threshold'; k = 'num'; cn = '门控通过阈值'; tip = '--gate-threshold；0.55（旧权重淘汰新权重要赢 55%）' },
        @{ n = 'HalfLife'; key = 'half_life'; k = 'num'; cn = '回放池半衰期'; tip = '--half-life；20 代' },
        @{ n = 'Procs'; key = 'procs'; k = 'int'; cn = '多进程自对弈'; tip = '--procs；0=单进程多线程（实测多进程更慢）' },
        @{ n = 'KeepGens'; key = 'keep_gens'; k = 'int'; cn = '保留最近几代权重'; tip = '--keep-gens；0=全留' },
        @{ n = 'KeepBufferGens'; key = 'keep_buffer_gens'; k = 'int'; cn = '回放池文件保留代数'; tip = '--keep-buffer-gens；10' }
    )
    $cppHtml = ''
    foreach ($f in $cppSpec) {
        $v = ''
        if ($cppDefaults -and ($cppDefaults.PSObject.Properties.Name -contains $f.key)) { $v = $cppDefaults.($f.key) }
        $cppHtml += "<label title='$($f.n)：$($f.tip)'>"
        $cppHtml += "<span class='fn'>" + $f.cn + " <em>($($f.n))</em></span>"
        $itype = if ($f.k -eq 'int' -or $f.k -eq 'num') { 'number' } else { 'text' }
        $istep = if ($f.k -eq 'num') { " step='any'" } else { '' }
        $cppHtml += "<input type='$itype' id='q_" + $f.n + "'$istep value='" + $v + "'>"
        $cppHtml += "<span class='tip'>" + $f.tip + "</span></label>"
    }
    $cppJsSpec = ($cppSpec | ForEach-Object { "  ['$($_.n)', '$($_.k)', '$($_.cn)', '$($_.tip)']" }) -join ",`n"
    # 共享字段的 C++ 默认值（JS 对象字面量）。取不到 gk.exe defaults 时留空，
    # 这时切引擎只切显示、不动数值，并在提示里说明。
    # ---- 【2026-10-28 加】配方档案（C++\peifang\配方NN-*.json）----
    # 用户要求：「训练控制里加一个填入配方按钮，可以选 peifang 里的配方」。
    # 每份档案的 args 原样烘进页面（snake_case 键 = gk.exe 开关名），前端用一张
    # 键→表单 id 的映射表回填 ⇒ 以后新增配方只要丢一个 json 进目录，不用改代码。
    # 15 秒 TTL 缓存：页面每 3 秒重新生成，别每次都去读盘（也别再犯 PATH 那类每轮副作用）。
    $nowTs = Get-Date
    if (-not $script:CppRecipesAt -or ($nowTs - $script:CppRecipesAt).TotalSeconds -gt 15) {
        $script:CppRecipesAt = $nowTs
        $script:CppRecipes = @()
        $peifangDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'C++\peifang'
        if (Test-Path $peifangDir) {
            foreach ($pf in (Get-ChildItem $peifangDir -Filter '*.json' | Sort-Object Name)) {
                try {
                    $pj = Get-Content $pf.FullName -Raw -Encoding utf8 | ConvertFrom-Json
                    $script:CppRecipes += [pscustomobject]@{
                        file    = $pf.Name
                        id      = $pj.id
                        name    = $pj.name
                        status  = $pj.status
                        created = $pj.created
                        args    = $pj.args
                        command = $pj.command
                        changed = $pj.changed_vs_recipe_1
                        lesson  = $pj.lesson
                    }
                } catch { }
            }
        }
    }
    $cppRecipesJs = '[]'
    if ($script:CppRecipes -and $script:CppRecipes.Count -gt 0) {
        try { $cppRecipesJs = ($script:CppRecipes | ConvertTo-Json -Depth 8 -Compress -AsArray) } catch { $cppRecipesJs = '[]' }
    }
    $cppSharedJs = ''
    if ($cppDefaults) {
        $pairs = @(
            @('Workers', 'workers'), @('Games', 'games'), @('MinSteps', 'min_steps'),
            @('EvalGames', 'eval_games'), @('EvalEvery', 'eval_every'),
            @('GensPerRound', 'gens'), @('Seed', 'seed'),
            @('BufferGens', 'buffer_gens'), @('HeuristicWeight', 'hw'), @('PolicyWeight', 'pw')
        )
        $kv = @()
        foreach ($p in $pairs) {
            $v = $cppDefaults.($p[1])
            if ($null -ne $v) { $kv += ("$($p[0]): $v") }
        }
        $cppSharedJs = ($kv -join ', ')
    # ---- 【2026-10-28 加】配方档案（C++\peifang\配方NN-*.json）----
    # 用户要求：「训练控制里加一个填入配方按钮，可以选 peifang 里的配方」。
    # 这里把每份档案的 args 原样烘进页面（snake_case 键 = gk.exe 的开关名），
    # 前端用一张 键→表单 id 的映射表回填，所以新增配方只要丢一个 json 进目录，不用改代码。
    # 15 秒 TTL 缓存：页面每 3 秒重生成一次，别每次都去读盘。
    $nowTs = Get-Date
    if (-not $script:CppRecipesAt -or ($nowTs - $script:CppRecipesAt).TotalSeconds -gt 15) {
        $script:CppRecipesAt = $nowTs
        $script:CppRecipes = @()
        $peifangDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'C++\peifang'
        if (Test-Path $peifangDir) {
            foreach ($pf in (Get-ChildItem $peifangDir -Filter '*.json' | Sort-Object Name)) {
                try {
                    $pj = Get-Content $pf.FullName -Raw -Encoding utf8 | ConvertFrom-Json
                    $script:CppRecipes += [pscustomobject]@{
                        file = $pf.Name
                        id = $pj.id
                        name = $pj.name
                        status = $pj.status
                        created = $pj.created
                        args = $pj.args
                        command = $pj.command
                        changed = $pj.changed_vs_recipe_1
                        lesson = $pj.lesson
                    }
                } catch { }
            }
        }
    }
    $cppRecipesJs = '[]'
    if ($script:CppRecipes -and $script:CppRecipes.Count -gt 0) {
        try { $cppRecipesJs = ($script:CppRecipes | ConvertTo-Json -Depth 8 -Compress -AsArray) } catch { $cppRecipesJs = '[]' }
    }    # ---- 【2026-10-26 加】C++ 运行目录列表 ----
    # 目录名本身说不清「谁该继续训」（run_v2 是从 run_v4amp 第 7 代接过来的，它自己的 gen 从 1 重开），
    # 所以下拉里直接显示 run.json 里的角色 + 换算后的全局代数。
    $cppModelRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'C++\model'
    $cppRuns = @()
    if (Test-Path $cppModelRoot) {
        foreach ($d in Get-ChildItem $cppModelRoot -Directory) {
            $gens = @(Get-ChildItem $d.FullName -Filter 'gen_*.bin' -ErrorAction SilentlyContinue)
            $hasBest = Test-Path (Join-Path $d.FullName 'best.bin')
            if (-not ($gens.Count -or $hasBest)) { continue }
            $role = ''; $note = ''; $off = 0
            $mf = Join-Path $d.FullName 'run.json'
            if (Test-Path $mf) {
                try {
                    $j = Get-Content $mf -Raw -Encoding UTF8 | ConvertFrom-Json
                    $role = [string]$j.role; $note = [string]$j.note; $off = [int]$j.gen_offset
                } catch { }
            }
            $last = 0
            if ($gens.Count) {
                $nb = ($gens | Sort-Object Name)[-1].BaseName
                $last = [int]($nb -replace '^gen_0*', '')
            }
            $cppRuns += [pscustomobject]@{
                dir = ('model/' + $d.Name); role = $role; note = $note
                total = ($last + $off); gens = $gens.Count; mtime = $d.LastWriteTime
            }
        }
    }
    $cppRuns = @($cppRuns | Sort-Object @{ e = { if ($_.role -eq '主线') { 0 } else { 1 } } }, @{ e = { $_.mtime }; Descending = $true })
    $recRun = $cppRuns | Where-Object { $_.role -eq '主线' } | Select-Object -First 1
    if (-not $recRun) { $recRun = $cppRuns | Where-Object { $_.role -ne '损坏' } | Select-Object -First 1 }
    $cppRunDefault = if ($recRun) { $recRun.dir } else { 'model/run' }
    $cppRunOptions = ''
    foreach ($r in $cppRuns) {
        $label = $r.dir
        if ($r.role) { $label += ' · ' + $r.role }
        if ($r.total -gt 0) { $label += ' · 全局第 ' + $r.total + ' 代' }
        $selAttr = if ($r.dir -eq $cppRunDefault) { ' selected' } else { '' }
        $cppRunOptions += "<option value='$($r.dir)'$selAttr>$label</option>"
    }
    $cppRunOptions += "<option value='__custom__'>＋ 其他（手动输入路径）</option>"
    }
    # 基线值：从「上次参数」文件读（接口拿到更新值后，JS 只改 value，不动布局）
    $ctlBase = @{}
    $lpf = Join-Path $PSScriptRoot 'checkpoints\last_train_params.json'
    if (Test-Path $lpf) {
        try {
            $lpj = Get-Content $lpf -Encoding utf8 -Raw | ConvertFrom-Json
            foreach ($p in $lpj.params.PSObject.Properties) { $ctlBase[$p.Name] = $p.Value }
        } catch { }
    }
    $ctlHtml = ''
    foreach ($f in $ctlSpec) {
        $v = if ($ctlBase.ContainsKey($f.n)) { $ctlBase[$f.n] } else { '' }
        $eng = if ($f.py) { " data-eng='python'" } else { '' }
        $ctlHtml += "<label$eng title='" + $f.n + "：" + $f.tip + "'>"
        $ctlHtml += "<span class='fn'>" + $f.cn + " <em>(" + $f.n + ")</em></span>"
        if ($f.k -eq 'bool') {
            $checked = if ($v -eq $true -or $v -eq 'True' -or $v -eq 'true') { ' checked' } else { '' }
            $ctlHtml += "<input type='checkbox' id='p_" + $f.n + "'$checked>"
        } else {
            # 【必须按类型分开】文字参数放进 type=number 会被浏览器判为非法值 ⇒ 显示成空白
            # （实测踩过：轮次名/模型目录两个框看起来"字不见了"）。数字才用 number。
            # 【2026-10-13】新增 'num'（浮点）：hw/pw 是 0.5 / 2.0 这种小数，
            # 以前被当成 'int' ⇒ 前端 parseInt('0.5') = 0 ⇒ **静默把 hw 改成 0** 发给服务端
            # （正是当初那次 -HeuristicWeight 0 事故）。type=number 默认 step=1 会把 0.5 判非法，
            # 所以浮点字段要 step='any'。
            $itype = if ($f.k -eq 'int' -or $f.k -eq 'num') { 'number' } else { 'text' }
            $istep = if ($f.k -eq 'num') { " step='any'" } else { '' }
            $ctlHtml += "<input type='$itype' id='p_" + $f.n + "'$istep value='" + $v + "'>"
        }
        $ctlHtml += "<span class='tip'>" + $f.tip + "</span></label>"
    }
    # 同一份 spec 注入给 JS（避免中英文两处维护、日后漂移）
    $ctlJsSpec = ($ctlSpec | ForEach-Object { "  ['$($_.n)', '$($_.k)', '$($_.cn)', '$($_.tip)']" }) -join ",`n"

    # ---- 现在训练的模型参数（用户 2026-10-12 要求：一眼看到"在训的东西是什么配方"）----
    $rcp = $s.recipe
    $rcpRows = ''
    if ($rcp) {
        foreach ($k in $rcp.rows.Keys) {
            $rcpRows += "<tr><td>$k</td><td><b>$($rcp.rows[$k])</b></td></tr>"
        }
        foreach ($k in $rcp.flags.Keys) {
            $rcpRows += "<tr><td>$k</td><td>$($rcp.flags[$k])</td></tr>"
        }
        foreach ($k in $rcp.live.Keys) {
            $rcpRows += "<tr><td>$k</td><td>$($rcp.live[$k])</td></tr>"
        }
    }
    if (-not $rcpRows) { $rcpRows = "<tr><td>—</td><td>没有可显示的参数</td></tr>" }
    $rcpWarn = ''
    if ($rcp -and $rcp.warn -and $rcp.warn.Count) {
        $rcpWarn = "<div style='background:linear-gradient(180deg,rgba(255,110,110,.22),rgba(255,80,80,.12));border:1px solid rgba(255,130,130,.55);border-radius:12px;padding:7px 10px;margin:6px 0'>" +
                   "⚠️ <b>配方与已验证的生产配方不一致</b>：" + (($rcp.warn) -join '；') +
                   "<br><span class='small'>hw/pw 是<b>引擎口径</b>：改了它就等于换引擎，跨口径的强度读数不可直接比较" +
                   "（见 <code>NOTES.md</code> §12）。已验收的 V3 稳妥版是 <b>hw=0.5 / pw=1.0</b>；" +
                   "表单里的值可能来自<b>浏览器草稿</b>（旧一次失败尝试填过 hw=0）。</span></div>"
    }
    $rcpTitle = if ($rcp -and $rcp.alive) { "🔴 现在训练的模型参数（$($rcp.phase)）" }
                else { '现在训练的模型参数（训练未在跑）' }
    $rcpRun = if ($rcp -and $rcp.run) { "轮次 <b>$($rcp.run)</b>　" } else { '' }
    $recipeHtml = @"
<div class="card" id="recipe"><h3>$rcpTitle</h3>
<div class='small muted'>$(if ($rcpRun) { $rcpRun })数据来源：$(if ($rcp) { $rcp.src } else { '—' })</div>
<div class='small muted' style='margin-top:4px'>生产配方基准（已验收）：<b>hw=0.5 / pw=1.0</b>　|　口径警告：hw/pw 是<b>引擎口径</b>，改了它 = 换了引擎，跨口径的强度读数不可直接比</div>
$rcpWarn
<table class="data" style="margin-top:6px">$rcpRows</table>
<div class='small muted' style='line-height:1.7'>
· 优先级：<b>正在跑的 train.py 命令行</b>（它真正收到的参数）&gt; <code>checkpoints/last_train_params.json</code>（上次启动表单值）&gt;
<code>train_log.jsonl</code> 最新一行（实际产生的 lr/代数/数据量）。三者含义不同，所以上面写明用了哪一路。<br>
· 想改参数：在上面<b>「训练控制」</b>面板改，然后点「开始训练」（会写进 <code>last_train_params.json</code>）。<br>
· <b>hw/pw 是引擎口径</b>：不同 hw/pw 的强度读数不能互相比较（改它 = 换引擎，见 <code>NOTES.md</code> §12）。
</div></div>

"@
    # ---- 各轮训练量（总局数 / 总时长）----
    $runRows = ''
    $tG = 0; $tS = 0; $tN = 0; $tA = 0.0; $tW = 0.0; $tGens = 0
    foreach ($x in $s.runs) {
        if ($x.dup) {
            $runRows += ("<tr style='color:var(--mut)'><td>{0}</td><td>gen {1}–{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td style='color:var(--mut)'>{9}</td></tr>" -f `
                    $x.name, $x.g0, $x.g1, ("{0:N0}" -f $x.games), ("{0:N0}" -f $x.steps), ("{0:N0}" -f $x.samples), (Format-Span $x.actualMin), (Format-Span $x.wallMin), '—', $x.note)
            continue
        }
        $tG += $x.games; $tS += $x.steps; $tN += $x.samples; $tA += $x.actualMin; $tW += $x.wallMin; $tGens += $x.gens
        $note = $x.note
        if ($x.pauseMin -gt 5) { $note += ("（其中停顿 " + (Format-Span $x.pauseMin) + "）") }
        $pg = if ($null -ne $x.perGen) { [string]$x.perGen } else { '—' }
        $runRows += ("<tr><td><b>{0}</b></td><td>gen {1}–{2}</td><td>{3:N0}</td><td>{4:N0}</td><td>{5:N0}</td><td><b>{6}</b></td><td>{7}</td><td>{8}</td><td style='color:var(--mut)'>{9}</td></tr>" -f `
                $x.name, $x.g0, $x.g1, $x.games, $x.steps, $x.samples, (Format-Span $x.actualMin), (Format-Span $x.wallMin), $pg, $note)
    }
    if ($runRows) {
        $runRows += ("<tr style='border-top:1px solid var(--line2)'><td><b>合计</b></td><td>{0} 代</td><td><b>{1:N0}</b></td><td>{2:N0}</td><td>{3:N0}</td><td><b>{4}</b></td><td>{5}</td><td>—</td><td></td></tr>" -f `
                $tGens, $tG, $tS, $tN, (Format-Span $tA), (Format-Span $tW))
    } else {
        $runRows = "<tr><td colspan=9 style='color:var(--mut)'>暂无</td></tr>"
    }
    # ---- 结果文件兜底表 ----
    $artRows = ''
    foreach ($a in $s.artifacts) {
        $artRows += ("<tr><td style='color:var(--mut)'>{0}</td><td>{1}</td><td>{2}</td><td style='color:var(--fg2)'>{3}</td></tr>" -f $a.at, $a.name, $a.kb, $a.sum)
    }
    if (-not $artRows) { $artRows = "<tr><td colspan=4 style='color:var(--mut)'>暂无</td></tr>" }
    # ---- 进程卡 ----
    $rows = ''
    foreach ($p in $s.procs) { $rows += "<tr><td>$($p.kind)</td><td>$($p.pid)</td><td>$($p.mins) 分钟</td></tr>" }
    if (-not $rows) { $rows = "<tr><td colspan=3 style='color:var(--bad)'>⚠ 没有任何训练/测量进程在跑</td></tr>" }
    # ---- 测量中卡 ----
    $measHtml = "<p style='color:var(--mut)'>当前没有在写的测量日志</p>"
    if ($s.meas -and $s.meas.plain) {
        $measHtml = "<h3>正在写：$($s.meas.file)</h3><p style='color:var(--fg2);font-size:13px'>$($s.meas.tail -replace '<','&lt;')</p><p style='color:var(--mut);font-size:12px'>最近写入 $($s.meas.age) 秒前（这一阶段没有逐局进度行，只能报输出）</p>"
    } elseif ($s.meas) {
        $pct = if ($s.meas.total) { [int]([math]::Round($s.meas.done / $s.meas.total * 100)) } else { 0 }
        $eta = if ($s.meas.eta) { $s.meas.eta.ToString('HH:mm:ss') } else { '计算中…' }
        $rate = if ($s.meas.rate) { "{0:N1} 局/分钟" -f $s.meas.rate } else { '计算中…' }
        $measHtml = @"
<h3>测量中：$($s.meas.file)</h3>
<div class="bar"><div class="fill" style="width:$pct%">$pct%</div></div>
<p><b>$($s.meas.done) / $($s.meas.total)</b> 局　速率 <b>$rate</b>　预计完成 <b>$eta</b></p>
"@
    }
    # ---- 结果表 ----
    $resRows = ''
    foreach ($r in $s.results) {
        $elo = if ($null -ne $r.elo) { "<b>{0:N1}</b> <span class='muted'>[{1:N1}, {2:N1}] {3}</span>" -f [double]$r.elo, [double]$r.lo, [double]$r.hi, $r.metric } else { '<span class="muted">—</span>' }
        $vc = switch -Regex ($r.verdict) { 'stronger' { 'ok' } 'weaker' { 'bad' } 'indistinguishable' { 'warn' } default { 'mut' } }
        $sc = if ($r.src -like '账本*') { 'acc' } else { 'mut' }
        $resRows += ("<tr><td class='muted'>{0}</td><td>{1}</td><td><span class='pill {2}'>{3}</span></td><td>{4}</td><td>{5}</td><td>{6}</td><td><span class='pill {7}'>{8}</span></td></tr>" -f `
                $r.at, $r.label, $sc, $r.src, $r.games, $r.score, $elo, $vc, $r.verdict)
    }
    if (-not $resRows) { $resRows = "<tr><td colspan=7 style='color:var(--mut)'>暂无</td></tr>" }
    # ---- champion ----
    $champHtml = "<p style='color:var(--mut)'>没有 champions/current.json</p>"
    if ($s.champ) {
        $champHtml = ("<div class='row' style='gap:12px'>" +
                "<span class='pill acc' style='font-size:13px;padding:5px 13px'>{0}</span>" +
                "<span class='small muted'>sha1_8 <code>{1}</code></span>" +
                "<span class='mono'>相对 root <b>{2}</b> <span class='muted'>[{3}, {4}]</span></span>" +
                "<span class='pill mut'>hops={5}</span>" +
                "<span class='small muted'>更新于 {6}</span></div>") -f `
                $s.champ.name, $s.champ.sha1_8, $s.champ.elo_vs_root, $s.champ.ci_lo, $s.champ.ci_hi, $s.champ.hops, $s.champ.updated_at
    }
    $gpuHtml = if ($s.gpu) { "$($s.gpu.pwr)W　利用率 $($s.gpu.util)%　显存 $($s.gpu.mem)MiB　温度 $($s.gpu.temp)℃" } else { '—' }
    # ↑ 保留：GPU 卡已并入「训练概览」（见 $gpuSub），这里只留一个完整的单行文本备用（暂无模板引用）
    $ev = ($s.events | Select-Object -Last 8 | ForEach-Object { "<div class='ev'>" + (($_ -replace '<', '&lt;') -replace '[\x00-\x08\x0b\x0c\x0e-\x1f]', '') + "</div>" }) -join ''
    $genEpoch = [long](([datetime]::UtcNow - [datetime]'1970-01-01').TotalSeconds)
    $jsLimit = [math]::Max(8, $Interval * 3)
    $jsInterval = [math]::Max(1000, $Interval * 1000)

    @"
<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<!-- 【2026-10-26 删掉 meta refresh】
     以前这里有一条 content="$Interval" 的 meta，页面靠"整页重载"拿新数据：
     重载期间整块面板消失再出现，用户形容"画面会消失一段时间再回来"。
     现在改成两条真正的原地同步路径（见文件底部）：
       · SSE /api/stream：看板文件一被重写，服务端立刻推一条 ⇒ 页面马上 patch
       · setInterval(update, IVL)：后台兜底轮询（SSE 断开时仍能更新）
     两者都只改**变了的那几个节点**，不重建卡片、不重载页面。 -->
<script>
/* 老版本为兼容 file:// 留下的"掐 meta"逻辑，保留但已无事可做（模板里不再有 meta）。 */
(function(){try{var m=document.querySelector('meta[http-equiv="refresh"]');
if(m&&m.parentNode){m.parentNode.removeChild(m);}window.__metaStripped=true;}catch(e){}})();
</script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>训练/测量 实时看板</title>
<style>
:root{
  /* 极简·白纸 + 严格单栏（定稿）。
     没有玻璃/渐变/阴影/圆角；层次由**字号·字重·留白·规则线**承担。
     ⚠️ --mut 压到 #6f6f6f：白底上 11px 小字用 #8c8c8c 对比度只有 ~3.5:1，发灰看不清。 */
  --bg:#ffffff; --bg2:#ffffff; --panel:#ffffff; --panel2:#ffffff;
  --line:rgba(0,0,0,.11); --line2:rgba(0,0,0,.24);
  --hair:rgba(0,0,0,.07);
  --hover:rgba(0,0,0,.022);
  --bar:rgba(0,0,0,.10);
  --invert-fg:#ffffff;
  --fg:#101010; --fg2:#3d3d3d; --mut:#6f6f6f;
  --acc:#101010; --acc2:#3d3d3d;
  --ok:#2c7a4b; --warn:#85650f; --bad:#ad352b; --purple:#5f52a8;
  /* 【2026-10-31 加】三条曲线各自一个颜色（白纸配色，只用在 1.5px 细线和 10% 浅底上，
     不动任何版面元素）：c1=策略 CE（蓝）c2=价值 MSE（紫）c3=Elo（绿）。
     三条同色会分不清谁是谁；全黑则整页发闷、看不出主次。饱和度都压低，跟 --ok/--bad 一个调。 */
  --c1:#2f6fb5; --c1s:rgba(47,111,181,.10);
  --c2:#7b52ab; --c2s:rgba(123,82,171,.10);
  --c3:#1f7a5a; --c3s:rgba(31,122,90,.10);
  --r:0px; --r2:0px;
  --blur:none; --glass:none; --sh:none; --hi:none;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Display","Segoe UI","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;padding:46px 28px 92px;background:var(--bg);color:var(--fg);
     font-family:var(--sans);font-size:13.5px;line-height:1.6;
     -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
     min-height:100vh;overflow-x:hidden}
.wrap{max-width:1240px;margin:0 auto}
a{color:var(--fg);text-decoration:underline;text-underline-offset:3px;text-decoration-color:var(--line2)}
a:hover{text-decoration-color:var(--fg)}
code{font-family:var(--mono);font-size:12.5px;color:var(--fg2)}
/* 报头：标题下压一条实黑线（刊头规则），把整页钉住 */
h1{font-size:22px;font-weight:600;margin:0;letter-spacing:-.015em;display:flex;align-items:baseline;gap:14px}
.sub{color:var(--mut);font-size:12px;margin:10px 0 0;padding-bottom:22px;
     border-bottom:1px solid var(--fg);letter-spacing:.01em}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

/* ══════════ 严格单栏（取代碎格）══════════
   【为什么撤掉碎格】"非常规布局"和"不杂乱"是冲突的：12 列里跨度 3~12 乱跳，
   再叠上 grid-auto-flow:dense（它会为了填空洞**重排**），眼睛找不到固定的左边缘，
   13 个区块又全是同一档发丝线 ⇒ 没有主次，看起来就是一锅。
   现在：**所有区块同宽、严格按 DOM 顺序、一条左边缘**；主次改由"章规则"承担 ——
   每个新章的第一张卡用一条**更重的实线 + 更大留白**开场（见下面的 #c-hist/#c-results/#c-champ）。 */
/* 严格单栏 —— 只开一个例外：**配方（左）与曲线（右）并成一行**（用户要求）。
   其余所有区块仍然整行，保证"一条左边缘"不乱。
   为什么要把 $evalHtml 挪到曲线之后：网格按 DOM 顺序从左往右填，
   配方与曲线**必须在 DOM 里相邻**才能落到同一行的左右两格
   （中间夹一张整行的卡，稀疏自动放置不会再回头填右格）。 */
#cards{display:grid;grid-template-columns:1fr 1fr;column-gap:46px;row-gap:0;align-items:start}
#cards>.card{grid-column:1/-1}
/* ⚠️ 选择器必须写 #cards>#recipe / #cards>#c-curve（2,0,0）：
   重置用的 #cards>.card 是 (1,1,0)，单个 #recipe (1,0,0) 会被它压成整行 ——
   和当初碎格那轮的 margin 失效是**同一个坑**，这次又踩了一次。 */
#cards>#recipe{grid-column:1/2}
#cards>#c-curve{grid-column:2/3}
/* 任一张缺席时，剩下一张吃满整行（:has() 现代 Chrome/Edge 都支持） */
#cards:not(:has(#recipe))>#c-curve{grid-column:1/-1}
#cards:not(:has(#c-curve))>#recipe{grid-column:1/-1}
/* 窄屏下并排会挤 ⇒ 退回整行 */
@media(max-width:1000px){
  #cards{display:block}
  #cards>#recipe,#cards>#c-curve{grid-column:auto}
}
.card{position:relative;padding:30px 0 8px;margin:0;border:0;border-top:1px solid var(--line);
      border-radius:0;background:none;box-shadow:none}
#cards>.card:last-child{border-bottom:1px solid var(--line);padding-bottom:30px}
/* 章首：更重的线 + 更大留白 */
#c-hist,#c-results,#c-champ{border-top:1px solid var(--fg2);margin-top:34px}
.card>h3{font-size:11px;font-weight:600;margin:0 0 18px;color:var(--mut);
         letter-spacing:.20em;display:flex;align-items:center;gap:8px}
.muted{color:var(--mut)} .small{font-size:12px}
/* ---- KPI：竖排"标签/大号等宽数字"，靠上方发丝线分栏 ---- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:22px 30px;margin-bottom:26px}
.kpi{padding-top:12px;border-top:1px solid var(--line);background:none}
.kpi .k{color:var(--mut);font-size:11px;letter-spacing:.12em}
.kpi .v{font-family:var(--mono);font-size:24px;font-weight:500;font-variant-numeric:tabular-nums;
        margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.02em}
.kpi .u{font-size:12px;color:var(--mut);font-weight:400;margin-left:4px}
.kpi.ok .v{color:var(--ok)} .kpi.acc .v{color:var(--fg)} .kpi.warn .v{color:var(--warn)}
.kpi.bad .v{color:var(--bad)} .kpi.pur .v{color:var(--purple)}
/* ---- 状态：不铺色块，一个小圆点 + 文字 ---- */
.pill{display:inline-block;font-size:11.5px;font-weight:500;border:0;background:none;
      padding:0;color:var(--fg2);white-space:nowrap}
.pill.ok::before,.pill.bad::before,.pill.warn::before{content:"";display:inline-block;
      width:5px;height:5px;border-radius:50%;background:currentColor;
      margin-right:7px;vertical-align:middle;position:relative;top:-1px}
.pill.ok{color:var(--ok)} .pill.bad{color:var(--bad)} .pill.warn{color:var(--warn)}
.pill.mut{color:var(--mut)} .pill.acc{color:var(--fg)}
/* ---- 进度条：2px 细线 ---- */
.bar{background:var(--bar);border:0;border-radius:0;overflow:hidden;height:2px;margin:14px 0}
.fill{background:var(--fg);height:100%;border-radius:0;transition:width .4s ease;min-width:2px}
.fill.alt{background:var(--mut)}
.plabel{display:flex;justify-content:space-between;gap:12px;font-size:11.5px;color:var(--mut);
        font-family:var(--mono);font-variant-numeric:tabular-nums}
/* ---- 表格：表头下一条线 + 极淡行线；数字等宽右对齐 ---- */
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:10.5px;letter-spacing:.12em;
   padding:0 12px 10px 0;border-bottom:1px solid var(--line2);white-space:nowrap;
   position:sticky;top:0;z-index:1;background:var(--bg)}
td{padding:7px 12px 7px 0;border-bottom:1px solid var(--hair);white-space:nowrap}
tbody tr:hover td,table tr:hover td{background:var(--hover)}
table .num,table td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
/* 【2026-10-27 修 · 用户报"C++ 训练历史的数据对不齐"】
   原来只有 td.num 右对齐，th 一律左对齐（上面那条 th{text-align:left}）⇒ 每一列的数字
   贴着右边、表头却贴着左边，列越宽错位越明显（C++ 每代一行有 14 列，"自对弈 322.9 s"
   这种宽列一撑开，表头就跑到数字左边去了）。表头必须和它下面那些格子同侧。 */
th.num,table th.num{text-align:right}
table.hist th:nth-child(2),table.hist th:nth-child(3),table.hist th:nth-child(4),
table.hist th:nth-child(5),table.hist th:nth-child(6),table.hist th:nth-child(9),
table.res th:nth-child(4),table.res th:nth-child(5),table.res th:nth-child(6),
table.runs th:nth-child(3),table.runs th:nth-child(4),table.runs th:nth-child(5),
table.runs th:nth-child(6),table.runs th:nth-child(7),table.runs th:nth-child(8),
table.art th:nth-child(3),
#cppHist th,#cppHist td{text-align:right}
/* 14 列在窄屏/小卡片里会被挤到重叠，给个横向滚动兜底，别让右边几列跑出卡片外 */
#cppHist{overflow-x:auto}
#cppHist table{min-width:860px}
table.hist td:nth-child(2),table.hist td:nth-child(3),table.hist td:nth-child(4),
table.hist td:nth-child(5),table.hist td:nth-child(6),table.hist td:nth-child(9),
table.res td:nth-child(4),table.res td:nth-child(5),table.res td:nth-child(6),
table.runs td:nth-child(3),table.runs td:nth-child(4),table.runs td:nth-child(5),
table.runs td:nth-child(6),table.runs td:nth-child(7),table.runs td:nth-child(8),
table.art td:nth-child(3){font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.card.flush{padding:30px 0 8px}
/* ---- 表单：下划线式输入 ---- */
#ctlGrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:24px 28px;font-size:12.5px}
/* 【2026-10-26 加】C++ 引擎的专属参数区：默认不显示，引擎切到 C++ 才出现；
   反过来，Python 专属字段（轮次名/模型目录/标尺/自动测量…）在 C++ 模式下隐藏，
   否则面板里一半参数对 C++ 无意义，看起来就像"切了引擎但参数还是 Python 的"。 */
#cppGrid{display:none;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:24px 28px;font-size:12.5px;margin-top:24px}
body[data-engine="cpp"] #cppGrid{display:grid}
body[data-engine="cpp"] #ctlGrid label[data-eng="python"]{display:none}
/* Python 专属的配方按钮（V3 稳妥版 / V4 配方 / 一键开 V4 / 沿用上轮参数）在 C++ 模式下必须藏起来：
   它们填的是 train.py 的数值，在 C++ 面板里点一下就会把 C++ 参数覆盖成 Python 的。 */
body[data-engine="cpp"] #btnV3Stable,body[data-engine="cpp"] #btnV4Fill,
body[data-engine="cpp"] #btnV4Go,body[data-engine="cpp"] #btnLast{display:none}
#cppGrid label{display:flex;flex-direction:column;gap:6px}
#ctlGrid label,#cppGrid label{display:flex;flex-direction:column;gap:6px}
#ctlGrid .fn,#cppGrid .fn{color:var(--fg2);font-weight:600}
#ctlGrid .fn em,#cppGrid .fn em{font-style:normal;color:var(--mut);font-weight:400;font-size:11px}
#ctlGrid .tip,#cppGrid .tip{color:var(--mut);font-size:11px;line-height:1.5}
#ctlGrid input[type=text],#ctlGrid input[type=number],
#cppGrid input[type=text],#cppGrid input[type=number]{
  width:100%;background:none;color:var(--fg);border:0;border-bottom:1px solid var(--line2);
  border-radius:0;padding:5px 0;font-family:var(--mono);font-size:13px;transition:border-color .15s}
#ctlGrid input[type=text]:hover,#ctlGrid input[type=number]:hover,
#cppGrid input[type=text]:hover,#cppGrid input[type=number]:hover{border-bottom-color:var(--fg2)}
#ctlGrid input[type=text]:focus,#ctlGrid input[type=number]:focus,
#cppGrid input[type=text]:focus,#cppGrid input[type=number]:focus{outline:none;border-bottom-color:var(--fg)}
#ctlGrid input[type=checkbox],#cppGrid input[type=checkbox]{width:15px;height:15px;accent-color:var(--fg);cursor:pointer}
/* ---- 按钮：方角描边；主按钮＝反白 ---- */
.btn{appearance:none;border:1px solid var(--line2);background:none;color:var(--fg);
     border-radius:0;padding:8px 20px;font-size:12.5px;font-weight:600;cursor:pointer;
     font-family:inherit;letter-spacing:.02em;transition:background .15s,border-color .15s,color .15s}
.btn:hover{background:var(--hover);border-color:var(--fg2)}
.btn.primary{background:var(--fg);border-color:var(--fg);color:var(--invert-fg)}
.btn.primary:hover{background:#000;border-color:#000}
.btn.danger{border-color:rgba(173,53,43,.5);color:var(--bad)}
.btn.danger:hover{background:rgba(173,53,43,.06);border-color:var(--bad)}
/* ---- 其它 ---- */
details.card>summary{cursor:pointer;color:var(--fg);font-weight:600;font-size:12.5px;
     list-style:none;letter-spacing:.10em}
details.card>summary::-webkit-details-marker{display:none}
details.card>summary::before{content:"+　";color:var(--mut)}
details.card[open]>summary::before{content:"−　"}
.ev{color:var(--mut);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    font-family:var(--mono);padding:4px 0;border-bottom:1px solid var(--hair)}
.ev:last-child{border-bottom:none}
/* 说明块：限制行宽（一行太长本身就是"杂乱感"的来源），行距收紧 */
.note{border:0;border-left:1px solid var(--line2);border-radius:0;background:none;
      padding:2px 0 2px 18px;margin-top:20px;font-size:11.5px;line-height:1.75;
      color:var(--mut);max-width:92ch}
.note b{color:var(--fg2)}
.note code{background:none;color:var(--fg2)}
.hb{font-size:11.5px;border:0;border-radius:0;padding:0;margin-left:auto;font-weight:500;
    background:none;color:var(--mut);letter-spacing:.04em}
.ok{color:var(--ok)} .bad{color:var(--bad)}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
/* 曲线图：inline 的固定 620px 是"不闪"的前提。但单栏后卡片有 1240px，
   620px 的图缩在中间只剩一大片空 —— 所以在**保持固定尺寸盒**的前提下放大：
   width 用 min(100%,840px)、height 改成 auto + aspect-ratio（盒子尺寸仍与图片加载无关
   ⇒ 依旧不会闪、不会跳），背景是 center/contain 所以不会变形。 */
#c-curve>div:last-child{margin-left:auto;margin-right:auto;
  width:min(100%,840px)!important;height:auto!important;aspect-ratio:1080/960}
/* 配方表是"标签/数值"两列，拉满 1240px 会让两者隔 800px 远、看着像联系不起来。
   限制在 660px：一列读下来是一组紧密的键值对。 */
#recipe table{max-width:660px}
/* 宽表在窄屏下内部滚动，别把数据切掉 */
#c-overview,#recipe,#c-hist,#c-results,#c-runs,#c-art{overflow-x:auto}
@media(max-width:820px){
  body{padding:26px 16px 60px} h1{font-size:19px}
  .card{padding:24px 0 6px}
  #c-hist,#c-results,#c-champ{margin-top:26px}
  .kpi .v{font-size:21px}
  .kpis{grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:18px 20px}
  .note{max-width:none}
}
/* ══════════ 版本切换：Python 版 / C++ 版 ══════════
   两块面板互斥显示，共用同一套 .card 与列宽，切版本时只有数据源换掉。
   开关状态存 localStorage，刷新后还停在原来那版。 */
#verBar{display:flex;gap:12px;align-items:center;margin:0 0 14px;flex-wrap:wrap}
#verBar .lbl{font-size:11px;letter-spacing:.20em;color:var(--mut)}
#verBar .seg{display:inline-flex;border:1px solid var(--line2)}
#verBar .seg .btn{border:0;border-radius:0;padding:6px 16px;font-size:12px}
#verBar .seg .btn+.btn{border-left:1px solid var(--line2)}
#verBar .seg .btn.on{background:var(--fg);color:var(--invert-fg)}
#verBar .seg .btn.on:hover{background:#000;border-color:#000}
#cppCards{display:none}
#cppCards.on{display:block}
#cppCards>.card:last-child{border-bottom:1px solid var(--line);padding-bottom:30px}
#cppCards .card{overflow-x:auto}
#cppSel{appearance:none;border:1px solid var(--line2);background:none;color:var(--fg);
  font-family:var(--mono);font-size:12px;padding:3px 8px}
/* 版本=C++ 时那张「C++ 引擎训练进度」小卡是重复信息（整个面板已经是 C++ 了），
   用 !important 压过它自己轮询时写的 inline display。 */
#c-cpp[data-ver-hidden]{display:none !important}
#cppTail{white-space:pre-wrap;line-height:1.75;color:var(--mut)}
</style></head><body><div class="wrap">
<h1>五子棋 · 训练/测量看板<span id="hb" class="hb ok">自检中…</span></h1>
<div class="sub">数据每 $Interval 秒自动更新　·　曲线图在轮边界重画　·　<a href="$curveUrl" target="_blank">打开曲线原图</a>$curveWhich</div>

<div class="row" style="margin-top:10px">
  <button type="button" class="btn" id="btnPlay">♟ 对弈（引擎可选 C++ / Python）</button>
</div>

<div id="playWrap" style="display:none;position:fixed;inset:0;z-index:9999;background:var(--bg);padding:10px;box-sizing:border-box;flex-direction:column">
  <div class="row" style="margin-bottom:8px">
    <button type="button" class="btn" id="btnPlayClose">← 返回看板</button>
    <span id="playMsg" class="small muted"></span>
  </div>
  <iframe id="playFrame" src="about:blank" style="flex:1;width:100%;border:1px solid var(--line);border-radius:10px;background:var(--bg)"></iframe>
</div>
<script>
// 对弈页是一个独立进程（web_play.py，默认 8770）：这里只负责拉起 + 用 iframe 铺满整屏，
// 看板自身的布局/配色一行不动。切回看板只是 display:none，不销毁 iframe（棋局状态保留）。
(function () {
  var wrap = document.getElementById('playWrap');
  var frame = document.getElementById('playFrame');
  var msg = document.getElementById('playMsg');
  function say(t) { msg.textContent = t || ''; }
  function openPlay() {
    wrap.style.display = 'flex';
    if (frame.src && frame.src.indexOf('about:blank') < 0) { return; }   // 已经开过：直接切回来
    say('检查对弈页…');
    fetch('/api/play/ping').then(function (r) { return r.json(); }).then(function (st) {
      if (st && st.running) { frame.src = st.url; say(''); return null; }
      say('正在启动对弈页（首次要加载模型，约十几秒）…');
      return fetch('/api/play/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
      }).then(function (r) { return r.json(); }).then(function (r2) {
        if (r2 && r2.ok) { frame.src = r2.url; say(''); }
        else { say('启动失败：' + ((r2 && r2.errors) || ['未知原因']).join(' ')); }
      });
    }).catch(function (e) { say('启动失败：' + e); });
  }
  document.getElementById('btnPlay').addEventListener('click', openPlay);
  document.getElementById('btnPlayClose').addEventListener('click', function () {
    wrap.style.display = 'none';
  });
})();
</script>

<details class="card" id="trainctl">
<summary>训练控制（展开：改参数 / 开始 / 停止）</summary>
<div class="row" style="margin-top:14px">
  <label style="display:flex;gap:8px;align-items:center;min-width:210px">
    <span class="fn">训练引擎</span>
    <select id="p_Engine" style="background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">
      <option value="python">Python · train.py</option>
      <option value="cpp">C++ · gk.exe loop</option>
    </select>
  </label>
  <label id="cppDirBox" style="display:none;min-width:300px">
    <span class="fn">C++ 运行目录 <em>(--dir)</em></span>
    <!-- 下拉里是「目录 · 角色 · 全局第几代」，角色与全局代数来自各目录的 run.json。
         选「其他」才显示输入框；平时不给人手打路径的机会（手打最容易打到一个空目录，
         那样 gk.exe 会用随机权重从零开训，等于把练好的东西扔掉）。 -->
    <select id="p_CppDirSel" data-rec="$cppRunDefault" style="width:100%;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">$cppRunOptions</select>
    <input type="text" id="p_CppDir" value="$cppRunDefault" style="width:100%;display:none;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">
    <span class="small muted" id="cppDirHint" style="display:block;margin-top:3px"></span>
  </label>
  <span class="small muted" id="engineHint"></span>
</div>
<div id="ctlGrid" style="margin-top:14px">$ctlHtml</div>
<!-- 【2026-10-26 加】C++ 引擎专属参数（默认值来自 gk.exe defaults ⇒ config.h）。
     只在引擎=C++ 时显示；同一时刻 Python 专属字段会被 CSS 隐藏，面板里不会两套参数混着。 -->
<div id="cppGrid">$cppHtml</div>
<!-- 共享字段（局数/步数…）在 Python 与 C++ 两套配方里数值不同：草稿或上次 Python 参数
     会把它们留在 Python 的数值上，一代就只产约 1 万样本而不是 2 万。这里显式提示并给一键恢复。 -->
<div id="cppDefBar" style="margin-top:10px;display:none">
  <span class="small" id="cppDefMsg"></span>
  <a href="#" class="small" id="btnCppDefaults" style="margin-left:10px;display:none">用 C++ 默认配方</a>
</div>
<div class="row" style="margin-top:14px">
  <!-- 【2026-10-28 加】配方档案：C++\peifang\配方NN-*.json（参数 + 动机 + 结果）。
       选一份点「填入配方」= 把整套参数回填到上面的表单（会自动切到 C++ 引擎），
       之后仍走「开始训练」那条干跑确认路径，不会绕过任何校验。 -->
  <span class="small" id="recipeBox" style="white-space:nowrap;margin-right:10px">
    <b>配方档案</b>
    <select id="p_RecipeSel" style="max-width:330px;margin-left:4px;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:3px 0;font-family:inherit"></select>
    <button type="button" class="btn" id="btnRecipeFill" style="margin-left:6px" title="把这份配方档案里的参数整套填进表单（不动权重/不动目录）">填入配方</button>
    <a href="#" class="small" id="recipeInfo" style="margin-left:8px">说明</a>
  </span>
  <button type="button" class="btn" id="btnLast">沿用上轮参数</button>
  <button type="button" class="btn" id="btnV3Stable">填入 V3 稳妥版</button>
  <button type="button" class="btn" id="btnV4Fill">填入 V4 配方</button>
  <button type="button" class="btn primary" id="btnV4Go">⚡ 一键开 V4</button>
  <button type="button" class="btn primary" id="btnStart">▶ 开始训练</button>
  <button type="button" class="btn danger" id="btnStop">■ 停止训练</button>
  <span id="ctlMsg" class="small muted"></span>
  <span id="ctlDraftTag" class="small muted" style="margin-left:auto"></span>
</div>
<div id="ctlHint" class="small muted" style="margin-top:10px"></div>
</details>

<div class="card" id="c-cpp" style="display:none">
  <h3>C++ 引擎训练进度<span class="small muted" id="cppHead"></span></h3>
  <div id="cppBody" class="small"></div>
</div>
<script>
// C++ 引擎的进度来源和 Python 完全不同（C++\model\<run>\train_log.jsonl），
// 所以单独一张卡、由这里轮询 /api/cpp-state 渲染；没有 C++ 训练/模型时整卡隐藏，
// 看板原有布局一个像素都不动。
(function () {
  var card = document.getElementById('c-cpp');
  var head = document.getElementById('cppHead');
  var body = document.getElementById('cppBody');
  if (!card) { return; }
  function n(v, d) { return (v === undefined || v === null) ? '—' : Number(v).toFixed(d === undefined ? 0 : d); }
  function row(label, val) {
    return '<tr><td class="muted">' + label + '</td><td>' + val + '</td></tr>';
  }
  function render(st) {
    if (!st || (!st.dir && !st.running)) { card.style.display = 'none'; return; }
    card.style.display = '';
    head.innerHTML = '　·　' + (st.dir || '?') + '　·　' +
      (st.running ? '<b style="color:var(--ok)">训练中</b>' : '<span class="muted">空闲</span>');
    var l = st.last;
    if (!l) {
      body.innerHTML = '<div class="muted">这个目录还没有 train_log.jsonl（等第 1 代跑完就会出现）</div>';
      return;
    }
    var html = '<table class="data" style="margin-top:8px">';
    html += row('最近一代', 'gen ' + n(l.gen) + '　·　已落盘 ' + (st.gens.length ? (st.gens[0] + '…' + st.gens[st.gens.length - 1]) : '—') +
                '　·　下一代 ' + n(st.next_gen));
    html += row('样本 / 手数', n(l.samples) + ' 条　·　' + n(l.games) + ' 局　·　' +
                (l.games ? (l.samples / l.games).toFixed(1) : '—') + ' 手/局');
    html += row('自对弈 / 训练', n(l.selfplay_s, 1) + ' s　·　' + n(l.train_s, 1) + ' s（' +
                n(l.steps) + ' 步 ⇒ ' + (l.steps ? (1000 * l.train_s / l.steps).toFixed(1) : '—') + ' ms/步）');
    html += row('loss / ce / mse', n(l.loss, 4) + '　·　' + n(l.ce, 4) + '　·　' + n(l.mse, 4));
    html += row('lr / 回放池', n(l.lr, 6) + '　·　' + n(l.buffer_total) + ' 条');
    if (l.gated) {
      html += row('门控', n(l.gate_wins) + '/' + n(l.gate_games) + ' = ' + n(l.gate_wr, 3) + '　·　' +
                  (l.gate_accepted ? '<b style="color:var(--ok)">采纳</b>' : '<span style="color:var(--bad)">拒绝</span>'));
    } else {
      html += row('门控', '<span class="muted">本代没跑</span>');
    }
    html += '</table>';
    if (st.recent && st.recent.length > 1) {
      html += '<div class="muted" style="margin-top:10px">最近 ' + st.recent.length + ' 代：' +
        st.recent.map(function (e) {
          return 'gen' + n(e.gen) + '（loss ' + n(e.loss, 3) + '）';
        }).join('　') + '</div>';
    }
    body.innerHTML = html;
  }
  function poll() {
    fetch('/api/cpp-state', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function () { card.style.display = 'none'; });
  }
  poll();
  setInterval(poll, 4000);
})();
</script>
<script>
// 【必须放在字段之后、且同步执行】一次性恢复"展开状态"与"未提交的草稿"。
// 字段已经由服务端渲染好（含基线值），这里只是改 value/open —— **不改变布局**，
// 所以在首帧之前就完成，用户看不到"消失再出现"或"下面内容跳动"。
(function () {
  var d = document.getElementById('trainctl');
  if (d) {
    try { if (localStorage.getItem('flashui_ctl_open') === '1') { d.open = true; } } catch (e) {}
    d.addEventListener('toggle', function () {
      try { localStorage.setItem('flashui_ctl_open', d.open ? '1' : '0'); } catch (e) {}
    });
  }
  var draft = null;
  try { var s = localStorage.getItem('flashui_ctl_draft'); draft = s ? JSON.parse(s) : null; } catch (e) {}
  if (draft) {
    for (var k in draft) {
      if (!Object.prototype.hasOwnProperty.call(draft, k)) { continue; }
      var el = document.getElementById('p_' + k);
      if (!el) { continue; }
      if (typeof draft[k] === 'boolean') { el.checked = draft[k]; } else { el.value = draft[k]; }
    }
  }
})();
</script>

<div id="verBar">
  <span class="lbl">版本</span>
  <div class="seg">
    <button class="btn small on" id="verPy" data-ver="python">Python</button>
    <button class="btn small" id="verCpp" data-ver="cpp">C++</button>
  </div>
  <span id="verHint" class="small muted"></span>
</div>

<div id="cards">
<div class="card" id="c-overview"><h3>训练概览</h3>$trainHtml$barHtml$roundLine</div>

$recipeHtml

$curveHtml

$evalHtml

<div class="card" id="c-running"><h3>正在跑</h3><table class="data">$rows</table></div>

<div class="card" id="c-meas">$measHtml</div>

<div class="card" id="c-hist"><h3>训练历史（各轮次混排，最近 10 代）</h3>
<table class="hist"><tr><th>轮次</th><th>代数</th><th>数据条数</th><th>局数</th><th>步数</th>
<th title="策略交叉熵：网络走子概率 vs MCTS 走子分布。越小越好；完全没学=ln(361)≈5.89">loss_p ⓘ</th>
<th title="价值均方误差：预测胜负 vs 实际胜负。越小越好；一律猜平局≈0.8~1.0">loss_v ⓘ</th>
<th>门控胜率</th><th>采纳</th><th>耗时(分)</th></tr>
$histRows</table>$histNote$lossNote</div>

<div class="card" id="c-results"><h3>测量结果（最近 $($s.results.Count) 条 · 全部测量都在这）</h3>
<table class="res"><tr><th>时间</th><th>测量</th><th>来源</th><th>局数</th><th>得分率</th><th>ΔElo [95%]</th><th>判定</th></tr>
$resRows</table></div>

<div class="card" id="c-runs"><h3>各轮训练量（总局数与总时长）</h3>
<table class="runs"><tr><th>轮次</th><th>代数范围</th><th>自对弈局数</th><th>训练步</th><th>样本数</th><th>实际训练时长</th><th>墙钟跨度</th><th>平均每代(分)</th><th>说明</th></tr>
$runRows</table>
<div class='small muted' style='line-height:1.8'>
· <b>实际训练时长</b> = 墙钟 − 停顿。停顿判据：<b>单代间隔 &gt; 20 分钟</b>（轮边界那代含硬门槛+200 局评测、本来就 10~12 分钟，所以不能按倍数筛，否则会少算真实工作量）<br>
· <b>墙钟跨度</b> = 首尾时间戳之差，含所有停顿与评测时间<br>
· 局数取自 <code>train_log.jsonl</code> 的 <code>n_games</code> 求和（不是"代数×每代局数"推算）<br>
· 这里只是<b>自对弈</b>局数；评测/门控对局另算（曲线账本 + 一次性测量）</div></div>

<div class="card" id="c-art"><h3>最近产出的结果文件（兜底：探针等自定义格式也在这）</h3>
<table class="art"><tr><th>时间</th><th>文件</th><th>KB</th><th>摘要</th></tr>
$artRows</table></div>

<div class="card" id="c-champ"><h3>当前 champion</h3>$champHtml</div>

<div class="card" id="c-events"><h3>最近事件</h3>$ev</div>
</div>

<!-- ══════════ C++ 版看板（版本=C++ 时显示；数据来自 /api/cpp-state） ══════════ -->
<div id="cppCards">
  <div class="card" id="c-cpp-overview">
    <h3>C++ 训练概览</h3>
    <div id="cppKpis" class="kpis"></div>
    <div id="cppBar"></div>
    <div id="cppRunLine" class="small muted" style="margin-top:10px;line-height:1.8"></div>
  </div>
  <div class="card" id="c-cpp-curve">
    <h3>C++ 损失曲线（每代 · CE = 策略交叉熵 / MSE = 价值均方误差）</h3>
    <div id="cppCurve" class="small muted">还没有数据</div>
  </div>
  <!-- 【2026-10-28 加】用户要求「加上 elo 曲线，起点由你定，可以参考 katago 的 elo 曲线设定」。
       参考 KataGo 的读法：① 必须有一个显式锚点（gen N = 0 Elo）并写在标题里，否则纵轴是无意义的绝对数；
       ② 每点带 95% 置信区间并画成竖须，区间重叠的两个版本就该被看成"没测出差别"；
       ③ 官方按训练代推进，所以 x 轴是代而不是时间。数据来自 C++\<run>\elo.json（gk.exe elo 生成）。
       【2026-10-29 加】用户要求「每次用新配方训练，这张图上要标出哪里用了新配方」：
       竖虚线 = 从这一代起换了配方（局数/模拟数/门控局数一起换），隔段底色标出同一配方的连续区间。
       跨过虚线的那一步里包含"换配方"本身，不能算训练变强；虚线两侧的点也不该直接互比。
       数据来自 train_log.jsonl 的 recipe_id（老日志没有该字段时退回"每代样本数跨 15000"推断）。 -->
  <div class="card" id="c-cpp-elo">
    <h3>C++ 棋力曲线（Elo · 每点 95% 置信区间 · 区间重叠 = 没测出差别）</h3>
    <div id="cppElo" class="small muted">还没有 Elo 数据（跑一次 <code>gk.exe elo --dir model/run_v2 --gens ...</code> 生成 elo.json）</div>

    <!-- 【2026-10-30 加】用户要求「在实时看板上增加 elo 对战功能，方便我看 elo 提升没有」。
         这一段就是那个按钮：选最近 N 代 + 锚点 → 后台跑 gk.exe elo → 进度条 → 跑完自动重画上面那张图，
         并给出「比上次 +X」。它不动任何训练状态，只读权重、写 elo.json。
         控件放进这张卡片而不是顶部控制区：它的产物就是上面这张图，放一起才看得出因果。 -->
    <div id="eloPanel" style="margin-top:16px;border-top:1px solid var(--hair);padding-top:14px">
      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end">
        <label class="small" style="display:flex;flex-direction:column;gap:4px">
          <span class="muted">最近几代</span>
          <input id="eloRecent" type="number" min="1" max="40" value="5" style="width:88px">
        </label>
        <label class="small" style="display:flex;flex-direction:column;gap:4px">
          <span class="muted">锚点（=0 Elo）</span>
          <input id="eloAnchor" type="number" min="0" style="width:104px">
        </label>
        <label class="small" style="display:flex;flex-direction:column;gap:4px">
          <span class="muted">每对局数</span>
          <input id="eloGames" type="number" min="2" max="200" step="2" value="30" style="width:88px">
        </label>
        <button id="eloBtn" class="btn primary">开始 Elo 对战</button>
        <button id="eloStopBtn" class="btn">停止</button>
      </div>
      <div id="eloHint" class="small muted" style="margin-top:10px;line-height:1.75"></div>
      <div id="eloBarWrap" style="display:none;margin-top:10px">
        <div style="height:6px;background:var(--hair);border-radius:3px;overflow:hidden">
          <div id="eloBar" style="height:100%;width:0;background:var(--ok);transition:width .3s"></div>
        </div>
        <div id="eloProg" class="small muted" style="margin-top:6px"></div>
      </div>
      <div id="eloResult" class="small" style="margin-top:10px;line-height:1.8"></div>
      <details style="margin-top:10px">
        <summary class="small muted" style="cursor:pointer">对战日志（最后几行）</summary>
        <pre id="eloTail" class="small" style="margin-top:8px;max-height:220px;overflow:auto;white-space:pre-wrap;color:var(--fg2)"></pre>
      </details>
    </div>
  </div>
  <div class="card" id="c-cpp-hist">
    <!-- 【2026-10-31 加】这张表 40 行 × 14 列，展开占满一屏还多，但只在"想查某一代"时才需要看。
         所以整块可折叠：折叠后标题行保留一行速览（最新代 / loss / 门控），不丢信息；
         状态记在 localStorage（gkCppHistOpen），刷新后保持上次的选择。标题可点、可键盘回车/空格。 -->
    <h3 id="cppHistTitle" tabindex="0" style="cursor:pointer;user-select:none;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap">
      <span id="cppHistArrow" class="small muted" style="display:inline-block;width:10px">▾</span>
      <span>C++ 训练历史（每代一行 · 来自 C++\model\&lt;run&gt;\train_log.jsonl）</span>
      <span id="cppHistPeek" class="small muted"></span>
      <span id="cppHistHint" class="small muted" style="opacity:.7"></span>
    </h3>
    <div id="cppHistBody" style="display:none">
    <div id="cppHist" class="small muted">还没有数据</div>
    <!-- 【2026-10-27 加】用户问「loss/ce/mse/lr/回放池这些参数有什么用」——
         列名本身不带含义，悬停依赖鼠标，所以常驻一张小抄。这里必须写清"不能拿它判断强弱"，
         否则每次 loss 降一点就会被当成进步（README §6 那条：mse 0.57 的新配方比 0.33 的旧配方更强）。 -->
    <div class="small muted" style="line-height:1.95;margin-top:14px;border-top:1px solid var(--hair);padding-top:12px">
      <b>这五列怎么读</b><br>
      · <code>loss</code>：这一代的训练目标值 = <code>ce + mse</code>（trainer.cpp:109）。它只说明"这批样本拟合得怎么样"，
         <b>不说明棋力</b>；模拟数 / 温度 / fresh 比例一换，跨代就不具可比性<br>
      · <code>ce</code>：策略头交叉熵 —— 网络预测分布与这一代 MCTS 访问分布（软目标）的差距。
         目标本身在动（搜索越深越尖），所以它主要反映"这批目标有多难"，只能同配方看趋势<br>
      · <code>mse</code>：价值头平方误差 —— <code>(value − 终局结果 z)²</code> 的均值。样本难度随策略漂移，
         它会自己上下晃；<b>本项目里 mse 更高的一代反而更强</b>（0.57 的新配方 vs 0.33 的旧配方，README §6）<br>
      · <code>lr</code>：这一代<b>实际</b>用的学习率。和面板上的配方对不上 ⇒ 被 lr 棘轮 / 退火 / 参数覆盖改过<br>
      · <code>回放池</code>：缓冲区总条数 <code>buffer_total</code> = 最近 <code>--buffer-gens</code>（现在 60）代的样本累加。
         先涨满、之后随旧代退出而平台化；变大只表示"池子装满了"，不等于学到东西<br>
      <span style="color:var(--fg2)">判断有没有变强只看 <b>门控</b>（<code>门控</code> 列的胜局/总局 + <code>采纳</code>）和 Elo；
      loss / mse 只用来判断"训练有没有崩"（突然飙高、NaN、或长期不动）。</span>
    </div>
    </div>
  </div>
  <div class="card" id="c-cpp-runs">
    <h3>C++ 各运行目录的训练量（总局数与总时长）</h3>
    <div id="cppRuns" class="small muted">还没有数据</div>
    <div class="small muted" style="line-height:1.8;margin-top:10px">
      · 局数取自 <code>train_log.jsonl</code> 的 <code>games</code> 求和（不是"代数×每代局数"推算）<br>
      · 「自对弈 / 训练」是累计墙钟时长，按量级自动换单位（≥1 小时显示 h、≥1 分钟显示 min），悬停可看精确秒数；训练步是 <code>steps</code> 求和<br>
      · 名称里带 <code>broken</code> 的配方产物已从列表里隐藏
    </div>
  </div>
  <div class="card" id="c-cpp-champ">
    <h3>C++ 当前 champion</h3>
    <div id="cppChamp" class="small muted">这个 run 还没有 best.bin</div>
  </div>
  <div class="card" id="c-cpp-log">
    <h3>C++ 运行日志（gk.exe loop 的 stdout 尾巴）</h3>
    <div id="cppTail">没有日志</div>
  </div>
</div>
<script>
// ── 更新方式：**原地替换变化了的那张卡**，而不是整页重载 ──
// 为什么：meta refresh 每 3 秒整页重载会把所有内容（包括 80 KB 的曲线图）重画一遍
// ⇒ 肉眼可见的闪，而且每次刷新都跳回页首。改成只替换内容真的变了的 .card，
// 没变的卡（比如曲线图）**完全不碰** ⇒ 不闪、不跳、不重复下载图片。
//
// 渐进增强：fetch 成功后才移除 meta refresh。
//   · http:// 打开 → fetch 可用 → 移除 meta，走原地更新（顺滑）
//   · file:// 打开 → fetch 被浏览器禁止 → 保留 meta，退回整页刷新（老行为，能用）
var GEN = $genEpoch;              // 服务端生成时刻（Unix 秒），每次抓取都会更新
var LIM = $jsLimit;               // 超过这么多秒没更新 = 数据过期
var IVL = $jsInterval;            // 更新间隔（毫秒）
var T0 = Date.now();
var okCount = 0, failCount = 0, lastOk = Date.now(), stripped = false;
var ctlWarnedEdit = false;          // 只提示一次"已暂停自动重载"

// 【2026-10-12 修 · 用户报"每隔几秒就弹重新加载"】
// 上一版留了两个问题：
//   ① stripMeta 只在第一次 fetch **成功之后**才移除 <meta refresh> ⇒ 那之前页面
//      照样每 $Interval 秒整页重载；
//   ② 我还加了 beforeunload 提示 ⇒ **每次 meta 重载都触发 Chrome 的"重新加载此网站？"确认框**。
// 根治法：**加载时立刻掐掉 meta refresh**（下面这段必须最早执行），不再依赖 fetch 成功；
// 并删掉 beforeunload —— 它保护不了数据（草稿另有内存/localStorage 双份），只制造弹窗。
(function () {
  try {
    var m = document.querySelector('meta[http-equiv="refresh"]');
    if (m && m.parentNode) { m.parentNode.removeChild(m); }
    window.__metaStripped = true;
  } catch (e) {}
})();
function stripMeta(){
  // 兜底：老版逻辑（fetch 成功后）保留，万一上面那段没跑到
  if (stripped) { return; }
  var m = document.querySelector('meta[http-equiv="refresh"]');
  if (m) { m.parentNode.removeChild(m); stripped = true; }
}

// ───────── 训练控制面板 ─────────
// 面板放在 #cards **外面**（而且默认折叠），所以自动刷新永远不会重建它
// ⇒ 用户正在输入 / 展开的状态不会被刷掉。按钮用 document 级委托绑定，
// 这样即使将来 DOM 被整体替换也不会失效。
// 字段表由 PowerShell 注入（同一份 spec，服务端渲染与这里共用，避免两处漂移）
var CTL_FIELDS = [
$ctlJsSpec
];
// C++ 引擎专属字段（同一份 spec，服务端渲染 #cppGrid 与这里共用）
var CTL_CPP_FIELDS = [
$cppJsSpec
];
// 两个引擎共用的字段（Workers/Games/…）：切到 C++ 时用 C++ 的默认值覆盖，
// 否则切过去还是 Python 的 160 局/320 步 —— 用户看到的就是「参数还是 python 的」。
var CPP_SHARED = { $cppSharedJs };
// ---- 【2026-10-28 加】配方档案（C++\peifang\配方NN-*.json，服务端 15 秒 TTL 重读）----
// 每份档案的 args 用 gk.exe 的开关名（snake_case）做键；这里只负责「键 → 表单 id」的翻译，
// 所以以后往 peifang 里丢一份新 json 就能直接在页面上选到，不用改这份 JS。
var CPP_RECIPES = $cppRecipesJs;
var CPP_ACTIVE_RECIPE = null;   // 当前填入的配方（决定「与默认不同」的比对基准）
var RECIPE_TO_FIELD = {
  games: 'p_Games', min_steps: 'p_MinSteps', eval_games: 'p_EvalGames', eval_every: 'p_EvalEvery',
  workers: 'p_Workers', buffer_gens: 'p_BufferGens', seed: 'p_Seed',
  hw: 'p_HeuristicWeight', pw: 'p_PolicyWeight',
  sims: 'q_Sims', batch_mcts: 'q_BatchMcts', batch: 'q_Batch', lr: 'q_LR', fresh_ratio: 'q_Fresh',
  eval_batch: 'q_EvalBatch', eval_threads: 'q_EvalThreads', half_life: 'q_HalfLife', procs: 'q_Procs',
  keep_gens: 'q_KeepGens', keep_buffer_gens: 'q_KeepBufferGens',
  recipe: 'q_Recipe', recipe_id: 'q_Recipe', lr_gen_decay: 'q_LrGenDecay', lr_floor: 'q_LrFloor',
  opening_plies: 'q_OpeningPlies', gate_threshold: 'q_GateThreshold'
};
function recipeByIdx(i){
  if (!CPP_RECIPES || i < 0 || i >= CPP_RECIPES.length) { return null; }
  return CPP_RECIPES[i];
}
function recipeLabel(r){
  var id = (r.id === null || r.id === undefined) ? '?' : r.id;
  var st = String(r.status || '').replace(/[\r\n]+/g, ' ').trim();
  if (st.length > 26) { st = st.slice(0, 26) + '…'; }
  return '配方' + id + ' · ' + (r.name || r.file) + (st ? ' 〔' + st + '〕' : '');
}
function recipeInit(){
  var sel = document.getElementById('p_RecipeSel');
  if (!sel) { return; }
  if (!CPP_RECIPES || !CPP_RECIPES.length) {
    sel.innerHTML = '<option value="-1">（没读到 C++\\peifang\\*.json）</option>';
    return;
  }
  var cur = sel.value;
  var h = '';
  for (var i = 0; i < CPP_RECIPES.length; i++) { h += '<option value="' + i + '">' + recipeLabel(CPP_RECIPES[i]) + '</option>'; }
  sel.innerHTML = h;
  // 默认选中 status 以「现行」开头的那份（配方 3），否则选最后一份
  var pick = CPP_RECIPES.length - 1;
  for (var j = 0; j < CPP_RECIPES.length; j++) {
    var st = String(CPP_RECIPES[j].status || '');
    if (st.indexOf('现行') === 0 || /^current/i.test(st)) { pick = j; break; }
  }
  if (cur !== '' && cur !== '-1' && parseInt(cur, 10) < CPP_RECIPES.length) { pick = parseInt(cur, 10); }
  sel.value = String(pick);
}
function recipeFill(){
  var sel = document.getElementById('p_RecipeSel');
  var r = recipeByIdx(sel ? parseInt(sel.value, 10) : -1);
  if (!r) { return; }
  var eng = document.getElementById('p_Engine');
  if (eng && eng.value !== 'cpp') { eng.value = 'cpp'; engineChanged(); }
  var args = r.args || {}, setN = 0, miss = [], setTxt = [];
  for (var k in args) {
    if (!Object.prototype.hasOwnProperty.call(args, k)) { continue; }
    var fid = RECIPE_TO_FIELD[k];
    var el = fid ? document.getElementById(fid) : null;
    if (!el) { miss.push(k); continue; }
    el.value = args[k];
    setN++; setTxt.push(k + '=' + args[k]);
  }
  CPP_ACTIVE_RECIPE = r;
  ctlHadEdit = true;
  ctlSaveDraft();
  cppDefCheck();
  var hint = document.getElementById('engineHint');
  if (hint && setTxt.length) {
    hint.innerHTML = '已填入 <b>' + recipeLabel(r) + '</b>：<code>' + setTxt.join(' ') + '</code>' +
      (miss.length ? '<br><b class="bad">表单里没有这些字段，未填入：</b><code>' + miss.join(', ') + '</code>' : '') +
      '<br>点「开始训练」仍会先干跑一次、把服务端拼好的 <code>gk.exe loop …</code> 命令行拿来确认。';
  }
}
function recipeShowInfo(){
  var sel = document.getElementById('p_RecipeSel');
  var r = recipeByIdx(sel ? parseInt(sel.value, 10) : -1);
  if (!r) { return; }
  var lines = [];
  lines.push(recipeLabel(r));
  var ch = r.changed || {};
  for (var k in ch) { if (Object.prototype.hasOwnProperty.call(ch, k)) { lines.push('· ' + k + '：' + ch[k]); } }
  if (r.lesson) { lines.push('教训：' + r.lesson); }
  if (r.command) { lines.push('命令行：' + r.command); }
  lines.push('文件：C++\\peifang\\' + r.file);
  window.alert(lines.join('\n'));
}
var ctlReady = false;
var ctlBound = false;   // #ctlGrid 的 input/change 监听器只挂一次
var CTL_DRAFT_KEY = 'flashui_ctl_draft';
// 【2026-10-12 修 · 用户报"网页改参数会被自动刷新回原来的值"】
// 原因：控制面板的草稿**只存在 localStorage**。整页刷新（meta refresh 兜底、
// "数据过期"自愈重载、手动 F5、浏览器重启）只能靠它恢复；localStorage 一旦失效
// 或草稿为空，服务器值就会把输入覆盖回去。
// 对策：① 编辑过就**禁止自动重载**（下面 hbTick 用这个标志），② 草稿额外存一份
// **内存副本**（不依赖 localStorage 是否可用），③ 有未提交修改时离开页面前给提示。
var ctlHadEdit = false;
var ctlDraftMem = null;
function ctlDraftPeek(){
  if (ctlDraftMem) { return ctlDraftMem; }
  try { var s = localStorage.getItem(CTL_DRAFT_KEY); return s ? JSON.parse(s) : null; } catch (e) { return null; }
}

function ctlRender(){
  // 【兜底】正常情况下字段已由服务端渲染好，这里什么都不做。
  // 只有 #ctlGrid 为空（例如手改了 HTML）才用 JS 追补一份，保证面板仍可用。
  var g = document.getElementById('ctlGrid');
  if (!g || ctlReady || g.children.length > 0) { ctlReady = true; return; }
  var html = '';
  for (var i = 0; i < CTL_FIELDS.length; i++) {
    var name = CTL_FIELDS[i][0], kind = CTL_FIELDS[i][1], cn = CTL_FIELDS[i][2], tip = CTL_FIELDS[i][3];
    html += '<label title="' + name + '：' + tip + '" style="display:flex;flex-direction:column;gap:3px">';
    html += '<span style="color:var(--acc2)">' + cn +
            ' <span style="color:var(--ok);font-size:11px">(' + name + ')</span></span>';
    if (kind === 'bool') {
      html += '<input type="checkbox" id="p_' + name + '" style="width:auto">';
    } else if (kind === 'int') {
      html += '<input type="number" id="p_' + name + '" style="width:100%;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">';
    } else if (kind === 'num') {
      html += '<input type="number" step="any" id="p_' + name + '" style="width:100%;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">';
    } else {
      html += '<input type="text" id="p_' + name + '" style="width:100%;background:none;color:inherit;border:0;border-bottom:1px solid rgba(128,128,128,.55);border-radius:0;padding:5px 0;font-family:inherit">';
    }
    html += '<span style="color:var(--mut);font-size:11px">' + tip + '</span></label>';
  }
  g.innerHTML = html;
  ctlBind();
  ctlReady = true;
}
function ctlBind(){
  // 【2026-10-13 修 · 这就是"改了参数又自己变回去"的根因】
  // 监听器原来挂在 ctlRender() 的**尾部**，而 ctlRender() 在"服务端已经把字段渲染好"
  // 时会**提前 return**（见上面那行 g.children.length > 0）⇒ 监听器**从来没挂上过**。
  // 后果：你在参数框里打字时 ctlSaveDraft() 根本不被调用 —— 草稿不存、ctlHadEdit 不置位，
  // 于是任何一次整页重载都会用服务器值把输入覆盖掉（"填入 V3 稳妥版后又变回去"同理）。
  // 现在独立成函数、**无条件调用**；ctlBound 保证只挂一次（避免重复绑定）。
  var g = document.getElementById('ctlGrid');
  if (!g || ctlBound) { return; }
  g.addEventListener('input', ctlSaveDraft);
  g.addEventListener('change', ctlSaveDraft);
  // 勾选框在某些浏览器下只触发 change；键盘输入触发 input —— 两个都挂才不漏
  ctlBound = true;
}
function ctlSaveDraft(){
  ctlHadEdit = true;                 // 用户动过表单 ⇒ 从此禁止自动整页重载
  try {
    ctlDraftMem = ctlCollect();      // 内存副本（不依赖 localStorage）
    localStorage.setItem(CTL_DRAFT_KEY, JSON.stringify(ctlDraftMem));
  } catch (e) {
    ctlDraftMem = ctlCollect();      // localStorage 不可用也要留住输入
  }
  // 【2026-10-13 加】**让"保护已生效"可见**。用户反复报"改了参数又被刷回原值"，
  // 而这个过程以前是完全静默的 —— 分不清"没保存"还是"保存了但被覆盖"。
  // 现在每存一次就在面板右侧打一个时间戳：看得见它，就能一眼判断保护有没有在工作。
  var tag = document.getElementById('ctlDraftTag');
  if (tag) {
    var d = new Date();
    var p2 = function(n){ return (n < 10 ? '0' : '') + n; };
    tag.style.color = 'var(--ok)';
    tag.textContent = '已自动保存你的修改 ' + p2(d.getHours()) + ':' + p2(d.getMinutes()) + ':' + p2(d.getSeconds()) + '（整页重载不会丢）';
  }
}
function ctlLoadDraft(){
  try {
    var s = localStorage.getItem(CTL_DRAFT_KEY);
    return s ? JSON.parse(s) : null;
  } catch (e) { return null; }
}
function ctlClearDraft(){
  ctlDraftMem = null;
  try { localStorage.removeItem(CTL_DRAFT_KEY); } catch (e) {}
}
// 有未提交修改时离开页面给个提示（避免误刷新丢掉刚填的参数）
// 【2026-10-12 删】原来这里挂了 beforeunload。它**保护不了数据**（草稿有内存+localStorage 两份），
// 却会在 <meta refresh> 每次触发整页重载时弹出 Chrome 的"重新加载此网站？"确认框
// —— 用户实测"每隔几秒就弹这个按钮"。移除；防丢数据靠草稿双份 + 加载时掐掉 meta refresh。
function ctlFill(p){
  for (var i = 0; i < CTL_FIELDS.length; i++) {
    var name = CTL_FIELDS[i][0], el = document.getElementById('p_' + name);
    if (!el) { continue; }
    if (CTL_FIELDS[i][1] === 'bool') { el.checked = !!p[name]; }
    else { el.value = (p[name] === undefined || p[name] === null) ? '' : p[name]; }
  }
  var eng = document.getElementById('p_Engine');
  if (eng && p.Engine) { eng.value = p.Engine; }
  var cdir = document.getElementById('p_CppDir');
  if (cdir && p.CppDir) { cdir.value = p.CppDir; }
  // C++ 专属字段（q_*）：草稿或旧参数里带了就恢复，否则保留服务端渲染的默认值
  for (var j = 0; j < CTL_CPP_FIELDS.length; j++) {
    var qn = CTL_CPP_FIELDS[j][0], qel = document.getElementById('q_' + qn);
    if (!qel || p[qn] === undefined || p[qn] === null || p[qn] === '') { continue; }
    qel.value = p[qn];
  }
  engineChanged(true);
  // 【2026-10-12 修】「本轮目标代数」永远显示一个**能用**的数：
  // 服务器说 p.StartTarget 是过期值（已退化成 0=自动）时，填 p.start_target_effective
  // （= 当前最新 gen + 每轮代数）。旧行为是把过期值原样填进表单，用户一点
  // 「开始训练」就被拒，网页上看起来就是"按钮没用"。
  // ⚠️ 这一步必须放在"草稿优先"之后也生效：否则用户浏览器里存着的旧草稿
  //（localStorage 里的 StartTarget）会把过期值又填回来 —— 那是同一个坑的第二个入口。
  var st = document.getElementById('p_StartTarget');
  var staleDraft = false;                    // 草稿里的目标代数是否已过期（< 当前最新代数）
  if (st && p) {
    var raw = String(st.value || '').trim();
    var n = raw === '' ? 0 : parseInt(raw, 10);
    staleDraft = (!isNaN(n) && n > 0 && n <= (p.current_gen || 0));
  }
  if (st && p && (p.start_target_stale || staleDraft)) {
    var eff = p.start_target_effective || p.suggested_start_target || '';
    st.value = eff;
    st.dataset.autoFilled = '1';
  } else if (st && p && p.start_target_effective && p.source === '运行中的训练') {
    st.value = p.start_target_effective;      // 训练在跑：显示它这一轮的真实目标
    delete st.dataset.autoFilled;
  } else if (st && !(p && p.start_target_stale)) {
    delete st.dataset.autoFilled;
  }
}
// 目标代数的提示文案（表单下方那行小字，每次刷新/启动后都更新）
function ctlTargetHint(p){
  if (!p) { return ''; }
  var st = document.getElementById('p_StartTarget');
  var v = st ? String(st.value || '').trim() : '';
  var n = v === '' ? 0 : parseInt(v, 10);
  var cur = p.current_gen || 0, gpr = p.GensPerRound || 15;
  if (!n || n <= cur) {
    return '「本轮目标代数」留空/0 = <b>自动</b>：按**提交那一刻**的最新代数算，' +
           '当前最新 gen = <b>' + cur + '</b> ⇒ 现在提交会训到 <b>' + (cur + gpr) + '</b>。';
  }
  return '当前最新 gen = <b>' + cur + '</b>　·　本轮目标 gen = <b>' + n + '</b>（' +
         (n - cur) + ' 代）' + (st && st.dataset.autoFilled ? '　·　<i>已自动改成能用的值</i>' : '');
}
function ctlCollect(){
  var o = {};
  for (var i = 0; i < CTL_FIELDS.length; i++) {    var name = CTL_FIELDS[i][0], el = document.getElementById('p_' + name);
    if (!el) { continue; }
    if (CTL_FIELDS[i][1] === 'bool') { o[name] = el.checked; }
    else if (CTL_FIELDS[i][1] === 'int') { o[name] = el.value === '' ? 0 : parseInt(el.value, 10); }
    // 【2026-10-13 修】浮点字段必须走 parseFloat：以前 hw/pw 声明成 int，
    // parseInt('0.5') = 0 ⇒ 草稿存 0、发给服务端的也是 0（静默改成"纯网络先验"）。
    else if (CTL_FIELDS[i][1] === 'num') { o[name] = el.value === '' ? 0 : parseFloat(el.value); }
    else { o[name] = el.value; }
  }
  // 【2026-10-25 加】引擎选择与 C++ 目录不在 CTL_FIELDS 里（那是 train.py 的参数表），
  // 但它们必须跟着草稿一起存/取，否则刷新后引擎会被打回 Python。
  var eng = document.getElementById('p_Engine');
  if (eng) { o.Engine = eng.value; }
  var cdir = document.getElementById('p_CppDir');
  if (cdir) { o.CppDir = cdir.value; }
  // 【2026-10-26 加】C++ 专属字段也要进草稿/请求体：否则切到 C++ 提交的还是这批默认值，
  // 用户在 C++ 面板里改的模拟数/学习率会被静默丢掉。
  for (var j = 0; j < CTL_CPP_FIELDS.length; j++) {
    var qn = CTL_CPP_FIELDS[j][0], qk = CTL_CPP_FIELDS[j][1];
    var qel = document.getElementById('q_' + qn);
    if (!qel) { continue; }
    if (qk === 'bool') { o[qn] = qel.checked; }
    else if (qk === 'int') { o[qn] = qel.value === '' ? 0 : parseInt(qel.value, 10); }
    else if (qk === 'num') { o[qn] = qel.value === '' ? 0 : parseFloat(qel.value); }
    else { o[qn] = qel.value; }
  }
  return o;
}
// 【2026-10-26 重写】引擎切换：显示和**数值**都要切。
// 为什么不能只切显示：C++ 与 Python 的默认配方完全不同（C++: 320 局/600 模拟/160 步/lr 3e-4；
// Python: 160 局/400 模拟/320 步/lr 1e-3）。只切显示的话，用户切到 C++ 看到的还是 Python 的数，
// 就会以为「切换没生效」（真实反馈就是这样）。
// 策略：每个引擎各存一份快照；来回切时先存下当前这份，再把目标那份放回去。
// 第一次切到 C++ 用 CPP_SHARED（= gk.exe defaults，即 config.h 的默认配方）。
var engSnap = { python: null, cpp: null, pyServer: null };
function engCollectAll(){
  var o = {}, i, el;
  for (i = 0; i < CTL_FIELDS.length; i++) {
    el = document.getElementById('p_' + CTL_FIELDS[i][0]);
    if (!el) { continue; }
    o[CTL_FIELDS[i][0]] = (CTL_FIELDS[i][1] === 'bool') ? el.checked : el.value;
  }
  for (i = 0; i < CTL_CPP_FIELDS.length; i++) {
    el = document.getElementById('q_' + CTL_CPP_FIELDS[i][0]);
    if (!el) { continue; }
    o[CTL_CPP_FIELDS[i][0]] = (CTL_CPP_FIELDS[i][1] === 'bool') ? el.checked : el.value;
  }
  return o;
}
function engApplyAll(o){
  if (!o) { return; }
  var i, el;
  for (i = 0; i < CTL_FIELDS.length; i++) {
    el = document.getElementById('p_' + CTL_FIELDS[i][0]);
    if (!el || !(CTL_FIELDS[i][0] in o)) { continue; }
    if (CTL_FIELDS[i][1] === 'bool') { el.checked = !!o[CTL_FIELDS[i][0]]; }
    else { el.value = (o[CTL_FIELDS[i][0]] === undefined || o[CTL_FIELDS[i][0]] === null) ? '' : o[CTL_FIELDS[i][0]]; }
  }
  for (i = 0; i < CTL_CPP_FIELDS.length; i++) {
    el = document.getElementById('q_' + CTL_CPP_FIELDS[i][0]);
    if (!el || !(CTL_CPP_FIELDS[i][0] in o)) { continue; }
    el.value = o[CTL_CPP_FIELDS[i][0]];
  }
}
// C++ 运行目录：下拉 ↔ 输入框双向同步，并提示这个目录是「续训」还是「从零开训」。
// 空目录最危险：gk.exe 找不到 gen_*.bin/best.bin 就用随机权重起步，
// 所以这里必须让「这是主线」和「这是空目录」一眼可辨。
function cppDirSync(fromSel){
  var sel = document.getElementById('p_CppDirSel');
  var inp = document.getElementById('p_CppDir');
  var hint = document.getElementById('cppDirHint');
  if (!sel || !inp) { return; }
  if (fromSel) {
    if (sel.value === '__custom__') {
      inp.style.display = '';
      if (inp.value === sel.getAttribute('data-rec')) { inp.value = ''; }
    } else {
      inp.style.display = 'none';
      inp.value = sel.value;
    }
  } else {
    var found = false, i;
    for (i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === inp.value) { sel.selectedIndex = i; found = true; break; }
    }
    if (!found) { sel.value = '__custom__'; inp.style.display = ''; }
  }
  if (hint) {
    var rec = sel.getAttribute('data-rec') || '';
    var cur = String(inp.value || '').replace(/^\s+|\s+$/g, '');
    if (cur && cur === rec) {
      hint.innerHTML = '<b>推荐</b>：主线目录，续训就选它';
    } else if (sel.value === '__custom__' || !cur) {
      hint.innerHTML = '自定义路径：目录不存在或为空时，gk.exe 会用<b>随机权重从零开训</b>（不是接着主线）';
    } else {
      hint.innerHTML = '';
    }
  }
}
// C++「共享字段」自检：局数/步数这类字段在 Python 与 C++ 两套配方里含义相同、数值不同
//（Python 160 局/320 步，C++ 320 局/160 步）。草稿或上次 Python 参数会把它们留在 Python 的
// 数值上，于是 C++ 一代只产 160 局×62 手≈1 万样本，而不是 320 局≈2 万。
// 这里把它显式说出来，并给一个一键恢复按钮，避免"以为换了引擎参数就全换了"。
function cppDefCheck(){
  var bar = document.getElementById('cppDefBar');
  var msg = document.getElementById('cppDefMsg');
  var eng = document.getElementById('p_Engine');
  var useCpp = eng && eng.value === 'cpp';
  if (bar) { bar.style.display = useCpp ? '' : 'none'; }
  if (!useCpp || !msg) { return; }
  var CN = { Games: '局数', MinSteps: '步数', Workers: '并行线程', EvalGames: '门控局数',
             EvalEvery: '门控间隔', GensPerRound: '本轮代数', Seed: '随机种子',
             BufferGens: '回放池代数', HeuristicWeight: '启发式权重', PolicyWeight: '策略权重',
             sims: '每手模拟', batch_mcts: '搜索批', batch: '训练批', lr: '学习率', fresh_ratio: '新数据比例',
             eval_batch: '评估合并上限', eval_threads: '评估线程', half_life: '半衰期', procs: '多进程',
             keep_gens: '保留权重代数', keep_buffer_gens: '回放池留代', recipe: '配方编号',
             lr_gen_decay: '学习率退火', lr_floor: '学习率地板', opening_plies: '配对开局' };
  var has = function (o, k) { return Object.prototype.hasOwnProperty.call(o, k); };
  // 比对基准：优先「刚填入的那份配方档案」，否则 C++ 默认配方（config.h）
  var pairs = [], baseName = 'C++ 默认配方';
  if (CPP_ACTIVE_RECIPE && CPP_ACTIVE_RECIPE.args) {
    var ra = CPP_ACTIVE_RECIPE.args;
    for (var rk in RECIPE_TO_FIELD) {
      if (!has(RECIPE_TO_FIELD, rk) || !has(ra, rk)) { continue; }
      pairs.push([RECIPE_TO_FIELD[rk], rk, ra[rk]]);
    }
    baseName = '配方' + (CPP_ACTIVE_RECIPE.id === null || CPP_ACTIVE_RECIPE.id === undefined ? '?' : CPP_ACTIVE_RECIPE.id);
  } else {
    for (var k2 in CPP_SHARED) { if (has(CPP_SHARED, k2)) { pairs.push(['p_' + k2, k2, CPP_SHARED[k2]]); } }
  }
  var bad = [];
  for (var i = 0; i < pairs.length; i++) {
    var el = document.getElementById(pairs[i][0]);
    if (!el) { continue; }
    var exp = String(pairs[i][2]), got = String(el.value).trim();
    var same = (exp === got);
    if (!same) {
      // 3e-4 / 0.0003 这类写法差异不该报「不同」
      var e1 = parseFloat(exp), g1 = parseFloat(got);
      if (!isNaN(e1) && !isNaN(g1) && Math.abs(e1 - g1) <= Math.abs(e1) * 1e-9 + 1e-12) { same = true; }
    }
    if (!same) {
      // 字段名从 id 反推（q_Games → Games），这样提示里是中文而不是 gk 的开关名
      var fk = pairs[i][0].replace(/^[pq]_/, '');
      bad.push((CN[fk] || CN[pairs[i][1]] || pairs[i][1]) + ' <b>' + got + '</b>（' + baseName + ' 为 ' + exp + '）');
    }
  }
  var btn = document.getElementById('btnCppDefaults');
  if (!bad.length) {
    var gEl = document.getElementById('p_Games'), sEl = document.getElementById('p_MinSteps');
    msg.innerHTML = '参数与 <b>' + baseName + '</b> 一致' +
      (gEl && sEl ? '（' + gEl.value + ' 局 / ' + sEl.value + ' 步）' : '');
    if (btn) { btn.style.display = 'none'; }
  } else {
    msg.innerHTML = '⚠ 与 <b>' + baseName + '</b> 不同：' + bad.join('、');
    if (btn) { btn.style.display = ''; }
  }
}
// init=true：只同步显示（ctlFill 里调，避免和「填值」互相触发）
function engineChanged(init){
  var eng = document.getElementById('p_Engine');
  var useCpp = eng && eng.value === 'cpp';
  var box = document.getElementById('cppDirBox');
  if (box) { box.style.display = useCpp ? '' : 'none'; }
  if (document.body) { document.body.setAttribute('data-engine', useCpp ? 'cpp' : 'python'); }
  if (init !== true) {
    if (useCpp) {
      engSnap.python = engCollectAll();          // 记下 Python 那一套
      if (engSnap.cpp) { engApplyAll(engSnap.cpp); }
      else { engApplyAll(CPP_SHARED); }           // 第一次切过来：用 config.h 的默认配方
      engSnap.cpp = engCollectAll();
    } else {
      engSnap.cpp = engCollectAll();             // 记下 C++ 那一套
      if (engSnap.python) { engApplyAll(engSnap.python); }
      else if (engSnap.pyServer) { engApplyAll(engSnap.pyServer); }  // 没有快照 ⇒ 用服务器的 Python 配方
    }
  }
  var hint = document.getElementById('engineHint');
  if (hint) {
    var hasDef = false;
    for (var k in CPP_SHARED) { if (Object.prototype.hasOwnProperty.call(CPP_SHARED, k)) { hasDef = true; break; } }
    hint.innerHTML = useCpp
      ? 'C++ 引擎：权重/回放池/日志都在 <b>C++\\model\\&lt;目录&gt;</b>；参数默认值来自 <b>config.h</b>' +
        (hasDef ? '（已填入 320 局 / 600 模拟 / 160 步 / lr 3e-4 这套保守配方）'
                : '（<b class="bad">没取到</b> gk.exe defaults：面板里是 Python 的值，请手动改）') +
        '；局数/步数/门控映射成 <code>gk.exe loop --games --min-steps --eval-games ...</code>'
      : 'Python 引擎：走 <b>train.py</b>（models/&lt;目录&gt;），参数含义与以前完全一致';
  }
  cppDirSync(false);
  cppDefCheck();
}
// 「沿用上轮参数」走这条：**先清草稿**再用服务器上的值填，
// 否则会被"草稿优先"逻辑挡住（用户点了按钮却发现没变，会以为按钮坏了）。
function ctlApplyFromServer(){
  ctlClearDraft();
  fetch('/api/last-params', { cache: 'no-store' })
    .then(function(r){ return r.json(); })
    .then(function(p){
      ctlFill(p);
      ctlSaveDraft();
      ctlHint('已填入服务器上的参数　·　来源：<b>' + (p.source || '?') + '</b>　·　' +
              (p.SaveDir || '?') + '　' + ctlTargetHint(p));
    })
    .catch(function(){ ctlHint('取参数失败：看板服务可能没在跑'); });
}
function ctlMsg(txt, cls){
  var m = document.getElementById('ctlMsg');
  if (m) { m.textContent = txt || ''; m.style.color = cls === 'bad' ? 'var(--bad)' : (cls === 'ok' ? 'var(--ok)' : 'var(--fg2)'); }
}
function ctlHint(txt){
  var h = document.getElementById('ctlHint');
  if (h) { h.innerHTML = txt || ''; }
}
function ctlLoad(){
  ctlRender();
  ctlBind();   // 【必须在这里也调一次】ctlRender() 在服务端已渲染时会提前 return，不会走到 ctlBind
  engineChanged(true);   // 草稿里可能存着 engine=cpp：先按它把 C++ 目录框显示出来（只切显示）
  recipeInit();          // 【2026-10-28 加】配方档案下拉（数据由服务端 15 秒 TTL 烘进页面）
  var draft = ctlLoadDraft();
  // 【2026-10-12 修】草稿可能来自**上一次编辑**（localStorage 长期留存）。
  // 只要存在草稿，就认为用户有未提交的修改 ⇒ 之后的自动重载要放行（否则整页刷新
  // 会把输入刷掉；而 localStorage 一旦失效/为空，参数就会被服务器值覆盖回去）。
  if (draft) { ctlHadEdit = true; }
  fetch('/api/last-params', { cache: 'no-store' })
    .then(function(r){ return r.json(); })
    .then(function(p){
      // 【草稿优先】用户改到一半刷新了 → 恢复他的输入，而不是用服务器上的值覆盖掉
      if (draft) {
        ctlFill(draft);
        // 【2026-10-13 用户要求删掉这里的长提示，别的不动】
        // 原来这个分支会写一句三行长的提示（含 ctlTargetHint 那一大段"当前最新 gen ⇒ 会训到 N"）。
        // 它比面板里任何一行都长、会折行 ⇒ 页面每次更新时整页高度抖动（用户："收了然后展开"）。
        // 草稿是否生效由按钮行右侧的「已自动保存你的修改 HH:MM:SS」显示，已够用。
      } else {
        ctlFill(p);
      }
      ctlLastParams = p;
      // 【2026-10-27 修】上面 ctlFill 灌的是 **Python 面板**那套配方 —— checkpoints/last_train_params.json
      // 里还是 2026-10-15 的 Python V4 记录（160 局 / 320 步）。引擎选 C++ 时继承它，一点"开始训练"
      // 就成了 160 局/320 步（2026-10-27 就这么白跑了 gen 11、gen 12 两代）。
      // C++ 的权威配方只有一个：config.h，也就是服务端刚问过 `gk.exe defaults` 的 CPP_SHARED。
      var engSel = document.getElementById('p_Engine');
      if (engSel && engSel.value === 'cpp') { engApplyAll(CPP_SHARED); }
      // 记下服务器的 Python 配方：切回 Python 且没有用户快照时用它复原
      engSnap.pyServer = p;
    })
    .catch(function(){
      if (draft) { ctlFill(draft); ctlHint('已恢复你上次未提交的修改（但取服务器参数失败：看板服务可能没在跑）'); }
      else { ctlHint('取参数失败：看板服务可能没在跑'); }
    });
}
var ctlLastParams = null;
function ctlStart(){
  var p = ctlCollect();
  if (p.Engine === 'cpp') { ctlStartCpp(p); return; }
  var cmd = 'auto_train.ps1 -RunName ' + p.RunName + ' -SaveDir ' + p.SaveDir +
            ' -Workers ' + p.Workers + ' -Games ' + p.Games + ' -MinSteps ' + p.MinSteps +
            ' -EvalGames ' + p.EvalGames + ' -EvalEvery ' + p.EvalEvery + ' -EvalWorkers ' + p.EvalWorkers +
            ' -GensPerRound ' + p.GensPerRound + ' -StartTarget ' + (p.StartTarget || '自动') +
            ' -RulerGames ' + p.RulerGames + ' -Seed ' + p.Seed + (p.AutoEval ? ' -AutoEval' : '') +
            // 【2026-10-13 加】额外参数也要出现在确认框里：它是 train.py 真正收到的开关，
            // 不显示的话用户没法在"按下确认"前核对配方（新开关全靠这一格）。
            (p.ExtraArgs ? ' -ExtraArgs ' + p.ExtraArgs : '');
  if (!confirm('确认开始训练？\n\n' + cmd + '\n\n（会从 ' + p.SaveDir + '/resume.pt 接着跑）')) { return; }
  ctlMsg('正在启动…');
  fetch('/api/start-train', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p)
  }).then(function(r){ return r.json(); }).then(function(j){
    if (j.ok) {
      ctlMsg('已启动（PID ' + j.pid + '，目标 gen ' + j.start_target + '，' +
             (j.detached ? '已脱离本机会话' : '子进程方式') + '）', 'ok');
      ctlSaveDraft();
      // 【2026-10-12 修】启动后立刻把参数刷成"运行中的训练"，
      // 否则表单里还留着启动前的旧代数（下次点就可能又被拒 → 用户会以为按钮坏了）
      setTimeout(function(){
        fetch('/api/last-params', { cache: 'no-store' }).then(function(r){ return r.json(); })
          .then(function(q){ ctlLastParams = q; ctlHint('训练已在跑。' + ctlTargetHint(q)); })
          .catch(function(){});
        update();
      }, 2500);
    }
    else {
      var errs = (j.errors || []).join('；');
      ctlMsg('启动被拒：' + errs, 'bad');
      // 【主动自救】目标代数过期 ⇒ 直接把表单改成"自动"，并把每轮代数/目录按服务器现值修正，
      // 用户再点一次就能起来（这正是 2026-10-12 用户报"点开始训练没用"的那个场景）
      if (j.error_kind === 'stale_start_target') {
        var st = document.getElementById('p_StartTarget');
        if (st) { st.value = j.auto_start_target || ''; st.dataset.autoFilled = '1'; }
        ctlHint('「本轮目标代数」原值已过期（当前最新 gen = <b>' + j.current_gen +
                '</b>），已自动改成 <b>' + j.auto_start_target + '</b>。请再点一次「开始训练」。');
        ctlSaveDraft();
      }
    }
  }).catch(function(e){ ctlMsg('请求失败：' + e, 'bad'); });
}
// C++ 引擎：先 dry_run 拿"服务端真正会执行的那条命令行"，确认后再真启动 ——
// 这样对话框里看到的就是真跑的那一条，不会两边各写一份命令而漂移。
function ctlStartCpp(p){
  p.dry_run = true;
  p.engine = 'cpp';   // 服务端按这个字段分流；只发 p.Engine 会被当成 Python 启动（静默跑错引擎）
  ctlMsg('正在生成 C++ 启动命令…');
  fetch('/api/start-train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p) })
    .then(function(r){ return r.json(); }).then(function(j){
      if (!j.ok) { ctlMsg('参数被拒：' + (j.errors || []).join('；'), 'bad'); return; }
      if (!confirm('确认用 C++ 引擎开始训练？\n\n' + j.cmd + '\n\n（在 ' + j.dir + ' 里续训，日志写该目录）')) {
        ctlMsg('已取消'); return;
      }
      var real = {};
      for (var k in p) { if (Object.prototype.hasOwnProperty.call(p, k)) { real[k] = p[k]; } }
      real.dry_run = false;
      real.engine = 'cpp';
      ctlMsg('正在启动 C++ 训练…');
      fetch('/api/start-train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(real) })
        .then(function(r2){ return r2.json(); }).then(function(j2){
          if (j2.ok) {
            ctlMsg('C++ 训练已启动（PID ' + j2.pid + '，目录 ' + j2.dir + '，目标 ' + j2.gens + ' 代）', 'ok');
            ctlSaveDraft();
            setTimeout(update, 2500);
          } else { ctlMsg('启动被拒：' + (j2.errors || []).join('；'), 'bad'); }
        }).catch(function(e){ ctlMsg('请求失败：' + e, 'bad'); });
    }).catch(function(e){ ctlMsg('请求失败：' + e, 'bad'); });
}
function ctlStop(){  if (!confirm('确认停止训练？\n\n会按「先杀监督器，再杀 train.py」的顺序停，已完成的代不会丢。')) { return; }
  ctlMsg('正在停止…');
  fetch('/api/stop-train', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
  }).then(function(r){ return r.json(); }).then(function(j){
    if (j.ok) {
      var k = j.killed || {};
      ctlMsg('已停止（监督器 ' + (k['监督器']||0) + '、训练 ' + (k['训练']||0) +
             '、子进程 ' + (k['训练子进程']||0) + '、评测 ' + (k['评测']||0) + '）', 'ok');
      setTimeout(update, 1200);
    } else { ctlMsg('停止后仍有残留，请看 NOTES 或手工处理', 'bad'); }
  }).catch(function(e){ ctlMsg('请求失败：' + e, 'bad'); });
}
document.addEventListener('click', function(e){
  var id = e.target && e.target.id;
  if (id === 'btnLast') { ctlApplyFromServer(); ctlMsg('已填入服务器上的参数'); }
  else if (id === 'btnV3Stable') { ctlFillV3Stable(); }
  else if (id === 'btnV4Fill') { ctlFillV4(); }
  else if (id === 'btnV4Go') { ctlStartV4(); }
  else if (id === 'btnStart') { ctlStart(); }
  else if (id === 'btnStop') { ctlStop(); }
  else if (id === 'btnCppDefaults') {
    e.preventDefault();
    engApplyAll(CPP_SHARED);       // 只恢复「两套配方共有」的字段（局数/步数/并行…）
    CPP_ACTIVE_RECIPE = null;      // 比对基准回到 config.h 的默认配方
    cppDefCheck(); ctlSaveDraft();
    ctlMsg('已恢复 C++ 默认配方：' + CPP_SHARED.Games + ' 局 / ' + CPP_SHARED.MinSteps + ' 步');
  }
  else if (id === 'btnRecipeFill') { e.preventDefault(); recipeFill(); }
  else if (id === 'recipeInfo') { e.preventDefault(); recipeShowInfo(); }
});
// 引擎下拉：切换时只改显示与提示（并立刻存草稿，刷新后不会打回 Python）
document.addEventListener('change', function(e){
  if (!e.target) { return; }
  if (e.target.id === 'p_Engine') { engineChanged(); ctlSaveDraft(); }
  else if (e.target.id === 'p_CppDirSel') { cppDirSync(true); ctlSaveDraft(); }
  else if (e.target.id && (e.target.id.indexOf('p_') === 0 || e.target.id.indexOf('q_') === 0)) {
    cppDefCheck();                    // 改任何一个参数都重算「与 C++ 默认配方差在哪」
  }
});
// 手输路径时也更新提示（判断这是不是主线目录 / 是不是空目录）
document.addEventListener('input', function(e){
  if (e.target && e.target.id === 'p_CppDir') { cppDirSync(false); }
});
// 输入值快照/还原：万一 #cards 容器被整块替换，也不会影响控制面板（它在 #cards 之外），
// 但为了将来 DOM 结构变化时不丢输入，这里仍然做一层保护
function ctlSnapshot(){
  var o = {};
  for (var i = 0; i < CTL_FIELDS.length; i++) {
    var el = document.getElementById('p_' + CTL_FIELDS[i][0]);
    if (el) { o[CTL_FIELDS[i][0]] = (CTL_FIELDS[i][1] === 'bool') ? el.checked : el.value; }
  }
  return o;
}
function ctlRestore(o){
  if (!o) { return; }
  for (var k in o) {
    if (!Object.prototype.hasOwnProperty.call(o, k)) { continue; }
    var el = document.getElementById('p_' + k);
    if (!el) { continue; }
    if (typeof o[k] === 'boolean') { el.checked = o[k]; } else if (el.value === '') { el.value = o[k]; }
  }
}

function setBadge(cls, txt){
  var hb = document.getElementById('hb');
  if (hb) { hb.className = 'hb ' + cls; hb.textContent = txt; }
}
// 【2026-10-12 加】一键填入 V3 **稳妥版**（第一次尝试失败后的修正配方）。
//
// 为什么稳妥版长这样（每一条都有依据）：
//   * 继承 resume.pt（回放池 12.6 万条 + 优化器 + lr 计数）：第一次崩坏（−478 Elo）的头号原因
//     就是**冷启动空回放池**，第 1 代只 9,736 条数据却训 320 步。
// ⚠️ 注释里**不要出现连续的 -- 破折号**：会被 JS 解析成自减运算符，
//    而 // 注释只到行尾 ⇒ 续行直接 SyntaxError（2026-10-12 实测踩到，node --check 抓到）。
//   * **一次只改一个变量**：这次只改"数据量"（games 80→160、min-steps 160→320）。
//   * hw=0.5 / pw=1.0 且**关退火**：V2 末期就是 0.5/1.0；若开着退火，代码会把
//     --heuristic-weight 当**退火起点**（传 6.0 就从 6.0 开始退）⇒ 等于又换了一个变量。
//   * BufferGens 60：数据翻倍后回放池涨得快，不放大就会边加边扔（并防止 V2 数据过早被挤掉）。
//   * AllowOverwrite：目录里已有 best.pt（起点的权重），train.py 默认拒绝 ⇒ 缺它 exit=3 秒退。
//
// ⚠️⚠️ **JS 注释里绝对不要用反引号**（本文件是 PowerShell @"..."@ here-string，
//      反引号是 PowerShell 的转义符：反引号+r 会被它先变成回车 ⇒ JS 收到的是**从中间断掉的一行**
//      ⇒ SyntaxError。2026-10-12 实测踩到，node --check 抓到）。也不要用连续的 -- 破折号。
function ctlFillV3Stable(){
  var preset = {
    RunName: 'V3', SaveDir: 'models/V3',
    Workers: 16, Games: 160, MinSteps: 320,
    EvalGames: 20, EvalEvery: 2, EvalWorkers: 8,
    GensPerRound: 30, StartTarget: 0,
    RulerGames: 200, Seed: 12345,
    BufferGens: 60, HeuristicWeight: 0.5, PolicyWeight: 1,
    NoAnneal: true, AllowOverwrite: true,
    AutoEval: true
  };
  ctlFill(preset);
  ctlSaveDraft();
  ctlMsg('已填入 V3 稳妥版（只改数据量：160 局 / 320 步；其余沿用 V2 末期口径）', 'ok');
  ctlHint('V3 稳妥版已就位：<b>每代 160 局 / 训练 320 步 / 回放池 60 代 / 固定 hw=0.5·pw=1.0（不退火）/ 允许写入已有产物</b>。' +
          '起点 = V2 gen_175 的权重 <b>+ V2 的 12.6 万条回放池</b>（不是冷启动）。' +
          '⚠️ 开跑后<b>第 1~2 代就要做一次对外体检</b>（别只看 loss）：' +
          '<code>ladder.py --candidate models/V3/gen_2.pt --anchor models/V2/gen_175.pt --games 200 ' +
          '--heuristic-weight 0.5 --policy-weight 1.0 --workers 8</code>，得分率 ≥0.5 才继续。');
}
// 【2026-10-13 加】V4 一键配方（= V3 的实配 + 只有一根轴：回放池最近性）。
//
// 配方本体写在**服务端**（serve_dashboard.py 的 V4_PRESET），这里只用 /api/v4-preset 取来用。
// 为什么不在 JS 里再抄一份：抄两份必然漂移，而"按钮跑的配方"与"文档写的配方"必须是同一份。
//
// 两个按钮分工：
//   * 「填入 V4 配方」只填表（和「填入 V3 稳妥版」同类），你可以再改，然后走「开始训练」。
//   * 「一键开 V4」是真正的一键：服务端先跑 prepare_run.ps1 继承 V3 的权重 + 回放池 + 优化器
//     状态（1.29 GB，约 1 到 3 分钟），再按同一份配方启动训练。
//     为什么把"继承起点"也包进按钮：忘了这一步，train.py 会因为目标目录里没有 resume.pt 而走
//     冷启动，那正是 V3 第一次尝试崩坏（负 478 Elo）的头号原因。
function ctlV4Info(cb){
  fetch('/api/v4-preset', { cache: 'no-store' })
    .then(function(r){ return r.json(); })
    .then(cb)
    .catch(function(){ ctlMsg('取 V4 配方失败：看板服务可能没在跑', 'bad'); });
}
function ctlV4Summary(j){
  var e = j.effective || j.preset, p = j.preset;
  return 'auto_train.ps1 -RunName ' + p.RunName + ' -SaveDir ' + p.SaveDir +
         ' -Workers ' + e.Workers + ' -Games ' + e.Games + ' -MinSteps ' + e.MinSteps +
         ' -EvalGames ' + e.EvalGames + ' -EvalEvery ' + e.EvalEvery + ' -EvalWorkers ' + e.EvalWorkers +
         ' -GensPerRound ' + e.GensPerRound + ' -StartTarget ' + (e.StartTarget || '自动') +
         ' -RulerGames ' + e.RulerGames + ' -Seed ' + e.Seed +
         (e.AutoEval ? ' -AutoEval' : '') + (e.NoAnneal ? ' -NoAnneal' : '') +
         ' -BufferGens ' + e.BufferGens + ' -HeuristicWeight ' + e.HeuristicWeight +
         ' -PolicyWeight ' + e.PolicyWeight + ' -AllowOverwrite' +
         (e.ExtraArgs ? ' -ExtraArgs ' + e.ExtraArgs : '');
}
function ctlFillV4(){
  ctlV4Info(function(j){
    if (!j || !j.preset) { ctlMsg('取 V4 配方失败', 'bad'); return; }
    ctlClearDraft();
    ctlFill(j.preset);
    ctlSaveDraft();
    ctlMsg('已填入 V4 配方（V3 实配 + 回放池最近性这一根轴）', 'ok');
    ctlHint('V4 = <b>V3 的实配</b>，只多一个开关：<code>' + j.preset.ExtraArgs + '</code>' +
            '（代权重按「2 的 负代龄比半衰期 次方」衰减 ⇒ 60 代池里最近 20 代占约 55%，均匀本应 32%）。' +
            '⚠️ 启动前必须先继承 V3 的权重与回放池 —— 直接点「⚡ 一键开 V4」，它把这一步和启动一起做。');
  });
}
function ctlStartV4(){
  ctlV4Info(function(j){
    if (!j || !j.preset) { ctlMsg('取 V4 配方失败', 'bad'); return; }
    var f = j.from || {};
    var msg = '确认一键开 V4？\n\n① 先继承起点（1.29 GB，实测约 10 到 20 秒）：\n' +
              '   prepare_run.ps1 -Dir ' + j.preset.SaveDir + ' -FromDir ' + f.dir +
              ' -FromGen ' + f.gen + '\n   （' + f.weights + ' 的权重 + ' + f.resume +
              ' 里的回放池与优化器状态，代数计数清零）\n\n② 再启动训练：\n   ' + ctlV4Summary(j) +
              '\n\n（会新建 ' + j.preset.SaveDir + '；若已有训练在跑会被拒绝）';
    if (!confirm(msg)) { return; }
    ctlMsg('正在继承 ' + f.dir + ' 的权重与回放池（约 10 到 20 秒，不要关页面）…');
    ctlHint('继承中：把 V3 的回放池与优化器状态搬到 ' + j.preset.SaveDir +
            '；完成后会自动启动训练，看板顶部「进度」卡几秒后显示当前阶段。');
    fetch('/api/start-v4', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: '{"dry_run":false,"confirm":"开始 V4"}'
    }).then(function(r){ return r.json(); }).then(function(k){
      if (k.ok) {
        ctlMsg('V4 已启动（PID ' + k.pid + '，目标 gen ' + k.start_target + '，' +
               (k.detached ? '已脱离本机会话' : '子进程方式') + '）', 'ok');
        ctlSaveDraft();
        setTimeout(function(){
          fetch('/api/last-params', { cache: 'no-store' }).then(function(r){ return r.json(); })
            .then(function(q){ ctlLastParams = q; ctlHint('训练已在跑。' + ctlTargetHint(q)); })
            .catch(function(){});
          update();
        }, 2500);
      } else {
        ctlMsg('V4 启动失败：' + ((k.errors || []).join('；') || '未知原因'), 'bad');
        ctlHint((k.prepare_summary ? k.prepare_summary + '<br>' : '') +
                '失败详情：<code>checkpoints/history/dashboard_prepare.log</code>（继承阶段）与 ' +
                '<code>checkpoints/history/dashboard_launch.log</code>（启动阶段）。' +
                (k.prepare ? '<br>继承输出末尾：<br><code>' +
                   String(k.prepare).split('<').join('&lt;').slice(-400) + '</code>' : ''));
      }
    }).catch(function(e){ ctlMsg('请求失败：' + e, 'bad'); });
  });
}
// 【2026-10-12 删】旧的「填入 V3 配方」预设已移除：它那套参数（冷启动 + hw=0 + 关退火）
// 实测把模型练成 −478 Elo，留在面板上只会让人误点。需要复现那个结论时，
// 用命令行显式给参数即可（依据与产物见 checkpoints/history/archive/V3_attempt1_negative_result.md）。
/* ══════════ 原地同步：只改真的变了的那几个文本/属性 ══════════
   以前是「整张卡 outerHTML 替换」：卡片里只要有时间/秒数在变，整棵子树就重建，
   一帧里大面积重排 ⇒ 看起来就是"闪一下 / 画面消失又回来"。
   现在按节点递归对比：标签结构一样就只改文本与属性，**元素身份保留**
   ⇒ 浏览器只重绘变了的几个像素，滚动位置、<details> 展开、输入焦点都不丢。 */
function patchNode(a, b){
  if (a.nodeType !== b.nodeType) {
    if (a.parentNode) a.parentNode.replaceChild(b.cloneNode(true), a);
    return;
  }
  if (a.nodeType === 3 || a.nodeType === 8) {          // 文本 / 注释
    if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue;
    return;
  }
  if (a.nodeType !== 1) return;
  if (a.tagName !== b.tagName) {
    if (a.parentNode) a.parentNode.replaceChild(b.cloneNode(true), a);
    return;
  }
  var i, at;
  for (i = a.attributes.length - 1; i >= 0; i--) {
    at = a.attributes[i];
    if (!b.hasAttribute(at.name)) a.removeAttribute(at.name);
  }
  for (i = 0; i < b.attributes.length; i++) {
    at = b.attributes[i];
    if (a.getAttribute(at.name) === at.value) continue;
    // 背景图：先把新图下载好再换，避免"旧图被撤掉、新图还没到"露出白底
    if (at.name === 'style' && /url\(/.test(at.value) &&
        /url\(/.test(a.getAttribute('style') || '')) {
      patchBackground(a, at.value);
    } else {
      a.setAttribute(at.name, at.value);
    }
  }
  var ac = a.childNodes, bc = b.childNodes;
  if (ac.length !== bc.length) {                       // 结构变了才整块换
    a.innerHTML = b.innerHTML;
    return;
  }
  for (i = 0; i < bc.length; i++) patchNode(ac[i], bc[i]);
}

function patchBackground(el, newStyle){
  var m = /url\((['"]?)([^'")]+)\1\)/.exec(newStyle);
  if (!m) { el.setAttribute('style', newStyle); return; }
  var img = new Image();
  var done = function(){ el.setAttribute('style', newStyle); };
  img.onload = done;
  img.onerror = done;
  img.src = m[2];
}

function patchInto(cont, newCont){
  var oldById = {}, i, el;
  for (i = 0; i < cont.children.length; i++) {
    el = cont.children[i];
    if (el.id) oldById[el.id] = el;
  }
  var order = [];
  for (i = 0; i < newCont.children.length; i++) {
    var nb = newCont.children[i];
    var ob = nb.id ? oldById[nb.id] : null;
    if (ob) { patchNode(ob, nb); delete oldById[nb.id]; order.push(ob); }
    else { order.push(nb.cloneNode(true)); }
  }
  Object.keys(oldById).forEach(function(id){
    var gone = oldById[id];
    if (gone.parentNode === cont) cont.removeChild(gone);
  });
  for (i = 0; i < order.length; i++) {
    if (cont.children[i] !== order[i]) cont.insertBefore(order[i], cont.children[i] || null);
  }
}

function update(){
  fetch(location.href.split('#')[0], { cache: 'no-store' })
    .then(function(r){ if (!r.ok) { throw new Error(r.status); } return r.text(); })
    .then(function(html){
      okCount++; failCount = 0; lastOk = Date.now(); stripped = true;
      stripMeta();
      var m = /var GEN = (\d+);/.exec(html);
      if (m) { GEN = parseInt(m[1], 10); }
      var doc = new DOMParser().parseFromString(html, 'text/html');
      var cont = document.getElementById('cards');
      var newCont = doc.getElementById('cards');
      if (!cont || !newCont) { return; }
      try {
        patchInto(cont, newCont);
      } catch (e) {
        // 兜底：万一某个卡的节点结构让 patch 走不通，退回整块替换。
        // 整块替换会闪一下，但总比"面板卡死不动"强。
        cont.innerHTML = newCont.innerHTML;
      }
    })
    .catch(function(){
      failCount++;
      if (stripped) {
        setBadge('bad', '更新失败（看板服务可能已停；最后成功 ' + Math.round((Date.now()-lastOk)/1000) + ' 秒前）');
      }
    });
}
function hbTick(){
  // 【2026-10-12 修】用户动过训练参数面板 ⇒ **绝不自动整页重载**（否则输入被刷回服务器值）
  if (ctlHadEdit) {
    if (!ctlWarnedEdit) {
      ctlWarnedEdit = true;
      ctlMsg('已检测到你对参数的修改：自动刷新已暂停（防止把你填的值刷掉）。提交或点「沿用上轮参数」后恢复自动刷新。');
    }
    setBadge('ok', '原地更新·参数编辑中（已暂停自动重载）');
    return;
  }
  if (!stripped) {
    // 【2026-10-12 改】fetch 不可用（例如 file:// 打开）：**不再强制重载**。
    // 原因：meta refresh 在加载时就被掐掉了（见文件里最早那段 IIFE），
    // 而"数据过期就 location.replace"会整页重载 ⇒ 触发 Chrome 的"重新加载此网站？"
    // 确认框（用户实测每几秒弹一次）。现在只**提示**，让用户自己按 F5。
    var age0 = Math.round(Date.now()/1000) - GEN;
    if (age0 > LIM && age0 < 3600) {
      setBadge('bad', '数据已过期 ' + age0 + ' 秒（自动重载已停用，请按 F5 手动刷新）');
    } else {
      setBadge('warn', '只读模式（fetch 不可用：请用 http:// 地址打开，file:// 下会闪）');
    }
    return;
  }
  if (failCount > 0) { return; }   // 失败提示由 update() 负责，别覆盖
  setBadge('ok', '原地更新·正常');
}
/* ══════════ 实时通道：SSE ══════════
   看板文件一被重写，服务端立刻推 event:changed ⇒ 这里 120 ms 防抖后 patch 一次，
   延迟从"一个轮询周期"降到百毫秒级，且不重载页面、不重建卡片。
   setInterval(update, IVL) 保留为兜底：SSE 断开、被代理掐掉或浏览器太老时照常更新。 */
var updTimer = null;
function scheduleUpdate(){
  if (updTimer) return;
  updTimer = setTimeout(function(){ updTimer = null; update(); }, 120);
}
function startStream(){
  if (typeof EventSource !== 'function') return;
  // ?stream=0：调试/截图时关掉长连接（无头浏览器会一直等"网络空闲"而卡住）
  if (/[?&]stream=0\b/.test(location.search)) return;
  var es = null;
  try { es = new EventSource('/api/stream'); } catch (e) { return; }
  es.addEventListener('changed', scheduleUpdate);
  // 断线浏览器会自动重连，这里不重试也不报警：中间的空白由轮询兜底
  es.onerror = function(){};
  window.__es = es;
}
update();
ctlLoad();
// 定时器统一在这里起：加载完草稿后才知道该不该"暂停自动重载"（见 hbTick）
setInterval(update, IVL);
setInterval(hbTick, 1000);
setTimeout(startStream, 300);
hbTick();
// 【2026-10-13 加】离开/重载前再存一次草稿。**不设 returnValue** ⇒ 不会弹
// Chrome 的"重新加载此网站？"确认框（那个弹窗正是当初删掉 beforeunload 的原因），
// 纯保存是安全的：多一道保险，就算 aria 之外的路径触发了整页重载，输入也不会丢。
try { window.addEventListener('beforeunload', ctlSaveDraft); } catch (e) {}

/* ══════════ 版本切换：Python 版 / C++ 版 ══════════
   两套数据源互斥：Python 版是服务端渲染的 #cards（train.py 的账本），
   C++ 版是这里按 /api/cpp-state 现画的 #cppCards（gk.exe loop 的日志）。
   顺手把「训练引擎」下拉带过去，避免「看的是 C++、点开始却起 Python」。
   选择存 localStorage；#verBar/#cppCards 都在 #cards 之外，原地更新不会砸掉它们。 */
var VER = 'python', cppTimer = null, cppDir = '', cppState = null;

function verApply(v, persist){
  VER = (v === 'cpp') ? 'cpp' : 'python';
  var py = document.getElementById('cards');
  var cp = document.getElementById('cppCards');
  var bPy = document.getElementById('verPy'), bCpp = document.getElementById('verCpp');
  if (py) py.style.display = (VER === 'python') ? '' : 'none';
  if (cp) { if (VER === 'cpp') cp.classList.add('on'); else cp.classList.remove('on'); }
  if (bPy) { if (VER === 'python') bPy.classList.add('on'); else bPy.classList.remove('on'); }
  if (bCpp) { if (VER === 'cpp') bCpp.classList.add('on'); else bCpp.classList.remove('on'); }
  var mini = document.getElementById('c-cpp');
  if (mini) {
    if (VER === 'cpp') mini.setAttribute('data-ver-hidden', '1');
    else mini.removeAttribute('data-ver-hidden');
  }
  if (persist !== false) { try { localStorage.setItem('gkVer', VER); } catch (e) {} }
  var sel = document.getElementById('p_Engine');
  if (sel && sel.value !== VER) {
    sel.value = VER;
    if (typeof engineChanged === 'function') engineChanged();
  }
  var hint = document.getElementById('verHint');
  if (hint) hint.textContent = (VER === 'cpp')
    ? '正在看：C++ 引擎（gk.exe · C++\\model\\<run>）'
    : '正在看：Python 引擎（train.py · models\\<run>）';
  if (VER === 'cpp') {
    cppPoll();
    if (!cppTimer) cppTimer = setInterval(cppPoll, 1200);
  } else if (cppTimer) {
    clearInterval(cppTimer); cppTimer = null;
  }
}

function cppKpi(k, v, cls){
  return '<div class="kpi"><div class="k">' + k + '</div><div class="v' +
         (cls ? ' ' + cls : '') + '">' + v + '</div></div>';
}
// 【2026-10-27 加】秒数改成人读的时长（用户："自对弈累计 6033 s 改成小时计数吧"）。
// 按量级选单位：≥1 小时 → "1.7 h"，≥1 分钟 → "12.3 min"，其余 → "45 s"。
// 为什么不全用小时：训练累计每代才几秒，全用 h 就全变成 "0.0 h"，等于把信息抹掉。
// 精确秒数留在 title 里，鼠标停上去还是能看到原值。
function cppDur(sec){
  if (sec === null || sec === undefined || sec === '' || !isFinite(Number(sec))) return '—';
  var s = Math.round(Number(sec));
  if (s <= 0) return '0';
  var txt;
  // 3540 s = 59 min 就切到小时，避免出现 "60.0 min" 这种尴尬值
  if (s >= 3540) txt = (s / 3600).toFixed(1) + ' h';
  else if (s >= 60) txt = (s / 60).toFixed(1) + ' min';
  else txt = s + ' s';
  return '<span title="' + s.toLocaleString('en-US') + ' 秒">' + txt + '</span>';
}
function cppN(v, d){
  if (v === null || v === undefined || v === '') return '—';
  var n = Number(v);
  if (!isFinite(n)) return '—';
  return (d === undefined) ? String(Math.round(n)) : n.toFixed(d);
}
function cppInt(v){
  if (v === null || v === undefined || v === '') return '—';
  var n = Number(v);
  if (!isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

function cppTicks(mn, mx, n){
  /* 「好看的」刻度：步长只取 1/2/2.5/5/10 × 10^k。
     原来直接用 (mx-mn)/n 当步长，刻度会变成 0.037、1.83 这种机器数，根本读不出来。 */
  var raw = (mx - mn) / Math.max(1, n);
  if (!(raw > 0)) return [mn, mx];
  var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
  var norm = raw / mag, step;
  if (norm <= 1) step = 1; else if (norm <= 2) step = 2; else if (norm <= 2.5) step = 2.5; else if (norm <= 5) step = 5; else step = 10;
  step *= mag;
  var out = [], t = Math.ceil(mn / step - 1e-9) * step, k = 0;
  while (t <= mx + step * 1e-9 && k++ < 40) { out.push(Math.round(t / step) * step); t += step; }
  return out;
}
function cppTickDec(step){
  var s = Math.abs(step);
  if (s >= 1) return (Math.abs(s - Math.round(s)) < 1e-9) ? 0 : 1;
  if (s >= 0.1) return 1;
  if (s >= 0.01) return 2;
  if (s >= 0.001) return 3;
  return 4;
}
function cppSpark(rows, key, label){
  // 【2026-10-27 改】x 用**代号**而不是数组下标（中间缺代时下标会画成等距，横向就是假的），
  //   并在配方断点画竖虚线 —— 否则 gen5–7 那次 mse 下探会被当成"模型变好"，其实只是每代样本少了一半。
  // 【2026-10-31 重画】原来只有一条光秃秃的折线 + 文字里的 min/max，确实难读，改了六处：
  //   ① y 轴刻度 + 横向网格（步长走 1/2/2.5/5×10^k 的整数档）；
  //   ② x 轴中间刻度（原来首尾各一个 gen 数字）；
  //   ③ 曲线下方填 5% 浅底，出面积感；
  //   ④ 每点 hover 提示（代数 / 数值 / 本代样本数），末点加圈标出；
  //   ⑤ 点数 ≥6 时叠一条滑动均线（虚线）——单代抖动不必再当趋势读；
  //   ⑥ SVG 改 width:100% + viewBox，窄屏不再被压扁。
  // 口径一行没动：还是按代号定位、还是同一套配方断点判据。
  var W = 520, H = 184, L = 58, R = 14, T = 20, B = 30;
  var pts = [];
  rows.forEach(function(r){
    var v = Number(r[key]);
    if (isFinite(v) && r[key] !== null && r[key] !== undefined) {
      // train_log 每行带 recipe_id（--recipe 写进去的）；老日志没有这个字段时 rid 为 NaN，退回样本数判据
      var rid = (r.recipe_id !== undefined && r.recipe_id !== null) ? Number(r.recipe_id) : NaN;
      pts.push({ g: Number(r.gen), v: v, s: Number(r.samples), r: rid });
    }
  });
  if (pts.length < 2) return '';
  var g0 = pts[0].g, g1 = pts[pts.length - 1].g;
  if (!isFinite(g0) || !isFinite(g1)) { g0 = 0; g1 = pts.length - 1; }
  var span = (g1 - g0) || 1;
  var vals = pts.map(function(p){ return p.v; });
  var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
  if (mx - mn < 1e-9) { var cc = Math.abs(mx) || 1; mn -= cc * 0.05; mx += cc * 0.05; }
  var vp = (mx - mn) * 0.12; mn -= vp; mx += vp;      // 上下留白：否则首尾点贴在框线上
  function X(g){ return L + (W - L - R) * (g - g0) / span; }
  function Y(v){ return (H - B) - (H - B - T) * (v - mn) / (mx - mn); }
  // 300 局一代 ~1.8–2 万样本，160 局一代 ~1 万 ⇒ 15000 是干净的判据（老日志的退路）
  var cuts = [], i, j;
  var useRecipe = pts.some(function(p){ return isFinite(p.r) && p.r > 0; });
  for (i = 1; i < pts.length; i++) {
    if (useRecipe) {
      var ra = pts[i - 1].r, rb = pts[i].r;
      if (isFinite(ra) && isFinite(rb) && ra !== rb) cuts.push({ g: pts[i].g, t: '配方' + rb });
    } else {
      var sa = pts[i - 1].s, sb = pts[i].s;
      if (isFinite(sa) && isFinite(sb) && ((sa < 15000) !== (sb < 15000))) cuts.push({ g: pts[i].g, t: '配方' });
    }
  }
  var svg = '';
  var C = (key === 'mse') ? 'var(--c2)' : 'var(--c1)';        // CE 用蓝、MSE 用紫
  var tk = cppTicks(mn, mx, 4);
  var dec = cppTickDec(tk.length > 1 ? (tk[1] - tk[0]) : ((mx - mn) / 4));
  tk.forEach(function(t){
    var y = Y(t).toFixed(1);
    svg += '<line x1="' + L + '" y1="' + y + '" x2="' + (W - R) + '" y2="' + y +
           '" stroke="var(--hair)" stroke-width="0.7" />' +
           '<text x="' + (L - 8) + '" y="' + y + '" font-size="9.5" fill="var(--mut)" ' +
           'text-anchor="end" dominant-baseline="middle">' + t.toFixed(dec) + '</text>';
  });
  var xn = Math.max(2, Math.min(7, Math.round((W - L - R) / 90))), seenX = {};
  for (i = 0; i < xn; i++) {
    var gx = Math.round(g0 + span * i / (xn - 1));
    if (seenX[gx]) continue;
    seenX[gx] = 1;
    var xx = X(gx).toFixed(1);
    svg += '<line x1="' + xx + '" y1="' + (H - B) + '" x2="' + xx + '" y2="' + (H - B + 3.5) +
           '" stroke="var(--line2)" stroke-width="0.8" />' +
           '<text x="' + xx + '" y="' + (H - B + 15) + '" font-size="9.5" fill="var(--mut)" text-anchor="middle">' + gx + '</text>';
  }
  svg += '<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) +
         '" stroke="var(--line2)" stroke-width="0.9" />' +
         '<text x="' + (W - R + 4) + '" y="' + (H - B + 4) + '" font-size="9" fill="var(--mut)" text-anchor="start">代</text>';
  cuts.forEach(function(c){
    var cx = X(c.g).toFixed(1);
    svg += '<line x1="' + cx + '" y1="' + T + '" x2="' + cx + '" y2="' + (H - B) +
           '" stroke="var(--fg)" stroke-width="0.8" opacity="0.35" stroke-dasharray="3 3" />' +
           '<text x="' + cx + '" y="' + (T - 6) + '" font-size="9" fill="var(--mut)" text-anchor="middle">' + c.t + '</text>';
  });
  var poly = pts.map(function(p){ return X(p.g).toFixed(1) + ',' + Y(p.v).toFixed(1); }).join(' ');
  svg += '<polygon points="' + X(g0).toFixed(1) + ',' + (H - B) + ' ' + poly + ' ' + X(g1).toFixed(1) + ',' + (H - B) +
         '" fill="' + C + '" opacity="0.09" />';
  var win = Math.min(5, Math.max(3, Math.round(pts.length / 6)));
  if (pts.length >= 6) {
    var ma = [];
    for (i = 0; i < pts.length; i++) {
      var acc = 0, cnt = 0;
      for (j = Math.max(0, i - win + 1); j <= i; j++) { acc += pts[j].v; cnt++; }
      ma.push(X(pts[i].g).toFixed(1) + ',' + Y(acc / cnt).toFixed(1));
    }
    svg += '<polyline fill="none" stroke="' + C + '" stroke-width="1.2" opacity="0.42" stroke-dasharray="4 3" ' +
           'stroke-linejoin="round" points="' + ma.join(' ') + '" />';
  }
  svg += '<polyline fill="none" stroke="' + C + '" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" points="' + poly + '" />';
  var showDots = pts.length <= 45;
  pts.forEach(function(p){
    var px = X(p.g).toFixed(1), py = Y(p.v).toFixed(1);
    if (showDots) svg += '<circle cx="' + px + '" cy="' + py + '" r="1.9" fill="' + C + '" opacity="0.55" />';
    svg += '<circle cx="' + px + '" cy="' + py + '" r="7" fill="none" pointer-events="all"><title>gen ' + p.g +
           ' · ' + label + ' ' + p.v.toFixed(4) +
           (isFinite(p.s) ? (' · 本代 ' + Math.round(p.s).toLocaleString('en-US') + ' 样本') : '') + '</title></circle>';
  });
  var lp = pts[pts.length - 1], lx = X(lp.g).toFixed(1), ly = Y(lp.v).toFixed(1);
  svg += '<circle cx="' + lx + '" cy="' + ly + '" r="6.5" fill="none" stroke="' + C + '" stroke-width="1.2" opacity="0.45" />' +
         '<circle cx="' + lx + '" cy="' + ly + '" r="3.2" fill="' + C + '" />';
  var dv = lp.v - pts[0].v;
  var pv = (Math.abs(pts[0].v) > 1e-12) ? (dv / Math.abs(pts[0].v) * 100) : 0;
  var best = pts[0];
  pts.forEach(function(p){ if (p.v < best.v) best = p; });      // CE / MSE 都是越低越好
  return '<div style="flex:1 1 330px;min-width:300px">' +
    '<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:2px">' +
    '<span style="display:inline-block;width:9px;height:9px;background:' + C + '"></span>' +
    '<b style="font-weight:600">' + label + '</b>' +
    '<span class="small muted">末值 ' + lp.v.toFixed(4) + ' · 首→末 ' + (dv >= 0 ? '+' : '−') + Math.abs(dv).toFixed(4) +
    '（' + (pv >= 0 ? '+' : '−') + Math.abs(pv).toFixed(1) + '%） · 最低 ' + best.v.toFixed(4) + ' @ gen ' + best.g +
    ' · ' + pts.length + ' 代' + (pts.length >= 6 ? (' · 虚线 = ' + win + ' 代均线') : '') + '</span></div>' +
    '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" ' +
    'style="display:block;width:100%;height:auto;font-family:var(--mono)">' + svg + '</svg></div>';
}

function cppRecipeName(rid){
  // 配方档案（C++\peifang\配方NN-*.json）里这个号叫什么。只用于 hover 提示，查不到返回空串。
  try {
    for (var i = 0; i < CPP_RECIPES.length; i++) {
      if (Number(CPP_RECIPES[i].id) === Number(rid)) return String(CPP_RECIPES[i].name || '');
    }
  } catch (e) {}
  return '';
}

function cppRecipeEpochs(history){
  // 【2026-10-29 加】把 train_log 折成「哪几代用哪个配方」的区间表（用户要求：
  //   每次用新配方训练，Elo 图上要标出哪里用了新配方）。
  // 为什么必须标：换配方是**外部干预**（局数/模拟数/门控局数/配对开局一起换），
  //   跨过断点的那一步不是"模型变强"。不标出来，换配方后的第一个点就会被读成
  //   "这一代涨了 40 Elo"——和当初 mse 被误读是同一类错误。
  // 数据源：train_log.jsonl 每行的 recipe_id（`gk loop --recipe N` 写进去的）。
  //   老日志（本项目 gen 1–43）没有这个字段 ⇒ 退回「每代样本数跨过 15000」的启发式，
  //   判据和损失曲线的虚线完全一致（320 局 ≈ 1.9 万样本，160 局 ≈ 1 万）。
  var rows = [];
  (history || []).forEach(function(r){
    var g = Number(r.gen);
    if (!isFinite(g)) return;
    var rid = (r.recipe_id === undefined || r.recipe_id === null) ? NaN : Number(r.recipe_id);
    var s = Number(r.samples), games = Number(r.games);
    var e;
    if (isFinite(rid) && rid > 0) {
      e = { key: 'r' + rid, rid: rid, label: '配方' + rid, exact: true };
    } else if (isFinite(s)) {
      e = { key: 'h' + (s < 15000 ? 'lo' : 'hi'), rid: null, label: '配方?', exact: false };
    } else {
      e = { key: 'h?', rid: null, label: '配方?', exact: false };
    }
    e.g = g;
    if (isFinite(games) && games > 0) e.games = games;
    rows.push(e);
  });
  rows.sort(function(a, b){ return a.g - b.g; });
  var eps = [];
  rows.forEach(function(r){
    var last = eps[eps.length - 1];
    if (last && last.key === r.key) {
      if (r.g < last.from) last.from = r.g;
      if (r.g > last.to) last.to = r.g;
      last.n += 1;
      if (last.games === undefined && r.games !== undefined) last.games = r.games;
    } else {
      eps.push({ key: r.key, rid: r.rid, label: r.label, exact: r.exact,
                 from: r.g, to: r.g, n: 1, games: r.games });
    }
  });
  return eps;
}

// ══════════════════════════════════════════════════════════════════════════
// 【2026-10-30 加】Elo 对战面板
//
// 为什么不复用 1200ms 的 cppPoll 来画进度：cppPoll 拉的是 /api/cpp-state，
// 那个接口要扫目录、解析训练日志、读 elo.json —— 开销不小，压到 1 秒级会白白烧 CPU，
// 而进度条只需要三个数。所以对战进行中另开一个 2 秒的轻量轮询走 /api/elo-status（只读一个 json）。
//
// 所有请求都显式带 dry_run —— 服务端要求这个字段必须有且是布尔。
// 这是照 V4 那条教训定的规矩：危险动作不能靠"字段缺省"来判断意图，
// 少写一个字段就静默变成"真启动"，那是最难查的一类 bug。
// ══════════════════════════════════════════════════════════════════════════
var ELO_TIMER = null;
var ELO_BUSY = false;
var ELO_ANCHOR_SET = false;

function eloNum(id, dflt){
  var el = document.getElementById(id);
  var v = el ? Number(el.value) : NaN;
  return isFinite(v) ? v : dflt;
}

function eloHMS(sec){
  sec = Math.max(0, Math.round(Number(sec) || 0));
  var m = Math.floor(sec / 60), s = sec % 60;
  return m ? (m + ' 分 ' + s + ' 秒') : (s + ' 秒');
}

function eloSigned(v){
  // 注意 Number(null) === 0、Number('') === 0 —— 直接 Number(v) 会把"没有值"
  // 显示成 +0.0，和真的测得 0.0 分不开。所以空值先挡掉。
  if (v === null || v === undefined || v === '') return '—';
  v = Number(v);
  if (!isFinite(v)) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(1);
}

// 把 elo_job 状态画到面板上。running / summary / error 三种情况分开处理，
// 因为用户真正想看的是"涨了没"，而不是"任务成功了吗"。
function eloPaintJob(job){
  var hint = document.getElementById('eloHint');
  var wrap = document.getElementById('eloBarWrap');
  var bar  = document.getElementById('eloBar');
  var prog = document.getElementById('eloProg');
  var res  = document.getElementById('eloResult');
  var tail = document.getElementById('eloTail');
  if (!hint) return;
  job = job || {};

  if (tail) tail.textContent = (job.tail || []).join('\n');

  if (job.running) {
    if (wrap) wrap.style.display = '';
    var tot = Number(job.pairs) || 0, done = Number(job.pairs_done) || 0;
    var pct = tot ? Math.min(99, Math.round(done * 100 / tot)) : 0;
    if (bar) bar.style.width = pct + '%';
    if (prog) {
      prog.textContent = '进行中 ' + done + ' / ' + tot + ' 对' +
        '（已用 ' + eloHMS(job.elapsed) +
        (tot ? '，预计还要 ' + eloHMS(job.eta - job.elapsed) : '') + '）';
    }
    hint.textContent = job.gens ? ('正在对战 gen ' + job.gens.join(', ') + '，锚点 gen_' + job.anchor) : '';
    return;
  }

  if (wrap && !job.summary) wrap.style.display = 'none';

  // 结束态：优先展示"涨了多少"，其次才是错误。
  if (job.summary) {
    var sm = job.summary;
    var d = job.delta;
    var color = 'var(--mut)', verdict = '';
    if (d !== null && d !== undefined && isFinite(Number(d))) {
      // 关键判据：95% 区间是否整体落在旧值之上/之下。区间重叠 ⇒ 只能说"没测出差别"，
      // 这和 cppEloSvg 里"区间重叠 = 没测出差别"是同一套读法，不能两处说法打架。
      var sig = sm.lo > (sm.prev_elo !== undefined ? sm.prev_elo : sm.lo);
      var sigDown = sm.hi < (sm.prev_elo !== undefined ? sm.prev_elo : sm.hi);
      if (sig) { color = 'var(--ok)'; verdict = '（区间整体在旧值之上，是真的涨了）'; }
      else if (sigDown) { color = 'var(--bad, #e5534b)'; verdict = '（区间整体在旧值之下，退步了）'; }
      else { color = 'var(--warn)'; verdict = '（95% 区间与旧值重叠 ⇒ 这次没测出差别，别当成涨/跌）'; }
    }
    var html = 'gen_' + sm.gen + ' 当前 <b>' + eloSigned(sm.elo) + ' Elo</b>' +
      '（95% 区间 ' + eloSigned(sm.lo) + ' ~ ' + eloSigned(sm.hi) + '）';
    if (d !== null && d !== undefined && isFinite(Number(d))) {
      html += '<br>比上次同代 <b style="color:' + color + '">' + eloSigned(d) + ' Elo</b>' + verdict;
    } else {
      html += '<br><span class="muted">上次没有测过 gen_' + sm.gen + '，无从对比（下次再跑就能算出差值）</span>';
    }
    html += '<br><span class="muted">锚点 ' + (sm.anchor_name || ('gen_' + job.anchor)) + ' = 0 Elo · ' +
      sm.versions + ' 个版本 · 用时 ' + eloHMS(job.elapsed) + '</span>';
    // 这次只测了几代，曲线上的其它代是原样保留的旧值 —— 必须说出来，
    // 否则用户会以为整条曲线都刚重算过（第一版在这里直接覆盖，把 40 代碾成了 3 代）。
    if (job.merge_note) {
      html += '<br><span class="muted">' + String(job.merge_note) + '</span>';
    }
    if (res) res.innerHTML = html;
    hint.textContent = '';
    return;
  }

  if (job.error) {
    if (res) res.innerHTML = '<span style="color:var(--bad,#e5534b)">' + String(job.error) + '</span>';
    hint.textContent = '';
    return;
  }
  if (job.stage === 'idle') {
    if (res) res.innerHTML = '<span class="muted">还没有跑过对战。上面填规模，点「开始 Elo 对战」。</span>';
  }
}

function eloSetBusy(b){
  ELO_BUSY = b;
  var btn = document.getElementById('eloBtn');
  var stop = document.getElementById('eloStopBtn');
  if (btn) { btn.disabled = b; btn.textContent = b ? '对战中…' : '开始 Elo 对战'; }
  if (stop) stop.disabled = !b;
}

function eloPoll(){
  fetch('/api/elo-status', { cache: 'no-store' })
    .then(function(r){ return r.json(); })
    .then(function(j){
      eloPaintJob(j);
      eloSetBusy(!!j.running);
      if (j.running && !ELO_TIMER) {
        ELO_TIMER = setInterval(eloPoll, 2000);
      } else if (!j.running && ELO_TIMER) {
        clearInterval(ELO_TIMER); ELO_TIMER = null;
        // 跑完了：把 /api/cpp-state 重拉一次，曲线立刻换成新数据（不等下一个 1200ms 周期）
        if (typeof cppPoll === 'function') cppPoll();
      }
    })
    .catch(function(e){
      var hint = document.getElementById('eloHint');
      if (hint) hint.textContent = '读取 /api/elo-status 失败：' + e;
    });
}

function eloPreview(){
  // 预览：只让服务端算计划（几对、几局、多久），不启动进程。
  var body = { dry_run: true,
               recent: eloNum('eloRecent', 5),
               games:  eloNum('eloGames', 30),
               anchor: eloNum('eloAnchor', 0) };
  if (typeof cppDir === "string" && cppDir) body.dir = cppDir;
  fetch('/api/elo-run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify(body) })
    .then(function(r){ return r.json(); })
    .then(function(j){
      var hint = document.getElementById('eloHint');
      if (!hint) return;
      if (!j.ok) { hint.innerHTML = '<span style="color:var(--bad,#e5534b)">' + (j.errors || ['预览失败']).join('；') + '</span>'; return; }
      hint.innerHTML = '将打 <b>' + j.pairs + '</b> 对（' + (j.gens || []).length + ' 个版本：' +
        (j.gens || []).join(', ') + '），每对 ' + j.games + ' 局 = <b>' + j.total_games + '</b> 局，约 ' +
        eloHMS(j.est_seconds) + '。锚点 gen_' + j.anchor + ' = 0 Elo。' +
        (j.anchor_note ? '<br><span style="color:var(--warn,#d29922)">' + j.anchor_note + '</span>' : '');
    })
    .catch(function(){});
}

function eloRun(){
  if (ELO_BUSY) return;
  var body = { dry_run: false,
               recent: eloNum('eloRecent', 5),
               games:  eloNum('eloGames', 30),
               anchor: eloNum('eloAnchor', 0) };
  if (typeof cppDir === "string" && cppDir) body.dir = cppDir;
  eloSetBusy(true);
  fetch('/api/elo-run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify(body) })
    .then(function(r){ return r.json(); })
    .then(function(j){
      if (!j.ok) {
        eloSetBusy(false);
        var hint = document.getElementById('eloHint');
        if (hint) hint.innerHTML = '<span style="color:var(--bad,#e5534b)">' +
          (j.errors || ['启动失败']).join('；') + '</span>';
        return;
      }
      if (!ELO_TIMER) ELO_TIMER = setInterval(eloPoll, 2000);
      eloPoll();
    })
    .catch(function(e){
      eloSetBusy(false);
      var hint = document.getElementById('eloHint');
      if (hint) hint.textContent = '启动失败：' + e;
    });
}

function eloStop(){
  fetch('/api/elo-stop', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                           body: JSON.stringify({}) })
    .then(function(r){ return r.json(); })
    .then(function(j){
      eloSetBusy(false);
      if (ELO_TIMER) { clearInterval(ELO_TIMER); ELO_TIMER = null; }
      eloPoll();
    })
    .catch(function(){ eloSetBusy(false); });
}

// 每次 /api/cpp-state 回来都会调这里：填默认锚点、补出可用代数、把任务状态刷上去。
// 默认锚点只填一次（ELO_ANCHOR_SET），否则用户手动改过的值会被每 1.2 秒覆盖一次。
function eloPaint(s){
  if (!s) return;
  var d = s.elo_defaults || {};
  var anchorEl = document.getElementById('eloAnchor');
  if (anchorEl && !ELO_ANCHOR_SET && d.anchor) {
    anchorEl.value = d.anchor;
    ELO_ANCHOR_SET = true;
  }
  var hint = document.getElementById('eloHint');
  // 【为什么要判 ELO_BUSY 和 !running】/api/cpp-state 每 1.2 秒来刷一次。
  // 对战进行中或刚结束时，hint 正显示"打哪几代 / 错误原因"，
  // 这时被这条静态说明覆盖掉，用户会以为对战没在进行。
  // 也不能像第一版那样用 dataset.ready 只写一次 —— 那样手动点了"停止"后
  // 提示永远停在停止前的文字上（自检 §6 抓到的就是这个）。
  if (hint && !ELO_BUSY && !(s.elo_job || {}).running && !(s.elo_job || {}).error
      && d.gens && d.gens.length) {
    hint.textContent = '可用 gen_' + d.gens[0] + ' … gen_' + d.gens[d.gens.length - 1] +
      '（共 ' + d.gens.length + ' 个版本）· 默认锚点 gen_' + d.anchor +
      '（来源：' + (d.anchor_from || '—') + '，换锚点会让旧曲线失去可比性）';
  }
  if (s.elo_job) eloPaintJob(s.elo_job);
}

function cppEloSvg(elo, history){
  // 参考 KataGo 的 Elo 图：显式锚点 + 95% 置信区间竖须 + x 轴按代。
  // 读法上最重要的一条：两个版本如果区间大面积重叠，就不该说"谁更强"——这正是我们这次
  // 200 局复核（gen43 0.535，区间 -24~+43）得到结论的图形版本。
  // 【2026-10-29 加】配方断点：每条竖虚线 = 从这一代起换了配方。跨过虚线的那一步里
  // 包含"换配方"本身，所以它不能算训练变强，虚线两侧的点也不该直接互比 —— 只和同一段里的比。
  // 【2026-10-31 重画】只动画法、不动口径：
  //   ① 95% 区间从"一堆散竖线"改成一条连续的浅色带（band），区间宽窄一眼可见；
  //   ② 点加白色光圈压在带上，不再和竖须糊成一团；每点 hover 出完整数值；
  //   ③ x 轴补中间刻度；y 轴刻度字号与网格对齐；
  //   ④ 加最小二乘趋势线（长虚线）+ 图内标注 Elo/代 —— 单点噪声大，趋势才是能读的东西；
  //   ⑤ SVG 改 width:100% + viewBox，不再固定 900px。
  var vs = (elo.versions || []).filter(function(v){ return isFinite(Number(v.elo)); });
  if (vs.length < 2) return '';
  var W = 1000, H = 252, L = 54, R = 16, T = 42, B = 32;
  var g0 = Number(vs[0].gen), g1 = Number(vs[vs.length - 1].gen);
  var span = (g1 - g0) || 1;
  var los = vs.map(function(v){ return Number(v.lo); }).filter(isFinite);
  var his = vs.map(function(v){ return Number(v.hi); }).filter(isFinite);
  var lo = los.length ? Math.min.apply(null, los) : 0;
  var hi = his.length ? Math.max.apply(null, his) : 0;
  lo = Math.min(lo, 0); hi = Math.max(hi, 0);
  var pad = Math.max(25, (hi - lo) * 0.10);
  lo -= pad; hi += pad;
  function X(g){ return L + (W - L - R) * (g - g0) / span; }
  function Y(e){ return (H - B) - (H - B - T) * (e - lo) / (hi - lo); }
  var y0 = Y(0), i, j;
  // ---- 横向网格 + y 刻度 ----
  var grid = '';
  var step = (hi - lo) > 600 ? 200 : ((hi - lo) > 300 ? 100 : 50);
  for (i = Math.ceil(lo / step) * step; i <= hi; i += step) {
    var gy = Y(i).toFixed(1);
    grid += '<line x1="' + L + '" y1="' + gy + '" x2="' + (W - R) + '" y2="' + gy +
            '" stroke="var(--hair)" stroke-width="0.7" />' +
            '<text x="' + (L - 8) + '" y="' + gy + '" font-size="9.5" fill="var(--mut)" text-anchor="end" ' +
            'dominant-baseline="middle">' + (i > 0 ? '+' + i : i) + '</text>';
  }
  // ---- x 轴刻度（原来只有首尾两个数字，中间那段没法定位） ----
  var xn = Math.max(2, Math.min(8, Math.round((W - L - R) / 120))), seenX = {}, xtick = '';
  for (i = 0; i < xn; i++) {
    var gx = Math.round(g0 + span * i / (xn - 1));
    if (seenX[gx]) continue;
    seenX[gx] = 1;
    var xx = X(gx).toFixed(1);
    xtick += '<line x1="' + xx + '" y1="' + (H - B) + '" x2="' + xx + '" y2="' + (H - B + 3.5) +
             '" stroke="var(--line2)" stroke-width="0.8" />' +
             '<text x="' + xx + '" y="' + (H - B + 16) + '" font-size="9.5" fill="var(--mut)" text-anchor="middle">' + gx + '</text>';
  }
  // ---- 配方分段：隔段底色 + 换配方处的竖虚线 ----
  // 按段读图：两条虚线之间是"同一个配方连续训出来的"，只有段内的涨跌才算训练本身。
  var eps = cppRecipeEpochs(history);
  function epLabel(ep){
    // 老日志只推断得出"每代多少局"，带上局数才能区分相邻的两段（否则 gen 5 和 gen 8 都显示"配方?"）
    return ep.label + (ep.exact ? '' : (ep.games ? ('·' + ep.games + '局') : ''));
  }
  var band = '', cut = '', cutX = [null, null], pending = [];
  for (i = 0; i < eps.length; i++) {
    var ep = eps[i], nx = eps[i + 1] || null;
    var ba = Math.max(ep.from, g0), bb = nx ? Math.min(nx.from, g1) : g1;
    if (i % 2 === 1 && bb > ba) {
      band += '<rect x="' + X(ba).toFixed(1) + '" y="' + T + '" width="' + (X(bb) - X(ba)).toFixed(1) +
              '" height="' + (H - B - T) + '" fill="var(--fg)" opacity="0.05" />';
    }
    if (i === 0) continue;
    if (ep.from > g1) { pending.push(ep); continue; }   // 断点已在 Elo 覆盖范围之外：只在文字里说
    var cx = X(ep.from);
    var row = (cutX[0] !== null && Math.abs(cx - cutX[0]) < 50) ? 1 : 0;   // 两条断点挨太近就错开一行
    cutX[row] = cx;
    var tip = (ep.exact ? ('从 gen ' + ep.from + ' 起用配方 ' + ep.rid) : ('gen ' + ep.from + ' 起每代数据量变了（推断为换配方）')) +
              (ep.games ? ('，每代 ' + ep.games + ' 局') : '') + '。跨过这一步的是配方，不是模型变强。';
    cut += '<line x1="' + cx.toFixed(1) + '" y1="' + T + '" x2="' + cx.toFixed(1) + '" y2="' + (H - B) +
           '" stroke="var(--fg)" stroke-width="1" opacity="0.5" stroke-dasharray="4 3" />' +
           '<text x="' + cx.toFixed(1) + '" y="' + (row ? 28 : 13) + '" font-size="9" fill="var(--fg2)" ' +
           'text-anchor="middle"><title>' + tip + '</title>' + epLabel(ep) + '</text>';
  }
  function epochOf(g){
    var hit = null;
    for (var k = 0; k < eps.length; k++) { if (eps[k].from <= g) hit = eps[k]; else break; }
    return hit;
  }
  // ---- 95% 区间：连续浅带（上沿正序 + 下沿倒序闭合成多边形） ----
  var ribbon = '';
  var haveCI = vs.every(function(v){ return isFinite(Number(v.lo)) && isFinite(Number(v.hi)); });
  if (haveCI) {
    var up = vs.map(function(v){ return X(Number(v.gen)).toFixed(1) + ',' + Y(Number(v.hi)).toFixed(1); });
    var dn = vs.map(function(v){ return X(Number(v.gen)).toFixed(1) + ',' + Y(Number(v.lo)).toFixed(1); }).reverse();
    ribbon = '<polygon points="' + up.concat(dn).join(' ') + '" fill="var(--c3)" opacity="0.10" />';
  }
  // ---- 趋势线：最小二乘拟合（单点 ±45 Elo 的噪声下，只有斜率能读） ----
  var trend = '', slopeTxt = '';
  if (vs.length >= 3) {
    var n = vs.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
    vs.forEach(function(v){ var g = Number(v.gen), e = Number(v.elo); sx += g; sy += e; sxx += g * g; sxy += g * e; });
    var den = n * sxx - sx * sx;
    if (Math.abs(den) > 1e-9) {
      var sl = (n * sxy - sx * sy) / den, ic = (sy - sl * sx) / n;
      var tyA = Y(sl * g0 + ic), tyB = Y(sl * g1 + ic);
      // 拟合线可能穿出画布（样本少时），裁掉出界的部分再画
      var tA = Math.max(T, Math.min(H - B, tyA)), tB = Math.max(T, Math.min(H - B, tyB));
      trend = '<line x1="' + L + '" y1="' + tA.toFixed(1) + '" x2="' + (W - R) + '" y2="' + tB.toFixed(1) +
              '" stroke="var(--c3)" stroke-width="1.2" opacity="0.5" stroke-dasharray="7 5" />';
      slopeTxt = '趋势 ' + (sl >= 0 ? '+' : '−') + Math.abs(sl).toFixed(1) + ' Elo/代';
    }
  }
  // ---- 点 + 竖须 ----
  var whisk = '', dots = '';
  vs.forEach(function(v){
    var g = Number(v.gen), x = X(g), yv = Y(Number(v.elo));
    var a = Y(Number(v.lo)), b = Y(Number(v.hi));
    if (isFinite(a) && isFinite(b)) {
      whisk += '<line x1="' + x.toFixed(1) + '" y1="' + b.toFixed(1) + '" x2="' + x.toFixed(1) +
               '" y2="' + a.toFixed(1) + '" stroke="var(--c3)" stroke-width="1" opacity="0.38" />';
    }
    var games = Number(v.games); var weak = isFinite(games) && games < 40;
    var epo = epochOf(g);
    var vtip = 'gen ' + g + '：' + (Number(v.elo) >= 0 ? '+' : '') + Number(v.elo).toFixed(0) + ' Elo' +
               '（95% 区间 ' + Number(v.lo).toFixed(0) + ' ~ ' + Number(v.hi).toFixed(0) + '）' +
               (isFinite(games) ? (' · ' + games + ' 局') : '') + (epo ? (' · ' + epo.label) : '');
    dots += '<circle cx="' + x.toFixed(1) + '" cy="' + yv.toFixed(1) + '" r="5.2" fill="#fff" opacity="0.92" />' +
            '<circle cx="' + x.toFixed(1) + '" cy="' + yv.toFixed(1) + '" r="' + (weak ? 2.6 : 3.3) + '" ' +
            'fill="' + (weak ? '#fff' : 'var(--c3)') + '" stroke="var(--c3)" stroke-width="1.4" />' +
            '<circle cx="' + x.toFixed(1) + '" cy="' + yv.toFixed(1) + '" r="9" fill="none" pointer-events="all">' +
            '<title>' + vtip + '</title></circle>';
  });
  // ---- 【2026-10-30 加】自动标注 Elo 前三高 ----
  // 用户要求「给 elo 曲线上自动标注前三高的模型」，方便一眼看出哪几代最强。
  // 但【只标名次会误导】：实测发现 gen_060 用 12 局就冲到了第一（+402.8），
  // 补测到 36 局后它变成 17 胜 19 负、对每个对手都落后 —— 小样本时 Elo 高低基本是抽签。
  // 所以每个标签都带上「局数」，样本不足的用警告色，让"最强"和"最可信"在图上就分得开。
  //
  // 【阈值怎么定】主曲线口径是「链式相邻 2 条边 + 锚点 1 条边，每边 12 局」：
  //   · 中间代 = 3 条边 = 36 局（满样本）
  //   · 两端点（gen_002 / gen_061）= 只有 2 条边（相邻 + 锚点）= 24 局（也是满样本）
  // 一开始只用了 36 一个数字，端点会被误标黄 —— 而端点本来就少一条边，
  // 拿中间代的标准去卡它属于口径用错。所以按"它应该有几条边"来判：
  // 用全图最大边数（3）当基准，某种点不足则按比例放宽。这里直接看实际值：
  // 该点局数 >= 12 * 它的边数 就算满样本（每条边 12 局）。
  // 拿不到边数时退回 36（保守）。
  var GAMES_PER_EDGE = 12;
  var GAMES_ENOUGH = 36;   // 兜底：中间代的满样本局数
  var ranked = vs.slice().sort(function(a, b){ return Number(b.elo) - Number(a.elo); });
  var topN = ranked.slice(0, 3);
  var medal = ['①', '②', '③'];
  var badge = '', badgeTxt = '', usedLabelX = [];
  topN.forEach(function(v, k){
    var g = Number(v.gen), el = Number(v.elo);
    var x = X(g), yv = Y(el);
    var games = Number(v.games);
    // 每个点的"满样本局数" = 12 × 它参与的边数。边数 = 相邻两条 + 锚点一条（端点只有一条相邻）。
    // elo.json 的 edge 里 a/b 是版本名，这里现算，避免把边数硬编码成 3 又把端点标黄。
    var nEdges = 0;
    (elo.edges || []).forEach(function(e){ if (e.a === v.name || e.b === v.name) nEdges++; });
    var need = nEdges > 0 ? (GAMES_PER_EDGE * nEdges) : GAMES_ENOUGH;
    var weak = isFinite(games) && games < need;
    var col = weak ? 'var(--warn)' : 'var(--c3)';
    var tip = 'Elo 第 ' + (k + 1) + ' 高：gen_' + g + '，' + (el >= 0 ? '+' : '') + el.toFixed(0) + ' Elo' +
              '（95% 区间 ' + Number(v.lo).toFixed(0) + ' ~ ' + Number(v.hi).toFixed(0) + '）' +
              (isFinite(games) ? (' · 累计 ' + games + ' 局') : '') +
              (weak ? (' · ⚠ 样本不足 ' + need + ' 局，这个名次不可靠') : ' · 样本足够');
    // 圆圈（带名次）恒定画；文字标签只在"和已画的标签不打架"时画（第 1 名永远画）
    var clash = false;
    usedLabelX.forEach(function(px){ if (Math.abs(px - x) < 58) clash = true; });
    var wantTxt = (k === 0) || !clash;
    if (wantTxt) usedLabelX.push(x);
    var above = yv > (T + 30);
    var ty = above ? (yv - 13) : (yv + 19);
    var anchor = (x > W - R - 70) ? 'end' : 'middle';
    badge += '<circle cx="' + x.toFixed(1) + '" cy="' + yv.toFixed(1) + '" r="' + (weak ? 6.5 : 6) +
             '" fill="#fff" stroke="' + col + '" stroke-width="1.4" />' +
             '<text x="' + x.toFixed(1) + '" y="' + (yv + 3).toFixed(1) + '" font-size="8.5" fill="' + col +
             '" text-anchor="middle">' + medal[k] + '</text>';
    if (wantTxt) {
      badge += '<text x="' + x.toFixed(1) + '" y="' + ty.toFixed(1) + '" font-size="9.5" fill="' + col +
               '" text-anchor="' + anchor + '" font-weight="600">' +
               '<title>' + tip + '</title>gen_' + g +
               (isFinite(games) && weak ? ('（' + games + '局）') : '') + '</text>';
    }
    badgeTxt += '<span title="' + tip + '"' + (weak ? ' style="color:var(--warn)"' : '') + '>' +
                medal[k] + ' gen_' + g + ' ' + (el >= 0 ? '+' : '') + el.toFixed(0) +
                (weak ? ('<b> ⚠仅' + games + '局</b>') : '') + '</span>' +
                (k < topN.length - 1 ? '　' : '');
  });
  // 图下方一行文字版排名：SVG 里的标签空间有限，这里把局数讲清楚
  var badgeLine = topN.length ? ('<div class="small" style="margin:2px 0 6px;line-height:1.6">' +
    '<b>Elo 前三高</b>：' + badgeTxt +
    '<span class="muted">（⚠ 标黄 = 累计对局没打满该点应有的场次，名次可能只是小样本抽签，别据此判定它最强）</span></div>') : '';
  var line = vs.map(function(v){ return X(Number(v.gen)).toFixed(1) + ',' + Y(Number(v.elo)).toFixed(1); }).join(' ');
  var gl = vs.map(function(v){ return Number(v.games); }).filter(isFinite);
  var gmin = gl.length ? Math.min.apply(null, gl) : 0;
  var gmax = gl.length ? Math.max.apply(null, gl) : 0;
  var gtxt = (gmin === gmax) ? (gmin + ' 局') : (gmin + '–' + gmax + ' 局');
  var top = vs[vs.length - 1];
  // 配方分段图例：直接把"哪几代是哪个配方"写出来，不用再去翻 train_log
  var eraTxt = eps.map(function(ep){
    var nm = ep.exact ? cppRecipeName(ep.rid) : '';
    var gs = (ep.from === ep.to) ? ('gen ' + ep.from) : ('gen ' + ep.from + '–' + ep.to);
    return '<span title="' + (ep.exact ? ('--recipe ' + ep.rid) : '老日志没有 recipe_id，按每代样本数跨 15000 推断') + '">' +
           gs + ' ' + epLabel(ep) + (nm ? ('「' + nm + '」') : '') + '</span>';
  }).join(' · ');
  var inferredTo = 0;
  eps.forEach(function(ep){ if (!ep.exact) inferredTo = Math.max(inferredTo, ep.to); });
  var pendTxt = pending.length ? ('<span style="color:var(--warn)">' +
    pending.map(function(ep){ return ep.label + ' 自 gen ' + ep.from + ' 起'; }).join('、') +
    '，Elo 图里还没有这些代（重跑 <code>gk elo</code> 后才会出现）</span>') : '';
  var recipeLine = eps.length ? ('<div class="small muted" style="margin:-2px 0 6px;line-height:1.6">' +
    '配方分段（竖虚线 = 从这里换成新配方，跨线那一步不能算训练变强）：' + eraTxt +
    (inferredTo ? (' · <span style="opacity:.75">gen≤' + inferredTo + ' 没有 recipe_id，段是按每代样本数推断的</span>') : '') +
    (pendTxt ? (' · ' + pendTxt) : '') +
    '</div>') : '';
  // 趋势线标注：贴在趋势线右端，往左偏一点免得压到 y 轴末位刻度
  var slopeMark = slopeTxt ? ('<text x="' + (W - R - 4) + '" y="' + (T + 12) + '" font-size="9.5" fill="var(--mut)" ' +
    'text-anchor="end">' + slopeTxt + '（虚线）</text>') : '';
  return '<div><div class="small muted" style="margin-bottom:4px">' +
    '<span style="display:inline-block;width:9px;height:9px;background:var(--c3);margin-right:6px"></span>' +
    '锚点 ' + (elo.anchor_name || ('gen_' + vs[0].gen)) + ' = 0 Elo · ' + vs.length + ' 个版本 · ' +
    '最新 ' + top.gen + ' 代 ' + (Number(top.elo) >= 0 ? '+' : '') + Number(top.elo).toFixed(0) + ' Elo' +
    ' · 各版本累计对局 ' + gtxt + ' · 实心=样本够，空心=样本偏少 · 浅带=95% 区间' +
    (elo.generated ? (' · ' + elo.generated) : '') + '</div>' + badgeLine + recipeLine +
    '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" ' +
    'style="display:block;width:100%;height:auto;font-family:var(--mono)">' +
    band + grid + xtick + cut + ribbon + trend +
    '<line x1="' + L + '" y1="' + y0.toFixed(1) + '" x2="' + (W - R) + '" y2="' + y0.toFixed(1) +
    '" stroke="var(--fg)" stroke-width="1" opacity="0.5" stroke-dasharray="6 4" />' +
    '<text x="' + (L + 4) + '" y="' + (y0 - 5).toFixed(1) + '" font-size="9" fill="var(--mut)">锚点 = 0</text>' +
    '<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) +
    '" stroke="var(--line2)" stroke-width="0.9" />' +
    '<polyline fill="none" stroke="var(--c3)" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" points="' + line + '" />' +
    whisk + dots + badge + slopeMark +
    '<text x="' + (W - R + 4) + '" y="' + (H - B + 4) + '" font-size="9" fill="var(--mut)" text-anchor="start">代</text>' +
    '<text x="' + (L - 8) + '" y="' + (T - 26) + '" font-size="9" fill="var(--mut)" text-anchor="end">Elo</text>' +
    '</svg></div>';
}

/* 【2026-10-31 加】训练历史卡折叠开关。
   为什么放在这里而不是用 <details>：页面每 1.2 s 原地重画 cppHist 的内容，
   <details> 的 open 属性会被重画冲掉（重设 innerHTML 不影响 details 本身，但整块卡片
   一旦被 patchNode 换掉就丢状态）。用 localStorage 存开关 + 每次渲染后重新贴一次状态，
   无论卡片被怎么重建都能保持。 */
function cppHistOpen(){
  try { return localStorage.getItem('gkCppHistOpen') === '1'; } catch (e) { return false; }
}
function cppHistApply(){
  var body = document.getElementById('cppHistBody');
  var arw = document.getElementById('cppHistArrow');
  var ttl = document.getElementById('cppHistTitle');
  var hint = document.getElementById('cppHistHint');
  if (!body || !ttl) { return; }
  var open = cppHistOpen();
  body.style.display = open ? '' : 'none';
  if (arw) { arw.textContent = open ? '▾' : '▸'; }
  if (hint) { hint.textContent = open ? '（点击折叠）' : '（点击展开）'; }
  ttl.setAttribute('title', open ? '点击折叠这张表' : '点击展开这张表');
}
function cppHistInit(){
  var ttl = document.getElementById('cppHistTitle');
  if (!ttl || ttl._bound) { return; }
  ttl._bound = true;
  function flip(){
    try { localStorage.setItem('gkCppHistOpen', cppHistOpen() ? '0' : '1'); } catch (e) {}
    cppHistApply();
  }
  ttl.addEventListener('click', flip);
  ttl.addEventListener('keydown', function(ev){
    if (ev.key === 'Enter' || ev.key === ' ' || ev.key === 'Spacebar') { ev.preventDefault(); flip(); }
  });
  cppHistApply();
}

function cppPoll(){
  var q = '/api/cpp-state' + (cppDir ? ('?dir=' + encodeURIComponent(cppDir)) : '');
  fetch(q, { cache: 'no-store' })
    .then(function(r){ if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function(s){ cppState = s; if (s.dir) cppDir = s.dir; cppRender(s); })
    .catch(function(e){
      var t = document.getElementById('cppTail');
      if (t) t.textContent = '读取 /api/cpp-state 失败：' + e +
        '\n（看板服务没起？重新打开 serve_dashboard.py 即可）';
    });
}

function cppRender(s){
  var kp = document.getElementById('cppKpis');
  if (!kp) return;
  // 【2026-10-26 加】数据没变就一个 DOM 都不碰。
  // 这里每个容器都是整块 innerHTML 重建，1.2 s 一次等于持续"闪"；
  // 用整份快照的签名当门槛，只有数字真的变了才重画。
  var sig = '';
  try { sig = JSON.stringify(s); } catch (e) { sig = ''; }
  if (sig && sig === cppRender._sig) return;
  cppRender._sig = sig;
  var ph = s.phase || {};
  var running = !!s.running;
  var last = s.last || null;
  var tot = s.totals || {};
  var gens = s.gens || [];
  var runTag = running ? '训练中' : '空闲';
  var genTxt = (last && last.gen !== undefined) ? ('gen ' + last.gen) : '—';
  var target = (ph.total ? ph.total : (s.next_gen || 1));
  kp.innerHTML =
    cppKpi('状态', runTag + (running && ph.text ? '<span class="small muted" style="font-size:11px;margin-left:6px">' + ph.text + '</span>' : '')) +
    cppKpi('当前代', genTxt) +
    cppKpi('目标代', String(target)) +
    cppKpi('已落盘', gens.length ? (gens[0] + '–' + gens[gens.length - 1]) : '—') +
    cppKpi('累计样本', cppInt(tot.samples)) +
    cppKpi('累计局数', cppInt(tot.games)) +
    cppKpi('自对弈累计', cppDur(tot.selfplay_s)) +
    cppKpi('训练累计', cppDur(tot.train_s)) +
    cppKpi('平均每代', cppDur(tot.avg_gen_s));

  var bar = document.getElementById('cppBar');
  if (bar) {
    // 只有在"知道目标"时才画进度：跑起来后从 stdout 的 `=== gen k/M ===` 拿 k/M；
    // 空闲时留空 —— 拿已落盘代数硬凑一个百分比只会误导（删过旧代就会算歪）。
    var pct = -1;
    if (ph.gen && ph.total) pct = Math.max(0, Math.min(100, Math.round(ph.gen * 100 / ph.total)));
    else if (running && gens.length && s.next_gen) {
      pct = Math.max(0, Math.min(100, Math.round(gens.length * 100 / s.next_gen)));
    }
    bar.innerHTML = (pct >= 0)
      ? '<div class="bar"><div class="fill" style="width:' + pct + '%"></div></div>'
      : '';
  }

  var line = document.getElementById('cppRunLine');
  if (line) {
    var runs = s.runs || [];
    var html = '运行目录 <code>' + (s.dir || '（还没有 C++ 产物）') + '</code>';
    if (runs.length > 1) {
      html += ' · 切换 <select id="cppSel">' + runs.map(function(r){
        return '<option value="' + r + '"' + (r === s.dir ? ' selected' : '') + '>' + r + '</option>';
      }).join('') + '</select>';
    }
    if (s.pids && s.pids.length) html += ' · PID ' + s.pids.join(', ');
    html += ' · 下一代 ' + (s.next_gen || 1);
    if (last && last.buffer_total !== undefined) html += ' · 回放池 ' + cppInt(last.buffer_total) + ' 条';
    line.innerHTML = html;
    var sel = document.getElementById('cppSel');
    if (sel) sel.onchange = function(){
      cppDir = sel.value;
      try { localStorage.setItem('gkCppDir', cppDir); } catch (e) {}
      cppPoll();
    };
  }

  var cv = document.getElementById('cppCurve');
  if (cv) {
    var hrows = s.history || [];
    if (hrows.length < 2) {
      cv.innerHTML = hrows.length ? '目前只有 1 代，曲线至少要 2 代' : '还没有数据';
    } else {
      // 两张损失图并排：每张 flex:1 1 330px，窄屏（<660px）自动折成上下两行
      cv.innerHTML = '<div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start">' +
        cppSpark(hrows, 'ce', '策略 CE') + cppSpark(hrows, 'mse', '价值 MSE') + '</div>';
    }
  }

  var eloBox = document.getElementById('cppElo');
  if (eloBox) {
    var eloData = s.elo;
    if (!eloData || !eloData.versions || eloData.versions.length < 2) {
      eloBox.innerHTML = '还没有 Elo 数据（跑一次 <code>gk.exe elo --dir ' + (s.dir || 'model/run_v2') +
        ' --gens ...</code> 生成 elo.json）';
    } else {
      eloBox.innerHTML = cppEloSvg(eloData, s.history);
    }
  }
  eloPaint(s);

  var hist = document.getElementById('cppHist');  if (hist) {
    var rows = (s.history || []).slice(-40).reverse();
    if (!rows.length) {
      hist.innerHTML = '还没有训练记录（<code>' + (s.dir || 'C++\\model') + '\\train_log.jsonl</code> 为空）';
    } else {
      var h = '<table><tr>' +
              '<th title="第几代">代</th>' +
              '<th title="这一代自对弈完成的局数（--games）">局</th>' +
              '<th title="这一代产出的决策样本条数（≈ 局数 × 每局手数）">样本</th>' +
              '<th title="这一代训练步数（--min-steps，或按新样本量算出的更大值）">步</th>' +
              '<th title="自对弈采样耗时（秒）">自对弈</th>' +
              '<th title="反向传播耗时（秒）">训练</th>' +
              '<th title="每训练步的毫秒数 = 训练毫秒 / 步数">ms/步</th>' +
              '<th title="训练目标值 = ce + mse。只反映拟合这批样本的程度，不能当棋力指标">loss</th>' +
              '<th title="策略头交叉熵：预测分布 vs 这一代 MCTS 访问分布（软目标）。目标越尖越大，跨配方不可比">ce</th>' +
              '<th title="价值头平方误差：(value - 终局结果)^2。目标随策略漂移，会自己上下晃，高不代表弱">mse</th>' +
              '<th title="这一代实际使用的学习率（与面板配方不符 = 被棘轮/退火改过）">lr</th>' +
              '<th title="缓冲区总条数 buffer_total = 最近 --buffer-gens 代的样本累加，先涨满后平台化">回放池</th>' +
              '<th title="与当前 champion 的 20 局门控：胜局/总局（胜率）">门控</th>' +
              '<th title="门控是否通过：胜率必须超过 0.55 才采纳">采纳</th></tr>';
      rows.forEach(function(r){
        var msp = (r.steps && r.train_s) ? (r.train_s * 1000 / r.steps) : null;
        var gate = r.gated ? ((r.gate_wins || 0) + '/' + (r.gate_games || 0) +
                   ' (' + (r.gate_wr ? Number(r.gate_wr).toFixed(3) : '—') + ')') : '—';
        var acc = r.gated ? (r.gate_accepted ? '<b>采纳</b>' : '拒绝') : '—';
        h += '<tr><td class="num">' + cppN(r.gen) + '</td><td class="num">' + cppN(r.games) +
             '</td><td class="num">' + cppInt(r.samples) + '</td><td class="num">' + cppN(r.steps) +
             '</td><td class="num">' + cppN(r.selfplay_s, 1) + ' s</td><td class="num">' +
             cppN(r.train_s, 1) + ' s</td><td class="num">' + cppN(msp, 1) + '</td>' +
             '<td class="num">' + cppN(r.loss, 5) + '</td><td class="num">' + cppN(r.ce, 5) +
             '</td><td class="num">' + cppN(r.mse, 5) + '</td><td class="num">' +
             (r.lr ? Number(r.lr).toExponential(1) : '—') + '</td><td class="num">' +
             cppInt(r.buffer_total) + '</td><td class="num">' + gate + '</td><td>' + acc + '</td></tr>';
      });
      hist.innerHTML = h + '</table>';
      // 折叠时标题行仍要能看出最新一代的状态，否则折叠就等于"信息被藏起来了"
      var r0 = rows[0] || null;
      var peek = document.getElementById('cppHistPeek');
      if (peek) {
        peek.innerHTML = r0 ? ('· ' + rows.length + ' 行 · 最新 gen ' + cppN(r0.gen) +
          ' · loss ' + cppN(r0.loss, 4) +
          (r0.gated ? (' · 门控 ' + (r0.gate_wins || 0) + '/' + (r0.gate_games || 0) +
                       (r0.gate_accepted ? ' 采纳' : ' 拒绝')) : '')) : '';
      }
    }
  }
  cppHistInit();

  var tail = document.getElementById('cppTail');
  if (tail) {
    var lines = s.tail || [];
    tail.textContent = lines.length ? lines.join('\n') : '（没有 loop_stdout.log：当前没在训练时不会写这个文件）';
  }

  var runsBox = document.getElementById('cppRuns');
  if (runsBox) {
    var rd = s.runs_detail || [];
    if (!rd.length) {
      rd = (s.runs || []).map(function(r){ return {dir: r}; });   // 老服务端兜底
    }
    if (!rd.length) {
      runsBox.innerHTML = '还没有 C++ 训练产物（<code>C++\\model</code> 下没有 run 目录）';
    } else {
      var rh = '<table><tr><th>运行目录</th><th>角色</th><th>全局第几代</th><th>代数</th>' +
               '<th>自对弈局数</th><th>训练步</th><th>样本数</th><th>自对弈累计</th>' +
               '<th>训练累计</th><th>说明</th></tr>';
      rd.forEach(function(r){
        var hm = cppDur(r.selfplay_s);
        var tm = cppDur(r.train_s);
        var role = r.role || '（未标注）';
        var star = r.recommended ? ' ★' : '';
        var roleCls = (role.indexOf('主线') >= 0) ? ' style="color:#7ee787;font-weight:600"' :
                      ((role.indexOf('损坏') >= 0) ? ' class="small muted"' : '');
        rh += '<tr><td><code>' + r.dir + '</code>' + star + '</td>' +
              '<td' + roleCls + '>' + role + '</td><td class="num">' +
              cppN(r.total_gen || r.last_gen) + '</td><td class="num">' +
              cppN(r.gens !== undefined ? r.gens : r.gen_files) + '</td><td class="num">' +
              cppInt(r.games) + '</td><td class="num">' + cppInt(r.steps) + '</td><td class="num">' +
              cppInt(r.samples) + '</td><td class="num">' + hm + '</td><td class="num">' + tm +
              '</td><td class="small muted">' + (r.note || '') + '</td></tr>';
      });
      runsBox.innerHTML = rh + '</table>';
    }
  }

  var champ = document.getElementById('cppChamp');
  if (champ) {
    var b = s.best;
    if (!b) {
      champ.innerHTML = '这个 run 还没有 <code>best.bin</code>' +
        '（门控第一次通过、或训练跑到结束时会写）';
    } else {
      var when = b.mtime ? new Date(b.mtime * 1000).toLocaleString('zh-CN') : '—';
      champ.innerHTML = '<code>' + b.path + '</code> · ' + b.mb + ' MB · 最近写入 ' + when +
        (b.gen ? (' · 对应第 ' + b.gen + ' 代') : '');
    }
  }
}

// 【2026-10-30 加】Elo 对战按钮的事件绑定。
// 放在这个 IIFE 外面单独绑定，而不是塞进 verInit：verInit 只在启动时跑一次，
// 而按钮在 DOM 里是静态的 —— 两者时机不同，混在一起容易漏绑（尤其切到 cpp 版之后）。
(function eloInit(){
  var btn  = document.getElementById('eloBtn');
  var stop = document.getElementById('eloStopBtn');
  var rec  = document.getElementById('eloRecent');
  var gm   = document.getElementById('eloGames');
  if (btn)  btn.onclick  = eloRun;
  if (stop) stop.onclick = eloStop;
  // 改规模就重新预览一次计划 —— 用户最想立刻看到的答案是"这些要打多久"
  if (rec)  rec.onchange  = eloPreview;
  if (gm)   gm.onchange   = eloPreview;
  var an = document.getElementById('eloAnchor');
  if (an) an.onchange = function(){ ELO_ANCHOR_SET = true; eloPreview(); };
  // 页面一加载就问一次状态：可能上次对战还在跑（服务那边是后台任务）
  eloPoll();
})();

(function verInit(){
  try { cppDir = localStorage.getItem('gkCppDir') || ''; } catch (e) {}
  var saved = 'python';
  try { saved = localStorage.getItem('gkVer') || 'python'; } catch (e) {}
  // 也支持用链接直接开某一版：...?ver=cpp / ...?ver=python（方便分享/截图）
  var qm = (location.search.match(/[?&]ver=(python|cpp)\b/) || [])[1];
  if (qm) saved = qm;
  var bPy = document.getElementById('verPy'), bCpp = document.getElementById('verCpp');
  if (bPy) bPy.onclick = function(){ verApply('python'); };
  if (bCpp) bCpp.onclick = function(){ verApply('cpp'); };
  verApply(saved, false);
  cppHistInit();          // 训练历史卡：先按上次的折叠状态摆好，避免展开的表闪一下才收起
  if (VER === 'cpp') { cppPoll(); if (!cppTimer) cppTimer = setInterval(cppPoll, 1200); }
})();
</script>
</div>
</body></html>
"@
}

while ($true) {
    $s = Get-State
    if ($Html) {
        $path = if ([System.IO.Path]::IsPathRooted($Html)) { $Html } else { Join-Path $PSScriptRoot $Html }
        # 原子替换：先写 .tmp 再改名 —— 否则浏览器正好在 Set-Content 清空/写入之间
        # 刷新，会看到一页空白（跟 resume.pt 用同样的纪律）
        $tmp = "$path.tmp"
        Format-Html $s | Set-Content -Path $tmp -Encoding utf8
        Move-Item -Force $tmp $path
        if (-not $Once) { Write-Output ("[{0}] 已刷新 {1}" -f (Get-Date -Format 'HH:mm:ss'), $path) }
    } else {
        # Clear-Host 在非交互宿主（重定向/无控制台）里会抛"句柄无效"，包一层
        try { Clear-Host } catch { }
        Format-Console $s | ForEach-Object { Write-Output $_ }
        Write-Output ""
        Write-Output "（每 $Interval 秒刷新；Ctrl+C 退出）"
    }
    # 【2026-10-14 加】同一份 $s 再落一份 JSON，给 serve_dashboard.py 的 GET /api/state 用。
    # 为什么不让 Python 侧自己读文件重算：那等于**第二套口径**，和页面迟早漂移
    # （看板刚在 2026-10-13 因为"每代耗时跨 run 混排"报错过 ETA，见 NOTES.md §23）。
    # 这里序列化的是**页面用的同一个对象**，不可能对不上。
    # 原子替换 + 深度够（$s 是嵌套 pscustomobject / 数组），并显式 -Compress 省空间。
    if ($Json) {
        $jp = if ([System.IO.Path]::IsPathRooted($Json)) { $Json } else { Join-Path $PSScriptRoot $Json }
        try {
            $jtmp = "$jp.tmp"
            # -Depth 8：$s 最深是 results[].detail 那一层，6 够用，留 2 层余量
            $s | ConvertTo-Json -Depth 8 -Compress | Set-Content -Path $jtmp -Encoding utf8
            Move-Item -Force $jtmp $jp
        } catch {
            # 落盘失败不能影响看板本身：页面已经写完了，这里只提示
            Write-Output ("[{0}] /api/state 快照写入失败：{1}" -f (Get-Date -Format 'HH:mm:ss'), $_.Exception.Message)
        }
    }
    if ($Once -gt 0) { break }
    if ($MaxMinutes -gt 0 -and ((Get-Date) - $started).TotalMinutes -ge $MaxMinutes) {
        Write-Output ("[{0}] 到 -MaxMinutes {1}，看板退出" -f (Get-Date -Format 'HH:mm:ss'), $MaxMinutes)
        break
    }
    Start-Sleep -Seconds $Interval
}
