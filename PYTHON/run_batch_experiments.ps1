# run_batch_experiments.ps1 —— 排队跑两个"批量 × 虚拟损失 / 模拟数"的同权重对照实验。
#
# 背景（NOTES §15.7）：`probe_batch_scan.py` 显示 `batch_mcts` 16→64 能让自对弈吞吐 +59%，
# 但同权重对照打到 17:3（batch 64 侧胜率仅 0.15）—— **固定模拟数下放大批量会让棋力崩**。
# 机制假说：`virtual_loss=3.0` 在 batch=64 时同时压住 64 个叶子，选点被摊平。
# 所以正确的做法是"放大批量 + 调小虚拟损失"，或者"放大批量 + 更多模拟"。
#
# 两个实验（都是同一个模型对自己，只改搜索配置；胜负与时间无关，不受背景负载影响）：
#   A: A侧 = batch16 / n_sim400 / vl3.0（生产基线）   B侧 = batch64 / n_sim400 / **vl1.0**
#      → 若打平，就是**免费拿到 +59% 吞吐**
#   B: A侧 = batch16 / n_sim400 / vl3.0            B侧 = batch64 / **n_sim800** / vl3.0
#      → 墙钟相当（batch64 每手 218ms、翻倍到 ~436ms ≈ batch16 的 538ms），
#        看 2× 模拟量能否抵消"批量摊平"的损失
#
# 排队策略：先等标尺守候的 200 局（ladder.py / auto_eval.ps1）跑完，避免三方抢机器。
# 产物：checkpoints/history/batch_exp_{A,B}.log（人读）、.json（机器读）、
#       batch_exp_summary.txt（一行一结论）
#
# 用法（脱离会话后台）：
#   Start-Process pwsh -ArgumentList '-NoLogo','-NoProfile','-File','run_batch_experiments.ps1'

param(
    [string]$Model = 'checkpoints/pilot/big/gen_45.pt',
    [int]$Games = 60,
    [double]$MaxWaitHours = 4
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
# 让原生程序（python）的 UTF-8 输出按 UTF-8 解码再写入文件，否则中文变乱码
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$hist = 'checkpoints/history'
New-Item -ItemType Directory -Force -Path $hist | Out-Null
$summary = Join-Path $hist 'batch_exp_summary.txt'

function Say([string]$m) { Write-Output ("{0} [EXP] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

Say "模型 $Model　每场 $Games 局　等标尺空闲…"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Get-Date) -lt $deadline) {
    $busy = @()
    $busy += Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*ladder.py*' }
    $busy += Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" |
        Where-Object { $_.CommandLine -like '*auto_eval.ps1*' -and $_.CommandLine -notlike '*CimInstance*' }
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
}
Say "标尺已空闲，开始实验"

function Run-Exp([string]$tag, [string]$desc, [string[]]$extra) {
    $json = Join-Path $hist "batch_exp_$tag.json"
    $log = Join-Path $hist "batch_exp_$tag.log"
    Say "[$tag] $desc"
    $t0 = Get-Date
    python -u compare.py --a $Model --b $Model --games $Games `
        --heuristic-weight 0.5 --policy-weight 1.0 --json $json @extra *> $log
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if (-not (Test-Path $json)) { Say "[$tag] 没有 JSON（看 $log）"; return }
    $j = Get-Content $json -Encoding utf8 -Raw | ConvertFrom-Json
    $rate = [math]::Round([double]$j.a_rate, 3)
    $msg = "[$tag] A(nsim=$($j.n_sim_a)/batch=$($j.mcts_batch_a)/vl=$($j.vl_a))" +
           " vs B(nsim=$($j.n_sim_b)/batch=$($j.mcts_batch_b)/vl=$($j.vl_b))" +
           " ｜ A胜 $($j.a_wins) : B胜 $($j.b_wins) 和 $($j.draws)" +
           " ｜ A胜率 $rate p=$([math]::Round([double]$j.p_value, 4)) ｜ $mins 分钟"
    Say $msg
    Add-Content -Path $summary -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) -Encoding utf8

    # ---- 判读规则（写死在这里，避免事后挑数字）----
    if ($tag -eq 'A') {
        if ($rate -ge 0.60) {
            Say "[A 判读] 基线(batch16)胜率 $rate → **batch64+vl1.0 更弱，不采用**（继续用 16）"
        } elseif ($rate -le 0.55) {
            Say "[A 判读] 基线胜率 $rate → **batch64+vl1.0 不弱于基线 → 可采用（免费 +59% 吞吐）**"
        } else {
            Say "[A 判读] 基线胜率 $rate 落在 0.55~0.60 灰区 → 样本不足，补 200 局再定"
        }
    } else {
        if ($rate -le 0.45) {
            Say "[B 判读] 基线胜率 $rate（B=batch64/800模拟 赢面大）→ 「更多模拟」路线成立，优先采用"
        } elseif ($rate -ge 0.55) {
            Say "[B 判读] 基线胜率 $rate → 2× 模拟没能抵消批量摊平，放弃该路线"
        } else {
            Say "[B 判读] 基线胜率 $rate → 两者相当（墙钟相当下选吞吐更好的那条）"
        }
    }
}

Run-Exp 'A' 'A=batch16/400模拟/vl3.0  vs  B=batch64/400模拟/vl1.0' `
    @('--n-sim', '400', '--mcts-batch-a', '16', '--mcts-batch-b', '64',
      '--vl-a', '3.0', '--vl-b', '1.0')

Run-Exp 'B' 'A=batch16/400模拟/vl3.0  vs  B=batch64/800模拟/vl3.0' `
    @('--n-sim-a', '400', '--n-sim-b', '800', '--mcts-batch-a', '16',
      '--mcts-batch-b', '64', '--vl-a', '3.0', '--vl-b', '3.0')

Say "两个实验结束，汇总在 $summary"
