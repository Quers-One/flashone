# auto_eval.ps1 —— 自动测量与晋升（KataGo 式连续 Elo 曲线）
#
# 【两个角色，别混】
#   root（记分基准）  = checkpoints/anchors/ROOT.txt 指向的模型（**永不变**，= gen_60）
#                       → 所有点都以它为 0 分，曲线连续、可跨代比较
#   champion（当前最强）= checkpoints/champions/current.pt
#                       → 比它显著更强就晋升（换掉它），它是"现在用的模型"
#
# 【为什么这样设计】旧的实现把两者合成一个"锚点"，每次晋升就换零点 → 曲线只能分段。
# 现在改成 **锚点链（ladder chaining）**：
#   候选 vs 当前 champion 打 200 局 → ΔElo(相对 champion)
#   加上 champion 相对 root 的累计分 → 候选相对 root 的 Elo（曲线上的一个点）
#   每 -ReanchorEvery 次（默认 3）直接对 root 打一次，用来消除链条累积误差并重置跳数。
#   晋升判定用的是"候选 vs champion"那一场（p 由 Wilson 区间给出）。
#
# 成本：每次 1 场 200 局（6×128 之间约 40 分钟）+ 硬门槛 ≈5 分钟。
#
# 用法：
#   pwsh -NoProfile -File auto_eval.ps1 -SaveDir checkpoints/pilot/big -Games 200
#   pwsh -NoProfile -File auto_eval.ps1 -SaveDir checkpoints/pilot/big -Games 8 -NoPromote   # 自测

param(
    [string]$RunName = 'big6x128',
    [string]$SaveDir = 'checkpoints/pilot/big',
    [int]$Games = 200,                 # 每场局数（200 ≈ ±50 Elo）
    [int]$CandGen = 0,                 # 0 = 取该目录里最新的 gen_N.pt
    [double]$MinGain = 25,             # 晋升还要求 ΔElo(相对 champion) ≥ 这个值
    [double]$MinScore = 0.55,          # 【任务 1，2026-10-13 新增】晋升还要求本场**得分率严格 >** 这个值。
                                       # "置信区间下界 >0.50" 由 verdict=stronger 等价给出（见 ⑤ 段注释）。
                                       # 0.55 得分率 ≈ ΔElo +34.9；200 局下约合 ≥110/200。
                                       # 想回到"只看 ΔElo≥25"就传 `-MinScore 0.50`。
    [int]$ReanchorEvery = 3,           # 每几跳直测一次 root（消漂移、重置跳数）
    [int]$LadderWorkers = 8,           # 标尺测量并行进程数（按局号切分，结果与串行逐局相同）。
                                       # 【2026-10-11 17:3x 加】训练同时跑时单进程尺子掉到
                                       # ~2 局/分钟（200 局要 90 分钟）；8 进程把它拉回 ~25 分钟。
                                       # 等价性验收：8 局串行 vs 4 进程并行，比分/Elo/手数/序列全同。
    [int]$ProdGames = 0,               # **生产配方**对照点局数（hw/pw=0.5/1.0）；**默认 0 = 不跑**。
                                       # 【2026-10-11 17:2x 改成 opt-in】原默认 60。理由：
                                       # 它每次多花 10–15 分钟（占一次 45 分钟测量的 ~25%），
                                       # 而迄今**两个口径从未给出不同判定**（全是 indistinguishable；
                                       # 手动跑的 V1 gen_60 / V2 gen_80 也都与尺子口径同向同量级）。
                                       # 真正需要"实战条件下的数"是**决策时刻**（准备晋升、
                                       # 或两臂咬得很紧）——那时手动加 `-ProdGames 60` 即可。
    [string]$HistDir = 'checkpoints/history',   # 账本目录（自测时换成临时目录）
    [switch]$NoPromote,
    [switch]$SkipGates,
    [switch]$SkipPlot
)

$ErrorActionPreference = 'Continue'
# 【安全闸 · 2026-10-11 22:2x 加】默认值 checkpoints/pilot/big 与 models/V1 是**同一个目录**
# （models\V1 是指向它的 Junction）。不带 -SaveDir 就写，会把评测产物塞进 V1 那条线。
if (-not $PSBoundParameters.ContainsKey('SaveDir')) {
    Write-Output "[ABORT] 必须显式指定 -SaveDir（旧默认值 checkpoints/pilot/big 就是 models/V1 本身）。"
    exit 4
}
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'

$events = Join-Path $SaveDir 'auto_events.log'
$hist = $HistDir                     # 可覆盖：自测时指到临时目录，避免污染真账本
$rootPtr = 'checkpoints/anchors/ROOT.txt'
$champPt = 'checkpoints/champions/current.pt'
$champJs = 'checkpoints/champions/current.json'
$champHist = 'checkpoints/champions/history.csv'
New-Item -ItemType Directory -Force -Path $hist, 'checkpoints/champions' | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SaveDir 'eval') | Out-Null

function Log([string]$ev, [string]$msg) {
    $line = "{0} {1} {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $ev, $msg
    Add-Content -Path $events -Value $line -Encoding utf8
    Write-Output $line
}

# ---- root / champion ----
if (-not (Test-Path $rootPtr)) { Log 'EVAL_ERR' "缺少 $rootPtr（记分基准指针）"; exit 1 }
$rootName = (Get-Content $rootPtr -Encoding utf8 -Raw).Trim()
$root = Join-Path 'checkpoints/anchors' $rootName
if (-not (Test-Path $root)) { Log 'EVAL_ERR' "ROOT 指向的模型不存在: $root"; exit 1 }
$champ = Get-Content $champJs -Encoding utf8 -Raw | ConvertFrom-Json
if (-not (Test-Path $champPt)) { Log 'EVAL_ERR' "缺少 $champPt"; exit 1 }

# ---- 候选 ----
if ($CandGen -le 0) {
    $cands = Get-ChildItem (Join-Path $SaveDir 'gen_*.pt') -ErrorAction SilentlyContinue |
        ForEach-Object { if ($_.BaseName -match '^gen_(\d+)$') { [pscustomobject]@{ N = [int]$Matches[1]; Path = $_.FullName } } } |
        Sort-Object N
    if (-not $cands) { Log 'EVAL_ERR' "在 $SaveDir 找不到 gen_N.pt"; exit 1 }
    $cand = $cands[-1]
} else {
    $p = Join-Path $SaveDir "gen_$CandGen.pt"
    if (-not (Test-Path $p)) { Log 'EVAL_ERR' "找不到 $p"; exit 1 }
    $cand = [pscustomobject]@{ N = $CandGen; Path = (Resolve-Path $p).Path }
}

# ---- 决定这一场打谁：收敛漂移时直测 root ----
$hops = [int]$champ.hops
$measureRoot = ($hops -ge $ReanchorEvery)
$opponent = if ($measureRoot) { $root } else { $champPt }
$oppName = if ($measureRoot) { "root:$rootName" } else { "champion:$($champ.name)" }
Log 'EVAL_START' ("run={0} gen={1} 对手={2} 局数={3} champion跳数={4}" -f $RunName, $cand.N, $oppName, $Games, $hops)

# ---- ① 硬门槛 ----
if (-not $SkipGates) {
    $t0 = Get-Date
    python -u test_tactics.py --model $cand.Path *> (Join-Path $SaveDir "eval/gen$($cand.N)_tactics.log")
    $tac = $LASTEXITCODE
    python -u test_selfplay.py --model $cand.Path *> (Join-Path $SaveDir "eval/gen$($cand.N)_selfplay.log")
    $sp = $LASTEXITCODE
    Log 'HARDGATE' ("gen={0} tactics_exit={1} selfplay_exit={2}（{3:N1} 分钟）" -f $cand.N, $tac, $sp, ((Get-Date) - $t0).TotalMinutes)
    if ($tac -ne 0 -or $sp -ne 0) {
        $msg = "gen $($cand.N) 硬门槛失败（tactics=$tac selfplay=$sp）—— 建议人工介入"
        Log 'HARDGATE_FAIL' $msg
        Set-Content -Path (Join-Path $hist 'HARDGATE_FAIL.txt') -Value $msg -Encoding utf8
        exit 2
    }
    Remove-Item (Join-Path $hist 'HARDGATE_FAIL.txt') -ErrorAction SilentlyContinue
}

# ---- ①b 锚定校准（2026-10-13 改）----
# 【旧实现的洞】$measureRoot 时只把**候选**直测 root，写对候选那一行、hops 归零，但
# **没有回写 champions/current.json**（那个文件只在下面的晋升分支被写）⇒ 下一场又从虚高的
# champion 基准继续累加，"消漂移"实际没生效。实测虚高：同一 gen30 直测 149.3 vs 链式 219.8
# （+70.5），到 gen110 涨到 +107.0（直测 251.9 vs 链式 358.9）——与"每跳各带 ±40 噪声随机游走"一致。
# 【现在的做法】锚定这一轮**先测 champion 本身 vs root**，把真值写回 current.json、hops 归零，
# 再照常测候选 ⇒ 本场两条读数都是直测，往后每段链条都从校正过的基准重新起算。
$csv = Join-Path $hist 'elo_history.csv'
$ledgerHead = 'at,run,gen,measured_vs,champion,root,games,wins,losses,draws,score,elo_vs_opponent,elo_vs_root,ci_lo_root,ci_hi_root,verdict_vs_opponent,hops,n_sim,hw,pw,engine_mcts,engine_full'
if (-not (Test-Path $csv)) { $ledgerHead | Set-Content -Path $csv -Encoding utf8 }

function Write-EloRow {
    param($at, $run, $gen, $measVs, $champName, $games, $wins, $losses, $draws, $score,
          $eloOpp, $eloRoot, $ciLo, $ciHi, $verdict, $hopN, $nSim, $hw, $pw, $engM, $engF)
    ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10:N4},{11:N1},{12:N1},{13:N1},{14:N1},{15},{16},{17},{18},{19},{20},{21}" -f `
            $at, $run, $gen, $measVs, $champName, $rootName, $games, $wins, $losses, $draws, `
            $score, $eloOpp, $eloRoot, $ciLo, $ciHi, $verdict, $hopN, $nSim, $hw, $pw, $engM, $engF) |
        Add-Content -Path $csv -Encoding utf8
}

if ($measureRoot) {
    $reJson = Join-Path $SaveDir ("eval/reanchor_{0}_vs_root.json" -f $champ.name)
    $reLog  = Join-Path $SaveDir ("eval/reanchor_{0}_ladder.log" -f $champ.name)
    $oldElo = [double]$champ.elo_vs_root
    Log 'REANCHOR_START' ("跳数={0} ≥ ReanchorEvery={1} ⇒ 先测 champion {2} vs root:{3} 校准基准（{4} 局）" -f `
            $hops, $ReanchorEvery, $champ.name, $rootName, $Games)
    python -u ladder.py --candidate $champPt --anchor $root --games $Games --workers $LadderWorkers --json $reJson *> $reLog
    if ($LASTEXITCODE -eq 0 -and (Test-Path $reJson)) {
        $jc = Get-Content $reJson -Encoding utf8 -Raw | ConvertFrom-Json
        $rcW = ([double]$jc.elo_hi - [double]$jc.elo_lo) / 2.0
        $rcLo = [double]$jc.elo - $rcW
        $rcHi = [double]$jc.elo + $rcW
        $rcEngF = if ($jc.PSObject.Properties.Name -contains 'engine_full') { $jc.engine_full } else { '?' }
        $reGen = if ($champ.name -match '_g(\d+)$') { [int]$Matches[1] } else { 0 }
        $reRun = if ($champ.source -match 'models[/\\]([^/\\]+)') { $Matches[1] } else { $RunName }
        # 这一行是**直测锚**（hops=0）⇒ 图上画成空心圆，与链式推算点一眼可分
        Write-EloRow $jc.at $reRun $reGen ("root:" + $rootName) $champ.name $jc.games $jc.wins $jc.losses `
            $jc.draws $jc.score $jc.elo $jc.elo $rcLo $rcHi $jc.verdict 0 $jc.n_sim `
            $jc.heuristic_weight $jc.policy_weight $jc.engine_fingerprint_mcts $rcEngF
        # 覆盖本进程内的 champion 记录（无论是否落盘），让本场的链式换算用上校正值
        $champ.elo_vs_root = [math]::Round([double]$jc.elo, 1)
        $champ.ci_lo = [math]::Round($rcLo, 1)
        $champ.ci_hi = [math]::Round($rcHi, 1)
        $champ.hops = 0
        if (-not $NoPromote) {
            $champ | Add-Member -NotePropertyName reanchor_at -NotePropertyValue $jc.at -Force
            $champ | Add-Member -NotePropertyName reanchor_note -NotePropertyValue `
                ("直测 root:{0} 得 {1:N1}（校正前链式值 {2:N1}，虚高 {3:N1}）" -f `
                    $rootName, $jc.elo, $oldElo, ($oldElo - [double]$jc.elo)) -Force
            ($champ | ConvertTo-Json -Depth 6) | Set-Content -Path $champJs -Encoding utf8
        }
        Log 'REANCHOR_DONE' ("champion={0} 直测 root:{1} = {2:N1} [{3:N1},{4:N1}]（旧链式 {5:N1}）⇒ hops 归零{6}" -f `
                $champ.name, $rootName, $jc.elo, $rcLo, $rcHi, $oldElo,
                $(if ($NoPromote) { '（-NoPromote：只算不落盘）' } else { '，已写回 current.json' }))
    } else {
        Log 'REANCHOR_ERR' ("champion 校准测量失败（exit={0}），本场退回旧链式路径继续；见 {1}" -f $LASTEXITCODE, $reLog)
    }
}

# ---- ② 一场对局：候选 vs 对手 ----
$jsonPath = Join-Path $SaveDir "eval/gen$($cand.N)_vs_$(if($measureRoot){'root'}else{'champ'}).json"
python -u ladder.py --candidate $cand.Path --anchor $opponent --games $Games --workers $LadderWorkers --json $jsonPath *> (Join-Path $SaveDir "eval/gen$($cand.N)_ladder.log")
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $jsonPath)) {
    Log 'EVAL_ERR' "ladder 失败（exit=$LASTEXITCODE），见 eval/gen_$($cand.N)_ladder.log"
    exit 3
}
$j = Get-Content $jsonPath -Encoding utf8 -Raw | ConvertFrom-Json

# ---- ③ 换算成"相对 root"的 Elo（锚点链）----
if ($measureRoot) {
    $eloRoot = [double]$j.elo
    $w = ([double]$j.elo_hi - [double]$j.elo_lo) / 2.0
    $newHops = 0
} else {
    $eloRoot = [double]$champ.elo_vs_root + [double]$j.elo
    $wc = ([double]$champ.ci_hi - [double]$champ.ci_lo) / 2.0
    $wm = ([double]$j.elo_hi - [double]$j.elo_lo) / 2.0
    $w = [math]::Sqrt($wc * $wc + $wm * $wm)
    $newHops = $hops + 1
}
$ciLo = $eloRoot - $w
$ciHi = $eloRoot + $w
# 【引擎全】= mcts.py + heuristics.py + gomoku_env.py 的组合短哈希（只改后两个时
# 旧的 engine_mcts **不会变**，但棋已经不一样了）。没这一项就发现不了那种静默作废。
$engFull = if ($j.PSObject.Properties.Name -contains 'engine_full') { $j.engine_full } else { '?' }
Log 'EVAL_DONE' ("gen={0} vs {1} 局数={2} 胜/负/和={3}/{4}/{5} 本场ΔElo={6:N1} [{7:N1},{8:N1}] 判定={9} | 相对 root({10}) Elo={11:N1} [{12:N1},{13:N1}] 跳数={14} 引擎={15} 引擎全={16}" -f `
        $cand.N, $oppName, $j.games, $j.wins, $j.losses, $j.draws, $j.elo, $j.elo_lo, $j.elo_hi, $j.verdict, `
        $rootName, $eloRoot, $ciLo, $ciHi, $newHops, $j.engine_fingerprint_mcts, $engFull)

# ---- ④ 曲线账本（连续曲线：每行都是"相对同一个 root"）----
# 【列只往尾部加】plot_progress.py 用 DictReader 按名字读，所以追加列不会破坏它；
# 老行少了最后一列时 DictReader 会把该字段读成空 —— 可接受。
# 表头在 ①b 里建好（$ledgerHead），写行走 Write-EloRow —— 校准那一行与这一行共用同一套格式。
Write-EloRow $j.at $RunName $cand.N $oppName $champ.name $j.games $j.wins $j.losses $j.draws `
    $j.score $j.elo $eloRoot $ciLo $ciHi $j.verdict $newHops $j.n_sim $j.heuristic_weight `
    $j.policy_weight $j.engine_fingerprint_mcts $engFull
Write-Output ("  曲线账本已追加: {0}" -f $csv)

# ---- ④b 生产配方对照点（hw/pw=0.5/1.0，默认 60 局）----
# 【为什么要有这一行】主曲线固定用 6.0/0.3（跨代可比、和锚点冻结口径一致），
# 但 gen 40+ 之后**生产自对弈配方**已经是 0.5/1.0（退火终点）。两个条件会给出
# 明显不同的读数（实测 gen_60：旧配方 +10.4 [−23.6,+44.4]，生产配方 30:30 s=0.500）
# —— 只记一个必然误导。所以**另记一条**，写进独立的账本，**绝不混进主曲线**。
# 失败只记日志、不影响主测量与晋升（与硬门槛同样的处理方式）。
if ($ProdGames -gt 0) {
    try {
        $prodJson = Join-Path $SaveDir ("eval/gen{0}_vs_{1}_prod.json" -f $cand.N, $(if ($measureRoot) { 'root' } else { 'champ' }))
        python -u ladder.py --candidate $cand.Path --anchor $opponent --games $ProdGames `
            --heuristic-weight 0.5 --policy-weight 1.0 --workers $LadderWorkers --json $prodJson `
            *> (Join-Path $SaveDir ("eval/gen{0}_ladder_prod.log" -f $cand.N))
        if ($LASTEXITCODE -eq 0 -and (Test-Path $prodJson)) {
            $jp = Get-Content $prodJson -Encoding utf8 -Raw | ConvertFrom-Json
            $engFullP = if ($jp.PSObject.Properties.Name -contains 'engine_full') { $jp.engine_full } else { '?' }
            Log 'EVAL_DONE_PROD' ("gen={0} vs {1} 局数={2} 胜/负/和={3}/{4}/{5} 本场ΔElo={6:N1} [{7:N1},{8:N1}] 判定={9} | 配方=生产(0.5/1.0) 引擎={10} 引擎全={11}" -f `
                    $cand.N, $oppName, $jp.games, $jp.wins, $jp.losses, $jp.draws, $jp.elo, $jp.elo_lo, $jp.elo_hi, $jp.verdict, `
                    $jp.engine_fingerprint_mcts, $engFullP)
            $csvP = Join-Path $hist 'elo_history_prod.csv'
            if (-not (Test-Path $csvP)) {
                "at,run,gen,measured_vs,champion,games,wins,losses,draws,score,elo_vs_opponent,ci_lo,ci_hi,verdict,n_sim,hw,pw,engine_mcts,engine_full" |
                    Set-Content -Path $csvP -Encoding utf8
            }
            ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9:N4},{10:N1},{11:N1},{12:N1},{13},{14},{15},{16},{17},{18}" -f `
                    $jp.at, $RunName, $cand.N, $oppName, $champ.name, $jp.games, $jp.wins, $jp.losses, $jp.draws, `
                    $jp.score, $jp.elo, $jp.elo_lo, $jp.elo_hi, $jp.verdict, $jp.n_sim, $jp.heuristic_weight, `
                    $jp.policy_weight, $jp.engine_fingerprint_mcts, $engFullP) | Add-Content -Path $csvP -Encoding utf8
            Write-Output ("  生产配方对照点已追加: {0}" -f $csvP)
        } else {
            Log 'EVAL_PROD_ERR' ("生产配方对照点失败（exit=$LASTEXITCODE），见 eval/gen{0}_ladder_prod.log；不影响主曲线" -f $cand.N)
        }
    } catch {
        Log 'EVAL_PROD_ERR' ("生产配方对照点异常：{0}（不影响主曲线）" -f $_.Exception.Message)
    }
}

# ---- ⑤ 晋升：显著强于 champion（= 得分率的 Wilson 下界 > 0.50）+ 得分率 >0.55 + 涨幅够 ----
# 任务 1 要求把晋升条件写成「严格 >0.55，且置信区间下界 >0.50」：
#   * "置信区间下界 >0.50" 与 `verdict -eq 'stronger'` **完全等价** —— ladder.py 的
#     verdict 就是按 elo_lo > 0 判的（§12 的 elo_ci 用的是得分率的 Wilson 区间，
#     而 Elo 的 0 点正对应得分率 0.50）；所以这一条不用另写。
#   * "严格 >0.55" 是**新增**的硬条件：原来只看 ΔElo ≥ $MinGain(=25，相当于得分率
#     0.5358)，现在还得得分率 >0.55（≈ +34.9 Elo）。200 局下这个门槛约合 ≥110/200。
$strong = ($j.verdict -eq 'stronger')
$scoreOk = ([double]$j.score -gt $MinScore)
if ($NoPromote) {
    Log 'PROMOTE_SKIP' 'NoPromote 指定，跳过晋升'
} elseif ($strong -and $scoreOk -and ([double]$j.elo -ge $MinGain)) {
    Copy-Item $cand.Path $champPt -Force
    $h = (Get-FileHash $cand.Path -Algorithm SHA1).Hash.Substring(0, 8).ToLower()
    $meta = [ordered]@{
        name = ("{0}_g{1}" -f $RunName, $cand.N); path = $champPt; sha1_8 = $h
        elo_vs_root = [math]::Round($eloRoot, 1); ci_lo = [math]::Round($ciLo, 1)
        ci_hi = [math]::Round($ciHi, 1); hops = $newHops
        source = $cand.Path; updated_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
        note = ("从 {0} 晋升；本场 {1:N1} 分（相对 {2}）" -f $champ.name, $j.elo, $oppName)
    }
    $meta | ConvertTo-Json | Set-Content -Path $champJs -Encoding utf8
    if (-not (Test-Path $champHist)) {
        "at,old_champion,new_champion,source_gen,elo_vs_root,ci_lo,ci_hi,gain_vs_old,games,run" |
            Set-Content -Path $champHist -Encoding utf8
    }
    ("{0},{1},{2},{3},{4:N1},{5:N1},{6:N1},{7:N1},{8},{9}" -f `
            (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $champ.name, $meta.name, $cand.N, `
            $eloRoot, $ciLo, $ciHi, $j.elo, $j.games, $RunName) | Add-Content -Path $champHist -Encoding utf8
    Log 'PROMOTE' ("champion: {0} -> {1}（相对 root {2:N1} 分，本场 +{3:N1}，得分率 {4:N3}）" -f $champ.name, $meta.name, $eloRoot, $j.elo, $j.score)
} elseif ($strong -and (-not $scoreOk)) {
    Log 'NO_PROMOTE' ("gen={0} 不晋升（verdict=stronger 但得分率 {1:N3} ≤ 门槛 {2:N2}；本场ΔElo={3:N1}）" -f $cand.N, $j.score, $MinScore, $j.elo)
} else {
    Log 'NO_PROMOTE' ("gen={0} 不晋升（verdict={1}, 本场ΔElo={2:N1} < 门槛 {3}）" -f $cand.N, $j.verdict, $j.elo, $MinGain)
}

# ---- ⑥ 刷新进度图 ----
if (-not $SkipPlot) {
    python -u plot_progress.py --dir $SaveDir *>> (Join-Path $SaveDir "eval/gen$($cand.N)_plot.log")
    Log 'PLOT' ("进度图已刷新（exit={0}）：{1}" -f $LASTEXITCODE, (Join-Path $SaveDir 'progress.png'))
}
exit 0
