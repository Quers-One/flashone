# prepare_run.ps1 —— 建/刷新一个"续训路线"的起点：**继承权重 + 回放池，但代数重新计数**
#
# 【为什么需要它】2026-10-12 V3 第一次尝试崩坏（−478 Elo）的主因被锁定为
# **冷启动（无继承回放池）+ 换先验**。正确做法是：从 V2 的 `resume.pt` 起步
# （回放池 12.6 万条 + 优化器状态 + lr 计数一起恢复），这样第 1 代就落在与 V2 同一分布上。
#
# 但 `resume.pt` 里带着 `gen=175` ⇒ train.py 会从 gen 176 继续编号，
# **与 V2 自己的 gen_176… 撞名**（两个 run 的模型文件名会互相覆盖，历史无法分辨）。
# 所以本脚本把计数清零：
#   * `gen`            → 0        （⇒ 第 1 代 = gen 1）
#   * `name_offset`    → 1        （⇒ 对外编号也不偏移）
#   * `buffer[].gen`   → 1..N     （只影响日志里的"回放池 N 代"，避免出现 gen 175 这种怪值）
#   * `summary`        → 清空代数类字段
# 权重（net/ema/best_net/optimizer/lr/lr_decay_count/rng）**原样继承**。
#
# ⚠️ **每次启动前都要跑一次**：train.py 每次写 `resume.pt` 时存的 gen 是"**启动时**的
# 计数 + 已训代数"，重启会把计数退回到启动时的值 ⇒ 会重训/覆盖已完成的代。
# 本脚本会自动取目标目录里**最大的 gen_N.pt** 作为新计数，所以重复运行是安全的。
#
# ② 需要**装了 torch 的 python**（改写 resume.pt 用它），而且**先写临时文件、成功才替换**：
#    2026-10-14 事故 = 裸 `python` 落到本机另一套 Python 3.14（无 torch）⇒ 在"权重已复制、
#    resume.pt 刚被覆盖"的状态下 exit 1（代数计数没重置），而训练进程因 -Detach（WMI 重建
#    环境）拿到的是有 torch 的 Python313 ⇒ 表现为"prepare 挂了但训练照跑"，极难查。
#    现在逐个候选解释器试 `import torch`，并把改写结果先落在 resume.pt.new 再原子替换。
#
# 用法：
#   pwsh -NoProfile -File prepare_run.ps1                      # 默认 V3 ← V2 最新
#   pwsh -NoProfile -File prepare_run.ps1 -DryRun
#   pwsh -NoProfile -File prepare_run.ps1 -Dir models/V4 -FromDir models/V2 -FromGen 175
param(
    [string]$Dir = 'models/V3',
    [string]$FromDir = 'models/V2',
    [int]$FromGen = 0,                  # 0 = 取 FromDir 里最新的一代
    [switch]$NoBuffer,                  # 只继承权重（= 冷启动，**不推荐**；留着做对照）
    [switch]$DryRun
)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

if ($Dir -notmatch '^models/[A-Za-z0-9_.\-]{1,40}$') { Write-Output "Dir 非法：$Dir"; exit 1 }
if ($Dir.TrimEnd('/').ToLower() -eq 'models/v2') { Write-Output "Dir 不能是 models/V2（那是正本）"; exit 1 }
$srcDir = Join-Path $PSScriptRoot $FromDir
$dstDir = Join-Path $PSScriptRoot $Dir

# ---- 源代数 ----
if ($FromGen -le 0) {
    $newest = Get-ChildItem (Join-Path $srcDir 'gen_*.pt') -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace 'gen_', '') } | Select-Object -Last 1
    if (-not $newest) { Write-Output "★ $FromDir 里找不到 gen_*.pt"; exit 1 }
    $FromGen = [int]($newest.BaseName -replace 'gen_', '')
}
$srcGen = Join-Path $srcDir "gen_$FromGen.pt"
$srcResume = Join-Path $srcDir 'resume.pt'
if (-not (Test-Path $srcGen)) { Write-Output "★ 找不到 $srcGen"; exit 1 }

# ---- 目标目录当前已有多少代（新计数从这里接着走，保证重复运行安全）----
$dstMaxGen = 0
if (Test-Path $dstDir) {
    foreach ($f in (Get-ChildItem (Join-Path $dstDir 'gen_*.pt') -ErrorAction SilentlyContinue)) {
        if ($f.BaseName -match '^gen_(\d+)$') { $dstMaxGen = [Math]::Max($dstMaxGen, [int]$Matches[1]) }
    }
}

Write-Output "=== 计划 ==="
Write-Output ("  权重:    {0}gen_{1}.pt → {2}\best.pt" -f ($FromDir + '\'), $FromGen, $Dir)
Write-Output ("  状态:    {0} → {1}\resume.pt" -f $(if ($NoBuffer) { '不继承（冷启动）' } else { ($FromDir + '\resume.pt（回放池+优化器+lr 计数）') }), $Dir)
Write-Output ("  代数计数: 重置为 {0} ⇒ 下一代 = gen {1}" -f $dstMaxGen, ($dstMaxGen + 1))
if ($DryRun) { Write-Output "[DryRun] 不写盘。"; exit 0 }

New-Item -ItemType Directory -Force -Path $dstDir, (Join-Path $dstDir 'logs'), (Join-Path $dstDir 'eval') | Out-Null

# ---- ① 权重（best.pt：train.py 在 --fresh 时用它做初始化）----
Copy-Item $srcGen (Join-Path $dstDir 'best.pt') -Force
Write-Output ("  ✅ {0}\best.pt ← gen_{1}（SHA1 {2}）" -f $Dir, $FromGen, (Get-FileHash $srcGen -Algorithm SHA1).Hash.Substring(0, 8))

# ---- ② resume.pt：继承回放池但计数清零 ----
$dstResume = Join-Path $dstDir 'resume.pt'
if ($NoBuffer) {
    if (Test-Path $dstResume) { Remove-Item $dstResume -Force; Write-Output "  ✅ 已删除 $Dir\resume.pt（冷启动）" }
} elseif (Test-Path $srcResume) {
    $py = @'
import sys, torch
path, newgen = sys.argv[1], int(sys.argv[2])
st = torch.load(path, map_location='cpu', weights_only=False)
old = st.get('gen')
st['gen'] = newgen
st['name_offset'] = 1
buf = st.get('buffer') or []
for i, g in enumerate(buf):
    g['gen'] = i + 1


def _n_samples(g):
    # 回放池有两种落盘格式：旧 = 原始数组（states），新 = npz 压缩块（n + blob）。
    # 2026-10-16 起 train.py 写的是新格式；这里两种都要能数，否则会 KeyError 而 exit 1。
    if not isinstance(g, dict):
        return 0
    if 'states' in g:
        return len(g['states'])
    return int(g.get('n', 0))


s = st.get('summary')
if isinstance(s, dict):
    for k in ('final_gen', 'final_win_rate', 'final_lr'):
        s.pop(k, None)
torch.save(st, path)
# 注意：不要在这里 print emoji/中文 —— Windows 控制台默认 GBK，emoji 会抛
# UnicodeEncodeError（实测把"保存成功"变成"exit 1"，很难查）。纯 ASCII 输出。
print('  [OK] resume.pt: gen %s -> %d, name_offset -> 1, buffer kept: %d gens / %d samples'
      % (old, newgen, len(buf), sum(_n_samples(g) for g in buf)))
'@
    $tmpPy = Join-Path $env:TEMP 'prepare_run_fix.py'
    Set-Content -Path $tmpPy -Value $py -Encoding utf8

    # ---- 挑一个**真的装了 torch** 的解释器（绝不猜裸 `python`）----
    # 本机有两套 Python：`%LOCALAPPDATA%\Python\bin\python.exe`（3.14，无 torch，PATH 里可能在前）
    # 与 `%LOCALAPPDATA%\Programs\Python\Python313\python.exe`（有 torch）。看板/计划任务继承的
    # PATH 不同 ⇒ 裸 `python` 会漂移。逐个候选试 `import torch`，第一个成功的才用。
    $pyExe = $null
    $cands = @()
    $cmdPy = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPy -and $cmdPy.Source) { $cands += $cmdPy.Source }
    $cands += @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
                "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
                "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")
    foreach ($c in $cands) {
        if (-not $c -or -not (Test-Path $c)) { continue }
        & $c -c "import torch" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $pyExe = $c; break }
    }
    if (-not $pyExe) {
        Write-Output "★ 找不到能 import torch 的 python（改写 resume.pt 必须用 torch）。试过："
        foreach ($c in $cands) { Write-Output ("    " + $c) }
        Write-Output "  ⇒ 未改动 $Dir\resume.pt（本次只刷新了 best.pt）"
        exit 1
    }

    # ---- 先写临时文件，成功才替换：失败时旧的 resume.pt 原样保留 ----
    $newResume = "$dstResume.new"
    if (Test-Path $newResume) { Remove-Item $newResume -Force -ErrorAction SilentlyContinue }
    Copy-Item $srcResume $newResume -Force
    Write-Output ("  解释器: {0}" -f $pyExe)
    & $pyExe $tmpPy $newResume $dstMaxGen
    if ($LASTEXITCODE -ne 0) {
        Remove-Item $newResume -Force -ErrorAction SilentlyContinue
        Write-Output "★ 改写 resume.pt 失败（$Dir\resume.pt 未被改动）"
        exit 1
    }
    Move-Item $newResume $dstResume -Force
    Write-Output ("  ✅ {0}\resume.pt ← {1}\resume.pt（回放池/优化器继承，计数重置）" -f $Dir, $FromDir)
} else {
    Write-Output "  ⚠ $FromDir 没有 resume.pt，只能冷启动"
}

# ---- ③ 记录来源（便于审计）----
[ordered]@{
    prepared_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    weights_from = "$FromDir/gen_$FromGen.pt"
    resume_inherited = (-not $NoBuffer -and (Test-Path $srcResume))
    counter_reset_to = $dstMaxGen
    why = '继承回放池/优化器以保持分布连续；代数清零以免与源 run 的 gen_N 撞名'
} | ConvertTo-Json | Set-Content -Path (Join-Path $dstDir 'INIT.json') -Encoding utf8
Write-Output ("  ✅ {0}\INIT.json" -f $Dir)
Write-Output ""
Write-Output "下一步（看板：点「填入 V3 稳妥版」；或你自己的终端）："
Write-Output ('  pwsh -NoProfile -File "' + (Join-Path $PSScriptRoot 'train_detached.ps1') +
    '" -RunName V3 -SaveDir models/V3 -GensPerRound 30 -Games 160 -MinSteps 320 -BufferGens 60 -HeuristicWeight 0.5 -PolicyWeight 1.0 -NoAnneal')
