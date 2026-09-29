# hft.ps1 - PowerShell helper for pumpfun-hft on Windows.
#
#   hft   <command>   runs on the SYNTHETIC data set: the fake market made by `hft synth`
#   hftr  <command>   runs on the REAL data set: Pump.fun events recorded by `hftr stream` / `hftr paper`
#
# Each data set is one folder that holds everything made from it: its events, the SQLite manifest,
# the DuckDB warehouse, trained models and backtest reports. The two never mix: a model trained with
# hftr is only used by hftr, `synth` refuses to touch recorded events, and the recording commands
# (stream, paper, live, collect-history) only run with hftr.
#
# Load it in the current PowerShell window:
#     . "C:\Users\sjd18\OneDrive\Documents\GitHub\Crypto-Trading\pumpfun-hft\hft.ps1"
# Load it in every PowerShell window from now on (VS Code's terminal included) - run once:
#     hft-install-profile
#
# hft-help     command cheat sheet
# hft-doctor   checks the paths below and shows what each data set holds

# ---- edit these four lines if your folders are elsewhere ---------------------------------------
$env:HFT_PYTHON    = 'C:\pumpfun\venv\Scripts\python.exe'   # python.exe of the virtual environment
$env:HFT_CONFIG    = 'C:\pumpfun\local.yaml'                # optional settings for both (skipped if missing)
$env:HFT_SYNTH_DIR = 'C:\pumpfun\data'                      # hft  -> synthetic data set folder
$env:HFT_REAL_DIR  = 'C:\pumpfun\real'                      # hftr -> real (recorded) data set folder
# --------------------------------------------------------------------------------------------------

$global:HftScriptPath = $PSCommandPath

function Invoke-PumpfunHft {
    param([string]$Dataset, [string]$Folder, [object[]]$Rest)
    if (-not (Test-Path $env:HFT_PYTHON)) {
        Write-Error "Python not found at $env:HFT_PYTHON. Create the venv or edit HFT_PYTHON at the top of hft.ps1."
        return
    }
    $cfg = @()
    if ($env:HFT_CONFIG -and (Test-Path $env:HFT_CONFIG)) { $cfg = @('--config', $env:HFT_CONFIG) }
    & $env:HFT_PYTHON -m pumpfun_hft.main @cfg --dataset $Dataset --data-root $Folder @Rest
}

function hft {
    <#
    .SYNOPSIS
        pumpfun-hft on the SYNTHETIC data set (folder $env:HFT_SYNTH_DIR).
    .EXAMPLE
        hft synth --hours 24
        hft backtest --strategy smart_money
    #>
    Invoke-PumpfunHft -Dataset 'synthetic' -Folder $env:HFT_SYNTH_DIR -Rest $args
}

function hftr {
    <#
    .SYNOPSIS
        pumpfun-hft on the REAL data set (folder $env:HFT_REAL_DIR): recorded Pump.fun events.
    .EXAMPLE
        hftr stream --minutes 60
        hftr data-info
        hftr backtest --strategy smart_money
    #>
    Invoke-PumpfunHft -Dataset 'real' -Folder $env:HFT_REAL_DIR -Rest $args
}

function hft-install-profile {
    <#
    .SYNOPSIS
        Load hft.ps1 in every PowerShell window (all hosts, so VS Code's terminal too).
    #>
    $prof = $PROFILE.CurrentUserAllHosts
    $line = ". '" + $global:HftScriptPath.Replace("'", "''") + "'"
    if (-not (Test-Path $prof)) { New-Item -ItemType File -Path $prof -Force | Out-Null }
    $text = Get-Content -Raw -Path $prof
    if ($text -and $text.Contains($global:HftScriptPath)) {
        Write-Host "Already loaded from $prof"
    } else {
        Add-Content -Path $prof -Value $line
        Write-Host "Added to $prof :"
        Write-Host "    $line"
    }
    foreach ($p in @($PROFILE.CurrentUserCurrentHost, $PROFILE.CurrentUserAllHosts)) {
        if ($p -and (Test-Path $p)) {
            foreach ($h in (Select-String -Path $p -Pattern '^\s*function\s+hftr?\b')) {
                Write-Warning ("{0} line {1} defines its own '{2}'. Delete that line so the hft.ps1 version is used." -f $h.Path, $h.LineNumber, $h.Line.Trim())
            }
        }
    }
    Write-Host "Open a new PowerShell window (or VS Code terminal) and run hft-doctor."
}

function hft-doctor {
    <#
    .SYNOPSIS
        Check the paths at the top of hft.ps1 and show what each data set holds.
    #>
    if (Test-Path $env:HFT_PYTHON) { Write-Host "python   $env:HFT_PYTHON  ok" }
    else { Write-Warning "python   $env:HFT_PYTHON  NOT FOUND - edit HFT_PYTHON at the top of hft.ps1"; return }
    if ($env:HFT_CONFIG -and (Test-Path $env:HFT_CONFIG)) { Write-Host "config   $env:HFT_CONFIG" }
    else { Write-Host "config   (none: built-in defaults)" }
    foreach ($name in @('hft', 'hftr')) {
        $cmd = Get-Command $name -CommandType Function -ErrorAction SilentlyContinue
        if (-not $cmd -or $cmd.Definition -notmatch 'Invoke-PumpfunHft') {
            Write-Warning "'$name' is not the hft.ps1 version (another definition, e.g. in your `$PROFILE, replaced it). Remove that definition."
        }
    }
    Write-Host ""
    Write-Host "=== hft  (synthetic) ===" -ForegroundColor Cyan
    hft data-info
    Write-Host ""
    Write-Host "=== hftr (real) ===" -ForegroundColor Cyan
    hftr data-info
}

function hft-help {
    @'
pumpfun-hft - command cheat sheet          (full docs: docs\ in this repo)

  hft  <command>   SYNTHETIC data set   folder: $env:HFT_SYNTH_DIR
  hftr <command>   REAL data set        folder: $env:HFT_REAL_DIR   (recorded Pump.fun events)
  Every command below works with either; each data set has its own events, models and reports.

SETUP AND DATA
  hft-doctor                                  check paths; what each data set holds
  hft-install-profile                         load hft / hftr in every new PowerShell window
  hftr init                                   create folders and .env (fill in RPC endpoints)
  hftr check-config                           validate config; show which secrets are set
  hftr data-info                              events, time range, models of the data set + a train/test split
  hftr find-data                              every event store on this PC, real / synthetic / mixed
  hftr import-data --from <folder>            copy recorded events from another folder into the real set
  hftr verify-data                            checksum the Parquet store, report slot gaps

RECORDING REAL DATA  (hftr only)
  hftr stream --minutes 60                    record the live feed (stops by itself after 60 min)
  hftr collect-history                        backfill Pump program history over RPC (resumable)
  hftr latency-probe                          measure your RPC / Metis latency

SYNTHETIC DATA  (hft only; results are not real)
  hft synth --hours 24                        generate a fake market (replaces the previous one)
  hft synth --hours 24 --seed 7               same, with a fixed seed

BACKTESTING
  hftr backtest                               run strategy.active from the config
  hftr backtest --strategy smart_money        run one strategy
  hftr backtest --strategy sniper --strategy momentum_ignition     several at once
  hftr backtest --start 2026-09-27T00:00:00Z --end 2026-09-28T00:00:00Z
  hftr backtest --seed 7 --no-montecarlo      reproducible, skip the Monte Carlo section
  hftr backtest --no-report                   metrics only, no HTML/PDF/CSV/JSON
  hftr report                                 rebuild the report for the latest run
  hftr montecarlo --sims 5000                 re-run Monte Carlo on the latest run

  Strategies: momentum_ignition  smart_money  sniper  volume_breakout
              mean_reversion     whale_follow  liquidity_sweep
              bonding_curve_scalp  migration   ml_signal (trained model)
              rug_avoidance (exit overlay)

OPTIMISATION AND VALIDATION
  hftr optimize --strategy smart_money --method bayesian --trials 32
  hftr optimize --strategy smart_money --final                 also score the sealed test set
  hftr walkforward --strategy momentum_ignition                rolling out-of-sample folds
      --method: grid | random | bayesian | genetic

MACHINE LEARNING  (train and test on the same data set)
  hftr data-info                              prints a ready --end (first 60 % of the events)
  hftr train-model --model lightgbm --target fwd_up --end <from data-info>
  hftr backtest --strategy ml_signal --start <training cut-off printed by train-model>
  hftr paper-replay --strategy ml_signal --start <cut-off>     same test, through the live engine
  hftr paper --strategy ml_signal --minutes 120                live data, simulated fills
  hftr --set strategy.params.ml_signal.top_frac=0.02 backtest --strategy ml_signal --start <cut-off>
      top_frac: share of the highest model scores to buy (train-model prints a table to choose it)
      --model: logistic | random_forest | xgboost | lightgbm | catboost
      --target: fwd_up (-> ml_signal) | migrate (-> ml_signal, set its model_path) | rug (-> rug_model)
  Rug model: hftr train-model --target rug, then in your config set rug_model.use_trained_model: true
             and rug_model.model_path: <file name it printed>

LOOKING AT RESULTS
  hftr dashboard                              serve http://127.0.0.1:8050 (Ctrl-C to stop)
  hftr dashboard-export                       one self-contained HTML file
  hftr wallets --top 25                       top-ranked wallets of the latest run
  hftr query "select kind, count(*) n from events group by 1 order by n desc"

PAPER TRADING
  hftr paper-replay --strategy smart_money    offline rehearsal on stored events
  hftr paper-replay --hours 3 --no-compare    shorter, without the backtest comparison
  hftr paper --strategy momentum_ignition --minutes 120        live data, simulated fills

LIVE TRADING - real money. Read docs\LIVE_TRADING.md first.
  hftr --set app.mode=live live --confirm-live --minutes 30    both flags are required

GLOBAL OPTIONS  (put them before the command)
  hftr --set sizing.fixed_sol=0.05 backtest   override any config key
  hftr <command> --help                       every option for one command
'@.Replace('$env:HFT_SYNTH_DIR', $env:HFT_SYNTH_DIR).Replace('$env:HFT_REAL_DIR', $env:HFT_REAL_DIR) | Write-Host
}

Write-Host "pumpfun-hft loaded: hft = synthetic data ($env:HFT_SYNTH_DIR), hftr = real data ($env:HFT_REAL_DIR). 'hft-help' lists the commands." -ForegroundColor Green
