# hft.ps1 — PowerShell helper for pumpfun-hft on Windows.
#
# Load it once per PowerShell window:
#     . "C:\Users\sjd18\OneDrive\Documents\GitHub\Crypto-Trading\pumpfun-hft\hft.ps1"
#
# To load it automatically in every new window, add that same line to your
# PowerShell profile (run `notepad $PROFILE` to edit it; create the file if asked).
#
# Then run commands as `hft <command>`, e.g. `hft synth --hours 24`.
# `hft-help` prints the command cheat sheet.

# Where the virtual environment and the config live. Edit these two lines if you
# installed somewhere else.
$env:HFT_PYTHON = 'C:\pumpfun\venv\Scripts\python.exe'
$env:HFT_CONFIG = 'C:\pumpfun\local.yaml'

function hft {
    <#
    .SYNOPSIS
        Run a pumpfun-hft command with the configured Python and config file.
    .EXAMPLE
        hft synth --hours 24
        hft backtest --strategy smart_money
    #>
    if (-not (Test-Path $env:HFT_PYTHON)) {
        Write-Error "Python not found at $env:HFT_PYTHON. Create the venv or edit hft.ps1."
        return
    }
    $cfg = @()
    if (Test-Path $env:HFT_CONFIG) { $cfg = @('--config', $env:HFT_CONFIG) }
    & $env:HFT_PYTHON -m pumpfun_hft.main @cfg @args
}

function hft-help {
    @'
pumpfun-hft — command cheat sheet          (full docs: docs\ in this repo)

SETUP
  hft init                                    create data / log / report folders and .env
  hft check-config                            validate config; show which secrets are set
  hft docs                                    regenerate docs\MODULES.md

SYNTHETIC DATA  (offline; results are not real)
  hft synth --hours 24                        generate a fake market into the event store
  hft synth --hours 24 --seed 7               same, with a fixed seed
  hft verify-data                             checksum the Parquet store, report slot gaps

BACKTESTING
  hft backtest                                run strategy.active from the config
  hft backtest --strategy smart_money         run one strategy
  hft backtest --strategy sniper --strategy momentum_ignition     several at once
  hft backtest --seed 7 --no-montecarlo       reproducible, skip the Monte Carlo section
  hft backtest --no-report                    metrics only, no HTML/PDF/CSV/JSON
  hft report                                  rebuild the report for the latest run
  hft report --run <run-id> --formats html,pdf
  hft montecarlo --sims 5000                  re-run Monte Carlo on the latest run

  Strategies: momentum_ignition  smart_money  sniper  volume_breakout
              mean_reversion     whale_follow  liquidity_sweep
              bonding_curve_scalp  migration   rug_avoidance (exit overlay)

OPTIMISATION AND VALIDATION
  hft optimize --strategy smart_money --method bayesian --trials 32
  hft optimize --strategy momentum_ignition --method grid
  hft optimize --strategy smart_money --final                 also score the sealed test set
  hft walkforward --strategy momentum_ignition                rolling out-of-sample folds
      --method: grid | random | bayesian | genetic

MACHINE LEARNING
  hft train-model --model lightgbm --target rug               rug-pull probability
  hft train-model --model xgboost --target fwd_up             forward return
      --model: logistic | random_forest | xgboost | lightgbm | catboost

LOOKING AT RESULTS
  hft dashboard                               serve http://127.0.0.1:8050 (Ctrl-C to stop)
  hft dashboard-export                        one self-contained HTML file
  hft wallets --top 25                        top-ranked wallets of the latest run
  hft query "select kind, count(*) n from events group by 1 order by n desc"

PAPER TRADING
  hft paper-replay --strategy smart_money     offline rehearsal on stored events (~20 s)
  hft paper-replay --hours 3 --no-compare     shorter, without the backtest comparison
  hft stream --minutes 5                      record the live feed only; show latency
  hft paper --strategy momentum_ignition --minutes 120        live data, simulated fills
  hft latency-probe                           measure your RPC / Metis latency

LIVE TRADING — real money. Read docs\LIVE_TRADING.md first.
  hft --set app.mode=live live --confirm-live --minutes 30    both flags are required

GLOBAL OPTIONS  (put them before the command)
  hft --set sizing.fixed_sol=0.05 backtest    override any config key
  hft --config C:\pumpfun\local.yaml backtest use a different config file
  hft <command> --help                        every option for one command
'@ | Write-Host
}

Write-Host "pumpfun-hft loaded. Run 'hft-help' for the command list." -ForegroundColor Green
