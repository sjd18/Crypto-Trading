# Research workflow

The platform is organised around one question: *would this have made money live, after costs,
without using information from the future and without fooling ourselves through selection?*

```
collect / synth ─> verify-data ─> backtest ─> optimize (train → validation) ─> walkforward
                                     │                  │
                                     │                  └─> DSR / PBO ─> unseal test + live-sim once
                                     ├─> montecarlo (trade-level + path-level stress)
                                     ├─> report / dashboard
                                     └─> train-model (purged CV, SHAP) ─> rug_model.model_path
```

## 1. Data

* `collect-history` backfills Pump program transactions (resumable checkpoints; failed fetches go
  to a retry queue; slot gaps are registered). `stream` records the live feed.
* `verify-data` re-hashes every Parquet file against the manifest and reports slot gaps.
* Token metadata (name, symbol, socials, anomalies) is joined point-in-time at creation.

## 2. Backtest

```bash
python -m pumpfun_hft.main backtest --strategy momentum_ignition --start 2026-09-01 --end 2026-09-08
```

The run directory `pumpfun_hft/reports/runs/<run_id>/` holds `result.json` (metrics, parameters,
config and data fingerprints, diagnostics), `trades / fills / equity / signals.parquet`, the
wallet and token snapshots, and `report/` (HTML, PDF, CSV, JSON). The same run is written to the
DuckDB warehouse (`query "select * from runs"`).

Metrics: total return, PnL, CAGR and Calmar (spans ≥ 30 days), return / max drawdown, Sharpe,
Sortino and Omega on `returns_bar_ms` bars, max drawdown, longest and total time under water,
win rate, profit factor, expectancy (SOL, return, R), average R multiple, Kelly fraction, payoff
ratio, median trade, MAE / MFE, streaks, fill / failure / drop rates, latency percentiles,
slippage vs the decision-time quote, and every cost component.

## 3. Optimisation without touching the test set

```bash
python -m pumpfun_hft.main optimize --strategy smart_money --method bayesian --trials 40
python -m pumpfun_hft.main optimize --strategy smart_money --method bayesian --trials 40 --final   # unseal once
```

1. **Splits.** Time is cut into `train / validation / test / live_sim` (`optimizer.splits`) with an
   embargo between them. Only tokens created inside a split trade in it; earlier data warms up
   wallet and creator state (past information only).
2. **Search on train** with grid, random, Bayesian (Gaussian process with an ARD Matérn-5/2
   kernel, expected improvement, constant-liar batches for parallel workers) or a genetic
   algorithm. The objective (`optimizer.objective`) is penalised below `min_trades` and above
   `max_drawdown_pct`.
3. **Select on validation**: the top-k train candidates are re-run on validation. With
   `selection: plateau`, candidates whose neighbourhood in parameter space also scores well are
   preferred over knife-edge optima.
4. **Selection-bias statistics** over *all* trials:
   * **Deflated Sharpe Ratio** — the probability that the selected Sharpe beats the Sharpe you
     would expect from the best of N random strategies, given skewness, kurtosis and sample length.
   * **PBO** (CSCV) — how often the in-sample winner lands below the out-of-sample median across
     combinatorial splits. PBO near 0.5 or above means the selection process is not finding
     anything that persists.
5. **Final estimate**: `--final` unseals `test` and `live_sim` exactly once; the unseal is logged
   with a reason in `study.json` (`unseal_log`), so every look at the hold-out is auditable.

Outputs: `pumpfun_hft/reports/optimize/<study_id>/{trials.parquet, study.json}`.

## 4. Walk-forward

```bash
python -m pumpfun_hft.main walkforward --strategy momentum_ignition --trials 20
```

Rolling (or anchored) folds of search → validation selection → out-of-sample test. The stitched
out-of-sample equity is the honest estimate; walk-forward efficiency (OOS / IS score) and the
stability of the chosen parameters across folds show whether the optimum is real.

## 5. Monte Carlo

```bash
python -m pumpfun_hft.main montecarlo --sims 10000 --paths 12
```

* **Trade level** (fast, Numba): resample closed trades (bootstrap, permutation or block
  bootstrap) and perturb each trade's slippage, fees, latency cost and size with log-normal
  multipliers. Outputs quantiles of total return, drawdown and longest drawdown, the probability
  of loss and the probability of ruin (equity below `ruin_equity_frac`), and an equity-path fan.
  Because the perturbations are adverse on average, this is a stress test around the realised
  trades, not a forecast.
* **Path level** (slow, full fidelity): re-run the event-driven backtest with new seeds and
  latency / failure multipliers drawn from `path_latency_scale` / `path_failure_scale`.

## 6. Machine learning

```bash
python -m pumpfun_hft.main train-model --model lightgbm --target rug
```

* **Dataset** (`ml/dataset.py`): for every token, snapshots at `rug_model.snapshot_delays_s` after
  creation, taken by replaying events through the same online state as live trading — a snapshot
  contains only what was known at that instant. Labels come strictly from later events:
  `rug` (liquidity falls ≥ `label_drawdown_pct` % below the snapshot within `label_horizon_s`),
  `fwd_return` / `fwd_up`, `migrate`.
* **Validation** (`ml/cv.py`): purged forward-chaining CV. Training rows must have their label
  window end before the test block starts, precede it by `embargo_s`, and belong to tokens that
  do not appear in the test block.
* **Models**: logistic regression, random forest, XGBoost, LightGBM, CatBoost. Reports AUC, log
  loss, Brier score and precision in the top decile per fold; importance as model-native,
  permutation (last fold) and mean |SHAP|.
* **Training cut-off**: every model records the end of its last label window. It refuses to
  score any event before it, so it can never be backtested on the data it learned from. Train on
  the older part of your data with `--end` and test on the rest with `--start` (both printed by
  `train-model`).

### Deploying a model

Which slot a model goes in depends on its target; each slot refuses the other kind.

**`fwd_up` (or `migrate`) → the `ml_signal` strategy**, which buys when the model's probability
is at least `min_prob`:

```bash
train-model --model lightgbm --target fwd_up --end 2026-09-20T00:00:00Z   # train on the older data
backtest     --strategy ml_signal --start <cut-off printed above>          # out-of-sample test
paper-replay --strategy ml_signal --start <cut-off>                        # same, through the live engine
paper        --strategy ml_signal --minutes 120                            # live data, simulated fills
```

* `strategy.params.ml_signal.model_path: latest` loads the newest `*-fwd_up-*.joblib` in
  `paths.models_dir`; set a file name to pin one model.
* It scores each token once per training snapshot delay (`rug_model.snapshot_delays_s`), the only
  moments the model was trained on, with the same feature code that built the training rows
  (`ml.dataset.model_features`).
* Exits: held for at most the label horizon (`ml.fwd_return_horizon_s`), plus the usual stop /
  take-profit / trailing stop and the rug-avoidance overlay. Entries are also gated by
  `max_rug_prob`, the overlay's veto and the cost gate
  (`p * up_return - (1 - p) * down_return` vs `strategy.cost_gate_multiple` x round-trip cost).
* After a run the CLI prints the funnel — scored, below `min_prob`, rug-blocked, signals,
  skipped before cut-off — so a run with no trades shows where the candidates went.
* `optimize --strategy ml_signal` tunes `min_prob` and `max_rug_prob`.

**`rug` → the rug scorer** used by every strategy's `max_rug_prob` gate and the overlay: set
`rug_model.use_trained_model: true` and `rug_model.model_path` to the saved file.

On the default synthetic market the rug model reaches a purged-CV AUC of about 0.82 — the
planted serial-rugger structure is learnable. Expect lower numbers on real data.

## 7. Notebooks

`pumpfun_hft/notebooks/` walks through the same workflow in code:

| Notebook | Shows |
|---|---|
| `01_data_and_features.ipynb` | event store, token lifecycle, online vs batch feature parity, wallet intelligence |
| `02_backtest_and_report.ipynb` | a backtest, the ledger reconciliation, costs, the HTML report |
| `03_optimization_and_validation.ipynb` | splits, Bayesian search, DSR, PBO, walk-forward, Monte Carlo |
| `04_rug_model_and_ml.ipynb` | point-in-time dataset, purged CV, SHAP, the look-ahead guard |
