# Dashboard and reports

## Dashboard

```bash
python -m pumpfun_hft.main dashboard            # FastAPI on http://127.0.0.1:8050 (dashboard.host / port)
python -m pumpfun_hft.main dashboard-export     # pumpfun_hft/reports/dashboard.html, every page as a tab
```

Everything is local: Plotly's JavaScript is served from the installed Python package (or inlined
in the export), so the dashboard works offline.

With a data set selected (`--dataset`, i.e. `hft dashboard` / `hftr dashboard`), the dashboard,
reports and exports use that data set's `reports/` folder instead of `pumpfun_hft/reports/`, so the
synthetic and real runs are listed separately. Both use the same port: run one at a time.

| Page | Shows |
|---|---|
| Strategy builder | build and run a trading session: pick the mode (`hft`, `hftr`, `hfts` (default), `hftl`), one or more strategies from `strategies/`, a duration and options; the page shows the exact PowerShell command (Copy) and can run it (**Run in dashboard**, **Stop**). Beside it: a status light for the RPC connection (e.g. Helius) and the session, the session's data (positions, PnL, latency) and its raw PowerShell output. See below |
| Overview | headline KPIs, equity and drawdown, per-strategy breakdown, provenance |
| Trades | every closed trade, exit-reason summary, every fill and failed attempt (sortable) |
| Equity curve | zoomable equity with drawdown panel, monthly returns |
| Drawdown | drawdown statistics and the deepest episodes |
| Heatmaps | PnL by weekday × UTC hour, PnL by hour, table view |
| Wallet explorer | ranked wallets (activity vs realised PnL coloured by smart score), label counts, searchable wallet table |
| Token explorer | price path of a traded token with our buys and sells, every token seen with its point-in-time outcome |
| Feature importance | model-native, permutation and SHAP importance of the latest trained model |
| Strategy comparison | indexed equity of the last 8 runs, metrics side by side |
| Live monitor | the running paper / live session (`hfts trade`, `hftl trade`, `hftr trade`): profile, equity, return, realised PnL, closed trades and win rate, open positions, fills / failed, stream events and reconnects, queue, latency percentiles and breaker state. The whole page refreshes every `dashboard.live_refresh_ms` from `/api/live/html`. Start the dashboard with the same command as the session (`hfts dashboard`) so it reads that data set |

JSON endpoints: `/api/runs`, `/api/runs/{run_id}/metrics`, `/api/runs/{run_id}/equity`,
`/api/live` (plus `/api/live/html`, the rendered Live monitor); interactive API docs at `/api/docs`.

### Strategy builder

`hfts dashboard`, then open http://127.0.0.1:8050/builder (the first tab).

* **Run in dashboard** starts the same command the page shows, with the Python that serves the
  dashboard, on the data-set folder `hft.ps1` points at (`HFT_SYNTH_DIR` / `HFT_REAL_DIR`, else
  `datasets.*` in the config) and the `HFT_CONFIG` file. One session at a time; its output streams
  into the black panel, and the data panel follows its snapshots.
* **Stop** ends a live-stream session the normal way (positions closed, final snapshot) through a
  stop file the CLI polls; a replay (`hft` / `hftr`) is terminated. Stopping the dashboard stops its
  session too.
* **`hftl` is never started from the page.** It builds the command; real-money trading is started by
  hand in PowerShell, where it still needs `app.mode=live` and `--confirm-live`.
* The status light: green = the RPC answers `getSlot` (round trip shown, checked at most every
  15 s) and the session is updating; amber = rate-limited, slow (> 1.5 s), no `SOLANA_WS_URL`, or a
  session that stopped updating; red = no RPC configured or not reachable. Only the provider's host
  name is shown, never the URL (it carries your API key).
* The endpoints that start or stop sessions (`/api/jobs`, `/api/jobs/stop`, `/api/builder/command`)
  answer only the dashboard's own page: a custom request header that other websites cannot send
  without a CORS preflight, plus a Host / Origin check against `dashboard.host` / `port`.

## Reports

Every `backtest` writes `pumpfun_hft/reports/runs/<run_id>/report/`:

| File | Content |
|---|---|
| `report.html` | self-contained: KPIs, equity + drawdown, return and R-multiple distributions, MAE / MFE, streaks, hourly and weekday × hour PnL, monthly returns, exit reasons, largest winners / losers, costs, order outcomes, slippage and latency by side, breakers, Monte Carlo (distributions, fan, quantiles), all trades, all metrics |
| `report.pdf` | the same essentials for printing (ReportLab + Matplotlib) |
| `csv/` | trades, fills, equity, monthly returns, hourly profile, signals |
| `report.json` | metrics, diagnostics, parameters, Monte Carlo summary, config / data fingerprints |

`python -m pumpfun_hft.main report --run <run_id>` regenerates it.

## Chart rules

The charts follow a small set of rules so that every figure reads the same way:

* **Categorical colour in a fixed order and fixed per entity.** Strategies keep their colour
  across pages and runs (`charts.strategy_color`); the palette was checked for colour-vision
  deficiency separation and lightness in both themes.
* **Polarity uses the diverging pair**: gains blue, losses red (never green / red), with a neutral
  midpoint in heatmaps.
* **One axis per chart.** Equity and drawdown are stacked panels sharing time, never a dual axis.
* Thin marks (2 px lines, rounded bar ends, 2 px surface gaps), recessive grids, legends for two or
  more series, hover on every mark, and a table next to each chart so no number is colour- or
  hover-only.
* **Light and dark themes**: pages follow the OS setting; dark mode swaps surfaces, grid, ink
  *and* every series colour for its validated dark-mode step.
* All text in tables is HTML-escaped: token names and symbols are chosen by anyone who launches a
  token and must be treated as hostile input.
