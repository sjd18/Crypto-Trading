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
| Overview | headline KPIs, equity and drawdown, per-strategy breakdown, provenance |
| Trades | every closed trade, exit-reason summary, every fill and failed attempt (sortable) |
| Equity curve | zoomable equity with drawdown panel, monthly returns |
| Drawdown | drawdown statistics and the deepest episodes |
| Heatmaps | PnL by weekday × UTC hour, PnL by hour, table view |
| Wallet explorer | ranked wallets (activity vs realised PnL coloured by smart score), label counts, searchable wallet table |
| Token explorer | price path of a traded token with our buys and sells, every token seen with its point-in-time outcome |
| Feature importance | model-native, permutation and SHAP importance of the latest trained model |
| Strategy comparison | indexed equity of the last 8 runs, metrics side by side |
| Live monitor | equity, positions, queue, latency percentiles and breaker state of the running paper / live session (polls `/api/live`) |

JSON endpoints: `/api/runs`, `/api/runs/{run_id}/metrics`, `/api/runs/{run_id}/equity`,
`/api/live`; interactive API docs at `/api/docs`.

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
