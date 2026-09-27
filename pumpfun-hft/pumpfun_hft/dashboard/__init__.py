"""Interactive analytics dashboard.

Purpose
    Local, offline dashboard over persisted runs, research artefacts and the live session.

Architecture
    app.py    FastAPI app (HTML pages + JSON API + offline Plotly JS) and ``static_export`` (one
              self-contained HTML file with every page as a tab, for sharing)
    views.py  page renderers: Overview, Trades (every fill), Equity curve, Drawdown, Heatmaps
              (weekday x hour PnL), Wallet explorer, Token explorer (price + our fills), Feature
              importance, Strategy comparison, Live monitor (polls the live snapshot)

Data flow
    reports/runs/<run_id>/{result.json, *.parquet, wallets.parquet, tokens.parquet}
    reports/ml/<model_id>/importance.parquet ; SQLite live_state (LiveTrader snapshots)
    events Parquet store (token price paths)

Inputs / Outputs
    Inputs: the files listed under Data flow plus the ``dashboard`` config section (host, port,
    table sizes, live refresh interval).
    Outputs: HTML pages, JSON endpoints (``/api/runs``, ``/api/runs/{id}/metrics``,
    ``/api/runs/{id}/equity``, ``/api/live``; strict JSON, NaN as null) and a single-file export.

Example
    python -m pumpfun_hft.main dashboard            # http://127.0.0.1:8050
    python -m pumpfun_hft.main dashboard-export     # -> reports/dashboard.html
"""
