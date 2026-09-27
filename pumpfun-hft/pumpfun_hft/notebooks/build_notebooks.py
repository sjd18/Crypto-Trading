"""Build (and execute) the research notebooks.

    python pumpfun_hft/notebooks/build_notebooks.py            # write + execute all four
    python pumpfun_hft/notebooks/build_notebooks.py --no-exec  # write only

Every notebook generates its own small synthetic market in memory, so it runs anywhere in a
minute or two and never touches your data directory. Each one shows how to swap in real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).parent

SETUP = """\
import warnings; warnings.filterwarnings("ignore")
import numpy as np, polars as pl, matplotlib.pyplot as plt
from pumpfun_hft.core.config import load_settings
from pumpfun_hft.collectors.synthetic import SyntheticMarket

pl.Config.set_tbl_rows(12); pl.Config.set_tbl_cols(14); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(160)
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#898781"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "font.size": 9})
"""

REAL_DATA_NOTE = """\
> **Real data instead of the synthetic market.** Replace the generation cell with
> ```python
> from pumpfun_hft.collectors.storage import ParquetEventStore
> from pumpfun_hft.backtester.replay import load_metadata
> events = ParquetEventStore(settings.paths.events_dir).read()
> metadata = load_metadata(settings.paths.metadata_dir / "tokens.parquet")
> ```
> after running `collect-history` (or `stream`) from the CLI.
"""


def nb01() -> nbformat.NotebookNode:
    c = [
        new_markdown_cell("# 01 · Data and features\n\nThe event store, a token's life on the bonding curve, the exact "
                          "curve math, online-vs-batch feature parity, and wallet / creator intelligence — all point-in-time.\n\n"
                          + REAL_DATA_NOTE),
        new_code_cell(SETUP),
        new_code_cell("""\
settings = load_settings(None, {"synthetic.duration_hours": 6})
market = SyntheticMarket(settings)
data = market.generate()                              # synthetic: plants informed wallets + serial ruggers
events, metadata, truth = data.events, data.metadata, data.truth
print(f"{events.height:,} events · {truth.height} launches · {int(truth['complete_ms'].is_not_null().sum())} graduations")
events.group_by("kind").len().sort("len", descending=True)"""),
        new_markdown_cell("## Every event, one schema\nSwaps carry the reserves *after* the trade, the fee bps and amounts "
                          "recorded on chain, and a total order `(slot, seq, ev_idx)`."),
        new_code_cell("""\
events.filter(pl.col("kind") == "trade").select("slot", "seq", "ts_ms", "mint", "user", "is_buy", "sol_amount",
                                                 "token_amount", "v_sol", "v_tok", "r_sol", "fee_bps", "fee").head(5)"""),
        new_markdown_cell("## A token's life\nPrice is `v_sol / v_tok` (SOL per whole token); real SOL liquidity is `r_sol`. "
                          "On a bonding curve the price can never fall below the launch price, which is why rugs are "
                          "labelled by **liquidity** drawdown, not price drawdown."),
        new_code_cell("""\
top = (events.filter(pl.col("kind") == "trade").group_by("mint").len().sort("len", descending=True)["mint"][0])
tok = (events.filter((pl.col("mint") == top) & (pl.col("kind") == "trade"))
       .with_columns((pl.col("v_sol") / pl.col("v_tok") / 1000).alias("price"), (pl.col("r_sol") / 1e9).alias("liq"),
                     pl.from_epoch("ts_ms", time_unit="ms").alias("t")))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a1.plot(tok["t"], tok["price"], color=BLUE, lw=1.4); a1.set_ylabel("SOL / token"); a1.set_yscale("log")
a1.set_title(f"{top[:8]}…  ({tok.height} swaps)", loc="left")
a2.plot(tok["t"], tok["liq"], color=AQUA, lw=1.4); a2.set_ylabel("real SOL")
plt.tight_layout(); plt.show()
truth.filter(pl.col("mint") == top).select("creator_kind", "quality", "viral", "rugged", "peak_multiple")"""),
        new_markdown_cell("## Exact curve math\nInteger arithmetic identical to the on-chain program (verified against 792 "
                          "golden cases from the official SDK). A round trip at launch costs the protocol + creator fee twice "
                          "plus our own price impact."),
        new_code_cell("""\
from pumpfun_hft.core.curve import BondingCurve
curve = BondingCurve.from_config(settings.protocol.curve, settings.protocol.curve_fee_tiers)
st = curve.new_state(); fees = curve.fees_for(st)
buy = curve.buy_with_budget(st, 1_000_000_000, fees)                       # spend exactly 1 SOL
after = curve.apply_buy(st, buy)
sell = curve.sell_proceeds_for_tokens(after, buy.tokens, fees)
print(f"1 SOL buys {buy.tokens / 1e6:,.0f} tokens at {buy.avg_price:.3e} SOL each (spot {st.price:.3e})")
print(f"fees in: {(buy.protocol_fee + buy.creator_fee) / 1e9:.5f} SOL · immediate sell returns {sell.net / 1e9:.5f} SOL")
print(f"round-trip cost at launch: {100 * (1 - sell.net / buy.total):.2f} %")"""),
        new_markdown_cell("## Online vs batch features: exact parity\nStrategies use the online `FeatureEngine`; research uses "
                          "the vectorised Polars pipeline. Both are computed from strictly past data, and they agree to "
                          "floating-point precision."),
        new_code_cell("""\
from pumpfun_hft.features.replay import online_features
from pumpfun_hft.features.batch import batch_features
cols = ["buy_sol_short", "sell_sol_medium", "volume_sol_long", "imbalance_medium", "vwap_dist_medium",
        "ret_short", "ret_long", "progress_pct", "liquidity_sol", "age_s"]
online = online_features(settings, events, cols)
batch = batch_features(events, settings.features, settings.protocol.curve.initial_real_token_reserves)
pl.DataFrame({"feature": cols, "max_abs_diff": [float(np.nanmax(np.abs(online[c].to_numpy() - batch[c].to_numpy()))) for c in cols]})"""),
        new_markdown_cell("## Wallet intelligence\nEach wallet's *smart score* is the posterior probability that its realised "
                          "round-trip edge is positive (shrunk towards zero for small samples), updated online. The synthetic "
                          "market knows which wallets were planted as informed, so we can check the ranking."),
        new_code_cell("""\
from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource
eng = BacktestEngine(settings, DataSource(frame=events), ["momentum_ignition"], metadata=metadata, synthetic=True)
res = eng.run()                                    # replays everything: wallet DB, creator book, outcomes
w = eng.wallets.to_frame(min_trades=3)
planted = {k: set(market.wallets.get(k, [])) for k in ("smart", "sniper", "bot", "whale")}   # ground truth
top = w.drop_nulls("smart_score").sort("smart_score", descending=True).head(50)
kinds = [next((k for k, s in planted.items() if a in s), "retail/other") for a in top["address"]]
print("planted class of the 50 highest-scored wallets:")
pl.Series("class", kinds).value_counts().sort("count", descending=True)"""),
        new_markdown_cell("Informed wallets rank highly — but so do fast copy-trading bots and snipers, which are "
                          "*profitable* in this market precisely because slower followers pay their impact. A high smart "
                          "score says a wallet makes money, not that following it will."),
        new_code_cell("""\
outcomes = eng.resolver.to_frame().join(truth.select("mint", "creator_kind"), on="mint")
(outcomes.group_by("creator_kind").agg(pl.len(), (pl.col("label") == "rug").mean().alias("labelled_rug"),
                                       (pl.col("label") == "success").mean().alias("labelled_success"))
 .sort("creator_kind"))"""),
        new_markdown_cell("Outcomes are resolved one hour after each launch using only what happened until then; they feed "
                          "the creator book (Beta posteriors) that strategies query point-in-time."),
    ]
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def nb02() -> nbformat.NotebookNode:
    c = [
        new_markdown_cell("# 02 · Backtest and report\n\nAn event-driven backtest with realistic execution, the lamport-exact "
                          "ledger, where the money goes, how sensitive the result is to latency, and the generated report.\n\n"
                          + REAL_DATA_NOTE),
        new_code_cell(SETUP),
        new_code_cell("""\
settings = load_settings(None, {"synthetic.duration_hours": 12})
data = SyntheticMarket(settings).generate()
events, metadata = data.events, data.metadata
from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource
runs = {}
for name in ["momentum_ignition", "smart_money", "volume_breakout"]:
    runs[name] = BacktestEngine(settings, DataSource(frame=events), [name], metadata=metadata, seed=42, synthetic=True).run()
pl.DataFrame([{"strategy": k, **{m: r.metrics.get(m) for m in ("total_return", "n_trades", "win_rate", "profit_factor",
              "max_drawdown", "fill_rate", "avg_latency_ms")}} for k, r in runs.items()])"""),
        new_markdown_cell("`smart_money` profits here **by construction**: the generator plants informed wallets. The others "
                          "lose after costs. Neither says anything about the live market."),
        new_markdown_cell("## The ledger reconciles to the lamport\nFinal equity = initial capital + the PnL of every closed "
                          "round trip − fees of failed transactions that never opened a position."),
        new_code_cell("""\
r = runs["momentum_ignition"]
lhs = r.metrics["final_equity_sol"]
rhs = r.initial_capital_sol + r.trades["pnl_sol"].sum() - r.metrics["unattributed_costs_sol"]
print(f"final equity {lhs:.9f} SOL  vs  capital + trade PnL - unattributed {rhs:.9f} SOL  → diff {abs(lhs - rhs) * 1e9:.0f} lamports")"""),
        new_markdown_cell("## Where the money goes"),
        new_code_cell("""\
f = r.fills
costs = {k: float(f[c].sum()) / 1e9 for k, c in [("protocol fees", "protocol_fee"), ("creator fees", "creator_fee"),
         ("platform fee (Mode A)", "platform_fee"), ("priority fees", "priority_fee"), ("base fees", "network_fee")]}
gross = r.trades["pnl_sol"].sum() + sum(costs.values())
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.barh(list(costs), list(costs.values()), color=BLUE, height=0.5)
for i, v in enumerate(costs.values()): ax.text(v, i, f" {v:.3f}", va="center")
ax.set_xlabel("SOL"); ax.set_title(f"costs paid · PnL before costs {gross:+.3f} SOL, after {r.metrics['pnl_sol']:+.3f} SOL", loc="left")
plt.tight_layout(); plt.show()"""),
        new_markdown_cell("## Execution quality\nPositive slippage is adverse (vs the quote at decision time). Buys during "
                          "bursts land behind everyone who was faster."),
        new_code_cell("""\
from pumpfun_hft.analytics.report import ReportGenerator, exit_categories
display(ReportGenerator(r).execution_quality())
exit_categories(r.trades)"""),
        new_markdown_cell("## Latency sensitivity\nThe same strategy with the whole latency distribution scaled. A strategy "
                          "whose edge disappears at 2× latency does not have an edge you can capture from a normal RPC."),
        new_code_cell("""\
rows = []
for scale in (0.5, 1.0, 2.0, 4.0):
    rr = BacktestEngine(settings, DataSource(frame=events), ["smart_money"], metadata=metadata, seed=42,
                        latency_scale=scale, synthetic=True).run()
    rows.append({"latency_scale": scale, "p50_latency_ms": rr.diagnostics["latency_ms"]["p50"],
                 "total_return": rr.metrics["total_return"], "n_trades": rr.metrics["n_trades"]})
lat = pl.DataFrame(rows); lat"""),
        new_code_cell("""\
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.plot(lat["p50_latency_ms"], 100 * lat["total_return"], "o-", color=BLUE, lw=2)
ax.axhline(0, color=GRAY, lw=0.8); ax.set_xlabel("median decision → landing latency (ms)"); ax.set_ylabel("total return %")
ax.set_title("smart_money vs latency", loc="left"); plt.tight_layout(); plt.show()"""),
        new_markdown_cell("## The report\nEvery CLI backtest writes this automatically; here we build one by hand."),
        new_code_cell("""\
import tempfile
from pumpfun_hft.analytics.montecarlo import trade_monte_carlo
mc = trade_monte_carlo(r.trades, r.initial_capital_sol, r.metrics["span_days"], settings.montecarlo, r.metrics["avg_latency_ms"], 2000)
out = ReportGenerator(r, mc).generate(tempfile.mkdtemp(), ["html", "json", "csv"])
print(out["html"]); print("P(loss) under resampling + cost stress:", round(mc.prob_loss, 3))"""),
    ]
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def nb03() -> nbformat.NotebookNode:
    c = [
        new_markdown_cell("# 03 · Optimisation and validation\n\nSearch on train, select on validation, measure selection "
                          "bias (Deflated Sharpe, PBO), unseal the test set exactly once, walk forward, and stress the "
                          "result with Monte Carlo.\n\n" + REAL_DATA_NOTE),
        new_code_cell(SETUP),
        new_code_cell("""\
import tempfile
settings = load_settings(None, {"synthetic.duration_hours": 24, "optimizer.n_workers": 2})
data = SyntheticMarket(settings).generate()
events = data.events
from pathlib import Path
meta_path = str(Path(tempfile.mkdtemp()) / "tokens.parquet"); data.metadata.write_parquet(meta_path)
from pumpfun_hft.optimizer.splits import make_splits
start, end = int(events["ts_ms"].min()), int(events["ts_ms"].max()) + 1
print(make_splits(start, end, settings.optimizer.splits).describe())"""),
        new_markdown_cell("Train / validation / test / live-sim are consecutive, embargoed time ranges. Test and live-sim are "
                          "**sealed**: reading them requires an explicit, logged `unseal`."),
        new_code_cell("""\
from pumpfun_hft.optimizer.study import OptimizationStudy
study = OptimizationStudy(settings, events, meta_path, "smart_money", method="bayesian", n_trials=12)
res = study.run(final_evaluation=True)
tr = pl.DataFrame([{"trial": t["number"], **t["params"], "score": t["score"], "trades": (t["metrics"] or {}).get("n_trades")}
                   for t in res.trials])
tr.sort("score", descending=True).head(8)"""),
        new_code_cell("""\
print("selected:", res.selected_params)
print(f"train return {res.selected_train_metrics.get('total_return', float('nan')):+.4f} · "
      f"validation return {res.selected_val_metrics.get('total_return', float('nan')):+.4f}")
print(f"Deflated Sharpe {res.dsr['dsr']:.3f} (per-period Sharpe {res.dsr['sharpe']:.3f} vs expected max {res.dsr['sr_star']:.3f})")
print(f"PBO {res.pbo['pbo']:.2f} over {int(res.pbo['n_combinations'])} CSCV combinations")
print(f"sealed test: return {res.test_metrics.get('total_return', float('nan')):+.4f}, trades {res.test_metrics.get('n_trades')}")
print("unseal log:", [u["reason"] for u in res.unseal_log])"""),
        new_markdown_cell("A PBO near or above 0.5 means the in-sample winner is no better than a coin flip out of sample: "
                          "the search is fitting noise, whatever the train score says."),
        new_code_cell("""\
from pumpfun_hft.optimizer.overfitting import pbo_cscv
rng = np.random.default_rng(0)
noise = rng.normal(0, 1, size=(400, 30))                          # 30 strategies with zero true edge
edge = noise.copy(); edge[:, 0] += 0.35                           # one strategy with a real edge
print("PBO, pure noise :", round(pbo_cscv(noise)["pbo"], 2))
print("PBO, one real edge:", round(pbo_cscv(edge)["pbo"], 2))"""),
        new_markdown_cell("## Walk-forward"),
        new_code_cell("""\
from pumpfun_hft.optimizer.study import WalkForwardAnalysis
wf = WalkForwardAnalysis(settings, events, meta_path, "smart_money", method="random", n_trials=6).run()
print(f"walk-forward efficiency {wf.wfe:.2f} · stitched out-of-sample return {wf.oos_total_return:+.4f}")
wf.frame().select([c for c in wf.frame().columns if c in ("fold", "is_score", "val_score", "oos_score", "oos_total_return", "oos_n_trades")
                   or c.startswith("p_")])"""),
        new_markdown_cell("## Monte Carlo\nResample the closed trades and perturb slippage, fees, latency and size. The "
                          "perturbations are adverse on average: this is a stress test, not a forecast."),
        new_code_cell("""\
from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource
from pumpfun_hft.analytics.montecarlo import trade_monte_carlo
bt = BacktestEngine(settings, DataSource(frame=events), ["smart_money"], metadata=data.metadata, synthetic=True).run()
mc = trade_monte_carlo(bt.trades, bt.initial_capital_sol, bt.metrics["span_days"], settings.montecarlo, bt.metrics["avg_latency_ms"], 5000)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.8))
a1.hist(100 * mc.distributions["total_return"], bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
a1.axvline(100 * bt.metrics["total_return"], color=ORANGE, lw=2, label="backtest"); a1.axvline(0, color=GRAY, lw=0.8)
a1.set_xlabel("total return %"); a1.legend(frameon=False)
q = np.quantile(mc.paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0); x = np.arange(1, q.shape[1] + 1)
a2.fill_between(x, q[0], q[4], color=BLUE, alpha=0.12, lw=0); a2.fill_between(x, q[1], q[3], color=BLUE, alpha=0.25, lw=0)
a2.plot(x, q[2], color=BLUE, lw=2); a2.set_xlabel("trade #"); a2.set_ylabel("equity (SOL)")
plt.tight_layout(); plt.show()
print(f"P(loss) {mc.prob_loss:.2f} · P(ruin) {mc.prob_ruin:.3f}")"""),
    ]
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def nb04() -> nbformat.NotebookNode:
    c = [
        new_markdown_cell("# 04 · Rug model and machine learning\n\nPoint-in-time snapshots, forward labels, purged "
                          "forward-chaining cross-validation, explainability, and the guard that stops a model from scoring "
                          "the data it was trained on.\n\n" + REAL_DATA_NOTE),
        new_code_cell(SETUP),
        new_code_cell("""\
settings = load_settings(None, {"synthetic.duration_hours": 24})
data = SyntheticMarket(settings).generate()
from pumpfun_hft.ml.dataset import build_snapshot_dataset, feature_columns
ds = build_snapshot_dataset(settings, data.events, {r["mint"]: r for r in data.metadata.iter_rows(named=True)})
cols = feature_columns(ds)
print(f"{ds.height:,} snapshots × {len(cols)} features · rug base rate {ds['rug'].mean():.3f}")
ds.select("mint", "delay_s", "snapshot_ms", "label_end_ms", "snap_liq", "min_liq", "rug", "fwd_return").head(6)"""),
        new_markdown_cell("Each row is a token observed `delay_s` seconds after launch, with features computed from the "
                          "online state at that instant. The `rug` label (liquidity falls ≥ 80 % below the snapshot within "
                          "30 minutes) comes only from later events, and `label_end_ms` lets cross-validation purge overlaps."),
        new_code_cell("""\
from pumpfun_hft.ml.models import train_and_evaluate
reports = {kind: train_and_evaluate(ds, cols, "rug", settings.ml.model_copy(update={"model": kind}), settings.app.seed)
           for kind in ("logistic", "lightgbm")}
pl.DataFrame([{"model": k, **{m: r.mean.get(m) for m in ("auc", "log_loss", "brier", "precision_top_decile", "base_rate")}}
              for k, r in reports.items()])"""),
        new_code_cell("""\
pl.DataFrame(reports["lightgbm"].folds).select("fold", "n_train", "n_test", "base_rate", "auc", "precision_top_decile")"""),
        new_markdown_cell("Training rows always end before the test block starts (purged + embargoed), and tokens never "
                          "appear on both sides. Later folds train on more history."),
        new_code_cell("""\
imp = reports["lightgbm"].importance.sort("shap", descending=True, nulls_last=True).head(12)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(imp["feature"][::-1], imp["shap"][::-1], color=BLUE, height=0.55)
ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title("what drives the rug score", loc="left"); plt.tight_layout(); plt.show()"""),
        new_markdown_cell("## The look-ahead guard\nA saved model records the last label time it saw. Scoring anything "
                          "earlier raises `LookAheadError`, so a model can never be backtested on its own training data."),
        new_code_cell("""\
import tempfile
from pathlib import Path
from pumpfun_hft.ml.rug_model import LookAheadError, TrainedRugModel
path = reports["logistic"].save(Path(tempfile.mkdtemp()) / "rug.joblib")
model = TrainedRugModel.load(path)
x = {f: 0.0 for f in model.features}
print("score after the cut-off:", round(model.predict(x, now_ms=model.train_end_ms + 1), 4))
try:
    model.predict(x, now_ms=model.train_end_ms - 1)
except LookAheadError as e:
    print("refused:", e)"""),
    ]
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


NOTEBOOKS = {"01_data_and_features.ipynb": nb01, "02_backtest_and_report.ipynb": nb02,
             "03_optimization_and_validation.ipynb": nb03, "04_rug_model_and_ml.ipynb": nb04}


def build(execute: bool = True, only: list[str] | None = None, timeout: int = 1200) -> list[Path]:
    out = []
    for name, fn in NOTEBOOKS.items():
        if only and not any(o in name for o in only):
            continue
        nb = fn()
        path = HERE / name
        if execute:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, path)
        out.append(path)
        print("wrote", path)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build(execute="--no-exec" not in sys.argv, only=args or None)
