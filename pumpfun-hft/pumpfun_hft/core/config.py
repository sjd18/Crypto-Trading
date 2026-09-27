"""Typed YAML configuration.

Purpose
    Single source of truth for every tunable parameter. ``configs/default.yaml`` must specify
    *every* field: the pydantic models below declare no defaults for tunables, so a missing key
    is a validation error rather than a silently hard-coded value (enforcing "no hardcoded
    parameters"). Unknown keys are rejected too (``extra="forbid"``) to catch typos.

Data flow
    default.yaml  ->  optional user YAML (deep-merged)  ->  dotted overrides  ->  Settings
    .env / environment  ->  Secrets (SecretStr, registered with the log redactor)

Example::

    s = load_settings(overrides={"backtest.initial_capital_sol": 5.0})
    s.backtest.initial_capital_sol
    secrets = Secrets.load()
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from pumpfun_hft.utils.hashing import stable_hash

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "default.yaml"

Commitment = Literal["processed", "confirmed", "finalized"]


class Strict(BaseModel):
    """Base model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- app / paths
class AppCfg(Strict):
    name: str
    mode: Literal["paper", "live"]
    seed: int


class PathsCfg(Strict):
    data_dir: str
    events_subdir: str
    metadata_subdir: str
    duckdb_file: str
    sqlite_file: str
    logs_dir: str
    reports_dir: str
    models_dir: str
    idl_dir: str

    def resolve(self, attr: str) -> Path:
        """Absolute path for a configured path attribute (relative paths are anchored at the project root)."""
        p = Path(getattr(self, attr))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def events_dir(self) -> Path:
        return self.resolve("data_dir") / self.events_subdir

    @property
    def metadata_dir(self) -> Path:
        return self.resolve("data_dir") / self.metadata_subdir


# ----------------------------------------------------------------------------- network
class RateLimitCfg(Strict):
    rps: float
    burst: int


class RetryCfg(Strict):
    max_attempts: int
    base_delay_s: float
    max_delay_s: float
    retry_statuses: list[int]


class HttpBreakerCfg(Strict):
    failure_threshold: int
    reset_timeout_s: float


class MetisCfg(Strict):
    mode: Literal["public", "authenticated"]
    public_url: str
    public_sunset_date: str
    public_platform_fee_bps: int
    timeout_s: float
    rate_limit: RateLimitCfg
    priority_fee_levels: dict[str, str]


class RpcCfg(Strict):
    timeout_s: float
    rate_limit: RateLimitCfg
    commitment: Commitment
    batch_size: int
    max_concurrency: int


class WsCfg(Strict):
    commitment: Commitment
    ping_interval_s: float
    ping_timeout_s: float
    reconnect_base_delay_s: float
    reconnect_max_delay_s: float
    max_queue: int


class FrontendCfg(Strict):
    enabled: bool
    base_url: str
    coin_path: str
    user_path: str
    timeout_s: float
    rate_limit: RateLimitCfg


class JwtCfg(Strict):
    algorithm: Literal["RS256", "ES256"]
    lifetime_s: int
    refresh_margin_s: int


class NetworkCfg(Strict):
    user_agent: str
    metis: MetisCfg
    rpc: RpcCfg
    ws: WsCfg
    pumpfun_frontend: FrontendCfg
    retry: RetryCfg
    http_breaker: HttpBreakerCfg
    jwt: JwtCfg


# ----------------------------------------------------------------------------- protocol / fees
class FeeTierCfg(Strict):
    mcap_sol: float
    protocol_bps: int
    creator_bps: int
    lp_bps: int


class CurveCfg(Strict):
    initial_virtual_token_reserves: int
    initial_virtual_sol_reserves: int
    initial_real_token_reserves: int
    token_total_supply: int


class ProtocolCfg(Strict):
    pump_program_id: str
    pump_amm_program_id: str
    pump_fees_program_id: str
    sol_mint: str
    token_decimals: int
    sol_decimals: int
    curve: CurveCfg
    curve_fee_tiers: list[FeeTierCfg]
    amm_fee_tiers: list[FeeTierCfg]
    amm_flat_fees: FeeTierCfg
    amm_event_reserves: Literal["pre", "post"]
    migration_fee_lamports: int

    @model_validator(mode="after")
    def _tiers_sorted(self) -> ProtocolCfg:
        for tiers in (self.curve_fee_tiers, self.amm_fee_tiers):
            th = [t.mcap_sol for t in tiers]
            if not tiers or th != sorted(th):
                raise ValueError("fee tiers must be non-empty and sorted by mcap_sol")
        return self


class FeesCfg(Strict):
    source: Literal["observed_if_available", "schedule"]
    base_fee_lamports_per_signature: int
    platform_fee_bps: int | None  # None = derive from network.metis (Mode A charges public_platform_fee_bps)
    token_account_rent_lamports: int
    charge_token_account_rent: bool
    close_token_account_on_exit: bool
    user_volume_accumulator_rent_lamports: int
    buy_instruction: Literal["buy_exact_sol_in", "buy"]


class SlippageCfg(Strict):
    buy_bps: int
    sell_bps: int
    exit_bps: int
    max_bps: int
    retry_widen_bps: int


class PriorityFeeCfg(Strict):
    mode: Literal["fixed", "dynamic"]
    compute_unit_limit: int
    fixed_micro_lamports: int
    dynamic_percentile: float
    min_micro_lamports: int
    max_micro_lamports: int
    urgency_multiplier: dict[str, float]
    retry_bump_factor: float
    refresh_interval_s: float


class JitoCfg(Strict):
    enabled: bool
    block_engine_url: str
    tip_mode: Literal["fixed", "floor_percentile"]
    tip_lamports: int
    tip_floor_url: str
    tip_floor_field: str
    min_tip_lamports: int
    max_tip_lamports: int
    rate_limit: RateLimitCfg


# ----------------------------------------------------------------------------- simulation
class LogNormalCfg(Strict):
    median_ms: float
    sigma: float


class LatencySimCfg(Strict):
    decision_ms: LogNormalCfg
    network_ms: LogNormalCfg
    rpc_ms: LogNormalCfg
    inclusion_ms: LogNormalCfg
    jito_inclusion_ms: LogNormalCfg
    confirmation_ms: LogNormalCfg
    priority_speedup: float
    spike_prob: float
    spike_multiplier: float


class FailureSimCfg(Strict):
    drop_prob: float
    landed_fail_prob: float
    congestion_events_per_s: float
    congestion_multiplier: float
    blockhash_ttl_ms: int
    jito_bundle_fail_prob: float


class OutageSimCfg(Strict):
    rate_per_hour: float
    mean_duration_s: float
    behavior: Literal["reject", "delay"]


class RateLimitSimCfg(Strict):
    max_orders_per_s: float
    burst: int
    behavior: Literal["reject", "delay"]


class RetrySimCfg(Strict):
    max_retries: int
    retry_on: list[Literal["dropped", "expired", "slippage", "failed", "rejected"]]
    backoff_ms: int


class SimulationCfg(Strict):
    slot_ms: int
    impact_model: Literal["persistent", "none"]
    latency: LatencySimCfg
    failures: FailureSimCfg
    outages: OutageSimCfg
    rate_limit: RateLimitSimCfg
    retries: RetrySimCfg


class BacktestCfg(Strict):
    initial_capital_sol: float
    replay_mode: Literal["event", "trade", "tick", "candle"]
    candle_interval_ms: int
    mark_to_market: Literal["liquidation", "mid"]
    equity_sample_ms: int
    sweep_interval_ms: int
    default_order_type: Literal["market", "limit", "ioc", "fok"]
    limit_offset_bps: int
    limit_ttl_ms: int
    force_close_at_end: bool
    min_order_sol: float
    cash_reserve_sol: float
    max_events: int | None
    annualization_days: float
    returns_bar_ms: int
    record_signals: bool


# ----------------------------------------------------------------------------- risk / sizing / position
class RiskLimitsCfg(Strict):
    daily_loss_sol: float
    hourly_loss_sol: float
    max_open_positions: int
    max_exposure_sol: float
    max_position_per_token_sol: float
    max_creator_exposure_sol: float
    max_sector_exposure_sol: float
    max_orders_per_token_per_min: int


class BreakersCfg(Strict):
    rpc_latency_p90_ms: float
    rpc_latency_window: int
    congestion_slot_ms: float
    slippage_bps_avg: float
    slippage_window: int
    failed_swaps_max: int
    failed_swaps_window_s: float
    drawdown_pct: float
    cooldown_s: float
    flatten_on_drawdown: bool


class SectorsCfg(Strict):
    keywords: dict[str, list[str]]
    default: str


class RiskCfg(Strict):
    limits: RiskLimitsCfg
    breakers: BreakersCfg
    sectors: SectorsCfg


class SizingCfg(Strict):
    method: Literal["fixed", "fixed_risk", "kelly", "volatility", "confidence", "max_exposure"]
    fixed_sol: float
    risk_per_trade_sol: float
    kelly_fraction: float
    kelly_cap_frac: float
    kelly_min_trades: int
    vol_target_sol: float
    vol_floor: float
    confidence_scaling: bool
    confidence_exponent: float
    min_confidence: float
    max_impact_bps: float
    max_equity_frac: float


class TakeProfitLevel(Strict):
    at_pct: float
    sell_frac: float


class TrailingCfg(Strict):
    enabled: bool
    activation_pct: float
    trail_pct: float


class PyramidCfg(Strict):
    enabled: bool
    max_adds: int
    add_trigger_pct: float
    add_size_frac: float
    min_confidence: float


class PositionCfg(Strict):
    stop_loss_pct: float
    take_profit: list[TakeProfitLevel]
    trailing_stop: TrailingCfg
    max_hold_s: float
    breakeven_after_first_tp: bool
    exit_on_curve_complete: bool
    pyramiding: PyramidCfg


# ----------------------------------------------------------------------------- research modules
class FeaturesCfg(Strict):
    short_window_ms: int
    medium_window_ms: int
    long_window_ms: int
    fast_half_life_ms: float
    slow_half_life_ms: float
    bar_ms: int
    atr_bars: int
    whale_trade_sol: float
    aggressive_impact_bps: float
    fresh_wallet_window_ms: int
    smart_score_threshold: float
    top_holders_k: int
    tps_half_life_ms: float


class MetadataCfg(Strict):
    enabled: bool
    ipfs_gateways: list[str]
    timeout_s: float
    max_concurrency: int
    cache_size: int


class DiscoveryCfg(Strict):
    resolution_horizon_s: float
    success_ath_multiple: float
    rug_drawdown_pct: float
    rug_creator_sold_pct: float
    creator_prior_alpha: float
    creator_prior_beta: float
    rug_prior_alpha: float
    rug_prior_beta: float
    experience_saturation: int
    duplicate_name_window_s: float
    score_weights: dict[str, float]
    metadata: MetadataCfg


class WalletIntelCfg(Strict):
    min_closed_for_rank: int
    prior_strength: float
    score_min_sd: float
    smart_label_min_score: float
    sniper_entry_s: float
    sniper_min_tokens: int
    sniper_min_early_frac: float
    whale_avg_trade_sol: float
    whale_min_buys: int
    mm_roundtrip_ratio: float
    mm_max_hold_s: float
    mm_min_closed: int
    bot_min_trades_per_hour: float
    bot_min_trades: int
    insider_min_cobuys: int
    rug_wallet_min_rugs: int
    max_wallets_tracked: int


class RugModelCfg(Strict):
    use_trained_model: bool
    model_path: str
    label_horizon_s: float
    label_drawdown_pct: float
    snapshot_delays_s: list[float]
    heuristic_weights: dict[str, float]


class StrategyCfg(Strict):
    active: list[str]
    exit_overlay: str | None
    cost_gate_multiple: float
    one_order_in_flight_per_token: bool
    reentry_cooldown_s: float
    params: dict[str, dict[str, Any]]


class ParamSpec(Strict):
    type: Literal["float", "int", "log_float", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical ParamSpec requires choices")
        elif self.low is None or self.high is None or self.low > self.high:
            raise ValueError("numeric ParamSpec requires low <= high")
        if self.type == "log_float" and (self.low is None or self.low <= 0):
            raise ValueError("log_float requires low > 0")
        return self


class SplitsCfg(Strict):
    train: float
    validation: float
    test: float
    live_sim: float
    embargo_s: float

    @model_validator(mode="after")
    def _sum(self) -> SplitsCfg:
        total = self.train + self.validation + self.test + self.live_sim
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"splits must sum to 1.0, got {total}")
        return self


class WalkForwardCfg(Strict):
    n_folds: int
    anchored: bool
    train_frac: float
    val_frac: float
    test_frac: float
    embargo_s: float


class BayesCfg(Strict):
    n_initial: int
    n_candidates: int
    xi: float
    restarts: int


class GeneticCfg(Strict):
    population: int
    generations: int
    crossover_prob: float
    mutation_prob: float
    mutation_scale: float
    tournament: int
    elite: int


class GridCfg(Strict):
    points_per_dim: int
    max_points: int


class OptimizerCfg(Strict):
    method: Literal["grid", "random", "bayesian", "genetic"]
    n_trials: int
    objective: Literal["sharpe", "sortino", "profit_factor", "expectancy", "calmar", "total_return", "composite"]
    min_trades: int
    max_drawdown_pct: float
    n_workers: int
    seed: int
    top_k: int
    selection: Literal["best_validation", "plateau"]
    splits: SplitsCfg
    walk_forward: WalkForwardCfg
    bayesian: BayesCfg
    genetic: GeneticCfg
    grid: GridCfg
    spaces: dict[str, dict[str, ParamSpec]]


class MonteCarloCfg(Strict):
    n_sims: int
    method: Literal["bootstrap", "permutation", "block_bootstrap"]
    block_size: int
    slippage_sigma: float
    fee_sigma: float
    latency_sigma: float
    latency_cost_bps_per_100ms: float
    sizing_sigma: float
    ruin_equity_frac: float
    quantiles: list[float]
    path_sims: int
    path_latency_scale: list[float]
    path_failure_scale: list[float]
    seed: int


class LiveCfg(Strict):
    quote_source: Literal["local", "metis"]
    swap_route: Literal["metis_swap", "metis_swap_instructions"]
    send_via: Literal["rpc", "jito"]
    queue_maxsize: int
    workers: int
    confirm_commitment: Commitment
    confirm_poll_ms: int
    confirm_timeout_ms: int
    rebroadcast_ms: int
    blockhash_refresh_ms: int
    max_retries: int
    quote_to_submit_budget_ms: float
    state_snapshot_s: float
    paper_initial_capital_sol: float
    curve_cache_ttl_ms: int
    skip_preflight: bool


class CollectorCfg(Strict):
    signatures_page_limit: int
    tx_batch_size: int
    max_concurrency: int
    fetch_block_hash: bool
    max_signatures: int
    flush_rows: int
    flush_interval_s: float
    large_trade_sol: float
    watch_wallets_max: int
    gap_slot_threshold: int
    live_latency_budget_ms: float
    backfill_on_reconnect: bool
    include_failed_tx: bool
    subscribe_amm: bool
    sol_price_source: Literal["static", "csv"]
    sol_price_static_usd: float
    sol_price_csv: str | None


class LoggingCfg(Strict):
    level: str
    max_bytes: int
    backup_count: int
    console: bool
    channels: list[str]


class DashboardCfg(Strict):
    host: str
    port: int
    live_refresh_ms: int
    max_table_rows: int


class SyntheticCfg(Strict):
    seed: int
    start: str
    duration_hours: float
    launches_per_hour: float
    n_creators: int
    creator_mix: dict[str, float]
    n_wallets: int
    wallet_mix: dict[str, float]
    max_token_life_s: float
    base_trade_rate_hz: float
    activity_floor: float
    activity_scale: float
    activity_exponent: float
    excitation: float
    excitation_decay_s: float
    dev_buy_sol: list[float]
    retail_trade_sol: list[float]
    whale_trade_sol: list[float]
    sniper_trade_sol: list[float]
    smart_trade_sol: list[float]
    sol_usd_start: float
    sol_usd_vol_daily: float
    amm_life_s: float
    life_scale_s: float
    social_prob: dict[str, float]
    rug_prob: dict[str, float]
    rug_delay_s: list[float]
    smart_signal_noise: float
    smart_entry_threshold: float
    sniper_participation: float
    sniper_slot0_frac: float
    whale_participation: float
    sell_pressure: float
    frontrun_intensity: float
    copytrade_mean: float
    quality_drift_s: float
    viral_prob: float
    viral_boost: float
    frontrun_trade_sol: list[float]


class MlCfg(Strict):
    model: Literal["logistic", "random_forest", "xgboost", "lightgbm", "catboost"]
    target: Literal["rug", "fwd_return", "migrate"]
    fwd_return_horizon_s: float
    fwd_return_threshold: float
    cv_folds: int
    embargo_s: float
    shap_samples: int
    permutation_repeats: int
    params: dict[str, dict[str, Any]]


class Settings(Strict):
    """Root configuration object."""

    app: AppCfg
    paths: PathsCfg
    network: NetworkCfg
    protocol: ProtocolCfg
    fees: FeesCfg
    slippage: SlippageCfg
    priority_fee: PriorityFeeCfg
    jito: JitoCfg
    simulation: SimulationCfg
    backtest: BacktestCfg
    risk: RiskCfg
    sizing: SizingCfg
    position: PositionCfg
    features: FeaturesCfg
    discovery: DiscoveryCfg
    wallet_intel: WalletIntelCfg
    rug_model: RugModelCfg
    strategy: StrategyCfg
    optimizer: OptimizerCfg
    montecarlo: MonteCarloCfg
    live: LiveCfg
    collector: CollectorCfg
    logging: LoggingCfg
    dashboard: DashboardCfg
    synthetic: SyntheticCfg
    ml: MlCfg

    @property
    def effective_platform_fee_bps(self) -> int:
        """Router fee on the SOL leg: explicit ``fees.platform_fee_bps`` or, when null, the public Metis fee in Mode A."""
        if self.fees.platform_fee_bps is not None:
            return int(self.fees.platform_fee_bps)
        return int(self.network.metis.public_platform_fee_bps) if self.network.metis.mode == "public" else 0

    def fingerprint(self) -> str:
        """Stable hash of the full configuration (for run reproducibility)."""
        return stable_hash(self.model_dump(mode="json"))


# ----------------------------------------------------------------------------- loading
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_dotted(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply ``{"a.b.c": value}`` style overrides (nested dict keys also accepted)."""
    out = copy.deepcopy(data)
    for key, value in overrides.items():
        if "." not in key:
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
            continue
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return out


def load_raw(path: str | Path | None = None) -> dict[str, Any]:
    """Load default.yaml, deep-merged with an optional user YAML file."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, encoding="utf-8") as fh:
            data = deep_merge(data, yaml.safe_load(fh) or {})
    return data


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load, merge and validate configuration."""
    data = load_raw(path)
    if overrides:
        data = apply_dotted(data, overrides)
    return Settings.model_validate(data)


def settings_to_yaml(settings: Settings) -> str:
    """Serialise settings back to YAML (for run snapshots)."""
    return yaml.safe_dump(json.loads(settings.model_dump_json()), sort_keys=False)


# ----------------------------------------------------------------------------- secrets
class Secrets(BaseModel):
    """Secrets loaded from ``.env`` / environment. Never logged (see utils.logging.redact)."""

    model_config = ConfigDict(extra="ignore")

    pumpfun_jwt: SecretStr | None = None
    qn_jwt_private_key_path: str | None = None
    qn_jwt_kid: str | None = None
    metis_url: SecretStr | None = None
    solana_rpc_url: SecretStr | None = None
    solana_ws_url: SecretStr | None = None
    private_key: SecretStr | None = None
    jito_auth_uuid: SecretStr | None = None
    pumpfun_frontend_jwt: SecretStr | None = None

    ENV_KEYS: ClassVar[tuple[str, ...]] = (
        "PUMPFUN_JWT",
        "QN_JWT_PRIVATE_KEY_PATH",
        "QN_JWT_KID",
        "METIS_URL",
        "SOLANA_RPC_URL",
        "SOLANA_WS_URL",
        "PRIVATE_KEY",
        "JITO_AUTH_UUID",
        "PUMPFUN_FRONTEND_JWT",
    )

    @classmethod
    def load(cls, env_file: str | Path | None = None, register: bool = True) -> Secrets:
        """Load from ``env_file`` (default ``<project>/.env``) then the process environment.

        Process environment variables take precedence over the file. Every non-empty secret is
        registered with the log redactor when ``register`` is true.
        """
        file_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        values: dict[str, str | None] = dict(dotenv_values(file_path)) if file_path.exists() else {}
        for key in cls.ENV_KEYS:
            if os.environ.get(key):
                values[key] = os.environ[key]
        cleaned = {k.lower(): (v if v not in ("", None) else None) for k, v in values.items() if k in cls.ENV_KEYS}
        obj = cls.model_validate(cleaned)
        if register:
            from pumpfun_hft.utils.logging import register_secret

            for v in obj.secret_values():
                register_secret(v)
                # also redact the token-bearing path segment of provider URLs
                if v.startswith(("http", "ws")):
                    for seg in v.split("/"):
                        if len(seg) >= 16:
                            register_secret(seg)
        return obj

    def secret_values(self) -> list[str]:
        out: list[str] = []
        for name in ("pumpfun_jwt", "metis_url", "solana_rpc_url", "solana_ws_url", "private_key", "jito_auth_uuid", "pumpfun_frontend_jwt"):
            v = getattr(self, name)
            if v is not None and v.get_secret_value():
                out.append(v.get_secret_value())
        return out

    def get(self, name: str) -> str | None:
        """Plain value of a secret field or None."""
        v = getattr(self, name)
        if v is None:
            return None
        return v.get_secret_value() if isinstance(v, SecretStr) else v

    def summary(self) -> dict[str, bool]:
        """Which secrets are present (values never exposed)."""
        return {name: self.get(name) is not None for name in type(self).model_fields}
