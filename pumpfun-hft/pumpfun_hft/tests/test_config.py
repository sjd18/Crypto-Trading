"""Configuration strictness, overrides, secrets handling and redaction."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pumpfun_hft.core.config import DEFAULT_CONFIG_PATH, PACKAGE_DIR, Secrets, Settings, load_raw, load_settings
from pumpfun_hft.utils.logging import JsonFormatter, redact, register_secret


def test_default_config_validates_and_every_section_is_required() -> None:
    raw = load_raw()
    Settings.model_validate(raw)
    for section in ("backtest", "risk", "simulation", "fees"):
        broken = {k: v for k, v in raw.items() if k != section}
        with pytest.raises(ValidationError):
            Settings.model_validate(broken)


def test_missing_nested_key_and_unknown_key_fail() -> None:
    raw = load_raw()
    del raw["backtest"]["initial_capital_sol"]
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)
    raw = load_raw()
    raw["backtest"]["initial_captial_sol"] = 5  # typo must not be silently ignored
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_dotted_overrides_and_user_yaml(tmp_path: Path) -> None:
    s = load_settings(None, {"backtest.initial_capital_sol": 3.5, "strategy.active": ["sniper"]})
    assert s.backtest.initial_capital_sol == 3.5 and s.strategy.active == ["sniper"]
    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump({"risk": {"limits": {"daily_loss_sol": 0.5}}}))
    u = load_settings(user)
    assert u.risk.limits.daily_loss_sol == 0.5
    assert u.risk.limits.max_open_positions == load_settings().risk.limits.max_open_positions  # deep merge keeps the rest
    assert u.fingerprint() != load_settings().fingerprint()


def test_platform_fee_follows_the_metis_mode() -> None:
    assert load_settings().effective_platform_fee_bps == 20
    assert load_settings(None, {"network.metis.mode": "authenticated"}).effective_platform_fee_bps == 0
    assert load_settings(None, {"fees.platform_fee_bps": 7}).effective_platform_fee_bps == 7


def test_no_secret_values_in_code_or_config() -> None:
    """The repository must not contain credentials: config has no secret keys, code has no JWT-looking strings."""
    def keys(node):  # every key anywhere in the YAML tree
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    secret_names = {k.lower() for k in Secrets.ENV_KEYS}
    assert not secret_names & set(keys(load_raw()))  # secrets are never configuration
    jwt_like = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")
    assert not jwt_like.search(DEFAULT_CONFIG_PATH.read_text())
    for f in PACKAGE_DIR.rglob("*.py"):
        if "tests" in f.parts:
            continue
        assert not jwt_like.search(f.read_text(encoding="utf-8")), f


def test_secrets_load_from_env_file_and_are_redacted(tmp_path: Path, monkeypatch) -> None:
    for k in Secrets.ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    fake_jwt = "eyJ" + "hbGciOiJSUzI1NiJ9" + "." + "eyJzdWIiOiJ0ZXN0In0" + "." + "c2lnbmF0dXJlLXNlY3JldA"
    env.write_text(f"PUMPFUN_JWT={fake_jwt}\n"
                   "SOLANA_RPC_URL=https://example.quiknode.pro/0123456789abcdef0123456789abcdef/\n")
    s = Secrets.load(env)
    assert s.summary()["pumpfun_jwt"] and not s.summary()["private_key"]
    assert "eyJ" not in repr(s) and "eyJ" not in str(s.model_dump())  # SecretStr never prints its value
    jwt = s.get("pumpfun_jwt")
    rec = logging.LogRecord("pumpfun_hft.api", logging.INFO, __file__, 1, f"auth header Bearer {jwt}", None, None)
    rec.data = {"url": s.get("solana_rpc_url")}
    out = JsonFormatter().format(rec)
    assert jwt not in out and "0123456789abcdef0123456789abcdef" not in out and "***" in out
    monkeypatch.setenv("PUMPFUN_JWT", "eyJenv.override.token-value")
    assert Secrets.load(env).get("pumpfun_jwt") == "eyJenv.override.token-value"  # the environment wins


def test_redact_helper() -> None:
    register_secret("super-secret-value")
    assert redact("x super-secret-value y") == "x *** y"
    register_secret("abc")  # too short to register safely
    assert redact("abc") == "abc"


async def test_jwt_minting_registers_the_token(tmp_path: Path) -> None:
    pytest.importorskip("jwt")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from pumpfun_hft.api.auth import JwtProvider

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "key.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    prov = JwtProvider(private_key_path=str(pem), kid="kid-1", algorithm="RS256", lifetime_s=600, refresh_margin_s=60)
    headers = await prov.headers()
    token = headers["Authorization"].removeprefix("Bearer ")
    import jwt

    assert jwt.get_unverified_header(token)["kid"] == "kid-1"
    assert redact(f"token={token}") == "token=***"
    assert await prov.token() == token  # cached until close to expiry
