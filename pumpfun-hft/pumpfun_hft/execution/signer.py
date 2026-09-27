"""Wallet signing (``solders``). The private key is read from ``PRIVATE_KEY`` in ``.env`` only.

Accepted key formats: base58 64-byte secret (Phantom / Solflare export) or a JSON byte array
(``solana-keygen`` file contents). The key is never logged (it is registered with the log
redactor by :class:`~pumpfun_hft.core.config.Secrets`).
"""

from __future__ import annotations

import json
from typing import Any

try:  # optional dependency (extra "live")
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction, VersionedTransaction

    HAVE_SOLDERS = True
except ImportError:  # pragma: no cover
    HAVE_SOLDERS = False

COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"
SYSTEM_PROGRAM = "11111111111111111111111111111111"


def _require() -> None:
    if not HAVE_SOLDERS:
        raise ImportError("live trading requires `solders` (pip install pumpfun-hft[live])")


class WalletSigner:
    """Holds the trading keypair and signs transactions.

    Example::

        signer = WalletSigner.from_secret(secrets.get("private_key"))
        signed = signer.sign(raw_unsigned_tx_bytes)
    """

    def __init__(self, keypair: Any) -> None:
        _require()
        self.keypair = keypair
        self.pubkey = str(keypair.pubkey())

    @classmethod
    def from_secret(cls, secret: str | None) -> WalletSigner:
        _require()
        if not secret:
            raise ValueError("PRIVATE_KEY is not set (required for live trading)")
        s = secret.strip()
        if s.startswith("["):
            return cls(Keypair.from_bytes(bytes(json.loads(s))))
        return cls(Keypair.from_base58_string(s))

    @classmethod
    def generate(cls) -> WalletSigner:
        _require()
        return cls(Keypair())

    def sign(self, raw_tx: bytes) -> bytes:
        """Sign a serialised versioned (or legacy) transaction built by a router."""
        try:
            vtx = VersionedTransaction.from_bytes(raw_tx)
            return bytes(VersionedTransaction(vtx.message, [self.keypair]))
        except Exception:  # noqa: BLE001 - fall back to legacy format
            tx = Transaction.from_bytes(raw_tx)
            tx.sign([self.keypair], tx.message.recent_blockhash)
            return bytes(tx)

    @staticmethod
    def signature_of(signed_tx: bytes) -> str:
        try:
            return str(VersionedTransaction.from_bytes(signed_tx).signatures[0])
        except Exception:  # noqa: BLE001
            return str(Transaction.from_bytes(signed_tx).signatures[0])

    @staticmethod
    def recent_blockhash_of(tx_bytes: bytes) -> str:
        try:
            return str(VersionedTransaction.from_bytes(tx_bytes).message.recent_blockhash)
        except Exception:  # noqa: BLE001
            return str(Transaction.from_bytes(tx_bytes).message.recent_blockhash)

    # ------------------------------------------------------------------ composing transactions
    @staticmethod
    def compute_budget_ixs(cu_limit: int, micro_lamports: int) -> list[Any]:
        from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price

        return [set_compute_unit_limit(cu_limit), set_compute_unit_price(micro_lamports)]

    def tip_ix(self, tip_account: str, lamports: int) -> Any:
        from solders.system_program import TransferParams, transfer

        return transfer(TransferParams(from_pubkey=self.keypair.pubkey(), to_pubkey=Pubkey.from_string(tip_account), lamports=lamports))

    @staticmethod
    def instruction_from_json(ix: dict[str, Any]) -> Any:
        """Instruction from a Metis ``swap-instructions`` item ({programId, keys, data})."""
        data = ix["data"]
        raw = bytes(data) if isinstance(data, list) else __import__("base64").b64decode(data)
        keys = [AccountMeta(Pubkey.from_string(k["pubkey"]), bool(k["isSigner"]), bool(k["isWritable"])) for k in ix["keys"]]
        return Instruction(Pubkey.from_string(ix["programId"]), raw, keys)

    def compose(self, instructions: list[Any], blockhash: str, cu_limit: int | None = None, micro_lamports: int | None = None,
                tip: tuple[str, int] | None = None) -> bytes:
        """Build and sign a v0 transaction: [compute budget] + instructions + [Jito tip]."""
        ixs = []
        if cu_limit is not None and micro_lamports is not None:
            ixs += self.compute_budget_ixs(cu_limit, micro_lamports)
        ixs += [i for i in instructions if str(i.program_id) != COMPUTE_BUDGET or cu_limit is None]
        if tip is not None:
            ixs.append(self.tip_ix(*tip))
        msg = MessageV0.try_compile(self.keypair.pubkey(), ixs, [], Hash.from_string(blockhash))
        return bytes(VersionedTransaction(msg, [self.keypair]))
