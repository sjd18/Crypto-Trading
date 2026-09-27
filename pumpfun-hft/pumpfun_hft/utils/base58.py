"""Base58 (Bitcoin alphabet) encoding used for Solana public keys and signatures.

The hot path (encoding 32-byte public keys while decoding events) uses ``solders`` when
available, which is implemented in Rust; the pure-Python implementation is exact and is used
as a fallback and for arbitrary-length payloads.
"""

from __future__ import annotations

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}

try:  # pragma: no cover - environment dependent
    from solders.pubkey import Pubkey as _Pubkey

    def _fast_pubkey(b: bytes) -> str:
        return str(_Pubkey.from_bytes(b))

    HAVE_SOLDERS = True
except Exception:  # noqa: BLE001
    _Pubkey = None
    HAVE_SOLDERS = False

    def _fast_pubkey(b: bytes) -> str:
        return b58encode(b)


def b58encode(data: bytes) -> str:
    """Encode bytes to base58."""
    n_zeros = len(data) - len(data.lstrip(b"\0"))
    num = int.from_bytes(data, "big")
    out = []
    while num:
        num, rem = divmod(num, 58)
        out.append(ALPHABET[rem])
    return "1" * n_zeros + "".join(reversed(out))


def b58decode(text: str) -> bytes:
    """Decode a base58 string to bytes. Raises ``ValueError`` on invalid characters."""
    num = 0
    for ch in text:
        try:
            num = num * 58 + _INDEX[ch]
        except KeyError as exc:
            raise ValueError(f"invalid base58 character {ch!r}") from exc
    n_zeros = len(text) - len(text.lstrip("1"))
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    return b"\0" * n_zeros + body


def pubkey_to_str(b: bytes) -> str:
    """Encode a 32-byte public key to its base58 string."""
    return _fast_pubkey(b)
