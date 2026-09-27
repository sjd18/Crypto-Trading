"""Program-derived addresses used by Pump / PumpSwap / SPL (requires ``solders``)."""

from __future__ import annotations

from functools import lru_cache

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
METAPLEX_PROGRAM = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"


def _pk(s: str):  # noqa: ANN202
    try:
        from solders.pubkey import Pubkey
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDA derivation requires `solders` (pip install pumpfun-hft[live])") from exc
    return Pubkey.from_string(s)


@lru_cache(maxsize=65536)
def find_pda(seeds: tuple[bytes, ...], program_id: str) -> str:
    from solders.pubkey import Pubkey

    addr, _bump = Pubkey.find_program_address(list(seeds), _pk(program_id))
    return str(addr)


def bonding_curve_pda(mint: str, pump_program_id: str) -> str:
    return find_pda((b"bonding-curve", bytes(_pk(mint))), pump_program_id)


def creator_vault_pda(creator: str, pump_program_id: str) -> str:
    return find_pda((b"creator-vault", bytes(_pk(creator))), pump_program_id)


def associated_token_address(owner: str, mint: str, token_program: str = TOKEN_PROGRAM) -> str:
    return find_pda((bytes(_pk(owner)), bytes(_pk(token_program)), bytes(_pk(mint))), ATA_PROGRAM)


def metaplex_metadata_pda(mint: str) -> str:
    return find_pda((b"metadata", bytes(_pk(METAPLEX_PROGRAM)), bytes(_pk(mint))), METAPLEX_PROGRAM)


def fee_config_pda(pump_program_id: str, fees_program_id: str) -> str:
    return find_pda((b"fee_config", bytes(_pk(pump_program_id))), fees_program_id)
