"""IDL-driven Borsh codec for Anchor programs (Anchor >= 0.30 IDL format).

Why IDL-driven: the Pump ``TradeEvent`` has grown from 8 to 34 fields over time, including a
``string`` and a ``vec`` in the middle of the layout. Hand-written offsets break on every
upgrade; decoding from the official IDL (bundled in ``core/idl``, refreshable with
``main.py update-idl``) keeps the collector correct, and *top-level truncation tolerance*
decodes historical (shorter) layouts: fields absent from older events come back as ``None``.

Supported types: bool, u8..u128, i8..i128, f32, f64, string, bytes, pubkey, option, vec,
array, defined structs (named and tuple) and enums (unit, tuple and struct variants).

Decoders are compiled once into closures per type for speed (the live path decodes every
Pump log line). An encoder is included for tests and the synthetic data generator.

Example::

    dec = IdlCodec.from_file("pumpfun_hft/core/idl/pump.json")
    name, fields = dec.decode_event(raw_bytes)          # ('TradeEvent', {...})
    name, acct = dec.decode_account(account_data)       # ('BondingCurve', {...})
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pumpfun_hft.utils.base58 import b58decode, pubkey_to_str

Reader = Callable[[bytes, int], tuple[Any, int]]
Writer = Callable[[Any, bytearray], None]

_PRIMS: dict[str, tuple[str, int]] = {
    "u8": ("<B", 1), "i8": ("<b", 1), "u16": ("<H", 2), "i16": ("<h", 2),
    "u32": ("<I", 4), "i32": ("<i", 4), "u64": ("<Q", 8), "i64": ("<q", 8),
    "f32": ("<f", 4), "f64": ("<d", 8),
}


class IdlDecodeError(ValueError):
    """Raised when bytes do not match the IDL layout."""


class IdlCodec:
    """Borsh encoder/decoder for one Anchor program IDL."""

    def __init__(self, idl: dict[str, Any]) -> None:
        self.idl = idl
        self.address: str = idl.get("address", "")
        self.types: dict[str, dict[str, Any]] = {t["name"]: t["type"] for t in idl.get("types", [])}
        self.event_by_disc: dict[bytes, str] = {bytes(e["discriminator"]): e["name"] for e in idl.get("events", [])}
        self.account_by_disc: dict[bytes, str] = {bytes(a["discriminator"]): a["name"] for a in idl.get("accounts", [])}
        self.ix_by_disc: dict[bytes, dict[str, Any]] = {bytes(i["discriminator"]): i for i in idl.get("instructions", [])}
        self.event_disc: dict[str, bytes] = {v: k for k, v in self.event_by_disc.items()}
        self.account_disc: dict[str, bytes] = {v: k for k, v in self.account_by_disc.items()}
        self._readers: dict[str, Reader] = {}
        self._struct_fields: dict[str, list[tuple[str, Reader]]] = {}
        self._writers: dict[str, Writer] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> IdlCodec:
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ------------------------------------------------------------------ reader compilation
    def _reader(self, ty: Any) -> Reader:  # noqa: C901 - a type switch is clearest here
        if isinstance(ty, str):
            if ty in _PRIMS:
                fmt, size = _PRIMS[ty]
                unpack = struct.Struct(fmt).unpack_from

                def rd_prim(buf: bytes, off: int, _u=unpack, _s=size) -> tuple[Any, int]:
                    return _u(buf, off)[0], off + _s

                return rd_prim
            if ty == "bool":
                def rd_bool(buf: bytes, off: int) -> tuple[Any, int]:
                    if off >= len(buf):
                        raise IdlDecodeError("eof")
                    return buf[off] != 0, off + 1
                return rd_bool
            if ty in ("u128", "i128"):
                signed = ty == "i128"

                def rd_128(buf: bytes, off: int, _signed=signed) -> tuple[Any, int]:
                    if off + 16 > len(buf):
                        raise IdlDecodeError("eof")
                    return int.from_bytes(buf[off:off + 16], "little", signed=_signed), off + 16
                return rd_128
            if ty == "pubkey":
                def rd_pk(buf: bytes, off: int) -> tuple[Any, int]:
                    if off + 32 > len(buf):
                        raise IdlDecodeError("eof")
                    return pubkey_to_str(bytes(buf[off:off + 32])), off + 32
                return rd_pk
            if ty in ("string", "bytes"):
                is_str = ty == "string"

                def rd_str(buf: bytes, off: int, _is_str=is_str) -> tuple[Any, int]:
                    if off + 4 > len(buf):
                        raise IdlDecodeError("eof")
                    (n,) = struct.unpack_from("<I", buf, off)
                    end = off + 4 + n
                    if end > len(buf):
                        raise IdlDecodeError("eof")
                    raw = bytes(buf[off + 4:end])
                    return (raw.decode("utf-8", "replace") if _is_str else raw), end
                return rd_str
            raise IdlDecodeError(f"unsupported primitive {ty}")
        if "option" in ty:
            inner = self._reader(ty["option"])

            def rd_opt(buf: bytes, off: int, _in=inner) -> tuple[Any, int]:
                if off >= len(buf):
                    raise IdlDecodeError("eof")
                if buf[off] == 0:
                    return None, off + 1
                return _in(buf, off + 1)
            return rd_opt
        if "vec" in ty:
            inner = self._reader(ty["vec"])

            def rd_vec(buf: bytes, off: int, _in=inner) -> tuple[Any, int]:
                if off + 4 > len(buf):
                    raise IdlDecodeError("eof")
                (n,) = struct.unpack_from("<I", buf, off)
                off += 4
                out = []
                for _ in range(n):
                    v, off = _in(buf, off)
                    out.append(v)
                return out, off
            return rd_vec
        if "array" in ty:
            elem, n = ty["array"]
            inner = self._reader(elem)

            def rd_arr(buf: bytes, off: int, _in=inner, _n=n) -> tuple[Any, int]:
                out = []
                for _ in range(_n):
                    v, off = _in(buf, off)
                    out.append(v)
                return out, off
            return rd_arr
        if "defined" in ty:
            name = ty["defined"]["name"] if isinstance(ty["defined"], dict) else ty["defined"]
            return self._defined_reader(name)
        raise IdlDecodeError(f"unsupported type {ty!r}")

    def _defined_reader(self, name: str) -> Reader:
        if name in self._readers:
            return self._readers[name]

        # placeholder allows recursive types
        def lazy(buf: bytes, off: int) -> tuple[Any, int]:
            return self._readers[name](buf, off)

        self._readers[name] = lazy
        tdef = self.types.get(name)
        if tdef is None:
            raise IdlDecodeError(f"unknown defined type {name}")
        kind = tdef["kind"]
        if kind == "struct":
            fields = tdef.get("fields", [])
            if fields and isinstance(fields[0], dict):
                compiled = [(f["name"], self._reader(f["type"])) for f in fields]
                self._struct_fields[name] = compiled

                def rd_struct(buf: bytes, off: int, _c=compiled) -> tuple[Any, int]:
                    out: dict[str, Any] = {}
                    for fname, rd in _c:
                        out[fname], off = rd(buf, off)
                    return out, off
                reader: Reader = rd_struct
            else:  # tuple struct
                compiled_t = [self._reader(f) for f in fields]

                def rd_tuple(buf: bytes, off: int, _c=compiled_t) -> tuple[Any, int]:
                    vals = []
                    for rd in _c:
                        v, off = rd(buf, off)
                        vals.append(v)
                    return (vals[0] if len(vals) == 1 else tuple(vals)), off
                reader = rd_tuple
        elif kind == "enum":
            variants = []
            for v in tdef["variants"]:
                vf = v.get("fields")
                if not vf:
                    variants.append((v["name"], None))
                elif isinstance(vf[0], dict):
                    variants.append((v["name"], [(f["name"], self._reader(f["type"])) for f in vf]))
                else:
                    variants.append((v["name"], [(str(i), self._reader(f)) for i, f in enumerate(vf)]))

            def rd_enum(buf: bytes, off: int, _v=variants) -> tuple[Any, int]:
                if off >= len(buf):
                    raise IdlDecodeError("eof")
                idx = buf[off]
                off += 1
                if idx >= len(_v):
                    raise IdlDecodeError(f"bad enum index {idx}")
                vname, vfields = _v[idx]
                if vfields is None:
                    return vname, off
                data: dict[str, Any] = {}
                for fname, rd in vfields:
                    data[fname], off = rd(buf, off)
                return {vname: data}, off
            reader = rd_enum
        else:
            raise IdlDecodeError(f"unsupported kind {kind}")
        self._readers[name] = reader
        return reader

    # ------------------------------------------------------------------ public decode API
    def decode_struct(self, name: str, data: bytes, offset: int = 0, tolerant: bool = True) -> dict[str, Any]:
        """Decode a named struct. With ``tolerant`` missing trailing fields are returned as None."""
        self._defined_reader(name)
        fields = self._struct_fields.get(name)
        if fields is None:
            value, _ = self._readers[name](data, offset)
            return value
        out: dict[str, Any] = {}
        off = offset
        n = len(data)
        for i, (fname, rd) in enumerate(fields):
            if off >= n:
                if not tolerant:
                    raise IdlDecodeError(f"{name}: truncated at field {fname}")
                for rest, _ in fields[i:]:
                    out[rest] = None
                break
            try:
                out[fname], off = rd(data, off)
            except (IdlDecodeError, struct.error, UnicodeDecodeError) as exc:
                if not tolerant:
                    raise IdlDecodeError(f"{name}.{fname}: {exc}") from exc
                for rest, _ in fields[i:]:
                    out[rest] = None
                break
        return out

    def decode_event(self, data: bytes) -> tuple[str, dict[str, Any]] | None:
        """Decode ``discriminator(8) + borsh`` event bytes; None for unknown discriminators."""
        name = self.event_by_disc.get(bytes(data[:8]))
        if name is None:
            return None
        return name, self.decode_struct(name, data, 8)

    def decode_account(self, data: bytes) -> tuple[str, dict[str, Any]] | None:
        name = self.account_by_disc.get(bytes(data[:8]))
        if name is None:
            return None
        return name, self.decode_struct(name, data, 8)

    def decode_instruction(self, data: bytes) -> tuple[str, dict[str, Any]] | None:
        ix = self.ix_by_disc.get(bytes(data[:8]))
        if ix is None:
            return None
        off = 8
        out: dict[str, Any] = {}
        for arg in ix.get("args", []):
            if off >= len(data):
                out[arg["name"]] = None
                continue
            out[arg["name"]], off = self._reader(arg["type"])(data, off)
        return ix["name"], out

    # ------------------------------------------------------------------ encoder (tests / synthetic)
    def _encode_value(self, ty: Any, value: Any, out: bytearray) -> None:  # noqa: C901
        if isinstance(ty, str):
            if ty in _PRIMS:
                out += struct.pack(_PRIMS[ty][0], value)
            elif ty == "bool":
                out.append(1 if value else 0)
            elif ty in ("u128", "i128"):
                out += int(value).to_bytes(16, "little", signed=ty == "i128")
            elif ty == "pubkey":
                raw = b58decode(value) if isinstance(value, str) else bytes(value)
                if len(raw) != 32:
                    raise IdlDecodeError("pubkey must be 32 bytes")
                out += raw
            elif ty == "string":
                raw = value.encode("utf-8")
                out += struct.pack("<I", len(raw)) + raw
            elif ty == "bytes":
                out += struct.pack("<I", len(value)) + bytes(value)
            else:
                raise IdlDecodeError(f"cannot encode {ty}")
            return
        if "option" in ty:
            if value is None:
                out.append(0)
            else:
                out.append(1)
                self._encode_value(ty["option"], value, out)
            return
        if "vec" in ty:
            out += struct.pack("<I", len(value))
            for v in value:
                self._encode_value(ty["vec"], v, out)
            return
        if "array" in ty:
            elem, n = ty["array"]
            if len(value) != n:
                raise IdlDecodeError("array length mismatch")
            for v in value:
                self._encode_value(elem, v, out)
            return
        if "defined" in ty:
            name = ty["defined"]["name"] if isinstance(ty["defined"], dict) else ty["defined"]
            self.encode_struct(name, value, out)
            return
        raise IdlDecodeError(f"cannot encode {ty!r}")

    def encode_struct(self, name: str, value: Any, out: bytearray | None = None) -> bytearray:
        buf = bytearray() if out is None else out
        tdef = self.types[name]
        if tdef["kind"] == "struct":
            fields = tdef.get("fields", [])
            if fields and isinstance(fields[0], dict):
                for f in fields:
                    self._encode_value(f["type"], value[f["name"]], buf)
            else:
                vals = value if isinstance(value, (list, tuple)) else [value]
                for f, v in zip(fields, vals, strict=True):
                    self._encode_value(f, v, buf)
        elif tdef["kind"] == "enum":
            vname = value if isinstance(value, str) else next(iter(value))
            for idx, var in enumerate(tdef["variants"]):
                if var["name"] == vname:
                    buf.append(idx)
                    if not isinstance(value, str):
                        payload = value[vname]
                        for i, f in enumerate(var.get("fields") or []):
                            if isinstance(f, dict):
                                self._encode_value(f["type"], payload[f["name"]], buf)
                            else:
                                self._encode_value(f, payload[str(i)], buf)
                    break
            else:
                raise IdlDecodeError(f"unknown variant {vname}")
        return buf

    def encode_event(self, name: str, value: dict[str, Any], truncate_after: str | None = None) -> bytes:
        """Encode an event; ``truncate_after`` emulates an older (shorter) on-chain layout."""
        buf = bytearray(self.event_disc[name])
        fields = self.types[name]["fields"]
        for f in fields:
            self._encode_value(f["type"], value[f["name"]], buf)
            if truncate_after is not None and f["name"] == truncate_after:
                break
        return bytes(buf)

    def struct_field_names(self, name: str) -> list[str]:
        return [f["name"] for f in self.types[name].get("fields", []) if isinstance(f, dict)]


def load_codecs(idl_dir: str | Path) -> dict[str, IdlCodec]:
    """Load every ``*.json`` IDL in ``idl_dir`` keyed by program address."""
    out: dict[str, IdlCodec] = {}
    for p in sorted(Path(idl_dir).glob("*.json")):
        codec = IdlCodec.from_file(p)
        if codec.address:
            out[codec.address] = codec
    return out
