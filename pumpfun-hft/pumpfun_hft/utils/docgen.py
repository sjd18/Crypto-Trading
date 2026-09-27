"""Generate developer documentation (docs/MODULES.md) from package and module docstrings."""

from __future__ import annotations

import ast
from pathlib import Path

from pumpfun_hft.core.config import PACKAGE_DIR

PACKAGES = ["core", "api", "collectors", "discovery", "features", "strategies", "backtester", "risk", "optimizer", "analytics",
            "ml", "execution", "dashboard", "database", "utils"]


def _doc(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except SyntaxError:
        return ""


def generate_module_docs(out: Path) -> Path:
    lines = ["# Module reference", "",
             "Generated from the source docstrings by `python -m pumpfun_hft.main docs`. Every package docstring states its "
             "purpose, architecture, data flow, inputs/outputs and an example.", ""]
    for pkg in PACKAGES:
        d = PACKAGE_DIR / pkg
        lines += [f"## `pumpfun_hft.{pkg}`", "", "```text", _doc(d / "__init__.py").strip(), "```", ""]
        for f in sorted(d.glob("*.py")):
            if f.name == "__init__.py":
                continue
            doc = _doc(f).strip()
            if doc:
                first = doc.split("\n\n")[0].replace("\n", " ")
                lines.append(f"- **`{pkg}/{f.name}`** — {first}")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
