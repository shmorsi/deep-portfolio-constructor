#!/usr/bin/env python3
"""Fail if reports/site_wealth.json no longer matches the committed results.

The website (shmorsi.github.io) reads only reports/site_wealth.json. If the
results in reports/results/wealth/ change and the export is not regenerated,
the site would show old numbers. CI runs:

    python -m dpc.cli wealth-site          # rewrites reports/site_wealth.json
    python scripts/check_site_export.py    # compares it with the committed copy

The "commit" and "source" fields name the commit the export ran at, so they
always differ between a fresh run and the committed file; they are ignored.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = "reports/site_wealth.json"
IGNORED = {"commit", "source"}


def diff(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if not path and k in IGNORED:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'committed' if k in a else 'fresh'} export")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items committed, {len(b)} fresh"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else [f"{path}: committed {a}, fresh {b}"]
    return [] if a == b else [f"{path}: committed {a!r}, fresh {b!r}"]


def main() -> int:
    committed = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{EXPORT}"],
                                          capture_output=True, text=True, check=True).stdout)
    fresh = json.loads((ROOT / EXPORT).read_text())
    problems = diff(committed, fresh)
    if problems:
        print(f"{EXPORT} is stale: run `python -m dpc.cli wealth-site` and commit it.", file=sys.stderr)
        for p in problems[:30]:
            print(f"  {p}", file=sys.stderr)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more", file=sys.stderr)
        return 1
    print(f"ok: {EXPORT} matches the committed results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
