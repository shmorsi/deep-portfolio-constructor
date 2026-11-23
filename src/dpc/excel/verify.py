"""Recalculate the workbooks with LibreOffice (headless) and compare every key cell to `models.py`.

`python -m dpc.cli excel` builds both files into excel/, recalculates copies, and writes
excel/verification.json with the largest difference found. It refuses to report success if any
cell differs from the Python model by more than 1e-6.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from . import build, models


def recalc(path: Path) -> Path:
    """Open and re-save with LibreOffice so every formula is computed; returns the recalculated copy."""
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office is None:
        raise RuntimeError("LibreOffice is needed to recalculate the workbooks")
    out = Path(tempfile.mkdtemp())
    subprocess.run([office, "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(out), str(path)],
                   check=True, capture_output=True, timeout=180)
    return out / path.name


def _diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def check_pension(path: Path) -> dict:
    ref = models.pension()
    wb = load_workbook(recalc(path), data_only=True)
    pr = wb["Projection"]
    n = models.PensionInputs().years
    col = lambda c: [pr[f"{c}{r}"].value for r in range(4, 5 + n)]
    d = {"liability": _diff(col("B"), ref["liability"]), "assets": _diff(col("C"), ref["assets"]),
         "special": _diff(col("F")[1:], ref["special"][1:]), "funded_ratio": _diff(col("H"), ref["funded_ratio"])}
    sh = wb["Rate shock"]
    shock = [models.rate_shock(models.PensionInputs(), bp / 1e4)["funded_ratio"] for bp in (-200, -100, -50, 0, 50, 100, 200)]
    d["rate_shock"] = _diff([sh[f"D{r}"].value for r in range(4, 11)], shock)
    return {"max_abs_difference": max(d.values()), "by_block": d,
            "funded_ratio_year_20": float(ref["funded_ratio"][-1]), "total_contributions": float((ref["normal_cost"] + ref["special"]).sum())}


def check_lbo(path: Path) -> dict:
    p = models.LBOInputs()
    ref = models.lbo(p)
    wb = load_workbook(recalc(path), data_only=True)
    md, rt = wb["Model"], wb["Returns"]
    row = lambda r: [md.cell(r, 2 + t).value for t in range(p.years + 1)]
    d = {"revenue": _diff(row(4), ref["revenue"]), "ebitda": _diff(row(6), ref["ebitda"]),
         "fcf": _diff(row(12)[1:], ref["fcf"][1:]), "senior": _diff(row(15), ref["senior"]), "cash": _diff(row(17), ref["cash"]),
         "moic": abs(rt["B11"].value - ref["moic"]), "irr": abs(rt["B12"].value - ref["irr"]),
         "irr_excel_function": abs(rt["B15"].value - ref["irr"])}
    grid = [[rt.cell(20 + i, 2 + j).value for j in range(len(p.grid))] for i in range(len(p.grid))]
    d["grid"] = _diff(grid, [[ref["grid_moic"][(e, x)] for x in p.grid] for e in p.grid])
    return {"max_abs_difference": max(d.values()), "by_block": d, "moic": float(ref["moic"]), "irr": float(ref["irr"]),
            "leverage_entry": float(ref["leverage"][0]), "leverage_exit": float(ref["leverage"][-1]),
            "min_coverage": float(np.nanmin(ref["coverage"][1:])), "senior_repaid_share": float(1 - ref["senior"][-1] / ref["senior"][0])}


def main(cfg, refresh: bool = False) -> int:
    out = cfg.root / "excel"
    pen = build.pension_workbook(out / "pension_funding_model.xlsx")
    lbo = build.lbo_workbook(out / "lbo_private_credit.xlsx")
    rep = {"pension_funding_model.xlsx": check_pension(pen), "lbo_private_credit.xlsx": check_lbo(lbo),
           "tolerance": 1e-6, "recalculated_with": "LibreOffice Calc (headless)"}
    (out / "verification.json").write_text(json.dumps(rep, indent=1) + "\n")
    bad = [k for k, v in rep.items() if isinstance(v, dict) and v["max_abs_difference"] > 1e-6]
    print(json.dumps({k: v["max_abs_difference"] for k, v in rep.items() if isinstance(v, dict)}, indent=1))
    return 1 if bad else 0
