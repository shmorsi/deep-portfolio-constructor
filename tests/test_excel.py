"""The spreadsheet models: Python twins obey their identities, and the workbooks compute the same numbers."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from dpc.excel import models

pytest.importorskip("openpyxl")


def test_pension_special_payment_amortises_the_shortfall():
    full = models.pension(models.PensionInputs(funded_ratio=1.2))
    assert full["special"][1] == 0.0                              # no shortfall, no special payment
    p = models.PensionInputs()
    out = models.pension(p)
    gap = p.liability * (1 - p.funded_ratio)
    r, m = p.discount_rate, p.amortisation_years
    assert out["special"][1] == pytest.approx(gap * r / (1 - (1 + r) ** -m))   # a level payment that repays the gap in m years


def test_rate_shock_is_symmetric_around_zero_only_through_convexity():
    p = models.PensionInputs()
    up, dn = models.rate_shock(p, 0.01), models.rate_shock(p, -0.01)
    assert up["funded_ratio"] > p.funded_ratio > dn["funded_ratio"]   # liability longer than assets
    assert models.rate_shock(p, 0.0)["funded_ratio"] == pytest.approx(p.funded_ratio)


def test_lbo_sources_equal_uses_and_cash_flows_reconcile():
    p = models.LBOInputs()
    o = models.lbo(p)
    ebitda0 = p.revenue0 * p.margin0
    uses = p.entry_multiple * ebitda0 * (1 + p.fees_pct) + p.min_cash
    assert o["equity0"] + o["senior"][0] + o["sub"][0] == pytest.approx(uses)
    assert np.allclose(np.diff(o["cash"]), o["fcf"][1:] - o["mandatory"][1:] - o["sweep"][1:])
    assert o["grid_moic"][(p.entry_multiple, p.exit_multiple)] == pytest.approx(o["moic"])
    assert o["grid_moic"][(8.0, 12.0)] > o["grid_moic"][(12.0, 8.0)]


@pytest.mark.skipif(not (shutil.which("soffice") or shutil.which("libreoffice")), reason="LibreOffice not installed")
def test_workbooks_match_python(tmp_path):
    from dpc.excel import build, verify

    pen = verify.check_pension(build.pension_workbook(tmp_path / "p.xlsx"))
    lbo = verify.check_lbo(build.lbo_workbook(tmp_path / "l.xlsx"))
    assert pen["max_abs_difference"] < 1e-6 and lbo["max_abs_difference"] < 1e-6
