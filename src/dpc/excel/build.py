"""Write the two workbooks with live formulas (openpyxl). Inputs are blue, formulas black, as banks format models.

    excel/pension_funding_model.xlsx   20-year funding projection, special payments, rate-shock table, chart
    excel/lbo_private_credit.xlsx      5-year LBO, senior debt with sweep, private-credit tranche, returns, sensitivity

Every formula mirrors a line of `models.py`; `verify.py` recalculates the files and compares.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .models import LBOInputs, PensionInputs

BLUE = Font(color="1F4E99")
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=14)
HEAD = PatternFill("solid", fgColor="DCE6F1")
THIN = Border(bottom=Side(style="thin", color="999999"))
PCT, PCT2, NUM, MULT = "0.0%", "0.00%", "#,##0.0", '0.00"x"'


def _inputs(ws, rows: list[tuple[str, float, str, str]], start: int = 3) -> dict[str, str]:
    """Write label / value / note rows; returns {key: absolute cell reference}."""
    refs = {}
    for i, (key, value, fmt, label) in enumerate(rows):
        r = start + i
        ws.cell(r, 1, label)
        c = ws.cell(r, 2, value)
        c.font, c.number_format = BLUE, fmt
        refs[key] = f"{ws.title}!$B${r}"
    return refs


def _header(ws, row: int, labels: list[str]) -> None:
    for j, lab in enumerate(labels, start=1):
        c = ws.cell(row, j, lab)
        c.font, c.fill, c.border = BOLD, HEAD, THIN
        c.alignment = Alignment(horizontal="center", wrap_text=True)


def pension_workbook(path: Path, p: PensionInputs = PensionInputs()) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"], ws["A1"].font = "Pension funding model: inputs (blue cells can be changed)", TITLE
    I = _inputs(ws, [
        ("L", p.liability, NUM, "Liability at the start"),
        ("FR", p.funded_ratio, PCT, "Funded ratio at the start"),
        ("r", p.discount_rate, PCT2, "Discount rate"),
        ("ben", p.benefits_pct, PCT2, "Benefit payments, % of last year's liability"),
        ("nc", p.normal_cost_pct, PCT2, "Normal cost, % of last year's liability"),
        ("eq", p.equity_share, PCT, "Equity share of assets"),
        ("req", p.equity_return, PCT2, "Expected equity return"),
        ("rb", p.bond_return, PCT2, "Expected bond return"),
        ("m", p.amortisation_years, "0", "Years to amortise a shortfall"),
        ("D", p.duration, NUM, "Liability duration (years)"),
        ("C", p.convexity, NUM, "Liability convexity"),
        ("bd", 8.0, NUM, "Bond portfolio duration (years)"),
    ])
    ws["A16"], ws["A16"].font = "Expected return on assets", BOLD
    ws["B16"] = f"={I['eq']}*{I['req']}+(1-{I['eq']})*{I['rb']}"
    ws["A17"], ws["A17"].font = "Annuity factor for the special payment", BOLD
    ws["B17"] = f"=(1-(1+{I['r']})^-{I['m']})/{I['r']}"
    ws["B16"].number_format, ws["B17"].number_format = PCT2, "0.000"
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 12

    pr = wb.create_sheet("Projection")
    pr["A1"], pr["A1"].font = "Funding projection (end-of-year flows)", TITLE
    _header(pr, 3, ["Year", "Liability", "Assets", "Normal cost", "Benefits", "Special payment", "Total contribution", "Funded ratio", "Surplus"])
    pr["A4"], pr["B4"], pr["C4"] = 0, f"={I['L']}", f"={I['L']}*{I['FR']}"
    pr["H4"], pr["I4"] = "=C4/B4", "=C4-B4"
    for t in range(1, p.years + 1):
        r, q = 4 + t, 3 + t
        pr[f"A{r}"] = t
        pr[f"D{r}"] = f"={I['nc']}*B{q}"
        pr[f"E{r}"] = f"={I['ben']}*B{q}"
        pr[f"F{r}"] = f"=MAX(B{q}-C{q},0)/Inputs!$B$17"
        pr[f"B{r}"] = f"=B{q}*(1+{I['r']})+D{r}-E{r}"
        pr[f"C{r}"] = f"=C{q}*(1+Inputs!$B$16)+D{r}+F{r}-E{r}"
        pr[f"G{r}"] = f"=D{r}+F{r}"
        pr[f"H{r}"] = f"=C{r}/B{r}"
        pr[f"I{r}"] = f"=C{r}-B{r}"
    last = 4 + p.years
    for col in "BCDEFGI":
        for r in range(4, last + 1):
            pr[f"{col}{r}"].number_format = NUM
    for r in range(4, last + 1):
        pr[f"H{r}"].number_format = PCT
    pr[f"A{last + 2}"], pr[f"A{last + 2}"].font = "Total contributions", BOLD
    pr[f"G{last + 2}"] = f"=SUM(G5:G{last})"
    pr[f"G{last + 2}"].number_format = NUM
    for j in range(1, 10):
        pr.column_dimensions[get_column_letter(j)].width = 14
    chart = LineChart()
    chart.title, chart.y_axis.title, chart.x_axis.title = "Funded ratio", "Funded ratio", "Year"
    chart.add_data(Reference(pr, min_col=8, min_row=3, max_row=last), titles_from_data=True)
    chart.set_categories(Reference(pr, min_col=1, min_row=4, max_row=last))
    chart.height, chart.width = 7.5, 16
    pr.add_chart(chart, "K3")

    sh = wb.create_sheet("Rate shock")
    sh["A1"], sh["A1"].font = "Immediate parallel move in yields", TITLE
    _header(sh, 3, ["Move (bp)", "Liability", "Assets", "Funded ratio", "Change (pts)"])
    for i, bp in enumerate((-200, -100, -50, 0, 50, 100, 200)):
        r = 4 + i
        sh[f"A{r}"] = bp
        sh[f"B{r}"] = f"={I['L']}*(1-{I['D']}*A{r}/10000+0.5*{I['C']}*(A{r}/10000)^2)"
        sh[f"C{r}"] = f"={I['L']}*{I['FR']}*({I['eq']}+(1-{I['eq']})*(1-{I['bd']}*A{r}/10000))"
        sh[f"D{r}"] = f"=C{r}/B{r}"
        sh[f"E{r}"] = f"=(D{r}-{I['FR']})*100"
        for col, fmt in (("B", NUM), ("C", NUM), ("D", PCT), ("E", "+0.0;-0.0;0.0")):
            sh[f"{col}{r}"].number_format = fmt
    for j in range(1, 6):
        sh.column_dimensions[get_column_letter(j)].width = 14
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def lbo_workbook(path: Path, p: LBOInputs = LBOInputs()) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Assumptions"
    ws["A1"], ws["A1"].font = "LBO with a private-credit tranche: assumptions (blue cells can be changed)", TITLE
    I = _inputs(ws, [
        ("rev0", p.revenue0, NUM, "Revenue, last twelve months"),
        ("m0", p.margin0, PCT, "EBITDA margin, last twelve months"),
        ("dm", p.margin_step, PCT2, "Margin expansion a year"),
        ("g", p.growth, PCT, "Revenue growth"),
        ("da", p.da_pct, PCT, "D&A, % of revenue"),
        ("capex", p.capex_pct, PCT, "Capex, % of revenue"),
        ("nwc", p.nwc_pct, PCT, "Working capital, % of the revenue increase"),
        ("tax", p.tax, PCT, "Tax rate"),
        ("em", p.entry_multiple, MULT, "Entry multiple (EV / EBITDA)"),
        ("xm", p.exit_multiple, MULT, "Exit multiple (EV / EBITDA)"),
        ("fees", p.fees_pct, PCT, "Transaction fees, % of EV"),
        ("sx", p.senior_x, MULT, "Senior term loan (x EBITDA)"),
        ("sr", p.senior_rate, PCT2, "Senior interest rate"),
        ("sa", p.senior_amort, PCT, "Senior mandatory amortisation, % of original a year"),
        ("sw", p.sweep, PCT, "Cash sweep, % of free cash flow after mandatory"),
        ("bx", p.sub_x, MULT, "Private-credit tranche (x EBITDA), bullet"),
        ("br", p.sub_rate, PCT2, "Private-credit coupon"),
        ("cash", p.min_cash, NUM, "Cash on the balance sheet at close"),
    ])
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 12

    md = wb.create_sheet("Model")
    md["A1"], md["A1"].font = "Operating model and debt schedule", TITLE
    cols = [get_column_letter(2 + t) for t in range(p.years + 1)]       # B..G = years 0..5
    _header(md, 3, ["", *[f"Year {t}" for t in range(p.years + 1)]])
    rows = ["Revenue", "EBITDA margin", "EBITDA", "D&A", "Interest (on opening debt)", "Taxes", "Capex", "Increase in working capital",
            "Free cash flow", "Mandatory amortisation", "Cash sweep", "Senior debt", "Private-credit tranche", "Cash",
            "Total debt / EBITDA", "EBITDA / interest"]
    R = {name: 4 + i for i, name in enumerate(rows)}
    for name, r in R.items():
        md.cell(r, 1, name).font = BOLD if name in ("EBITDA", "Free cash flow", "Senior debt") else Font()
    c0 = cols[0]
    md[f"{c0}{R['Revenue']}"] = f"={I['rev0']}"
    md[f"{c0}{R['EBITDA margin']}"] = f"={I['m0']}"
    md[f"{c0}{R['EBITDA']}"] = f"={c0}{R['Revenue']}*{c0}{R['EBITDA margin']}"
    md[f"{c0}{R['Senior debt']}"] = f"={I['sx']}*{c0}{R['EBITDA']}"
    md[f"{c0}{R['Private-credit tranche']}"] = f"={I['bx']}*{c0}{R['EBITDA']}"
    md[f"{c0}{R['Cash']}"] = f"={I['cash']}"
    md[f"{c0}{R['Total debt / EBITDA']}"] = f"=({c0}{R['Senior debt']}+{c0}{R['Private-credit tranche']})/{c0}{R['EBITDA']}"
    for t in range(1, p.years + 1):
        c, q = cols[t], cols[t - 1]
        f = {
            "Revenue": f"={q}{R['Revenue']}*(1+{I['g']})",
            "EBITDA margin": f"={I['m0']}+{I['dm']}*{t}",
            "EBITDA": f"={c}{R['Revenue']}*{c}{R['EBITDA margin']}",
            "D&A": f"={I['da']}*{c}{R['Revenue']}",
            "Interest (on opening debt)": f"={q}{R['Senior debt']}*{I['sr']}+{q}{R['Private-credit tranche']}*{I['br']}",
            "Taxes": f"=MAX({c}{R['EBITDA']}-{c}{R['D&A']}-{c}{R['Interest (on opening debt)']},0)*{I['tax']}",
            "Capex": f"={I['capex']}*{c}{R['Revenue']}",
            "Increase in working capital": f"={I['nwc']}*({c}{R['Revenue']}-{q}{R['Revenue']})",
            "Free cash flow": f"={c}{R['EBITDA']}-{c}{R['Interest (on opening debt)']}-{c}{R['Taxes']}-{c}{R['Capex']}-{c}{R['Increase in working capital']}",
            "Mandatory amortisation": f"=MIN({I['sa']}*{c0}{R['Senior debt']},{q}{R['Senior debt']})",
            "Cash sweep": f"=MIN(MAX({c}{R['Free cash flow']}-{c}{R['Mandatory amortisation']},0)*{I['sw']},{q}{R['Senior debt']}-{c}{R['Mandatory amortisation']})",
            "Senior debt": f"={q}{R['Senior debt']}-{c}{R['Mandatory amortisation']}-{c}{R['Cash sweep']}",
            "Private-credit tranche": f"={q}{R['Private-credit tranche']}",
            "Cash": f"={q}{R['Cash']}+{c}{R['Free cash flow']}-{c}{R['Mandatory amortisation']}-{c}{R['Cash sweep']}",
            "Total debt / EBITDA": f"=({c}{R['Senior debt']}+{c}{R['Private-credit tranche']})/{c}{R['EBITDA']}",
            "EBITDA / interest": f"={c}{R['EBITDA']}/{c}{R['Interest (on opening debt)']}",
        }
        for name, formula in f.items():
            md[f"{c}{R[name]}"] = formula
    for name, r in R.items():
        fmt = PCT if name == "EBITDA margin" else MULT if "/" in name else NUM
        for c in cols:
            md[f"{c}{r}"].number_format = fmt
    md.column_dimensions["A"].width = 32
    for c in cols:
        md.column_dimensions[c].width = 12
    last = cols[-1]

    rt = wb.create_sheet("Returns")
    rt["A1"], rt["A1"].font = "Returns to equity and sensitivity", TITLE
    lines = [
        ("Entry enterprise value", f"={I['em']}*Model!{c0}{R['EBITDA']}", NUM),
        ("Fees", f"={I['fees']}*B3", NUM),
        ("Debt raised", f"=Model!{c0}{R['Senior debt']}+Model!{c0}{R['Private-credit tranche']}", NUM),
        ("Cash funded at close", f"={I['cash']}", NUM),
        ("Equity invested", "=B3+B4+B6-B5", NUM),
        ("Exit enterprise value", f"={I['xm']}*Model!{last}{R['EBITDA']}", NUM),
        ("Net debt at exit", f"=Model!{last}{R['Senior debt']}+Model!{last}{R['Private-credit tranche']}-Model!{last}{R['Cash']}", NUM),
        ("Equity at exit", "=B8-B9", NUM),
        ("MOIC", "=B10/B7", MULT),
        ("IRR", f"=B11^(1/{p.years})-1", PCT),
    ]
    for i, (lab, formula, fmt) in enumerate(lines):
        rt.cell(3 + i, 1, lab).font = BOLD if lab in ("MOIC", "IRR") else Font()
        c = rt.cell(3 + i, 2, formula)
        c.number_format = fmt
    # the same IRR from the cash-flow row, with Excel's own IRR function
    rt["A14"], rt["A14"].font = "Equity cash flows", BOLD
    for t in range(p.years + 1):
        col = get_column_letter(2 + t)
        rt[f"{col}14"] = "=-B7" if t == 0 else ("=B10" if t == p.years else 0)
        rt[f"{col}14"].number_format = NUM
    rt["A15"], rt["B15"] = "IRR (Excel IRR function)", f"=IRR(B14:{get_column_letter(2 + p.years)}14)"
    rt["B15"].number_format = PCT
    # sensitivity: MOIC by entry (rows) and exit (columns) multiple
    rt["A18"], rt["A18"].font = "MOIC by entry multiple (rows) and exit multiple (columns)", BOLD
    for j, x in enumerate(p.grid):
        c = rt.cell(19, 2 + j, x)
        c.font, c.number_format, c.fill = BLUE, MULT, HEAD
    for i, e in enumerate(p.grid):
        c = rt.cell(20 + i, 1, e)
        c.font, c.number_format, c.fill = BLUE, MULT, HEAD
        for j in range(len(p.grid)):
            col = get_column_letter(2 + j)
            rt[f"{col}{20 + i}"] = (f"=({col}$19*Model!{last}{R['EBITDA']}-$B$9)/"
                                   f"($A{20 + i}*Model!{c0}{R['EBITDA']}*(1+{I['fees']})+{I['cash']}-$B$5)")
            rt[f"{col}{20 + i}"].number_format = MULT
    rt.column_dimensions["A"].width = 34
    for j in range(2, 8):
        rt.column_dimensions[get_column_letter(j)].width = 12
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
