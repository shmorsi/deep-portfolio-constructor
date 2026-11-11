"""The risk study's documents, all rendered from reports/risk/summary.json: results.md, the README
block and the site export. Nothing here computes a result, so a test can rebuild every document
from the committed summary and compare."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MINUS = "−"
START, END = "<!-- risk:start -->", "<!-- risk:end -->"
PROTOCOL = "docs/risk_protocol.md"
TSFM = "https://github.com/shmorsi/tsfm-from-scratch"
VERDICT_ACCURACY = {"better": "better than HAR", "worse": "worse than HAR", "no detectable difference": "not detectably different from HAR"}
VERDICT_RISK = {"lower risk": "lower risk than HAR's portfolio", "higher risk": "higher risk than HAR's portfolio",
                "no detectable difference": "no detectable difference from HAR's portfolio"}


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    text = f"{abs(100 * x):.{digits}f}"
    if float(text) == 0:
        return f"{text}%"
    return (("+" if x > 0 else MINUS) if sign else (MINUS if x < 0 else "")) + text + "%"


def num(x: float, digits: int = 2) -> str:
    text = f"{abs(x):.{digits}f}"
    return (MINUS if x < 0 and float(text) != 0 else "") + text


def money(x: float) -> str:
    return f"${x:,.0f}" if x >= 10 else f"${x:,.2f}"


def ratio(r: dict) -> str:
    return f"{num(r['ratio'], 3)} [{num(r['ci_low'], 3)}, {num(r['ci_high'], 3)}]"


def order(s: dict) -> list[str]:
    """Forecasters in the order they are shown: classical, then each checkpoint's two variants, then the other seeds."""
    names = list(s["accuracy"])
    classical = [n for n in ("last", "mean60", "ewma", "har") if n in names]
    return classical + [n for n in names if n not in classical]


def md_table(headers: list[str], rows: list[list[str]], bold_first: tuple[str, ...] = ()) -> str:
    """A Markdown table: the first column left-aligned, the rest right-aligned except verdicts."""
    align = ["---"] + ["---" if h == "Verdict" else "---:" for h in headers[1:]]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        first = f"**{r[0]}**" if r[0] in bold_first else r[0]
        out.append("| " + " | ".join([first, *r[1:]]) + " |")
    return "\n".join(out)


ACCURACY_HEADERS = ["Forecaster", "Quantile loss, ratio to HAR [95% interval]", "Verdict", "Absolute error, ratio to HAR", "QLIKE, ratio to HAR", "Inside its 80% band"]
PORTFOLIO_HEADERS = ["Portfolio fed by", "Realized volatility a year", "Against HAR's portfolio: volatility, t", "Verdict", "$1 became", "Worst fall", "Traded a year"]


def accuracy_rows(s: dict) -> list[list[str]]:
    rows = []
    for name in order(s):
        a = s["accuracy"][name]
        if name == "har":
            rows.append([s["labels"][name], f"1 (the standard to beat; loss {num(a['wql'], 4)})", "", "1", "1", pct(a["cover80"], 0)])
            continue
        v = a["vs_har"]
        rows.append([s["labels"][name], ratio(v["wql"]), v["wql"]["verdict"], num(v["mae"]["ratio"], 3), num(v["qlike"]["ratio"], 3), pct(a["cover80"], 0)])
    return rows


def portfolio_rows(s: dict, rule: str) -> list[list[str]]:
    rows = []
    P = s["portfolios"][rule]
    for name in [*order(s), "oracle"]:
        p = P[name]
        if name == "har":
            versus, verdict = "the standard to beat", ""
        else:
            c = p["vs_har"]
            versus, verdict = f"{pct(c['volatility_change'], 1, sign=True)}, {num(c['t_nw'])}", c["verdict"]
        rows.append([s["labels"][name], pct(p["realized_volatility"]), versus, verdict, money(p["terminal_wealth"]), pct(p["max_drawdown"], 0), pct(p["traded_per_year"], 0)])
    m = s["references"]["market"]
    rows.append([s["labels"]["market"], pct(m["realized_volatility"]), "", "", money(m["terminal_wealth"]), pct(m["max_drawdown"], 0), pct(m["traded_per_year"], 0)])
    return rows


def accuracy_table(s: dict) -> str:
    return md_table(ACCURACY_HEADERS, accuracy_rows(s), bold_first=(s["labels"]["har"],))


def portfolio_table(s: dict, rule: str) -> str:
    return md_table(PORTFOLIO_HEADERS, portfolio_rows(s, rule))


def decade_table(s: dict) -> str:
    names = [n for n in order(s) if n != "har"]
    return md_table(["Decade", *[s["labels"][n] for n in names]], [[decade, *[num(row[n], 3) for n in names]] for decade, row in s["decades_wql_vs_har"].items()])


def model_sentence(s: dict, checkpoint: str) -> str:
    """What the protocol's decision rule says about one checkpoint, in a sentence that follows the numbers."""
    m = s["models"][checkpoint]
    who = f"the {m['label']} ({m['parameters']:,} parameters)"
    if m["status"] != "run":
        return f"{who[0].upper()}{who[1:]} has not been run: its weights are {m['weights']}, and the protocol scores it without change when they are."
    ft, zero = f"tsfm_ft_{checkpoint}", f"tsfm_zero_{checkpoint}"
    a, z = s["accuracy"][ft]["vs_har"]["wql"], s["accuracy"][zero]["vs_har"]["wql"]
    c = s["portfolios"]["min_variance"][ft]["vs_har"]
    lead = "**Yes**" if m["earns_its_place"] else "**No**"
    return (f"{lead} for {who}. Fine-tuned, its quantile loss was {ratio(a)} times HAR's: {VERDICT_ACCURACY[a['verdict']]} "
            f"(zero-shot: {ratio(z)}). The minimum-variance portfolio built on its forecasts ran at "
            f"{pct(s['portfolios']['min_variance'][ft]['realized_volatility'])} a year against "
            f"{pct(s['portfolios']['min_variance']['har']['realized_volatility'])} for HAR's ({pct(c['volatility_change'], 1, sign=True)}, "
            f"*t* = {num(c['t_nw'])}): {VERDICT_RISK[c['verdict']]}.")


def oracle_sentence(s: dict) -> str:
    P = s["portfolios"]["min_variance"]
    c = P["oracle"]["vs_har"]
    return (f"A perfect forecast (the oracle, which is handed next month's volatilities and cannot be invested in) would have run the same rule at "
            f"{pct(P['oracle']['realized_volatility'])} a year against {pct(P['har']['realized_volatility'])} for HAR's "
            f"({pct(c['volatility_change'], 1, sign=True)}, *t* = {num(c['t_nw'])}: {c['verdict']}). That is all the room any forecaster has with this rule.")


def classical_sentence(s: dict) -> str:
    best = min((n for n in ("last", "mean60", "ewma") if n in s["accuracy"]), key=lambda n: s["accuracy"][n]["vs_har"]["wql"]["ratio"])
    v = s["accuracy"][best]["vs_har"]["wql"]
    rival = {"last": "last month's volatility", "mean60": "the 60-month mean", "ewma": "the RiskMetrics average"}[best]
    return (f"Among the classical forecasters, HAR is the standard: the closest rival, {rival}, has a quantile loss "
            f"{ratio(v)} times HAR's ({v['verdict']}).")


VERDICT_WORD = {"better": "better", "worse": "worse", "no detectable difference": "no detectable difference"}
RIVAL = {"last": "last month's volatility", "mean60": "the 60-month mean", "ewma": "the RiskMetrics average"}


def described_sentence(s: dict, checkpoint: str) -> str:
    """After the test, not pre-registered: where the fine-tuned model stands among the other forecasters."""
    d = s.get("described_after_the_test", {}).get("finetuned_quantile_loss_vs", {}).get(checkpoint)
    if not d:
        return ""
    parts = [f"{ratio(d[n])} times that of {RIVAL[n]} ({VERDICT_WORD[d[n]['verdict']]})" for n in ("ewma", "last", "mean60") if n in d]
    zero = d.get(f"tsfm_zero_{checkpoint}")
    tail = f" Against its own zero-shot run: {ratio(zero)} ({VERDICT_WORD[zero['verdict']]})." if zero else ""
    return (f"Described after the test, not pre-registered: fine-tuned, the {s['models'][checkpoint]['label']}'s quantile loss was "
            + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "." + tail)


def described_rows(s: dict) -> tuple[list[str], list[list[str]]]:
    rows = []
    for checkpoint, d in s.get("described_after_the_test", {}).get("finetuned_quantile_loss_vs", {}).items():
        for other, v in d.items():
            rows.append([f"{s['labels'][f'tsfm_ft_{checkpoint}']}", s["labels"][other], ratio(v), v["verdict"]])
    return ["Fine-tuned model", "Against", "Quantile loss, ratio [95% interval]", "Verdict"], rows


def seeds_sentence(s: dict) -> str:
    """The CPU model's other pretraining seeds, fine-tuned: the robustness check the protocol asks for."""
    names = sorted(n for n in s["accuracy"] if n.startswith("tsfm_ft_cpu_s0_seed"))
    if not names or "tsfm_ft_cpu_s0" not in s["accuracy"]:
        return ""
    P = s["portfolios"]["min_variance"]
    primary = (s["accuracy"]["tsfm_ft_cpu_s0"]["vs_har"]["wql"]["verdict"], P["tsfm_ft_cpu_s0"]["vs_har"]["verdict"])
    same = all((s["accuracy"][n]["vs_har"]["wql"]["verdict"], P[n]["vs_har"]["verdict"]) == primary for n in names)
    parts = [f"seed {n[-1]}: {ratio(s['accuracy'][n]['vs_har']['wql'])} times HAR's loss ({s['accuracy'][n]['vs_har']['wql']['verdict']}), portfolio "
             f"{pct(P[n]['vs_har']['volatility_change'], 1, sign=True)} against HAR's ({P[n]['vs_har']['verdict']})" for n in names]
    lead = "The CPU model's other pretraining seeds give the same verdicts" if same else "The CPU model's other pretraining seeds do not all give the same verdicts"
    return f"{lead}, fine-tuned: " + "; ".join(parts) + "."


def finetune_sentence(s: dict, checkpoint: str) -> str:
    m = s["models"][checkpoint]
    if m["status"] != "run":
        return ""
    kept = m["refits_kept_pretrained"]
    chose = ("at every one the validation years preferred a fine-tuned model to the untouched one" if kept == 0 else
             f"at {kept} of them the validation years preferred the untouched pretrained model, which was kept")
    return f"Fine-tuning was refit {m['refits']} times, every five years on past data only; {chose}."


MEASURE_NOTE = ("Two measures of a portfolio's risk appear side by side. \"Realized volatility a year\" is the root of the average monthly variance. The comparison "
                "with HAR's portfolio is the average difference in log variance, month by month, shown as the change in a typical month's volatility; "
                "it is the one the verdict is based on.")


def readme_section(s: dict, commit: str) -> str:
    w = s["window"]
    lines = [
        START,
        f"<!-- Generated by `python -m dpc.cli risk-site` from reports/risk/summary.json. Edit src/dpc/risk/report.py, not this block. -->",
        "",
        f"**The question** ([pre-registered]({PROTOCOL}), committed before any forecast was scored): does a time-series foundation model pretrained from scratch "
        f"([tsfm-from-scratch]({TSFM})) forecast next month's volatility of each of the {w['industries']} industries better than the standard methods, and does a long-only "
        f"portfolio built on its forecasts end up less volatile? Scored on {w['months']} months, {w['first_month']} to {w['last_month']}, the wealth study's window.",
        "",
        "**The answer, by the rule fixed in advance** (better than HAR on the quantile loss, and a lower-risk portfolio):",
        "",
        *[f"* {model_sentence(s, c)}" for c in s["models"]],
        "",
        accuracy_table(s),
        "",
        f"Lower is better; 1 is HAR. Each ratio pools {w['industries']} industries and {w['months']} months; the interval is a moving-block bootstrap over months. "
        f"\"Better\" needs the whole interval below 1.",
        "",
        "**What the forecasts are worth.** Each forecaster's volatilities go into the same rule: the long-only, fully invested portfolio with the smallest forecast "
        f"variance, at most {pct(s['tests']['max_weight'], 0)} in one industry, traded monthly at {s['tests']['cost_bps']:g} bps. The correlation matrix is the same for all of them.",
        "",
        portfolio_table(s, "min_variance"),
        "",
        f"* {oracle_sentence(s)}",
        f"* {classical_sentence(s)}",
        *[f"* {t}" for t in (finetune_sentence(s, c) for c in s["models"]) if t],
        *[f"* {t}" for t in (seeds_sentence(s),) if t],
        *[f"* {t}" for t in (described_sentence(s, c) for c in s["models"]) if t],
        f"* {MEASURE_NOTE}",
        f"* Every table (the second rule, the decades, the other seeds) is in [`reports/risk/results.md`](reports/risk/results.md). "
        f"Run `python -m dpc.cli risk-study`; the model's forecasts come from [`scripts/risk_tsfm_forecast.py`](scripts/risk_tsfm_forecast.py), run in tsfm-from-scratch's environment.",
        END,
    ]
    return "\n".join(lines)


def results_md(s: dict, commit: str) -> str:
    w = s["window"]
    parts = [
        "# Risk study: results",
        "",
        f"Generated by `python -m dpc.cli risk-evaluate` from the forecast files in `reports/risk/forecasts/` (commit {commit}). The protocol is "
        f"[`{PROTOCOL}`](../../{PROTOCOL}). {w['months']} months, {w['first_month']} to {w['last_month']}, {w['industries']} industries.",
        "",
        "## The decision rule, applied",
        "",
        *[f"* {model_sentence(s, c)}" for c in s["models"]],
        "",
        "## Forecast accuracy",
        "",
        accuracy_table(s),
        "",
        "## Minimum-variance portfolio",
        "",
        portfolio_table(s, "min_variance"),
        "",
        f"{oracle_sentence(s)}",
        "",
        "## Inverse-volatility portfolio (the second rule)",
        "",
        portfolio_table(s, "inverse_volatility"),
        "",
        "## Quantile loss by decade, ratio to HAR",
        "",
        decade_table(s),
        "",
        "## The other seeds",
        "",
        seeds_sentence(s) or "Not run.",
        "",
        "## Described after the test (not pre-registered)",
        "",
        "The protocol compares every forecaster with HAR. These rows compare the fine-tuned model with each other forecaster, by the same loss and bootstrap.",
        "",
        md_table(*described_rows(s)),
        "",
        MEASURE_NOTE,
        "",
    ]
    return "\n".join(parts)


def test_count(root: Path) -> int:
    """Test functions in tests/test_risk.py (counted from the file, as the Pension Lab does, so the export does not depend on what is installed)."""
    file = root / "tests" / "test_risk.py"
    return sum(1 for line in file.read_text().splitlines() if line.startswith("def test_")) if file.is_file() else 0


def export(s: dict, commit: str, root: Path) -> dict:
    """reports/risk/site_risk.json: the figures, tables and sentences the website shows."""
    where = "deep-portfolio-constructor reports/risk/summary.json"
    stats: dict[str, dict] = {"tests": {"label": "Tests of the risk study's code", "display": str(test_count(root)), "source": "deep-portfolio-constructor tests/test_risk.py"}}

    def put(key: str, label_: str, display: str, path: str) -> None:
        stats[key] = {"label": label_, "display": display, "source": f"{where} [{path}]"}

    w, P = s["window"], s["portfolios"]["min_variance"]
    put("months", "Months scored", str(w["months"]), "window.months")
    put("industries", "Industries", str(w["industries"]), "window.industries")
    put("forecasts", "Forecasts scored per forecaster", f"{w['months'] * w['industries']:,}", "window.months x window.industries")
    put("volHar", "Realized volatility a year, minimum variance on HAR's forecasts", pct(P["har"]["realized_volatility"]), "portfolios.min_variance.har.realized_volatility")
    put("volOracle", "Realized volatility a year, minimum variance on perfect forecasts", pct(P["oracle"]["realized_volatility"]), "portfolios.min_variance.oracle.realized_volatility")
    put("volMarket", "Realized volatility a year, the market", pct(s["references"]["market"]["realized_volatility"]), "references.market.realized_volatility")
    put("oracleVsHar", "Volatility of the oracle's portfolio against HAR's", pct(P["oracle"]["vs_har"]["volatility_change"], 1, sign=True), "portfolios.min_variance.oracle.vs_har.volatility_change")
    put("oracleVerdict", "Verdict, oracle against HAR", P["oracle"]["vs_har"]["verdict"], "portfolios.min_variance.oracle.vs_har.verdict")
    for checkpoint, m in s["models"].items():
        sfx = "".join(part.capitalize() for part in checkpoint.split("_"))
        put(f"status{sfx}", f"{m['label']}: status", m["status"], f"models.{checkpoint}.status")
        put(f"params{sfx}", f"{m['label']}: parameters", f"{m['parameters']:,}", f"models.{checkpoint}.parameters")
        if m["status"] != "run":
            continue
        for variant, tag in (("ft", "Ft"), ("zero", "Zero")):
            name = f"tsfm_{variant}_{checkpoint}"
            a, c = s["accuracy"][name]["vs_har"]["wql"], P[name]["vs_har"]
            put(f"wql{tag}{sfx}", f"Quantile loss against HAR, {s['labels'][name]}", f"{num(a['ratio'], 3)}×", f"accuracy.{name}.vs_har.wql.ratio")
            put(f"wqlCi{tag}{sfx}", f"95% interval, {s['labels'][name]}", f"[{num(a['ci_low'], 3)}, {num(a['ci_high'], 3)}]", f"accuracy.{name}.vs_har.wql.ci_low, ci_high")
            put(f"wqlVerdict{tag}{sfx}", f"Verdict on accuracy, {s['labels'][name]}", a["verdict"], f"accuracy.{name}.vs_har.wql.verdict")
            put(f"vol{tag}{sfx}", f"Realized volatility a year, minimum variance on {s['labels'][name]}", pct(P[name]["realized_volatility"]), f"portfolios.min_variance.{name}.realized_volatility")
            put(f"volVsHar{tag}{sfx}", f"Portfolio volatility against HAR's, {s['labels'][name]}", pct(c["volatility_change"], 1, sign=True), f"portfolios.min_variance.{name}.vs_har.volatility_change")
            put(f"volT{tag}{sfx}", f"Newey-West t, {s['labels'][name]}", num(c["t_nw"]), f"portfolios.min_variance.{name}.vs_har.t_nw")
            put(f"volVerdict{tag}{sfx}", f"Verdict on portfolio risk, {s['labels'][name]}", c["verdict"], f"portfolios.min_variance.{name}.vs_har.verdict")
        for other, v in s.get("described_after_the_test", {}).get("finetuned_quantile_loss_vs", {}).get(checkpoint, {}).items():
            tag = {"last": "Last", "mean60": "Mean60", "ewma": "Ewma"}.get(other, "Zero")
            path = f"described_after_the_test.finetuned_quantile_loss_vs.{checkpoint}.{other}"
            put(f"ftVs{tag}{sfx}", f"Fine-tuned {m['label']}: quantile loss against {s['labels'][other]} (described after the test)", f"{num(v['ratio'], 3)}×", f"{path}.ratio")
            put(f"ftVs{tag}Ci{sfx}", f"95% interval, fine-tuned {m['label']} against {s['labels'][other]}", f"[{num(v['ci_low'], 3)}, {num(v['ci_high'], 3)}]", f"{path}.ci_low, ci_high")
            put(f"ftVs{tag}Verdict{sfx}", f"Verdict, fine-tuned {m['label']} against {s['labels'][other]}", v["verdict"], f"{path}.verdict")
        put(f"earns{sfx}", f"{m['label']}: its risk forecasts help this investor", "yes" if m["earns_its_place"] else "no", f"models.{checkpoint}.earns_its_place")
        put(f"refits{sfx}", f"{m['label']}: fine-tuning refits", str(m["refits"]), f"models.{checkpoint}.refits")
        put(f"refitsKept{sfx}", f"{m['label']}: refits that kept the pretrained model", f"{m['refits_kept_pretrained']} of {m['refits']}", f"models.{checkpoint}.refits_kept_pretrained")
    return {
        "source": f"deep-portfolio-constructor {commit}: reports/risk/summary.json", "commit": commit, "protocol": PROTOCOL,
        "stats": stats,
        "sentences": {**{f"model_{c}": model_sentence(s, c).replace("**", "").replace("*", "") for c in s["models"]},
                      **{f"described_{c}": described_sentence(s, c) for c in s["models"] if described_sentence(s, c)},
                      "oracle": oracle_sentence(s).replace("*", ""), "classical": classical_sentence(s), "seeds": seeds_sentence(s)},
        "tables": {
            "accuracy": {"caption": f"Forecasts of next month's volatility, {s['window']['industries']} industries and {s['window']['months']} months pooled. Lower is better; 1 is HAR. "
                                    "Better needs the whole interval below 1.",
                         "headers": ACCURACY_HEADERS, "rows": accuracy_rows(s), "source": f"{where} [accuracy]"},
            "min_variance": {"caption": "One long-only minimum-variance rule, fed by each forecaster's volatilities. Realized volatility is measured from the portfolio's own daily returns.",
                             "headers": PORTFOLIO_HEADERS, "rows": portfolio_rows(s, "min_variance"), "source": f"{where} [portfolios.min_variance, references.market]"},
        },
    }


def head_commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def write_reports(root: Path, s: dict, commit: str | None = None) -> None:
    commit = commit or head_commit(root)
    out = root / "reports" / "risk"
    (out / "results.md").write_text(results_md(s, commit) + "\n")
    (out / "site_risk.json").write_text(json.dumps(export(s, commit, root), indent=1, ensure_ascii=False) + "\n")
    readme = root / "README.md"
    text = readme.read_text() if readme.is_file() else ""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        readme.write_text(head + readme_section(s, commit) + rest.split(END, 1)[1])


def stage_site(root: Path) -> None:
    write_reports(root, json.loads((root / "reports" / "risk" / "summary.json").read_text()))
