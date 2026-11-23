"""The anomaly zoo: fifteen published equity anomalies, their decay after publication, and combining them out of sample.

Data: Kenneth French's value-weighted decile portfolios formed on fifteen characteristics (CRSP
stocks, NYSE breakpoints). Each anomaly is the top-minus-bottom decile spread, signed the way its
paper found it profitable. Gross of trading costs; long-short; value-weighted (so microcaps carry
little weight). Not pre-registered: the questions are those of McLean and Pontiff (2016) and
Harvey, Liu and Zhu (2016), asked once of these data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anomaly:
    key: str
    name: str
    file: str
    long: str            # which decile is bought: "hi" or "lo"
    paper: str
    year: int            # publication year of the paper usually cited for it


ZOO: tuple[Anomaly, ...] = (
    Anomaly("size", "Size (small minus big)", "Portfolios_Formed_on_ME_CSV.zip", "lo", "Banz", 1981),
    Anomaly("value", "Book-to-market", "Portfolios_Formed_on_BE-ME_CSV.zip", "hi", "Fama and French", 1992),
    Anomaly("profitability", "Operating profitability", "Portfolios_Formed_on_OP_CSV.zip", "hi", "Novy-Marx", 2013),
    Anomaly("investment", "Investment (conservative minus aggressive)", "Portfolios_Formed_on_INV_CSV.zip", "lo", "Titman, Wei and Xie", 2004),
    Anomaly("momentum", "Momentum (12-2)", "10_Portfolios_Prior_12_2_CSV.zip", "hi", "Jegadeesh and Titman", 1993),
    Anomaly("st_reversal", "Short-term reversal (1-0)", "10_Portfolios_Prior_1_0_CSV.zip", "lo", "Jegadeesh", 1990),
    Anomaly("lt_reversal", "Long-term reversal (60-13)", "10_Portfolios_Prior_60_13_CSV.zip", "lo", "De Bondt and Thaler", 1985),
    Anomaly("accruals", "Accruals", "Portfolios_Formed_on_AC_CSV.zip", "lo", "Sloan", 1996),
    Anomaly("low_beta", "Low beta", "Portfolios_Formed_on_BETA_CSV.zip", "lo", "Black, Jensen and Scholes", 1972),
    Anomaly("net_issuance", "Net share issuance", "Portfolios_Formed_on_NI_CSV.zip", "lo", "Pontiff and Woodgate", 2008),
    Anomaly("low_vol", "Low volatility", "Portfolios_Formed_on_VAR_CSV.zip", "lo", "Ang, Hodrick, Xing and Zhang", 2006),
    Anomaly("low_resvol", "Low residual volatility", "Portfolios_Formed_on_RESVAR_CSV.zip", "lo", "Ang, Hodrick, Xing and Zhang", 2006),
    Anomaly("earnings_yield", "Earnings-to-price", "Portfolios_Formed_on_E-P_CSV.zip", "hi", "Basu", 1977),
    Anomaly("cashflow_yield", "Cash-flow-to-price", "Portfolios_Formed_on_CF-P_CSV.zip", "hi", "Lakonishok, Shleifer and Vishny", 1994),
    Anomaly("dividend_yield", "Dividend yield", "Portfolios_Formed_on_D-P_CSV.zip", "hi", "Litzenberger and Ramaswamy", 1979),
)
