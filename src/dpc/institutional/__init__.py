"""Institutional toolkit: the machinery a pension fund or asset manager runs around a portfolio.

    risk_model      a fundamental factor risk model (Fama-French factors), ex-ante risk and its decomposition
    black_litterman equilibrium returns implied by the benchmark, blended with views
    active          a tracking-error-budgeted, long-only optimiser and a walk-forward active-management backtest
    attribution     Brinson-Fachler attribution with Carino linking, in pandas and in SQL (DuckDB)
    curve           Nelson-Siegel-Svensson yield-curve fits, curve PCA, key-rate durations
    liabilities     a pension cohort with mortality (Gompertz-Makeham), solvency and going-concern valuations
    ldi             a swap overlay: hedge ratio, leverage, collateral, and the liquidity stress of a yield spike
    fx              the currency hedge ratio for a Canadian investor holding US equities
    privates        private assets: Takahashi-Alexander cash flows, commitment pacing, the denominator effect, desmoothing

None of this was pre-registered: it is a toolkit with tests, run once on public data by
`python -m dpc.cli institutional`, and described as such in reports/institutional/.
"""
