"""The risk study: does a forecasting model's volatility forecast help a long-only investor?

docs/risk_protocol.md fixes the question, the data, the forecasters, the scores and the
decision rule; it was committed before any forecast was scored. The classical forecasters
and every score live here. The foundation model's forecasts are produced by
scripts/risk_tsfm_forecast.py, which runs in the tsfm-from-scratch environment and writes
files in the same format, so this package never imports that project.
"""
