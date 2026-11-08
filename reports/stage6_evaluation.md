# Stage 6 - Evaluation (all results out-of-sample, walk-forward, net of costs)

## Performance

|                                |   ann_return |   ann_vol |   sharpe |   sharpe_gross |   sortino |   max_drawdown |   hit_rate_monthly |   alpha_ann_ff6 |   alpha_t_ff6 |   beta_mkt |   r2_ff6 |   turnover_ann |   cost_drag_ann |
|:-------------------------------|-------------:|----------:|---------:|---------------:|----------:|---------------:|-------------------:|----------------:|--------------:|-----------:|---------:|---------------:|----------------:|
| Equal weight (LO)              |        0.128 |     0.191 |    0.724 |          0.728 |     1.021 |         -0.485 |              0.671 |           0.029 |         3.832 |      0.918 |    0.945 |          0.580 |           0.001 |
| Min variance (LO)              |        0.076 |     0.139 |    0.599 |          0.625 |     0.834 |         -0.381 |              0.640 |          -0.006 |        -0.421 |      0.695 |    0.691 |          3.625 |           0.004 |
| Mean-variance LW (LO)          |        0.081 |     0.149 |    0.599 |          0.679 |     0.843 |         -0.369 |              0.622 |          -0.007 |        -0.578 |      0.730 |    0.714 |         11.738 |           0.012 |
| Risk parity (LO)               |        0.115 |     0.172 |    0.722 |          0.726 |     1.014 |         -0.450 |              0.676 |           0.020 |         2.424 |      0.849 |    0.917 |          0.749 |           0.001 |
| HRP (LO)                       |        0.109 |     0.161 |    0.720 |          0.740 |     1.013 |         -0.413 |              0.671 |           0.018 |         2.053 |      0.788 |    0.887 |          3.110 |           0.003 |
| L/S naive signal               |       -0.087 |     0.184 |   -0.404 |         -0.274 |    -0.544 |         -0.877 |              0.453 |          -0.078 |        -3.470 |     -0.163 |    0.515 |         23.746 |           0.024 |
| L/S classical (LW + optimiser) |       -0.006 |     0.080 |   -0.032 |          0.119 |    -0.045 |         -0.319 |              0.498 |          -0.012 |        -0.690 |      0.016 |    0.148 |         11.962 |           0.012 |
| L/S deep risk + optimiser      |       -0.009 |     0.095 |   -0.047 |          0.080 |    -0.065 |         -0.363 |              0.480 |          -0.020 |        -1.241 |      0.053 |    0.331 |         11.993 |           0.012 |
| L/S end-to-end deep            |        0.047 |     0.086 |    0.582 |          0.714 |     0.837 |         -0.276 |              0.613 |           0.031 |         1.475 |      0.193 |    0.289 |         11.140 |           0.011 |

## Risk decomposition (headline)

Ex-ante shares from an independent Barra-style model (6 style + 11 sector factors); realised shares split each day's P&L into factor and stock-specific parts.

|                                |   exante_share_specific |   exante_share_style |   exante_share_sector |   exante_vol_ann |   realised_share_factor |   realised_share_specific |   realised_cov_term_share |   mean_abs_beta_mkt |   mean_abs_beta_smb |   mean_abs_beta_hml |   max_abs_sector_net |
|:-------------------------------|------------------------:|---------------------:|----------------------:|-----------------:|------------------------:|--------------------------:|--------------------------:|--------------------:|--------------------:|--------------------:|---------------------:|
| Equal weight (LO)              |                   0.014 |                0.663 |                 0.323 |            0.177 |                   1.000 |                     0.000 |                    -0.000 |               0.927 |               0.102 |               0.097 |                0.173 |
| Min variance (LO)              |                   0.084 |                0.089 |                 0.827 |            0.133 |                   0.939 |                     0.061 |                    -0.047 |               0.596 |               0.128 |               0.131 |                0.273 |
| Mean-variance LW (LO)          |                   0.097 |                0.213 |                 0.691 |            0.138 |                   0.932 |                     0.068 |                    -0.009 |               0.656 |               0.133 |               0.149 |                0.282 |
| Risk parity (LO)               |                   0.016 |                0.522 |                 0.461 |            0.159 |                   1.000 |                     0.000 |                     0.000 |               0.842 |               0.113 |               0.096 |                0.144 |
| HRP (LO)                       |                   0.020 |                0.467 |                 0.513 |            0.151 |                   0.997 |                     0.003 |                    -0.009 |               0.805 |               0.120 |               0.100 |                0.162 |
| L/S naive signal               |                   0.174 |                0.693 |                 0.132 |            0.164 |                   0.926 |                     0.074 |                     0.036 |               0.214 |               0.127 |               0.332 |                0.283 |
| L/S classical (LW + optimiser) |                   0.614 |                0.340 |                 0.046 |            0.071 |                   0.379 |                     0.621 |                     0.091 |               0.038 |               0.039 |               0.041 |                0.050 |
| L/S deep risk + optimiser      |                   0.412 |                0.556 |                 0.032 |            0.093 |                   0.658 |                     0.342 |                    -0.010 |               0.039 |               0.042 |               0.044 |                0.050 |
| L/S end-to-end deep            |                   0.471 |                0.476 |                 0.053 |            0.078 |                   0.544 |                     0.456 |                     0.094 |               0.047 |               0.040 |               0.044 |                0.050 |

## 2008

|                                |   return |   vol_ann |   sharpe |   max_drawdown |
|:-------------------------------|---------:|----------:|---------:|---------------:|
| Equal weight (LO)              |   -0.290 |     0.408 |   -0.707 |         -0.463 |
| Min variance (LO)              |   -0.226 |     0.282 |   -0.841 |         -0.295 |
| Mean-variance LW (LO)          |   -0.203 |     0.292 |   -0.695 |         -0.298 |
| Risk parity (LO)               |   -0.267 |     0.372 |   -0.719 |         -0.425 |
| HRP (LO)                       |   -0.235 |     0.340 |   -0.681 |         -0.376 |
| L/S naive signal               |    0.206 |     0.244 |    0.952 |         -0.129 |
| L/S classical (LW + optimiser) |    0.047 |     0.123 |    0.468 |         -0.072 |
| L/S deep risk + optimiser      |    0.067 |     0.128 |    0.610 |         -0.084 |
| L/S end-to-end deep            |    0.006 |     0.133 |    0.115 |         -0.149 |

## 2020

|                                |   return |   vol_ann |   sharpe |   max_drawdown |
|:-------------------------------|---------:|----------:|---------:|---------------:|
| Equal weight (LO)              |    0.137 |     0.351 |    0.542 |         -0.357 |
| Min variance (LO)              |    0.108 |     0.307 |    0.487 |         -0.352 |
| Mean-variance LW (LO)          |    0.092 |     0.309 |    0.438 |         -0.351 |
| Risk parity (LO)               |    0.135 |     0.325 |    0.553 |         -0.342 |
| HRP (LO)                       |    0.131 |     0.321 |    0.543 |         -0.334 |
| L/S naive signal               |   -0.149 |     0.310 |   -0.360 |         -0.321 |
| L/S classical (LW + optimiser) |    0.061 |     0.096 |    0.655 |         -0.071 |
| L/S deep risk + optimiser      |   -0.022 |     0.131 |   -0.099 |         -0.179 |
| L/S end-to-end deep            |    0.091 |     0.091 |    1.003 |         -0.063 |

## 2022

|                                |   return |   vol_ann |   sharpe |   max_drawdown |
|:-------------------------------|---------:|----------:|---------:|---------------:|
| Equal weight (LO)              |   -0.078 |     0.212 |   -0.278 |         -0.207 |
| Min variance (LO)              |   -0.020 |     0.157 |   -0.049 |         -0.198 |
| Mean-variance LW (LO)          |   -0.047 |     0.173 |   -0.192 |         -0.177 |
| Risk parity (LO)               |   -0.068 |     0.190 |   -0.276 |         -0.191 |
| HRP (LO)                       |   -0.067 |     0.178 |   -0.302 |         -0.194 |
| L/S naive signal               |    0.017 |     0.188 |    0.183 |         -0.123 |
| L/S classical (LW + optimiser) |   -0.029 |     0.089 |   -0.291 |         -0.099 |
| L/S deep risk + optimiser      |   -0.006 |     0.100 |   -0.014 |         -0.073 |
| L/S end-to-end deep            |   -0.113 |     0.095 |   -1.219 |         -0.152 |

## Paired tests on monthly net returns (Newey-West)

|                                 |   mean_monthly_diff |   t_stat |   n_months |
|:--------------------------------|--------------------:|---------:|-----------:|
| e2e - ls_classical              |              0.0043 |   1.7049 |   225.0000 |
| ls_deep_twostage - ls_classical |             -0.0002 |  -0.2209 |   225.0000 |
| e2e - ls_deep_twostage          |              0.0046 |   1.7130 |   225.0000 |

## Sensitivity: transaction costs (Sharpe, same trades re-priced)

|                                |   sharpe@0bps |   sharpe@5bps |   sharpe@10bps |   sharpe@20bps |   sharpe@40bps |
|:-------------------------------|--------------:|--------------:|---------------:|---------------:|---------------:|
| L/S naive signal               |         -0.27 |         -0.34 |          -0.40 |          -0.53 |          -0.79 |
| L/S classical (LW + optimiser) |          0.12 |          0.04 |          -0.03 |          -0.18 |          -0.48 |
| L/S deep risk + optimiser      |          0.08 |          0.02 |          -0.05 |          -0.17 |          -0.42 |
| L/S end-to-end deep            |          0.71 |          0.65 |           0.58 |           0.45 |           0.19 |

## Sensitivity: constraints (re-optimised; deep models not retrained)

|                                                          |   sharpe |   ann_vol |   max_drawdown |   turnover_ann |
|:---------------------------------------------------------|---------:|----------:|---------------:|---------------:|
| ('max_weight=0.03', 'L/S classical (LW + optimiser)')    |    0.039 |     0.064 |         -0.255 |         11.630 |
| ('max_weight=0.03', 'L/S deep risk + optimiser')         |    0.003 |     0.077 |         -0.283 |         11.725 |
| ('max_weight=0.03', 'L/S end-to-end deep')               |    0.541 |     0.070 |         -0.231 |         10.358 |
| ('max_weight=0.05', 'L/S classical (LW + optimiser)')    |   -0.032 |     0.080 |         -0.319 |         11.962 |
| ('max_weight=0.05', 'L/S deep risk + optimiser')         |   -0.047 |     0.095 |         -0.363 |         11.993 |
| ('max_weight=0.05', 'L/S end-to-end deep')               |    0.582 |     0.086 |         -0.276 |         11.140 |
| ('max_weight=0.1', 'L/S classical (LW + optimiser)')     |   -0.084 |     0.098 |         -0.472 |         12.000 |
| ('max_weight=0.1', 'L/S deep risk + optimiser')          |   -0.109 |     0.115 |         -0.510 |         12.000 |
| ('max_weight=0.1', 'L/S end-to-end deep')                |    0.671 |     0.099 |         -0.285 |         11.517 |
| ('sector_limit=0.025', 'L/S classical (LW + optimiser)') |   -0.033 |     0.079 |         -0.319 |         11.951 |
| ('sector_limit=0.025', 'L/S deep risk + optimiser')      |   -0.070 |     0.093 |         -0.368 |         11.990 |
| ('sector_limit=0.025', 'L/S end-to-end deep')            |    0.606 |     0.083 |         -0.269 |         11.122 |
| ('sector_limit=0.05', 'L/S classical (LW + optimiser)')  |   -0.032 |     0.080 |         -0.319 |         11.962 |
| ('sector_limit=0.05', 'L/S deep risk + optimiser')       |   -0.047 |     0.095 |         -0.363 |         11.993 |
| ('sector_limit=0.05', 'L/S end-to-end deep')             |    0.582 |     0.086 |         -0.276 |         11.140 |
| ('sector_limit=0.1', 'L/S classical (LW + optimiser)')   |   -0.032 |     0.082 |         -0.331 |         11.974 |
| ('sector_limit=0.1', 'L/S deep risk + optimiser')        |   -0.034 |     0.099 |         -0.369 |         11.992 |
| ('sector_limit=0.1', 'L/S end-to-end deep')              |    0.579 |     0.090 |         -0.285 |         11.137 |

## Look-ahead checks

```json
{
  "engine_truncation_invariance_ls_classical": true,
  "deep_feature_truncation_invariance": true,
  "oracle_sharpe": 10.565780817438494
}
```
