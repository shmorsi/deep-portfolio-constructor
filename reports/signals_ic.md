# Stage 2 - Signal information coefficients

IC = monthly Spearman rank correlation between the signal at month-end t and the
stock return over month t+1. t-statistics use Newey-West errors with 6 lags. Q5-Q1 = equal-weight top-quintile minus bottom-quintile return.

## Full sample

| signal   |   mean_IC |   IC_t_stat_NW |   IC_IR_ann |   IC_hit_rate |   Q5_minus_Q1_monthly |   Q5_Q1_t_stat_NW |   n_months |
|:---------|----------:|---------------:|------------:|--------------:|----------------------:|------------------:|-----------:|
| momentum |    0.0192 |         1.4900 |      0.2746 |        0.5404 |                0.0029 |            0.9488 |   272.0000 |
| reversal |    0.0056 |         0.4799 |      0.1014 |        0.5110 |               -0.0001 |           -0.0302 |   272.0000 |
| low_vol  |   -0.0408 |        -2.7607 |     -0.5386 |        0.4338 |               -0.0115 |           -3.4032 |   272.0000 |
| combined |   -0.0030 |        -0.2465 |     -0.0487 |        0.5147 |               -0.0027 |           -0.9834 |   272.0000 |

## Out-of-sample window only (from 2008-01-01)

| signal   |   mean_IC |   IC_t_stat_NW |   IC_IR_ann |   IC_hit_rate |   Q5_minus_Q1_monthly |   Q5_Q1_t_stat_NW |   n_months |
|:---------|----------:|---------------:|------------:|--------------:|----------------------:|------------------:|-----------:|
| momentum |    0.0161 |         1.1512 |      0.2287 |        0.5402 |                0.0024 |            0.6717 |   224.0000 |
| reversal |    0.0015 |         0.1157 |      0.0268 |        0.5045 |               -0.0007 |           -0.2797 |   224.0000 |
| low_vol  |   -0.0355 |        -2.0823 |     -0.4591 |        0.4464 |               -0.0112 |           -2.8238 |   224.0000 |
| combined |   -0.0039 |        -0.2800 |     -0.0599 |        0.5223 |               -0.0032 |           -0.9934 |   224.0000 |

## Correlation between signals (pooled z-scores)

|          |   momentum |   reversal |   low_vol |   combined |
|:---------|-----------:|-----------:|----------:|-----------:|
| momentum |       1.00 |      -0.03 |     -0.09 |       0.50 |
| reversal |      -0.03 |       1.00 |      0.05 |       0.60 |
| low_vol  |      -0.09 |       0.05 |      1.00 |       0.55 |
| combined |       0.50 |       0.60 |      0.55 |       1.00 |
