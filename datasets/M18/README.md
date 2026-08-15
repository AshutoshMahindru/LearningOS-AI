# M18 datasets

`checkout_experiment_daily.csv` is a deterministic teaching fixture for a
fictional checkout experiment. It contains grouped counts, not real people or
production telemetry. Variant A has 72 conversions from 600 sessions (12%);
variant B has 84 from 600 (14%). The observed +2 percentage-point difference is
deliberately compatible with both a useful effect and ordinary sampling noise.

`seasonal_correlation.csv` is a synthetic monthly fixture in which temperature
is an explicit common cause of ice-cream sales and swimming exposure. Its strong
sales/drownings correlation is not evidence that either variable causes the
other.

Both files are committed so the lab has no runtime network dependency. The
fixtures are instructional simulations and must not be represented as empirical
business evidence.
