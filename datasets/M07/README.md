# M07 customer renewal fixture

`customer_renewals.csv` is a deterministic synthetic teaching fixture created
for M07. It contains 60 fictional customer rows, a unique identifier, three
pre-outcome numerical features, three pre-outcome categorical features and a
binary renewal target.

Blank values are intentional. They exercise numerical and categorical
imputation. The rows do not describe real people or a production population,
and model accuracy on this fixture must not be presented as business evidence.

Feature availability contract:

| Column | Role | Available before outcome? |
| --- | --- | --- |
| `customer_id` | identifier, excluded | yes |
| `tenure_months` | numerical feature | yes |
| `monthly_spend` | numerical feature | yes |
| `support_tickets` | numerical feature | yes |
| `plan` | categorical feature | yes |
| `region` | categorical feature | yes |
| `signup_channel` | categorical feature | yes |
| `renewed` | target, excluded | no |

The lab creates inference-only categories in memory; it does not rewrite this
fixture or require network access.
