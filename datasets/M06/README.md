# M06 datasets

Both CSV files are deterministic synthetic teaching fixtures. They contain no real people, organizations, support records, or library programs. They are committed locally so M06 has no runtime network dependency.

## `support_tickets.csv`

Guided question: which observable signals are associated with escalation in this sample, and what must be resolved before modeling?

Row grain: one closed support ticket.

| Field | Meaning | Timing / quality note |
|---|---|---|
| `ticket_id` | synthetic unique identifier | available at opening; identity, not a model signal |
| `opened_week` | synthetic week number 1–8 | available at opening |
| `channel` | chat, email, phone, or web | available at opening |
| `region` | synthetic service region | available at opening |
| `issue_type` | broad issue category | available after initial categorization |
| `customer_tenure_months` | months since account opening | available at opening; a few values are missing |
| `first_response_minutes` | minutes to first staff response | known only after first response; contains legitimate extreme-delay candidates |
| `messages_count` | total messages over the closed case | post-case measure; unavailable when a ticket first opens |
| `satisfaction_score` | optional 1–5 post-case survey | post-case and missing for nonrespondents |
| `escalated` | closed-case escalation outcome | binary target for the interrogation exercise |
| `post_case_priority` | administrative label assigned after closure | deliberately mirrors the outcome and is a leakage trap |

Designed properties:

- 48 records and a minority escalation class;
- missing tenure and satisfaction values;
- a right-tailed response-time distribution with IQR outlier candidates;
- group and record-level relationships worth questioning;
- `post_case_priority` is unavailable at decision time and deliberately has perfect target association.

These properties support pedagogy, not external validity. They must not be reported as findings about real support operations.

## `community_programs_fresh.csv`

This fixture is reserved for `missions/M06/no_ai_gate.md`.

Row grain: one synthetic community program. It includes day type, delivery format, topic, registration, attendance, and promotion channel. The weekday/weekend groups differ in composition, so an aggregate rate alone cannot justify a schedule change.

The README intentionally does not provide the result or recommend a chart. The learner must choose the chart, compute correct denominators, interpret the visible evidence, and state limitations independently.
