# No-AI gate

Complete this transfer task without AI-generated code.

You receive a fresh ten-row inventory CSV with these columns:
`sku`, `item_name`, `warehouse`, `quantity`, `unit_cost`, `received_date`.
It contains blanks, whitespace, one exact duplicate, one reused SKU with
conflicting quantities, currency punctuation, a malformed quantity, mixed date
formats and one unusually large but documented shipment.

Tasks:

1. Record predicted defects before cleaning.
2. Declare a data contract and accepted date formats.
3. Load every field losslessly as raw text.
4. Separate exact duplicates from SKU conflicts.
5. Preserve raw values and source-row provenance.
6. Normalize, parse and attach explicit issue codes.
7. Make and justify each missingness decision.
8. Flag but do not automatically delete the large shipment.
9. Produce an analysis-ready table plus a quarantine/review view.
10. Reconcile raw, removed-exact-duplicate, ready and review counts.
11. Write and run at least six substantive invariant assertions.
12. Explain one unresolved uncertainty and who should resolve it.

Passing requires runnable code, row-accounting evidence and a plain-language
explanation of why no conflicting or malformed record was silently lost.
