# M19 — Make Models Learn with Gradients

This mission makes the parameter-to-loss connection visible before introducing derivative notation. The learner first changes one weight by hand, predicts what will happen, and observes the complete loss curve. Finite differences then turn “which way is downhill?” into a numerical estimate. Only after that observation does the notebook derive the analytic gradient and use it for one update, several updates, and finally a two-parameter model.

The controlled failure uses the wrong update sign. Its rising loss is evidence to trace through `parameter → prediction → loss → gradient → update`, not a warning to memorize.

The package is deterministic, CPU-only, offline, and uses a five-row synthetic fixture. Implementation status does not represent learner completion; learner evidence is intentionally empty.
