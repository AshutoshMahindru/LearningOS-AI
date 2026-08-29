# WP-122: Mission and Stage Runtime State Machine

## 1. Mission Session State Machine

```
               ┌──────────┐
               │  NOT_    │
               │ STARTED  │
               └────┬─────┘
                    │ start_session()
                    ▼
               ┌──────────┐   pause()    ┌──────────┐
               │  ACTIVE  ├─────────────►│  PAUSED  │
               └────┬─────┴◄─────────────┴──────────┘
                    │          resume()
                    │ evaluate_gate() [PASSED]
                    ▼
               ┌──────────┐
               │COMPLETED │
               └──────────┘
```

## 2. Stage State Machine

```
               ┌──────────┐
               │  LOCKED  │
               └────┬─────┘
                    │ prerequisites_met()
                    ▼
               ┌──────────┐
               │  READY   │
               └────┬─────┘
                    │ enter_stage()
                    ▼
               ┌──────────┐    retry()
               │  ACTIVE  ├───────────────┐
               └────┬─────┴◄──────────────┘
                    │ submit_action()
                    ▼
               ┌──────────┐
               │SUBMITTED │
               └────┬─────┘
                    │ evaluate_rubric()
         ┌──────────┴──────────┐
         ▼                     ▼
   ┌──────────┐          ┌──────────┐
   │ COMPLETED│          │  REPAIR_ │
   │ (PASSED) │          │ REQUIRED │
   └──────────┘          └──────────┘
```

## 3. Invariants
- A stage cannot transition to `ACTIVE` until its immediate predecessor in the mission DAG is `COMPLETED`.
- A session cannot transition to `COMPLETED` until all required stages are `COMPLETED` and the `competency_gate` returns `status: "PASSED"`.
