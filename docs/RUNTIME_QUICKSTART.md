# Runtime Quickstart

The runtime is deliberately local-first and standard-library-first.

## Install

```bash
python -m pip install -e .
```

## Start a mission

```bash
learning-os start M01
```

## Record evidence

```bash
learning-os evidence M01 \
  --type artifact \
  --summary "My AI/ML system map and explanation" \
  --competency "system mapping" \
  --no-ai --transfer --explanation
```

## Run the gate

```bash
learning-os gate M01
```

A PASS requires at least one deliverable plus explanation, unseen transfer, and no-AI evidence. PARTIAL triggers targeted repair. FAIL keeps the mission active.

## Ask the runtime what to do next

```bash
learning-os next
```

The current engine returns ADVANCE, CONTINUE, ZOOM_IN, or COMPLETE. Later tutor integration will enrich these decisions without changing the evidence contract.
