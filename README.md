# LearningOS V3

LearningOS V3 is a schema-driven, local-first pedagogy engine for software engineering. Built on the core principle of **"Transfer without AI"**, this system aims to force the learner into the driver's seat by stripping away instant answers and replacing them with deep, Socratic interrogation and rigorous code execution bounds.

## Core Philosophy

> "Start with the useful whole. Map it. Interrogate it. Descend only to the narrowest blocker. Decompose it. Rebuild it. Break it. Explain it. Transfer without AI. Prove competence. Return to the system."

**Key Principles:**
- **Zero Mission-Specific Logic:** The platform has no hardcoded knowledge of the "missions" it runs. Everything is driven by a unified `MissionSchema`.
- **Local-First Verification:** No internet is required to run the primary loop. Cryptographic hashes of predictions and local test execution seal the learner's evidence ledger on their local machine.
- **Socratic Friction:** The integrated AI Tutor daemon is strictly forbidden from providing code answers. It uses generative AI only to increase the learner's cognitive load by forcing them to articulate their hypotheses.

## Architecture

LearningOS V3 consists of a three-tier architecture:

1. **Frontend (`platform/frontend`)**
   - React 18 + Vite SPA built with Tailwind CSS.
   - Features a premium "Deep Space" glassmorphic UI.
   - Responsible for rendering dynamic stages (`CodeReadingStage`, `InterrogateStage`, `CompetencyGateStage`) mapped from the backend schema.
   
2. **Backend Server (`platform/backend`)**
   - FastAPI server using `uvicorn`.
   - Manages the local SQLite database (`learningos.db`) containing the `MissionSchema`, `MissionSessions`, and `EvidenceLedger`.
   - Hosts the `TutorChat` route which proxies requests to the OpenAI API (or falls back to a local heuristic pattern-matcher).

3. **Execution Worker Daemon (`platform/backend/worker_daemon.py`)**
   - A standalone Python script running in the background.
   - Listens on a Unix Domain Socket (`/tmp/learningos_worker.sock`).
   - Safely isolates arbitrary code execution from the main API process.

## Setup & Running

This project uses `uv` for Python dependency management and `npm` for Node.js.

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) installed on your system.
- Node.js (v18+) and npm installed.

### 2. Configuration
If you want the Socratic Tutor to use generative AI, export your OpenAI API key in your terminal:
```bash
export OPENAI_API_KEY="sk-..."
```
*(If you do not provide this, the system will gracefully fall back to a local heuristic pattern-matching engine.)*

### 3. Launching the System
We provide a unified startup script that uses background job control to launch the Frontend, Backend, and Worker concurrently.

Run the following command from the repository root:
```bash
./start.sh
```

**Services Started:**
- **Frontend UI:** `http://localhost:5173`
- **Backend API:** `http://127.0.0.1:8000`
- **Execution Socket:** `/tmp/learningos_worker.sock`

To shut down all services cleanly, press `Ctrl+C`.

---
*Built incrementally over multiple pair-programming sessions.*
