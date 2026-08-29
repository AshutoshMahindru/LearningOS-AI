# WP-112: User Roles and Usage Model

## 1. Primary Roles

### A. The Learner / Apprentice (Primary Actor)
- Navigates the 42-mission spine and flagship system versions.
- Conducts whole-system tracing, code-reading, hypothesis-driven experimentation, failure diagnosis, and no-AI transfer challenges.
- Maintains a verifiable portfolio of architectural decisions (ADRs) and proven competencies.

### B. The Curriculum Author / Content Architect
- Authors declarative mission packages (YAML/JSON), datasets, test fixtures, rubrics, and failure injection scenarios.
- Utilizes the Mission Definition Language (MDL v1) without writing custom frontend or server endpoint code.

### C. The Reviewer / Principal Mentor (Human or Orchestrated Agent)
- Evaluates submitted architectural decisions, code reviews, and Feynman explanations against structured rubrics.
- Approves or prescribes targeted repair tasks.

### D. The Platform Administrator / Integrator
- Manages local runtime installations, dependency environments, database backups, and provider API keys (e.g. OpenAI, Anthropic, Gemini, local Ollama).

## 2. Usage Modes
1. **Focused Mission Execution**: Step-by-step progress through a mission's stage pipeline with real-time feedback.
2. **Flagship Build Mode**: Iterative construction of the 12-version Operations Intelligence System across multiple missions.
3. **Targeted Repair Mode**: Precise, isolated remediation of a specific weak knowledge node without repeating previous stages.
4. **Spaced Retention Review**: Daily 10-minute micro-drills to prevent competency decay.
