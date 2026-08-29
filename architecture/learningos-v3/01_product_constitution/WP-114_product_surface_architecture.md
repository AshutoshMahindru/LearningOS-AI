# WP-114: Product Surface Architecture

## The 8 Canonical Product Surfaces
The LearningOS V3 user interface comprises exactly 8 coherent, top-level surfaces. No mission-specific pages exist outside these surfaces.

1. **Home / Dashboard (`/`)**:
   - Current mission status, active flagship version progress, daily "Today" recommended action, and retention review queue.
2. **Mission Player (`/missions/:id`)**:
   - The primary learning environment. Orchestrates the 11 canonical stage types dynamically based on the mission's declarative schema.
3. **Workbench & Lab Studio (`/workbench`)**:
   - Interactive code editor, dataset viewer, tensor inspector, trace debugger, and sandbox runner.
4. **Socratic Tutor Panel (`/tutor` or slide-over drawer)**:
   - Role-based assistance with runtime enforcement (Navigator, Socratic Tutor, Debugger, Feynman Reviewer).
5. **Evidence Ledger (`/evidence`)**:
   - Complete cryptographic provenance record of all submitted artifacts, experiment results, and gate certifications.
6. **Competency Graph (`/competencies`)**:
   - Interactive visualization of 253 knowledge nodes and 76 observable competencies (L0 to L5 levels).
7. **Reviews & Flagship Portfolio (`/reviews`)**:
   - Engineering reviews, Architecture Decision Records (ADRs), and Flagship System release management.
8. **Settings & Diagnostics (`/settings`)**:
   - Local runtime health, LLM provider API keys, database backups, and worker configuration.
