# WP-118: Out-of-Scope Register & Product Boundaries

## Explicit Out-of-Scope Boundaries
To protect LearningOS V3 from feature creep, cloud sprawl, and architectural bloat, the following items are formally declared **OUT OF SCOPE**:

1. **General-Purpose Learning Management System (LMS)**:
   - No multi-tenant enterprise grading systems, SCORM packaging, bulk gradebook sync, or classroom seat management.
2. **Cloud-Hosted Multi-User Web Application**:
   - V3 is strictly a local-first single-user application running on the builder's workstation. No centralized cloud user authentication, multi-tenant DBs, or hosted cloud GPU clusters.
3. **Full Cloud IDE Replacement**:
   - LearningOS provides a focused workbench and lab runner, not a clone of VS Code or JetBrains (learners may use their preferred external IDEs alongside the local runtime).
4. **Automated AI Video / Voice Streaming**:
   - No real-time avatar video synthesis or automated voice agents in V3 baseline.
5. **Pre-Populated Learner Progress**:
   - Learner progress must always begin fresh. No artificial seeding of completed missions or pre-approved gate certificates.
