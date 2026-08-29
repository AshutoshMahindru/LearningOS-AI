# WP-143: Desktop Packaging & Distribution Strategy

## 1. Web-First, Desktop-Ready Architecture
LearningOS V3 is designed as a standalone web platform (FastAPI + React SPA) with a roadmap towards single-binary / desktop distribution (Tauri / Electron) in WP-800.

## 2. Desktop Packaging Roadmap
- **Phase A (V3 Core Runtime)**: Launched locally via lightweight Python CLI launcher (`learningos start`), opening the default browser to `http://127.0.0.1:8765`.
- **Phase B (Desktop Native Wrapper)**: Packaged with Tauri, embedding the frontend assets and orchestrating the Python runtime and SQLite store without requiring developer terminal usage.
