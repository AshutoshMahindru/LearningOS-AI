# Retrieval Architecture

The retrieval layer turns mission state and blockers into bounded learning context.

## Flow

```text
Mission + learner state + gate evidence
        |
        v
PrerequisiteGraph ----> unmet blocking mission / zoom target
        |
        v
ContentRouter
        |
        +--> SourceRegistry
        |
        +--> RetrievalBackend
                 |
                 +--> LocalVectorRetriever (dependency-free fallback)
                 +--> future external vector backend
        |
        v
MissionContextAssembler
        |
        v
Tutor / CLI / future dashboard
```

## Current scope

- The runtime mission graph covers M01-M42.
- The canonical 253-node knowledge graph is migrated and validated by `tools/validate_repo.py`.
- The source/content registry is a curated canonical routing layer; mission-local content files extend it without requiring every mission detail to live in one global registry.
- Local retrieval uses bag-of-words vectors with cosine similarity. It is a graceful-degradation fallback, not a replacement for embeddings.
- M01-M22 now have repository-executable notebooks and validated mission packages. M23-M42 remain specification-only until implemented and validated.

## Contract for external retrieval

A richer vector backend only needs to implement `RetrievalBackend.search(query, items, top_k)` and return `SearchHit` records. This keeps Qdrant or another vector store optional rather than coupling the learning runtime to infrastructure.

## Routing invariant

Retrieve the minimum content needed to continue the mission. A blocker opens a bounded zoom-in path; once the blocker is evidenced as resolved, return to the exact mission step.
