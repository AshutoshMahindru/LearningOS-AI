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
- A bootstrap concept graph is present to exercise graph contracts.
- The canonical 253-node knowledge graph is **not yet migrated**.
- The source/content registry is a curated starter set, not the full canonical content map.
- Local retrieval uses bag-of-words vectors with cosine similarity. It is a graceful-degradation fallback, not a replacement for embeddings.
- No notebook is currently present on this remote branch. M01, M02, M03 and M08 are recorded only as executable source artifacts awaiting migration.

## Contract for external retrieval

A richer vector backend only needs to implement `RetrievalBackend.search(query, items, top_k)` and return `SearchHit` records. This keeps Qdrant or another vector store optional rather than coupling the learning runtime to infrastructure.

## Routing invariant

Retrieve the minimum content needed to continue the mission. A blocker opens a bounded zoom-in path; once the blocker is evidenced as resolved, return to the exact mission step.
