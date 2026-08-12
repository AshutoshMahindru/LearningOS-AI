import json
import unittest
from pathlib import Path

from learning_os.content_router import ContentRouter
from learning_os.lab_registry import LabRegistry
from learning_os.mission_loader import MissionRepository
from learning_os.prerequisite_graph import PrerequisiteGraph
from learning_os.source_registry import SourceRegistry


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def test_source_references_resolve(self):
        registry = SourceRegistry(ROOT)
        for content in registry.content():
            resolved = registry.sources(content.get("source_ids", []))
            self.assertEqual(len(resolved), len(content.get("source_ids", [])))

    def test_semantic_search_routes_embeddings_for_m33(self):
        mission = MissionRepository(ROOT).get("M33")
        results = ContentRouter(ROOT).route(mission, blocker="embeddings semantic similarity", top_k=3)
        self.assertTrue(results)
        self.assertTrue(any("embeddings" in item.get("topics", []) or "semantic-search" in item.get("topics", []) for item in results))

    def test_prerequisites_are_blocking_and_helpful(self):
        graph = PrerequisiteGraph(ROOT)
        self.assertEqual(set(graph.blocking("M34")), {"M31", "M33"})
        self.assertIn("M15", graph.helpful("M13"))

    def test_remote_lab_status_is_not_overclaimed(self):
        labs = LabRegistry(ROOT)
        for mission in ["M01", "M02", "M03", "M08"]:
            status = labs.status(mission)
            self.assertFalse(status["repository_executable"])
            self.assertTrue(status["source_artifact_available"])


if __name__ == "__main__":
    unittest.main()
