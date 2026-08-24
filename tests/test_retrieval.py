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

    def test_repository_lab_status_matches_implemented_wave(self):
        labs = LabRegistry(ROOT)
        for number in range(1, 40):
            mission = f"M{number:02d}"
            self.assertTrue(labs.status(mission)["repository_executable"], mission)

        for number in range(40, 43):
            mission = f"M{number:02d}"
            self.assertFalse(labs.status(mission)["repository_executable"], mission)

    def test_source_package_status_remains_historical_provenance(self):
        labs = LabRegistry(ROOT)
        source_executable = {"M01", "M02", "M03", "M08"}
        for number in range(1, 43):
            mission = f"M{number:02d}"
            status = labs.status(mission)
            self.assertEqual(status["source_artifact_available"], mission in source_executable, mission)
            self.assertEqual(status["specification_only_in_source_package"], mission not in source_executable, mission)


if __name__ == "__main__":
    unittest.main()
