import unittest
from pathlib import Path

from learning_os.knowledge_graph import KnowledgeGraph


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(ROOT)

    def test_canonical_node_count(self):
        self.assertEqual(len(self.graph), 253)

    def test_canonical_nodes_exist(self):
        self.assertEqual(self.graph.get("K001")["concept"], "program execution")
        self.assertEqual(self.graph.get("K252")["concept"], "AI architecture")
        self.assertEqual(self.graph.get("KTXT")["concept"], "strings and text")

    def test_concept_resolution_and_search(self):
        node = self.graph.get("cosine similarity")
        self.assertEqual(node["concept"], "cosine similarity")
        results = self.graph.search("embedding similarity")
        self.assertTrue(results)

    def test_relationships_resolve(self):
        prerequisites = self.graph.prerequisites("embeddings")
        self.assertTrue(prerequisites)
        enables = self.graph.enables("attention")
        self.assertTrue(isinstance(enables, list))

    def test_mission_mapping(self):
        nodes = self.graph.nodes_for_mission("M33")
        self.assertTrue(nodes)


if __name__ == "__main__":
    unittest.main()
