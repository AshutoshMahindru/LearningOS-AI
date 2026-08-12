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
        self.assertEqual(self.graph.get("K252")["concept"], "architecture diagram")
        self.assertEqual(self.graph.get("KTXT")["concept"], "strings and text")

    def test_concept_resolution_and_search(self):
        node = self.graph.get("cosine similarity")
        self.assertEqual(node["concept"], "cosine similarity")
        embedding = self.graph.get("embedding")
        self.assertEqual(embedding["id"], "K178")
        results = self.graph.search("embedding similarity")
        self.assertTrue(results)

    def test_relationships_resolve(self):
        prerequisites = self.graph.prerequisites("embedding")
        self.assertTrue(prerequisites)
        self.assertEqual(prerequisites[0]["concept"], "vector")
        enables = self.graph.enables("attention")
        self.assertTrue(enables)
        self.assertIn("scaled dot-product attention", [item["concept"] for item in enables])

    def test_mission_mapping(self):
        nodes = self.graph.nodes_for_mission("M33")
        self.assertTrue(nodes)


if __name__ == "__main__":
    unittest.main()
