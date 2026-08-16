from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M21"
NOTEBOOK = ROOT / "labs" / "M21_train_neural_network_black_box.ipynb"


def load_core():
    spec = importlib.util.spec_from_file_location("m21_training_core", MISSION / "training_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M21 training core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()


class M21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = CORE.train_black_box()
        cls.seed_change = CORE.train_black_box(model_seed=2102)
        cls.under = CORE.train_black_box(max_iter=1)
        cls.tiny = CORE.train_black_box(hidden_units=4)
        cls.shuffle = CORE.train_black_box(shuffle_labels=True, label_seed=2121)

    def test_reference_beats_baseline(self):
        self.assertGreater(self.reference.test_accuracy, 0.90)
        self.assertGreater(self.reference.macro_f1, 0.90)
        self.assertLess(self.reference.majority_baseline_accuracy, 0.11)

    def test_seed_behaviour(self):
        self.assertEqual(self.reference, CORE.train_black_box())
        self.assertNotEqual(self.reference, self.seed_change)
        self.assertGreater(self.seed_change.test_accuracy, 0.90)

    def test_failures_are_observable(self):
        self.assertLess(self.under.test_accuracy, 0.60)
        self.assertGreater(self.reference.test_accuracy - self.under.test_accuracy, 0.40)
        self.assertLess(self.shuffle.test_accuracy, 0.20)
        self.assertLess(abs(self.shuffle.test_accuracy - self.shuffle.majority_baseline_accuracy), 0.06)
        self.assertLess(self.tiny.test_accuracy, self.reference.test_accuracy - 0.10)

    def test_confusion_matrix(self):
        self.assertEqual(sum(map(sum, self.reference.confusion_matrix)), self.reference.test_size)
        self.assertNotEqual(CORE.most_confused_pair(self.reference)[0], CORE.most_confused_pair(self.reference)[1])

    def test_notebook_clean(self):
        nb = json.loads(NOTEBOOK.read_text())
        self.assertEqual(len(nb["cells"]), 36)
        ids = [cell["id"] for cell in nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
