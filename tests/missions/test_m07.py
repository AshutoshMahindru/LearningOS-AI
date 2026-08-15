from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M07"
NOTEBOOK = ROOT / "labs" / "M07_reusable_pipeline.ipynb"
DATASET = ROOT / "datasets" / "M07" / "customer_renewals.csv"
REQUIREMENTS = ROOT / "requirements" / "m07.txt"

NUMERIC_FEATURES = ("tenure_months", "monthly_spend", "support_tickets")
CATEGORICAL_FEATURES = ("plan", "region", "signup_channel")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "renewed"

RUNTIME_IMPORTS = ("joblib", "numpy", "pandas", "sklearn")
MISSING_RUNTIME_IMPORTS = tuple(
    name for name in RUNTIME_IMPORTS if importlib.util.find_spec(name) is None
)
RUNTIME_DEPENDENCIES_AVAILABLE = not MISSING_RUNTIME_IMPORTS
RUNTIME_SKIP_REASON = (
    "M07 runtime dependencies are not installed: "
    + ", ".join(MISSING_RUNTIME_IMPORTS)
)


def dotted_name(node: ast.AST) -> str:
    """Return a dotted name for simple Name/Attribute AST nodes."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class M07StaticContractTests(unittest.TestCase):
    """Dependency-free checks that must run in the repository's bare CI."""

    def test_standard_package_and_required_review_artifacts_exist(self) -> None:
        mission_files = {
            "__init__.py",
            "pipeline.py",
            "manifest.yaml",
            "README.md",
            "content.yaml",
            "experiments.yaml",
            "code_reading.md",
            "no_ai_gate.md",
            "controlled_failure.md",
            "assessment.yaml",
            "evidence_contract.yaml",
            "flagship_integration.md",
            "status.yaml",
            "review_brief.md",
            "adr_prompt.md",
        }
        self.assertEqual(
            mission_files,
            {path.name for path in MISSION.iterdir() if path.is_file()},
        )
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((DATASET.parent / "README.md").is_file())
        self.assertTrue(REQUIREMENTS.is_file())

    def test_manifest_declares_runtime_review_and_completion_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        expected = [
            "id: M07",
            "phase: P1",
            "flagship: V01",
            "formal_engineering_review: true",
            "adr_required: true",
            "cpu_only: true",
            "paid_api_required: false",
            "network_required: false",
            "manual_vs_pipeline_comparison",
            "cross_validation_through_pipeline",
            "serialize_reload_equivalence",
            "controlled_failure_diagnosis",
        ]
        for value in expected:
            self.assertIn(value, text)

    def test_fixture_exercises_mixed_types_missingness_and_binary_target(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["customer_id"] for row in rows}), 60)
        self.assertEqual({row[TARGET_COLUMN] for row in rows}, {"0", "1"})
        self.assertEqual(
            {
                value: sum(row[TARGET_COLUMN] == value for row in rows)
                for value in ("0", "1")
            },
            {"0": 30, "1": 30},
        )
        self.assertTrue(
            any(row[column] == "" for row in rows for column in NUMERIC_FEATURES)
        )
        self.assertTrue(
            any(row[column] == "" for row in rows for column in CATEGORICAL_FEATURES)
        )

    def test_notebook_json_ids_outputs_syntax_and_offline_safety(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 28)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        forbidden_imports = {
            "httpx",
            "openai",
            "requests",
            "socket",
            "subprocess",
            "urllib",
        }
        forbidden_calls = {
            "eval",
            "exec",
            "os.system",
            "urlopen",
            "urllib.request.urlopen",
        }
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            compile(source, f"M07-cell-{index}", "exec")
            tree = ast.parse(source, filename=f"M07-cell-{index}")
            imported_roots: set[str] = set()
            called_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Call):
                    called_names.add(dotted_name(node.func))

            self.assertTrue(forbidden_imports.isdisjoint(imported_roots))
            self.assertTrue(forbidden_calls.isdisjoint(called_names))
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])

        all_code = "\n".join("".join(cell["source"]) for cell in code_cells)
        for forbidden_text in [
            "http://",
            "https://",
            "get_ipython(",
            "!pip",
            "%pip",
            "api_key",
            "os.environ",
        ]:
            self.assertNotIn(forbidden_text, all_code)

        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for phrase in [
            "Predict before running",
            "Manual vs pipeline",
            "fit vs transform",
            "Cross-validation through the pipeline",
            "Serialize and reload",
            "Controlled failure",
            "Formal engineering review",
            "ADR",
            "No-AI transfer gate",
        ]:
            self.assertIn(phrase, markdown)

    def test_controlled_failure_no_ai_and_assessment_contracts_are_substantive(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        assessment = (MISSION / "assessment.yaml").read_text(
            encoding="utf-8"
        ).lower()

        for term in [
            "independent encoding drift",
            "record a prediction",
            "exception type and message",
            "smallest root cause",
            "fitted pipeline",
            "do not",
        ]:
            self.assertIn(term, controlled)
        for term in [
            "without ai-generated code",
            "fresh local table",
            "unseen category",
            "serializes and reloads",
            "leakage argument",
        ]:
            self.assertIn(term, no_ai)
        for term in [
            "formal_engineering_review: required",
            "adr: required",
            "transfer_required: true",
            "cross_validate_complete_pipeline",
            "diagnose_independent_encoding_mismatch",
        ]:
            self.assertIn(term, assessment)

    def test_review_adr_and_evidence_contract_cover_consequential_boundaries(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").lower()
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")

        for term in [
            "architecture under review",
            "alternatives considered",
            "failure modes and controls",
            "residual uncertainty",
            "untrusted artifacts",
        ]:
            self.assertIn(term, review)
        for term in [
            "consequential decision",
            "required alternatives",
            "fit, transform, cv and inference lifecycle",
            "positive and negative consequences",
            "rollback triggers",
        ]:
            self.assertIn(term, adr)
        for term in [
            "required_evidence:",
            "controlled_failure_repair",
            "serialization_equivalence",
            "formal_review",
            "- id: adr",
            "no_ai_transfer",
            "prefilled_learner_evidence: prohibited",
        ]:
            self.assertIn(term, evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_(evidence|response)\s*:", evidence))

    def test_requirements_declare_all_runtime_and_validation_dependencies(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        for package in [
            "joblib",
            "jupyter",
            "nbformat",
            "numpy",
            "pandas",
            "pytest",
            "scikit-learn",
        ]:
            self.assertRegex(requirements, rf"(?m)^{re.escape(package)}[<=>]")


@unittest.skipUnless(RUNTIME_DEPENDENCIES_AVAILABLE, RUNTIME_SKIP_REASON)
class M07RuntimePipelineTests(unittest.TestCase):
    """Substantive runtime checks executed when M07 requirements are installed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.np = importlib.import_module("numpy")
        cls.pd = importlib.import_module("pandas")

        from sklearn.compose import ColumnTransformer
        from sklearn.exceptions import NotFittedError
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.utils.validation import check_is_fitted

        cls.ColumnTransformer = ColumnTransformer
        cls.NotFittedError = NotFittedError
        cls.LogisticRegression = LogisticRegression
        cls.StratifiedKFold = StratifiedKFold
        cls.cross_validate = staticmethod(cross_validate)
        cls.Pipeline = Pipeline
        cls.check_is_fitted = staticmethod(check_is_fitted)

        pipeline_spec = importlib.util.spec_from_file_location(
            "learningos_m07_pipeline", MISSION / "pipeline.py"
        )
        if pipeline_spec is None or pipeline_spec.loader is None:
            raise ImportError("could not load the M07 pipeline implementation")
        cls.pipeline_module = importlib.util.module_from_spec(pipeline_spec)
        pipeline_spec.loader.exec_module(cls.pipeline_module)

        cls.frame = cls.pipeline_module.load_dataset(DATASET)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = (
            cls.pipeline_module.train_test_frames(cls.frame)
        )

    def test_pipeline_has_column_transformer_and_separate_feature_treatments(self) -> None:
        pipeline = self.pipeline_module.build_pipeline()
        self.assertIsInstance(pipeline, self.Pipeline)
        self.assertEqual(list(pipeline.named_steps), ["preprocess", "model"])
        self.assertIsInstance(
            pipeline.named_steps["preprocess"], self.ColumnTransformer
        )
        self.assertIsInstance(
            pipeline.named_steps["model"], self.LogisticRegression
        )

        transformer_specs = {
            name: (transformer, tuple(columns))
            for name, transformer, columns in pipeline.named_steps[
                "preprocess"
            ].transformers
        }
        self.assertEqual(transformer_specs["numeric"][1], NUMERIC_FEATURES)
        self.assertEqual(transformer_specs["categorical"][1], CATEGORICAL_FEATURES)
        self.assertEqual(
            list(transformer_specs["numeric"][0].named_steps), ["impute", "scale"]
        )
        self.assertEqual(
            list(transformer_specs["categorical"][0].named_steps),
            ["impute", "encode"],
        )
        self.assertEqual(
            transformer_specs["categorical"][0]
            .named_steps["encode"]
            .handle_unknown,
            "ignore",
        )

    def test_feature_allow_list_excludes_identifier_target_and_post_outcome_data(self) -> None:
        augmented = self.frame.copy()
        augmented["post_outcome_contact"] = augmented[TARGET_COLUMN]
        features, target = self.pipeline_module.split_features_target(augmented)

        self.assertEqual(tuple(features.columns), MODEL_FEATURES)
        self.assertNotIn("customer_id", features.columns)
        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertNotIn("post_outcome_contact", features.columns)
        self.pd.testing.assert_series_equal(
            target.reset_index(drop=True),
            augmented[TARGET_COLUMN].astype("int64").reset_index(drop=True),
        )

    def test_fit_learns_state_while_transform_and_predict_reuse_it(self) -> None:
        pipeline = self.pipeline_module.build_pipeline()
        with self.assertRaises(self.NotFittedError):
            self.pipeline_module.transform_features(pipeline, self.X_test)

        pipeline.fit(self.X_train, self.y_train)
        preprocess = pipeline.named_steps["preprocess"]
        numeric = preprocess.named_transformers_["numeric"]
        categorical = preprocess.named_transformers_["categorical"]
        before = {
            "imputer": numeric.named_steps["impute"].statistics_.copy(),
            "scale": numeric.named_steps["scale"].mean_.copy(),
            "categories": tuple(
                values.copy()
                for values in categorical.named_steps["encode"].categories_
            ),
            "coefficients": pipeline.named_steps["model"].coef_.copy(),
        }

        first_transform = self.pipeline_module.transform_features(
            pipeline, self.X_test
        )
        second_transform = self.pipeline_module.transform_features(
            pipeline, self.X_test
        )
        first_predictions = pipeline.predict(self.X_test)
        second_predictions = pipeline.predict(self.X_test)

        self.np.testing.assert_allclose(first_transform, second_transform)
        self.np.testing.assert_array_equal(first_predictions, second_predictions)
        self.np.testing.assert_allclose(
            before["imputer"], numeric.named_steps["impute"].statistics_
        )
        self.np.testing.assert_allclose(
            before["scale"], numeric.named_steps["scale"].mean_
        )
        for expected, observed in zip(
            before["categories"], categorical.named_steps["encode"].categories_
        ):
            self.np.testing.assert_array_equal(expected, observed)
        self.np.testing.assert_allclose(
            before["coefficients"], pipeline.named_steps["model"].coef_
        )

    def test_raw_inference_with_unseen_categories_preserves_transformation_width(self) -> None:
        pipeline = self.pipeline_module.build_pipeline().fit(
            self.X_train, self.y_train
        )
        inference = self.X_test.iloc[[0]].copy()
        inference.loc[:, "plan"] = "enterprise"
        inference.loc[:, "region"] = "central"
        inference.loc[:, "signup_channel"] = "partner"

        training_width = self.pipeline_module.transform_features(
            pipeline, self.X_train.iloc[[0]]
        ).shape[1]
        inference_matrix = self.pipeline_module.transform_features(
            pipeline, inference
        )
        predictions = pipeline.predict(inference)
        probabilities = pipeline.predict_proba(inference)

        self.assertEqual(inference_matrix.shape, (1, training_width))
        self.assertEqual(predictions.shape, (1,))
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_cross_validation_clones_and_fits_the_complete_pipeline_per_fold(self) -> None:
        pipeline = self.pipeline_module.build_pipeline()
        cv = self.StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        results = self.cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="accuracy",
            return_estimator=True,
        )

        self.assertEqual(results["test_score"].shape, (5,))
        self.assertTrue(self.np.isfinite(results["test_score"]).all())
        self.assertTrue(
            ((results["test_score"] >= 0) & (results["test_score"] <= 1)).all()
        )
        with self.assertRaises(self.NotFittedError):
            self.check_is_fitted(pipeline.named_steps["preprocess"])
        for fitted_fold in results["estimator"]:
            self.check_is_fitted(fitted_fold.named_steps["preprocess"])
            self.assertGreater(
                fitted_fold.named_steps["preprocess"].get_feature_names_out().size,
                len(MODEL_FEATURES),
            )

    def test_serialized_pipeline_reloads_with_identical_inference_behavior(self) -> None:
        fitted = self.pipeline_module.build_pipeline().fit(
            self.X_train, self.y_train
        )
        expected_transform = self.pipeline_module.transform_features(
            fitted, self.X_test
        )
        expected_prediction = fitted.predict(self.X_test)
        expected_probability = fitted.predict_proba(self.X_test)

        with tempfile.TemporaryDirectory(prefix="m07-test-") as directory:
            path = self.pipeline_module.save_pipeline(
                fitted, Path(directory) / "pipeline.joblib"
            )
            reloaded = self.pipeline_module.load_pipeline(path)

        self.np.testing.assert_allclose(
            expected_transform,
            self.pipeline_module.transform_features(reloaded, self.X_test),
        )
        self.np.testing.assert_array_equal(
            expected_prediction, reloaded.predict(self.X_test)
        )
        self.np.testing.assert_allclose(
            expected_probability, reloaded.predict_proba(self.X_test)
        )

    def test_controlled_manual_encoding_failure_is_real_and_pipeline_repairs_it(self) -> None:
        numeric_fill = self.X_train[list(NUMERIC_FEATURES)].median()
        category_fill = self.X_train[list(CATEGORICAL_FEATURES)].mode().iloc[0]

        def manual_matrix(frame):
            prepared = frame.copy()
            prepared.loc[:, list(NUMERIC_FEATURES)] = prepared[
                list(NUMERIC_FEATURES)
            ].fillna(numeric_fill)
            prepared.loc[:, list(CATEGORICAL_FEATURES)] = prepared[
                list(CATEGORICAL_FEATURES)
            ].fillna(category_fill)
            return self.pd.get_dummies(
                prepared,
                columns=list(CATEGORICAL_FEATURES),
                dtype=float,
            )

        manual_train = manual_matrix(self.X_train)
        manual_model = self.LogisticRegression(max_iter=1_000).fit(
            manual_train, self.y_train
        )
        unseen = self.X_test.iloc[[0]].copy()
        unseen.loc[:, "plan"] = "enterprise"
        unseen.loc[:, "region"] = "central"
        unseen.loc[:, "signup_channel"] = "partner"
        mismatched_inference = manual_matrix(unseen)

        self.assertNotEqual(
            tuple(manual_train.columns), tuple(mismatched_inference.columns)
        )
        with self.assertRaises(ValueError):
            manual_model.predict(mismatched_inference)

        safe_pipeline = self.pipeline_module.build_pipeline().fit(
            self.X_train, self.y_train
        )
        self.assertEqual(safe_pipeline.predict(unseen).shape, (1,))

    def test_feature_treatment_configuration_is_explicit_and_validated(self) -> None:
        standard = self.pipeline_module.build_pipeline(scaler="standard")
        robust = self.pipeline_module.build_pipeline(scaler="robust")
        passthrough = self.pipeline_module.build_pipeline(scaler="passthrough")
        self.assertEqual(
            list(
                standard.named_steps["preprocess"]
                .transformers[0][1]
                .named_steps
            ),
            ["impute", "scale"],
        )
        self.assertEqual(
            type(
                robust.named_steps["preprocess"]
                .transformers[0][1]
                .named_steps["scale"]
            ).__name__,
            "RobustScaler",
        )
        self.assertEqual(
            list(
                passthrough.named_steps["preprocess"]
                .transformers[0][1]
                .named_steps
            ),
            ["impute"],
        )
        with self.assertRaisesRegex(ValueError, "numeric_strategy"):
            self.pipeline_module.build_pipeline(
                numeric_strategy="constant"  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "scaler"):
            self.pipeline_module.build_pipeline(
                scaler="normalize"  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
