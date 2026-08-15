from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MISSION = ROOT / "missions" / "M07"
NOTEBOOK = ROOT / "labs" / "M07_reusable_pipeline.ipynb"
DATASET = ROOT / "datasets" / "M07" / "customer_renewals.csv"

PIPELINE_SPEC = importlib.util.spec_from_file_location(
    "learningos_m07_pipeline", MISSION / "pipeline.py"
)
if PIPELINE_SPEC is None or PIPELINE_SPEC.loader is None:
    raise ImportError("could not load the M07 pipeline implementation")
PIPELINE_MODULE = importlib.util.module_from_spec(PIPELINE_SPEC)
PIPELINE_SPEC.loader.exec_module(PIPELINE_MODULE)

CATEGORICAL_FEATURES = PIPELINE_MODULE.CATEGORICAL_FEATURES
MODEL_FEATURES = PIPELINE_MODULE.MODEL_FEATURES
NUMERIC_FEATURES = PIPELINE_MODULE.NUMERIC_FEATURES
TARGET_COLUMN = PIPELINE_MODULE.TARGET_COLUMN
build_pipeline = PIPELINE_MODULE.build_pipeline
load_dataset = PIPELINE_MODULE.load_dataset
load_pipeline = PIPELINE_MODULE.load_pipeline
save_pipeline = PIPELINE_MODULE.save_pipeline
split_features_target = PIPELINE_MODULE.split_features_target
train_test_frames = PIPELINE_MODULE.train_test_frames
transform_features = PIPELINE_MODULE.transform_features


class M07MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset(DATASET)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_frames(
            cls.frame
        )

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
        self.assertTrue((ROOT / "requirements" / "m07.txt").is_file())

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
        self.assertEqual(len(self.frame), 60)
        self.assertTrue(self.frame["customer_id"].is_unique)
        self.assertEqual(set(self.frame[TARGET_COLUMN]), {0, 1})
        self.assertEqual(self.frame[TARGET_COLUMN].value_counts().to_dict(), {0: 30, 1: 30})
        self.assertGreater(self.frame[list(NUMERIC_FEATURES)].isna().sum().sum(), 0)
        self.assertGreater(
            self.frame[list(CATEGORICAL_FEATURES)].isna().sum().sum(), 0
        )

    def test_pipeline_has_column_transformer_and_separate_feature_treatments(self) -> None:
        pipeline = build_pipeline()
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(list(pipeline.named_steps), ["preprocess", "model"])
        self.assertIsInstance(pipeline.named_steps["preprocess"], ColumnTransformer)
        self.assertIsInstance(pipeline.named_steps["model"], LogisticRegression)

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
        features, target = split_features_target(augmented)

        self.assertEqual(tuple(features.columns), MODEL_FEATURES)
        self.assertNotIn("customer_id", features.columns)
        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertNotIn("post_outcome_contact", features.columns)
        pd.testing.assert_series_equal(
            target.reset_index(drop=True),
            augmented[TARGET_COLUMN].astype("int64").reset_index(drop=True),
        )

    def test_fit_learns_state_while_transform_and_predict_reuse_it(self) -> None:
        pipeline = build_pipeline()
        with self.assertRaises(NotFittedError):
            transform_features(pipeline, self.X_test)

        pipeline.fit(self.X_train, self.y_train)
        preprocess = pipeline.named_steps["preprocess"]
        numeric = preprocess.named_transformers_["numeric"]
        categorical = preprocess.named_transformers_["categorical"]
        before = {
            "imputer": numeric.named_steps["impute"].statistics_.copy(),
            "scale": numeric.named_steps["scale"].mean_.copy(),
            "categories": tuple(
                values.copy() for values in categorical.named_steps["encode"].categories_
            ),
            "coefficients": pipeline.named_steps["model"].coef_.copy(),
        }

        first_transform = transform_features(pipeline, self.X_test)
        second_transform = transform_features(pipeline, self.X_test)
        first_predictions = pipeline.predict(self.X_test)
        second_predictions = pipeline.predict(self.X_test)

        np.testing.assert_allclose(first_transform, second_transform)
        np.testing.assert_array_equal(first_predictions, second_predictions)
        np.testing.assert_allclose(
            before["imputer"], numeric.named_steps["impute"].statistics_
        )
        np.testing.assert_allclose(before["scale"], numeric.named_steps["scale"].mean_)
        for expected, observed in zip(
            before["categories"], categorical.named_steps["encode"].categories_
        ):
            np.testing.assert_array_equal(expected, observed)
        np.testing.assert_allclose(
            before["coefficients"], pipeline.named_steps["model"].coef_
        )

    def test_raw_inference_with_unseen_categories_preserves_transformation_width(self) -> None:
        pipeline = build_pipeline().fit(self.X_train, self.y_train)
        inference = self.X_test.iloc[[0]].copy()
        inference.loc[:, "plan"] = "enterprise"
        inference.loc[:, "region"] = "central"
        inference.loc[:, "signup_channel"] = "partner"

        training_width = transform_features(pipeline, self.X_train.iloc[[0]]).shape[1]
        inference_matrix = transform_features(pipeline, inference)
        predictions = pipeline.predict(inference)
        probabilities = pipeline.predict_proba(inference)

        self.assertEqual(inference_matrix.shape, (1, training_width))
        self.assertEqual(predictions.shape, (1,))
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_cross_validation_clones_and_fits_the_complete_pipeline_per_fold(self) -> None:
        pipeline = build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="accuracy",
            return_estimator=True,
        )

        self.assertEqual(results["test_score"].shape, (5,))
        self.assertTrue(np.isfinite(results["test_score"]).all())
        self.assertTrue(((results["test_score"] >= 0) & (results["test_score"] <= 1)).all())
        with self.assertRaises(NotFittedError):
            check_is_fitted(pipeline.named_steps["preprocess"])
        for fitted_fold in results["estimator"]:
            check_is_fitted(fitted_fold.named_steps["preprocess"])
            self.assertGreater(
                fitted_fold.named_steps["preprocess"].get_feature_names_out().size,
                len(MODEL_FEATURES),
            )

    def test_serialized_pipeline_reloads_with_identical_inference_behavior(self) -> None:
        fitted = build_pipeline().fit(self.X_train, self.y_train)
        expected_transform = transform_features(fitted, self.X_test)
        expected_prediction = fitted.predict(self.X_test)
        expected_probability = fitted.predict_proba(self.X_test)

        with tempfile.TemporaryDirectory(prefix="m07-test-") as directory:
            path = save_pipeline(fitted, Path(directory) / "pipeline.joblib")
            reloaded = load_pipeline(path)

        np.testing.assert_allclose(
            expected_transform, transform_features(reloaded, self.X_test)
        )
        np.testing.assert_array_equal(expected_prediction, reloaded.predict(self.X_test))
        np.testing.assert_allclose(
            expected_probability, reloaded.predict_proba(self.X_test)
        )

    def test_controlled_manual_encoding_failure_is_real_and_pipeline_repairs_it(self) -> None:
        numeric_fill = self.X_train[list(NUMERIC_FEATURES)].median()
        category_fill = self.X_train[list(CATEGORICAL_FEATURES)].mode().iloc[0]

        def manual_matrix(frame: pd.DataFrame) -> pd.DataFrame:
            prepared = frame.copy()
            prepared.loc[:, list(NUMERIC_FEATURES)] = prepared[
                list(NUMERIC_FEATURES)
            ].fillna(numeric_fill)
            prepared.loc[:, list(CATEGORICAL_FEATURES)] = prepared[
                list(CATEGORICAL_FEATURES)
            ].fillna(category_fill)
            return pd.get_dummies(
                prepared,
                columns=list(CATEGORICAL_FEATURES),
                dtype=float,
            )

        manual_train = manual_matrix(self.X_train)
        manual_model = LogisticRegression(max_iter=1_000).fit(
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

        safe_pipeline = build_pipeline().fit(self.X_train, self.y_train)
        self.assertEqual(safe_pipeline.predict(unseen).shape, (1,))

    def test_feature_treatment_configuration_is_explicit_and_validated(self) -> None:
        standard = build_pipeline(scaler="standard")
        robust = build_pipeline(scaler="robust")
        passthrough = build_pipeline(scaler="passthrough")
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
            build_pipeline(numeric_strategy="constant")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "scaler"):
            build_pipeline(scaler="normalize")  # type: ignore[arg-type]

    def test_notebook_is_substantive_stable_output_free_and_network_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 28)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            compile(source, f"M07-cell-{index}", "exec")
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])

        all_code = "\n".join("".join(cell["source"]) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "urlopen",
        ]:
            self.assertNotIn(forbidden, all_code)

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
        self.assertIn("required_evidence:", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_(evidence|response)\s*:", evidence))

    def test_requirement_file_names_runtime_and_validation_dependencies(self) -> None:
        requirements = (ROOT / "requirements" / "m07.txt").read_text(
            encoding="utf-8"
        )
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


if __name__ == "__main__":
    unittest.main()
