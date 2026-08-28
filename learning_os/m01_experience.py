from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from math import log, sqrt
from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop


EXPERIMENT_ORDER = ["E1", "E2", "E3", "E4", "E5"]


class M01Experience:
    """Reference-quality learner experience for M01.

    Whole-first uses a concrete demonstration transaction with different inputs
    from the later experiments. The experiment sequence remains prediction ->
    run -> observe -> explain, with outcomes withheld until a prediction exists.
    """

    def __init__(self, root: str | Path, loop: LearningLoop) -> None:
        self.root = Path(root).resolve()
        self.loop = loop
        self.notebook = next(iter(sorted((self.root / "labs").glob("M01_*.ipynb"))), None)

    def _require_notebook(self) -> Path:
        if self.notebook is None:
            raise ValueError("M01 repository notebook is missing")
        return self.notebook

    def _state(self) -> dict[str, Any]:
        payload = self.loop.store.read("m01_experience.json", {})
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("whole_run", None)
        payload.setdefault("whole_observation", "")
        payload.setdefault("initial_map", "")
        payload.setdefault("questions", "")
        payload.setdefault("experiments", {})
        payload.setdefault("controlled_failure", "")
        payload.setdefault("explanation", "")
        payload.setdefault("no_ai_submission", "")
        payload.setdefault("transfer_submission", "")
        return payload

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.loop.store.write("m01_experience.json", state)

    def runtime(self) -> dict[str, Any]:
        available = importlib.util.find_spec("jupyter") is not None and importlib.util.find_spec("nbconvert") is not None
        return {
            "python": sys.executable,
            "notebook_found": self.notebook is not None,
            "whole_demo_available": True,
            "jupyter_available": available,
            "install_command": f"{sys.executable} -m pip install jupyter nbconvert ipykernel",
        }

    def _notebook_payload(self) -> dict[str, Any]:
        path = self._require_notebook()
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _text(source: Any) -> str:
        if isinstance(source, list):
            return "".join(str(item) for item in source)
        return str(source or "")

    def experiments(self) -> list[dict[str, Any]]:
        if self.notebook is None:
            return []
        payload = self._notebook_payload()
        rows: dict[str, dict[str, Any]] = {}
        for cell in payload.get("cells", []):
            if not isinstance(cell, dict):
                continue
            meta = cell.get("metadata") or {}
            eid = meta.get("experiment_id")
            if eid not in EXPERIMENT_ORDER:
                continue
            row = rows.setdefault(eid, {"id": eid, "title": eid, "prediction_prompt": "", "action_cell": None})
            text = self._text(cell.get("source"))
            if meta.get("role") == "prediction":
                lines = [line.strip("# ") for line in text.splitlines() if line.strip()]
                if lines:
                    row["title"] = next((line for line in lines if line.startswith(eid)), lines[0])
                marker = "Prediction checkpoint:"
                plain = text.replace("**", "")
                row["prediction_prompt"] = plain.split(marker, 1)[1].strip() if marker in plain else plain.strip()
            if meta.get("role") == "action" and cell.get("cell_type") == "code":
                row["action_cell"] = cell
        return [rows[eid] for eid in EXPERIMENT_ORDER if eid in rows]

    def view(self) -> dict[str, Any]:
        state = self._state()
        experiment_state = state.get("experiments", {})
        experiments = []
        for index, spec in enumerate(self.experiments()):
            item = dict(experiment_state.get(spec["id"], {}))
            previous_done = index == 0 or bool(experiment_state.get(EXPERIMENT_ORDER[index - 1], {}).get("reflection"))
            status = (
                "complete" if item.get("reflection") else
                "observed" if item.get("result") else
                "predicted" if item.get("prediction") else
                "ready" if previous_done else "locked"
            )
            experiments.append({
                "id": spec["id"],
                "title": spec["title"],
                "prediction_prompt": spec["prediction_prompt"],
                "status": status,
                "prediction": item.get("prediction", ""),
                "result": item.get("result"),
                "reflection": item.get("reflection", ""),
            })
        complete_count = sum(1 for item in experiments if item["status"] == "complete")
        return {
            "mission_id": "M01",
            "runtime": self.runtime(),
            "whole": {"run": state.get("whole_run"), "observation": state.get("whole_observation", "")},
            "map": {"content": state.get("initial_map", "")},
            "interrogate": {"questions": state.get("questions", "")},
            "experiments": experiments,
            "experiment_progress": {"complete": complete_count, "total": len(experiments)},
            "controlled_failure": state.get("controlled_failure", ""),
            "explanation": state.get("explanation", ""),
            "no_ai_submission": state.get("no_ai_submission", ""),
            "transfer_submission": state.get("transfer_submission", ""),
            "gate": self._contract_status(),
        }

    def save_stage(self, stage: str, content: str) -> dict[str, Any]:
        key_map = {
            "whole": "whole_observation",
            "map": "initial_map",
            "interrogate": "questions",
            "break": "controlled_failure",
            "explain": "explanation",
            "no_ai": "no_ai_submission",
            "transfer": "transfer_submission",
        }
        if stage not in key_map:
            raise ValueError(f"Unsupported M01 stage: {stage}")
        text = str(content or "").strip()
        if len(text.split()) < 3:
            raise ValueError("Write a substantive learner response before saving this stage")
        state = self._state()
        if stage == "whole" and not state.get("whole_run"):
            raise ValueError("Run the whole system once before writing your observation")
        state[key_map[stage]] = text
        self._save(state)
        self._record_stage_evidence(stage)
        return self.view()

    def _record_stage_evidence(self, stage: str) -> None:
        existing = self.loop.evidence.for_mission("M01")
        marker = f"m01:{stage}:"
        if any(str(item.get("summary", "")).startswith(marker) for item in existing):
            return
        if stage == "map":
            self.loop.record_evidence("M01", "artifact", marker + " learner-created system map", ["system mapping"], False, False, False)
        elif stage == "break":
            self.loop.record_evidence("M01", "review", marker + " controlled failure diagnosis and repaired architecture", ["system mapping"], False, False, True)
        elif stage == "explain":
            self.loop.record_evidence("M01", "note", marker + " mechanism and boundary explanation", ["system mapping"], False, False, True)
        elif stage == "no_ai":
            self.loop.record_evidence("M01", "artifact", marker + " independent no-AI reconstruction", ["system mapping"], True, False, True)
        elif stage == "transfer":
            self.loop.record_evidence("M01", "note", marker + " unseen architecture transfer assessment", ["system mapping"], False, True, True)

    @staticmethod
    def _tok(text: str) -> list[str]:
        return [word.strip(".,!?;:").lower() for word in text.split() if word.strip(".,!?;:")]

    @classmethod
    def _train_classifier(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        docs: Counter[str] = Counter()
        words: defaultdict[str, Counter[str]] = defaultdict(Counter)
        vocab: set[str] = set()
        for row in rows:
            docs[row["label"]] += 1
            for word in cls._tok(row["text"]):
                words[row["label"]][word] += 1
                vocab.add(word)
        return {
            "labels": sorted(docs),
            "docs": dict(docs),
            "words": {key: dict(words[key]) for key in sorted(docs)},
            "vocab": sorted(vocab),
        }

    @staticmethod
    def _digest(model: dict[str, Any]) -> str:
        return sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def _predict(cls, model: dict[str, Any], text: str) -> tuple[str, dict[str, float]]:
        scores: dict[str, float] = {}
        total_docs = sum(model["docs"].values())
        vocab_size = max(1, len(model["vocab"]))
        for label in model["labels"]:
            counts = model["words"][label]
            total_words = sum(counts.values())
            score = log(model["docs"][label] / total_docs)
            for word in cls._tok(text):
                score += log((counts.get(word, 0) + 1) / (total_words + vocab_size))
            scores[label] = score
        return max(scores, key=scores.get), scores

    @classmethod
    def _whole_demo(cls) -> dict[str, Any]:
        train = [
            {"id": 1, "priority": "normal", "text": "reset password account", "label": "account"},
            {"id": 2, "priority": "normal", "text": "cannot sign in account", "label": "account"},
            {"id": 3, "priority": "urgent", "text": "card charged twice invoice", "label": "billing"},
            {"id": 4, "priority": "normal", "text": "copy of invoice billing", "label": "billing"},
            {"id": 5, "priority": "normal", "text": "app crashes dashboard", "label": "technical"},
            {"id": 6, "priority": "urgent", "text": "upload service error", "label": "technical"},
        ]
        test = [
            {"text": "password sign in", "label": "account"},
            {"text": "invoice duplicate charge", "label": "billing"},
            {"text": "upload crashes error", "label": "technical"},
        ]
        docs = [
            {"id": "K1", "text": "Reset password from account security. Sign in problems may require recovery."},
            {"id": "K2", "text": "For duplicate card charge compare invoice identifiers before billing review."},
            {"id": "K3", "text": "For upload errors check file size and retry after reopening the application."},
        ]

        model = cls._train_classifier(train)
        demo_request = {
            "id": 301,
            "priority": "normal",
            "text": "account security password recovery",
        }
        prediction, scores = cls._predict(model, demo_request["text"])

        retrieval_vocab = sorted({word for document in docs for word in cls._tok(document["text"])})

        def embed(text: str) -> list[float]:
            counts = Counter(cls._tok(text))
            return [float(counts.get(word, 0)) for word in retrieval_vocab]

        def cosine(left: list[float], right: list[float]) -> float:
            dot = sum(x * y for x, y in zip(left, right))
            left_norm = sqrt(sum(x * x for x in left))
            right_norm = sqrt(sum(y * y for y in right))
            return 0.0 if not left_norm or not right_norm else dot / (left_norm * right_norm)

        doc_vectors = {document["id"]: embed(document["text"]) for document in docs}
        query_vector = embed(demo_request["text"])
        ranked = sorted(
            [
                {**document, "score": cosine(query_vector, doc_vectors[document["id"]])}
                for document in docs
            ],
            key=lambda row: (-row["score"], row["id"]),
        )
        retrieved = ranked[0]

        memory_before: dict[str, Any] = {}
        memory_after = {
            "runs": 1,
            "last_intent": prediction,
            "last_ticket": demo_request["id"],
        }
        tool_result = None
        application_result = {
            "intent": prediction,
            "knowledge_document": retrieved["id"],
            "tool_result": tool_result,
        }
        trace = [
            {"event": "inference", "value": prediction},
            {"event": "retrieval", "value": retrieved["id"]},
            {"event": "memory", "value": memory_after},
        ]

        evaluated = []
        for row in test:
            predicted, _ = cls._predict(model, row["text"])
            evaluated.append({"input": row["text"], "expected": row["label"], "predicted": predicted, "correct": predicted == row["label"]})
        correct = sum(1 for row in evaluated if row["correct"])

        return {
            "training_data": {
                "count": len(train),
                "labels": sorted({row["label"] for row in train}),
                "examples": [{"text": row["text"], "label": row["label"]} for row in train],
            },
            "training": {
                "operation": "learn token statistics from labelled examples",
                "output": "MODEL",
            },
            "model_state": {
                "id": "MODEL",
                "digest": cls._digest(model),
                "labels": model["labels"],
                "vocabulary_size": len(model["vocab"]),
                "description": "learned token statistics used later by inference",
            },
            "request": demo_request,
            "inference": {
                "input": "MODEL + represented demo request",
                "output": "label scores",
            },
            "prediction": {
                "label": prediction,
                "scores": {label: round(value, 3) for label, value in sorted(scores.items())},
            },
            "retrieval": {
                "query": demo_request["text"],
                "document_id": retrieved["id"],
                "score": round(retrieved["score"], 3),
                "context": retrieved["text"],
            },
            "controller": {
                "decision": "continue with retrieved knowledge; no external tool used in this demo request",
                "tool_called": False,
                "trace": trace,
            },
            "application": application_result,
            "memory": {"before": memory_before, "after": memory_after},
            "evaluation": {
                "correct": correct,
                "total": len(evaluated),
                "accuracy": round(correct / len(evaluated), 3),
                "cases": evaluated,
            },
            "infrastructure": {
                "compute": "single local CPU process",
                "network_calls": 0,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "flow": [
                "training_data", "training", "model_state", "request", "inference",
                "prediction", "controller", "retrieval", "application", "memory",
                "evaluation", "infrastructure",
            ],
        }

    def run_whole(self, timeout_seconds: int = 180) -> dict[str, Any]:
        del timeout_seconds  # whole-first is a bounded in-process deterministic demo
        demo = self._whole_demo()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = self.root / "tracking" / "lab_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"M01_WHOLE_{stamp}.json"
        ran_at = datetime.now(timezone.utc).isoformat()
        artifact.write_text(
            json.dumps({"mission_id": "M01", "ran_at": ran_at, "demo": demo}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        state = self._state()
        state["whole_run"] = {
            "status": "PASS",
            "demo": demo,
            "artifact": artifact.relative_to(self.root).as_posix(),
            "ran_at": ran_at,
        }
        self._save(state)
        return self.view()

    def save_prediction(self, experiment_id: str, prediction: str) -> dict[str, Any]:
        eid = str(experiment_id or "").upper()
        if eid not in EXPERIMENT_ORDER:
            raise ValueError("Unknown M01 experiment")
        text = str(prediction or "").strip()
        if len(text.split()) < 3:
            raise ValueError("Commit a prediction before running the experiment")
        state = self._state()
        if not state.get("initial_map") or not state.get("questions"):
            raise ValueError("Complete the Map and Interrogate stages before starting experiments")
        experiments = state["experiments"]
        index = EXPERIMENT_ORDER.index(eid)
        if index and not experiments.get(EXPERIMENT_ORDER[index - 1], {}).get("reflection"):
            raise ValueError(f"Complete {EXPERIMENT_ORDER[index - 1]} before starting {eid}")
        item = experiments.setdefault(eid, {})
        if item.get("result"):
            raise ValueError("Prediction is locked after the experiment has run")
        item["prediction"] = text
        item["predicted_at"] = datetime.now(timezone.utc).isoformat()
        self._save(state)
        return self.view()

    def _execute_subset(self, subset: dict[str, Any], output_name: str, timeout: int) -> Path:
        runtime = self.runtime()
        if not runtime["jupyter_available"]:
            raise ValueError("Notebook runtime is not installed. Stop LearningOS, run: " + runtime["install_command"] + ", then restart LearningOS.")
        output_dir = self.root / "tracking" / "lab_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="learningos-m01-") as temp_dir:
            source = Path(temp_dir) / "M01_guided.ipynb"
            source.write_text(json.dumps(subset), encoding="utf-8")
            command = [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(source),
                "--output", output_name, "--output-dir", str(output_dir), f"--ExecutePreprocessor.timeout={timeout}",
            ]
            completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=timeout + 30, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Experiment execution failed")[-4000:]
            raise ValueError(detail)
        return output_dir / output_name

    def _subset_notebook(self, experiment_id: str) -> dict[str, Any]:
        payload = self._notebook_payload()
        target_index = EXPERIMENT_ORDER.index(experiment_id)
        cells: list[dict[str, Any]] = []
        for cell in payload.get("cells", []):
            if not isinstance(cell, dict):
                continue
            meta = cell.get("metadata") or {}
            eid = meta.get("experiment_id")
            if eid in EXPERIMENT_ORDER and EXPERIMENT_ORDER.index(eid) > target_index:
                continue
            if cell.get("cell_type") == "markdown" and eid in EXPERIMENT_ORDER:
                continue
            source = self._text(cell.get("source"))
            if "SYSTEM_MAP=" in source or "## No-AI gate" in source:
                continue
            cells.append(cell)
        return {
            "cells": cells,
            "metadata": payload.get("metadata", {}),
            "nbformat": payload.get("nbformat", 4),
            "nbformat_minor": payload.get("nbformat_minor", 5),
        }

    @staticmethod
    def _experiment_output_text(notebook_payload: dict[str, Any], experiment_id: str) -> str:
        pieces: list[str] = []
        for cell in notebook_payload.get("cells", []):
            if not isinstance(cell, dict):
                continue
            meta = cell.get("metadata") or {}
            if meta.get("experiment_id") != experiment_id or meta.get("role") != "action":
                continue
            for output in cell.get("outputs", []):
                if not isinstance(output, dict):
                    continue
                text = output.get("text")
                if isinstance(text, list):
                    pieces.extend(str(item) for item in text)
                elif isinstance(text, str):
                    pieces.append(text)
                data = output.get("data") or {}
                plain = data.get("text/plain") if isinstance(data, dict) else None
                if isinstance(plain, list):
                    pieces.extend(str(item) for item in plain)
                elif isinstance(plain, str):
                    pieces.append(plain)
        return "".join(pieces).strip()[-6000:]

    def run_experiment(self, experiment_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
        eid = str(experiment_id or "").upper()
        state = self._state()
        item = state["experiments"].get(eid, {})
        if not item.get("prediction"):
            raise ValueError("Commit your prediction before running the experiment")
        timeout = max(30, min(int(timeout_seconds), 600))
        subset = self._subset_notebook(eid)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_name = f"M01_{eid}_{stamp}.ipynb"
        executed_path = self._execute_subset(subset, output_name, timeout)
        executed = json.loads(executed_path.read_text(encoding="utf-8"))
        result = {
            "status": "PASS",
            "output": self._experiment_output_text(executed, eid),
            "executed_notebook": executed_path.relative_to(self.root).as_posix(),
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }
        state = self._state()
        state["experiments"].setdefault(eid, {})["result"] = result
        self._save(state)
        return self.view()

    def save_reflection(self, experiment_id: str, reflection: str) -> dict[str, Any]:
        eid = str(experiment_id or "").upper()
        text = str(reflection or "").strip()
        if len(text.split()) < 5:
            raise ValueError("Explain what the observation tells you before completing the experiment")
        state = self._state()
        item = state["experiments"].get(eid, {})
        if not item.get("result"):
            raise ValueError("Run the experiment before recording your explanation")
        item["reflection"] = text
        item["reflected_at"] = datetime.now(timezone.utc).isoformat()
        self._save(state)
        if eid == "E5":
            self._record_stage_evidence("break")
        if all(state["experiments"].get(key, {}).get("reflection") for key in EXPERIMENT_ORDER):
            existing = self.loop.evidence.for_mission("M01")
            if not any(str(record.get("summary", "")).startswith("m01:experiments:") for record in existing):
                self.loop.record_evidence(
                    "M01", "lab", "m01:experiments: E1-E5 prediction-run-observe-explain sequence completed",
                    ["AI systems vocabulary", "system mapping"], False, False, True,
                )
        return self.view()

    def _contract_status(self) -> dict[str, Any]:
        state = self._state()
        experiments = state["experiments"]
        checks = [
            {"id": "system_map", "label": "System map", "complete": bool(state.get("initial_map"))},
            {"id": "toy_system_trace", "label": "E1-E4 experiment trace", "complete": all(experiments.get(eid, {}).get("reflection") for eid in EXPERIMENT_ORDER[:4])},
            {"id": "controlled_failure_diagnosis", "label": "Controlled failure diagnosis", "complete": bool(state.get("controlled_failure") or experiments.get("E5", {}).get("reflection"))},
            {"id": "no_ai_gate", "label": "No-AI reconstruction", "complete": bool(state.get("no_ai_submission")) and state.get("no_ai_submission") != "__ACTIVE__"},
            {"id": "transfer_assessment", "label": "Unseen transfer assessment", "complete": bool(state.get("transfer_submission"))},
        ]
        return {"ready": all(item["complete"] for item in checks), "checks": checks}
