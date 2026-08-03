from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


aggregate_metrics = load_script("aggregate_metrics")
export_benchmark = load_script("export_benchmark")
prepare_benchmark = load_script("prepare_benchmark")


class PromptSnapshotTests(unittest.TestCase):
    def test_manifest_counts(self):
        manifest = json.loads((ROOT / "prompts" / "manifest.json").read_text())
        observed = {}
        for name, spec in manifest["benchmarks"].items():
            observed[name] = len((ROOT / spec["file"]).read_text().splitlines())
        self.assertEqual(
            observed,
            {"geneval": 553, "dpgbench": 1065, "geneval2": 800},
        )


class AggregateTests(unittest.TestCase):
    def test_deltas_and_imagereward_percentage(self):
        rows = [
            {"method_id": "without_loop", "benchmark": "x", "prompt_id": "p0", "metric": "score", "value": "2"},
            {"method_id": "without_loop", "benchmark": "x", "prompt_id": "p1", "metric": "score", "value": "4"},
            {"method_id": "dense_token_loop", "benchmark": "x", "prompt_id": "p0", "metric": "score", "value": "4"},
            {"method_id": "dense_token_loop", "benchmark": "x", "prompt_id": "p1", "metric": "score", "value": "5"},
            {"method_id": "without_loop", "benchmark": "x", "prompt_id": "p0", "metric": "ImageReward", "value": "-1"},
            {"method_id": "dense_token_loop", "benchmark": "x", "prompt_id": "p0", "metric": "ImageReward", "value": "0"},
        ]
        output = aggregate_metrics.aggregate_rows(rows, "without_loop")
        score = next(
            row for row in output
            if row["method_id"] == "dense_token_loop" and row["metric"] == "score"
        )
        self.assertAlmostEqual(score["value"], 4.5)
        self.assertAlmostEqual(score["delta"], 1.5)
        self.assertAlmostEqual(score["delta_pct"], 50.0)
        reward = next(
            row for row in output
            if row["method_id"] == "dense_token_loop" and row["metric"] == "ImageReward"
        )
        self.assertIsNone(reward["delta_pct"])

    def test_duplicate_rejected(self):
        row = {"method_id": "without_loop", "benchmark": "x", "prompt_id": "p0", "metric": "score", "value": "2"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_metrics.aggregate_rows([row, dict(row)], "without_loop")


class GenerationRecordTests(unittest.TestCase):
    def test_plan_is_prompt_matched(self):
        class Resolved:
            generation = {"seed": 42}
            method_id = "sparse_token_loop"

            @staticmethod
            def to_record():
                return {"method_id": "sparse_token_loop"}

        rows = prepare_benchmark.build_plan(
            ["first", "second"], "geneval", Resolved(), Path("images")
        )
        self.assertEqual(rows[1]["prompt_id"], "prompt00001")
        self.assertEqual(rows[1]["prompt"], "second")
        self.assertEqual(rows[1]["seed"], 42)

    def test_record_validation_rejects_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "image.png"
            image.write_bytes(b"not decoded by this CPU-only contract test")
            records = root / "records.jsonl"
            records.write_text(
                json.dumps(
                    {
                        "benchmark": "geneval",
                        "method_id": "without_loop",
                        "seed": 42,
                        "prompt_id": "prompt00000",
                        "prompt": "wrong",
                        "output_image": str(image),
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "prompt text mismatch"):
                export_benchmark.validated_rows(
                    records,
                    ["right"],
                    "geneval",
                    "without_loop",
                    42,
                    False,
                )


if __name__ == "__main__":
    unittest.main()
