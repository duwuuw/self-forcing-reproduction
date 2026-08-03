from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_loop import (  # noqa: E402
    LoopConfig,
    blend_loop_guidance,
    canonical_method_id,
    dense_token_loop,
    resolve_run_config,
    sparse_token_loop,
)
from diffusion_loop.loop_strategies import euler_loop  # noqa: E402
from diffusion_loop.token_selection import score_to_topk_mask  # noqa: E402


class LoopConfigTests(unittest.TestCase):
    def test_all_release_configs(self):
        seen = set()
        method_ids = set()
        for path in sorted((ROOT / "configs").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["model"]["weights_included"])
            for name, raw in payload["methods"].items():
                config = LoopConfig.from_dict(raw)
                seen.add(config.token_operator)
                method_ids.add(name)
                if name == "without_loop":
                    self.assertFalse(config.enabled)
        self.assertEqual(seen, {"dense", "sparse"})
        self.assertEqual(
            method_ids,
            {
                "without_loop",
                "dense_token_loop",
                "sparse_token_loop",
                "loop_guidance_dense_token_loop",
                "loop_guidance_sparse_token_loop",
            },
        )

    def test_dense_and_sparse_both_compose_with_loop_guidance(self):
        dense = LoopConfig(block_indices=[1], token_operator="dense", selector="all")
        sparse = LoopConfig(
            block_indices=[1],
            token_operator="sparse",
            selector="random",
            selection_ratio=0.5,
        )
        for base in (dense, sparse):
            guided = base.with_loop_guidance(3.5)
            self.assertEqual(guided.token_operator, base.token_operator)
            self.assertTrue(guided.loopguidance_enabled)
            self.assertEqual(guided.loopguidance_weight, 3.5)
            self.assertFalse(guided.without_loop_guidance().loopguidance_enabled)

    def test_raev2_resolves_full_composition_matrix(self):
        payload = json.loads((ROOT / "configs" / "raev2.json").read_text(encoding="utf-8"))
        expected = {
            ("dense", False): "dense_token_loop",
            ("sparse", False): "sparse_token_loop",
            ("dense", True): "loop_guidance_dense_token_loop",
            ("sparse", True): "loop_guidance_sparse_token_loop",
        }
        for (token_loop, guidance), method_id in expected.items():
            resolved = resolve_run_config(
                payload,
                token_loop=token_loop,
                loop_guidance=guidance,
            )
            self.assertEqual(resolved.method_id, method_id)
            self.assertEqual(resolved.loop_config.token_operator, token_loop)
            self.assertEqual(resolved.loop_config.loopguidance_enabled, guidance)
            self.assertTrue(resolved.paper_preset)

    def test_unreported_scale_sparse_guidance_requires_explicit_weight(self):
        payload = json.loads(
            (ROOT / "configs" / "scale_rae_1p5b.json").read_text(encoding="utf-8")
        )
        with self.assertRaisesRegex(ValueError, "not reported"):
            resolve_run_config(payload, token_loop="sparse", loop_guidance=True)
        resolved = resolve_run_config(
            payload,
            token_loop="sparse",
            loop_guidance=True,
            loop_guidance_weight=4.0,
        )
        self.assertEqual(resolved.method_id, "loop_guidance_sparse_token_loop")
        self.assertEqual(resolved.preset_id, "sparse_token_loop")
        self.assertFalse(resolved.paper_preset)
        self.assertEqual(resolved.loop_config.token_operator, "sparse")
        self.assertEqual(resolved.loop_config.loopguidance_weight, 4.0)

    def test_all_requested_hyperparameters_are_overridable(self):
        payload = json.loads(
            (ROOT / "configs" / "scale_rae_1p5b.json").read_text(encoding="utf-8")
        )
        resolved = resolve_run_config(
            payload,
            token_loop="sparse",
            loop_guidance=True,
            outer_steps=37,
            num_loops=6,
            lambda_loop=0.75,
            loop_active=(0.2, 0.8),
            loop_layers=(3, 7),
            loop_guidance_weight=5.5,
            selector="condition_aware",
            selection_ratio=0.6,
            selector_seed=123,
            token_domain="all_tokens",
        )
        config = resolved.loop_config
        self.assertEqual(resolved.generation["num_inference_steps"], 37)
        self.assertEqual(config.num_loops, 6)
        self.assertEqual(config.lambda_value, 0.75)
        self.assertEqual((config.start_frac, config.end_frac), (0.2, 0.8))
        self.assertEqual(config.block_indices, [3, 4, 5, 6, 7])
        self.assertEqual(config.token_operator, "sparse")
        self.assertEqual(config.loopguidance_weight, 5.5)
        self.assertEqual(config.selector, "condition_aware")
        self.assertEqual(config.selection_ratio, 0.6)
        self.assertFalse(resolved.paper_preset)

    def test_transfer_config_rejects_unimplemented_sparse_adapter(self):
        payload = json.loads((ROOT / "configs" / "pixart.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "does not support sparse"):
            resolve_run_config(payload, token_loop="sparse")

    def test_sparse_selector_capabilities_are_backend_specific(self):
        payload = json.loads((ROOT / "configs" / "raev2.json").read_text(encoding="utf-8"))
        split = resolve_run_config(
            payload,
            token_loop="sparse",
            selector="image_condition_split",
            token_domain="image_vs_condition",
        )
        self.assertEqual(split.loop_config.selector, "image_condition_split")
        self.assertIsNone(split.loop_config.selection_ratio)
        with self.assertRaisesRegex(ValueError, "does not support sparse selector"):
            resolve_run_config(
                payload,
                token_loop="sparse",
                selector="condition_aware",
            )

    def test_method_ids_are_orthogonal(self):
        self.assertEqual(canonical_method_id("dense", False), "dense_token_loop")
        self.assertEqual(
            canonical_method_id("dense", True),
            "loop_guidance_dense_token_loop",
        )
        self.assertEqual(canonical_method_id("sparse", False), "sparse_token_loop")
        self.assertEqual(
            canonical_method_id("sparse", True),
            "loop_guidance_sparse_token_loop",
        )

    def test_sampling_progress_gating(self):
        config = LoopConfig(
            block_indices=[1],
            num_loops=4,
            start_frac=0.0,
            end_frac=0.5,
        )
        self.assertEqual([config.resolve_k(i, 5) for i in range(5)], [4, 4, 4, 0, 0])
        self.assertEqual(config.lambda_for_k(4), 0.25)

    def test_sparse_requires_selector(self):
        with self.assertRaises(ValueError):
            LoopConfig(block_indices=[1], token_operator="sparse", selector="all")

    def test_config_rejects_unknown_fields_and_non_ranges(self):
        with self.assertRaisesRegex(ValueError, "unknown LoopConfig fields"):
            LoopConfig.from_dict({"enabled": False, "loop_count": 4})
        with self.assertRaisesRegex(ValueError, "contiguous"):
            LoopConfig(block_indices=[1, 3])

    def test_image_condition_split_has_no_ratio(self):
        config = LoopConfig(
            block_indices=[1],
            token_operator="sparse",
            token_domain="image_vs_condition",
            selector="image_condition_split",
        )
        self.assertEqual(config.selector_ratio, 1.0)
        with self.assertRaisesRegex(ValueError, "forbids selection_ratio"):
            LoopConfig(
                block_indices=[1],
                token_operator="sparse",
                token_domain="image_vs_condition",
                selector="image_condition_split",
                selection_ratio=0.5,
            )


class LoopRuntimeTests(unittest.TestCase):
    def test_loop_guidance_is_prediction_space_and_token_loop_agnostic(self):
        reference = torch.tensor([1.0, 2.0])
        looped = torch.tensor([2.0, 4.0])
        guided = blend_loop_guidance(reference, looped, 3.0)
        self.assertTrue(torch.equal(guided, torch.tensor([4.0, 8.0])))

    def test_dense_euler(self):
        state = torch.ones(1, 2, 1)

        def block(value, *args, **kwargs):
            return 2.0 * value

        result = euler_loop(block, state, (), {}, 2, 0.5)
        self.assertTrue(torch.allclose(result, torch.full_like(state, 2.25)))
        self.assertIs(euler_loop, dense_token_loop)

    def test_sparse_selection_uses_paper_ceiling_rule(self):
        score = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        selected = score_to_topk_mask(score, ratio=0.5)
        self.assertEqual(int(selected.sum().item()), 3)

    def test_sparse_caches_complement(self):
        calls = []
        state = torch.zeros(1, 3, 1)
        masks = {
            "selected": torch.tensor([[[True], [False], [False]]]),
            "complement": torch.tensor([[[False], [True], [False]]]),
        }

        def selected_forward(value, group):
            calls.append(group)
            mask = masks["selected"] | masks["complement"] if group == "all" else masks[group]
            return torch.where(mask, value + 1.0, value)

        result = sparse_token_loop(
            selected_forward,
            state,
            masks,
            operator="euler",
            step_size=1.0,
            num_loops=3,
        )
        self.assertEqual(calls, ["all", "selected", "selected"])
        self.assertTrue(torch.equal(result, torch.tensor([[[3.0], [3.0], [0.0]]])))


if __name__ == "__main__":
    unittest.main()
