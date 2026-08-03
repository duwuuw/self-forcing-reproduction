from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_loop import LoopConfig  # noqa: E402
from diffusion_loop.flux2 import apply_flux2_loop_patch, set_flux2_step_metadata  # noqa: E402
from diffusion_loop.pixart import apply_pixart_loop_patch, set_pixart_step_metadata  # noqa: E402
from diffusion_loop.raev2 import (  # noqa: E402
    apply_raev2_loop_patch,
    set_raev2_step_metadata,
    wrap_raev2_model_fn,
)
from diffusion_loop.scale_rae import apply_scale_rae_loop_patch  # noqa: E402
from diffusion_loop.scale_rae.loopguidance import loopguidance_prediction  # noqa: E402


class AddOneBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, *args, **kwargs):
        self.calls += 1
        return hidden_states + 1.0


class ScaleBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([AddOneBlock()])


class RAEBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([AddOneBlock()])
        self.num_enc_blocks = 1
        self.s_embedder = types.SimpleNamespace(num_patches=2)


class PixArtTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([AddOneBlock()])

    def forward(self, hidden_states, **kwargs):
        return self.transformer_blocks[0](hidden_states, **kwargs)


class FluxTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([])
        self.single_transformer_blocks = torch.nn.ModuleList([AddOneBlock()])

    def forward(self, hidden_states):
        return self.single_transformer_blocks[0](hidden_states)


class FluxDoubleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states,
        *,
        encoder_hidden_states,
        temb_mod_img=None,
        temb_mod_txt=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        self.calls += 1
        return encoder_hidden_states + 1.0, hidden_states + 1.0


class FluxSingleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states,
        *,
        encoder_hidden_states=None,
        temb_mod=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        self.calls += 1
        if encoder_hidden_states is not None:
            raise AssertionError("single-stream input must already be concatenated")
        return hidden_states + 1.0


class Flux2SignatureTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FluxDoubleBlock()])
        self.single_transformer_blocks = torch.nn.ModuleList([FluxSingleBlock()])

    def forward(self, hidden_states, encoder_hidden_states):
        text, image = self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        text_tokens = text.shape[1]
        joint = torch.cat([text, image], dim=1)
        joint = self.single_transformer_blocks[0](
            hidden_states=joint,
            encoder_hidden_states=None,
        )
        return joint[:, text_tokens:]


class AdapterTests(unittest.TestCase):
    @staticmethod
    def dense_config() -> LoopConfig:
        return LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=1.0,
        )

    def test_scale_rae_dense(self):
        model = ScaleBackbone()
        stats = apply_scale_rae_loop_patch(model, self.dense_config())
        model._loop_step_index = 0
        model._loop_total_steps = 1
        result = model.layers[0](torch.zeros(1, 4, 2))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertEqual(model.layers[0].calls, 2)
        self.assertEqual(stats.extra_block_calls, 1)

    def test_scale_rae_sparse_fallback(self):
        model = ScaleBackbone()
        config = LoopConfig(
            block_indices=[0],
            num_loops=3,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=1.0,
            token_operator="sparse",
            selector="random",
            selection_ratio=0.5,
        )
        stats = apply_scale_rae_loop_patch(model, config)
        model._loop_step_index = 0
        model._loop_total_steps = 1
        model.layers[0](torch.zeros(1, 4, 2))
        self.assertEqual(model.layers[0].calls, 3)
        self.assertEqual(stats.sparse_observations, 1)
        self.assertEqual(stats.sparse_selected_tokens, 2)
        self.assertEqual(stats.sparse_complement_tokens, 2)

    def test_scale_rae_sparse_loop_with_loop_guidance(self):
        model = ScaleBackbone()
        config = LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=1.0,
            token_operator="sparse",
            selector="random",
            selection_ratio=0.5,
        ).with_loop_guidance(2.0)
        stats = apply_scale_rae_loop_patch(model, config)
        model._loop_step_index = 0
        model._loop_total_steps = 1

        def denoiser(value, timestep):
            return model.layers[0](value)

        result = loopguidance_prediction(
            backbone=model,
            config=config,
            model=denoiser,
            x_t=torch.zeros(1, 4, 2),
            t=torch.zeros(1),
            denoised_fn=None,
            model_kwargs=None,
        )
        self.assertEqual(result.shape, (1, 4, 2))
        self.assertEqual(stats.loopguidance_reference_predictions, 1)
        self.assertEqual(stats.loopguidance_loop_predictions, 1)
        self.assertEqual(stats.sparse_observations, 1)

    def test_raev2_dense(self):
        model = RAEBackbone()
        stats = apply_raev2_loop_patch(model, self.dense_config())
        set_raev2_step_metadata(model, step_index=0, total_steps=1)
        result = model.blocks[0](torch.zeros(1, 4, 2))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertEqual(model.blocks[0].calls, 2)
        self.assertEqual(stats.actual_block_calls, 2)

    def test_raev2_sparse_loop_with_loop_guidance(self):
        model = RAEBackbone()
        config = LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=1.0,
            token_operator="sparse",
            token_domain="image_prefix_only",
            selector="random",
            selection_ratio=0.5,
        ).with_loop_guidance(2.0)
        stats = apply_raev2_loop_patch(model, config)

        def model_fn(value):
            return model.blocks[0](value)

        guided_model_fn = wrap_raev2_model_fn(
            model_fn,
            model,
            total_steps=1,
            config=config,
        )
        result = guided_model_fn(torch.zeros(1, 4, 2))
        self.assertEqual(result.shape, (1, 4, 2))
        self.assertEqual(stats.loopguidance_reference_predictions, 1)
        self.assertEqual(stats.loopguidance_loop_predictions, 1)
        self.assertEqual(stats.sparse_observations, 1)

    def test_raev2_image_condition_split(self):
        model = RAEBackbone()
        config = LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=1.0,
            token_operator="sparse",
            token_domain="image_vs_condition",
            selector="image_condition_split",
        )
        stats = apply_raev2_loop_patch(model, config)
        set_raev2_step_metadata(model, step_index=0, total_steps=1)
        model.blocks[0](torch.zeros(1, 4, 2))
        self.assertEqual(stats.sparse_selected_tokens, 2)
        self.assertEqual(stats.sparse_complement_tokens, 2)

    def test_pixart_dense(self):
        transformer = PixArtTransformer()
        stats = apply_pixart_loop_patch(transformer, self.dense_config())
        set_pixart_step_metadata(transformer, total_steps=1)
        result = transformer(torch.zeros(1, 4, 2))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertEqual(transformer.transformer_blocks[0].calls, 2)
        self.assertEqual(stats.active_block_invocations, 1)

    def test_pixart_loop_guidance_respects_active_interval(self):
        transformer = PixArtTransformer()
        config = LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=0.0,
        ).with_loop_guidance(2.0)
        stats = apply_pixart_loop_patch(transformer, config)
        set_pixart_step_metadata(transformer, total_steps=2)
        transformer(torch.zeros(1, 4, 2))
        transformer(torch.zeros(1, 4, 2))
        self.assertEqual(transformer.transformer_blocks[0].calls, 4)
        self.assertEqual(stats.loopguidance_reference_predictions, 1)
        self.assertEqual(stats.loopguidance_loop_predictions, 1)

    def test_flux2_dense(self):
        transformer = FluxTransformer()
        stats = apply_flux2_loop_patch(transformer, self.dense_config())
        set_flux2_step_metadata(transformer, total_steps=1)
        result = transformer(torch.zeros(1, 4, 2))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertEqual(transformer.single_transformer_blocks[0].calls, 2)
        self.assertEqual(stats.active_block_invocations, 1)

    def test_flux2_loop_guidance_respects_active_interval(self):
        transformer = FluxTransformer()
        config = LoopConfig(
            block_indices=[0],
            num_loops=2,
            lambda_value=1.0,
            start_frac=0.0,
            end_frac=0.0,
        ).with_loop_guidance(2.0)
        stats = apply_flux2_loop_patch(transformer, config)
        set_flux2_step_metadata(transformer, total_steps=2)
        transformer(torch.zeros(1, 4, 2))
        transformer(torch.zeros(1, 4, 2))
        self.assertEqual(transformer.single_transformer_blocks[0].calls, 4)
        self.assertEqual(stats.loopguidance_reference_predictions, 1)
        self.assertEqual(stats.loopguidance_loop_predictions, 1)

    def test_flux2_double_and_single_stream_signatures(self):
        transformer = Flux2SignatureTransformer()
        config = self.dense_config()
        config.block_indices = [0, 1]
        stats = apply_flux2_loop_patch(transformer, config)
        set_flux2_step_metadata(transformer, total_steps=1)
        result = transformer(
            torch.zeros(1, 3, 2),
            encoder_hidden_states=torch.zeros(1, 2, 2),
        )
        self.assertTrue(torch.equal(result, torch.full_like(result, 2.0)))
        self.assertEqual(transformer.transformer_blocks[0].calls, 2)
        self.assertEqual(transformer.single_transformer_blocks[0].calls, 2)
        self.assertEqual(stats.active_block_invocations, 2)


if __name__ == "__main__":
    unittest.main()
