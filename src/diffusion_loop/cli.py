from __future__ import annotations

import argparse
from typing import Any

from .config_resolution import resolve_run_config
from .loop_config import VALID_SELECTORS, VALID_TOKEN_DOMAINS


def add_runtime_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared paper-method and hyperparameter flags to a CLI."""
    parser.add_argument(
        "--token-loop",
        choices=("none", "dense", "sparse"),
        required=True,
        help="Base method: no loop, Dense Token Loop, or Sparse Token Loop.",
    )
    parser.add_argument(
        "--loop-guidance",
        action="store_true",
        help="Compose Loop Guidance with the selected Dense/Sparse Token Loop.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        help="Override generation.num_inference_steps (outer sampler steps N).",
    )
    parser.add_argument(
        "--loop-count",
        "--num-loops",
        dest="num_loops",
        type=int,
        help="Override the inner loop count K.",
    )
    parser.add_argument(
        "--lambda-loop",
        type=float,
        help="Override total loop strength lambda_loop (divided over K rounds).",
    )
    parser.add_argument(
        "--loop-active",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="Override the normalized inclusive loop-active sampling interval.",
    )
    parser.add_argument(
        "--loop-layers",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="Override the inclusive loop layer range [FIRST, LAST].",
    )
    parser.add_argument(
        "--loop-guidance-weight",
        type=float,
        help="Override Loop Guidance scale g_lg; requires --loop-guidance.",
    )
    parser.add_argument(
        "--selector",
        choices=VALID_SELECTORS,
        help="Override the Sparse Token Loop selector.",
    )
    parser.add_argument(
        "--selection-ratio",
        type=float,
        help="Override the selected-token fraction rho_sel.",
    )
    parser.add_argument(
        "--selector-seed",
        type=int,
        help="Override the Sparse Token Loop routing seed.",
    )
    parser.add_argument(
        "--token-domain",
        choices=VALID_TOKEN_DOMAINS,
        help="Override the token domain eligible for sparse replay.",
    )


def resolve_from_args(payload: dict[str, Any], args: argparse.Namespace):
    """Resolve the shared argparse namespace without mutating the JSON preset."""
    return resolve_run_config(
        payload,
        token_loop=args.token_loop,
        loop_guidance=args.loop_guidance,
        outer_steps=args.outer_steps,
        num_loops=args.num_loops,
        lambda_loop=args.lambda_loop,
        loop_active=tuple(args.loop_active) if args.loop_active is not None else None,
        loop_layers=tuple(args.loop_layers) if args.loop_layers is not None else None,
        loop_guidance_weight=args.loop_guidance_weight,
        selector=args.selector,
        selection_ratio=args.selection_ratio,
        selector_seed=args.selector_seed,
        token_domain=args.token_domain,
    )
