# Benchmark prompt snapshots

These UTF-8 files freeze the exact generation prompt order used for all paper
methods. Blank lines are not permitted and prompt IDs are zero-based:
`prompt00000`, `prompt00001`, and so on.

| Benchmark | Prompt file | Count | SHA-256 |
|---|---|---:|---|
| GenEval | `geneval_553.txt` | 553 | `55b6e3650e980c13daf30569eb666e000faf19bfc2b7b4cbe999fcdb7aa9d680` |
| DPG-Bench | `dpgbench_1065.txt` | 1065 | `42d0f5f1b63b5c50a84b2579e54806f72f57bc364e2d4a2b272044f8bea3cc59` |
| GenEval2 | `geneval2_800.txt` | 800 | `f7a0ad395fcb21864ac0e6240d1a91b1151826bdc16305a35395af132b83fe18` |

`dpgbench_1065_items.jsonl` preserves the public DPG-Bench item IDs and prompt
mapping (SHA-256
`11ad3a130566af1fc7458de544524fb2253f46d306f1a9248078791d2aecda26`).
Ground-truth annotations and evaluator answers are not included; obtain them
from each official benchmark release. See `THIRD_PARTY.md` for licensing scope.
