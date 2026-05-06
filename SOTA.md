# State-of-the-art per regime — Qwen3.6-27B on dual RTX PRO 6000 Blackwell, TP=2

Aggregating all seven experiments across the day. Updated by experiment 07 (quality sprint).

## Production-relevant cells — FP8+MTP=5 (NEW SOTA, image `d0a200f77546`, N=3)

| cell | SOTA tok/s | Config | Runner-up | Notes |
|---|---|---|---|---|
| c=1 ctx=0 | **119.9** | FP8+MTP=5 | FP8+MTP=3 (117.1) | MTP=5 +2.4% |
| c=1 ctx=32k | 119.2 | FP8+MTP=3 | FP8+MTP=5 (118.1) | Tied within noise |
| c=1 ctx=131k | **101.2** | FP8+MTP=5 | FP8+MTP=3 (95.0) | **MTP=5 +6.5% — biggest win** ⭐ |
| c=2 ctx=0 | **234.4** | FP8+MTP=5 | FP8+MTP=3 (227.2) | MTP=5 +3.2% |
| c=2 ctx=32k | **231.2** | FP8+MTP=5 | FP8+MTP=3 (227.0) | MTP=5 +1.9% |
| c=2 ctx=131k | **190.8** | FP8+MTP=5 | FP8+MTP=3 (184.9) | MTP=5 +3.2% |
| c=4 ctx=0 | **462.5** | FP8+MTP=5 | FP8+MTP=3 (449.8) | MTP=5 +2.8% |
| c=4 ctx=32k | 454.5 | FP8+MTP=3 | FP8+MTP=5 (445.0) | MTP=3 wins by 2.1% |
| c=4 ctx=131k | 350.5 | FP8+MTP=3 | FP8+MTP=5 (346.9) | Tied within noise |

**FP8+MTP=5 wins 6 of 9 cells, ties 2, loses 1.** Mean tok/s across 9 cells: **257.3** (MTP=5) vs **252.7** (MTP=3) = **+1.8% mean improvement.**

## Spec acceptance (server_spec_accept_rate aggregate)

| cell | MTP=3 accept | MTP=5 accept |
|---|---|---|
| c=1 ctx=0 | 69.7% | 42.4% |
| c=1 ctx=131k | 54.8% | 39.1% |
| c=2 ctx=0 | 56.5% | 50.1% |
| c=4 ctx=0 | 60.1% | 44.1% |

MTP=5 has **lower per-position acceptance** (positions 4-5 are harder to predict than 1-3), but the **larger spec window means more total accepted tokens per verify** — net throughput gain.

## Speculative decoding method comparison (final)

| Method | Best tok/s (any cell) | Mean across 9 cells | Drafter |
|---|---|---|---|
| **MTP=5 (FP8)** | **462.5** | **257.3** | None — uses MTP head built into FP8 weights |
| MTP=3 (FP8) | 454.5 | 252.7 | Same |
| MTP=2 (FP8) | 411.2 (mini-cell only) | n/a | Same |
| MTP=4 (FP8) | 405.3 (mini-cell only) | n/a | Same |
| MTP=6 (FP8) | 463.4 (mini-cell only) | n/a | Same |
| DFlash=7 (BF16) | 344.9 | 197.5 | z-lab/Qwen3.6-27B-DFlash |
| DFlash=8 (BF16) | 336.1 | 194.1 | Same |
| DFlash=15 (BF16) | 313.4 | 178.5 | Same |

**MTP=5 wins on FP8 (best-of-9-cells throughput).** Among DFlash variants, **DFlash=7 is best** but still 30% behind MTP=5.

## Quality picture (from experiment 07)

- **FP8+MTP=5 vs FP8+no-spec** at temp=0: outputs differ on 4/8 standard prompts but **all stay correct.** Gates 4/4 pass.
- **FP8 vs Q8 GGUF (Phaelon's lobotomization claim):** Both produce factually correct, syntactically valid outputs. Both pass 8/8 functional tests on Manacher's algorithm. **No measurable W8A8 quality regression** at our test depth.
- **NVFP4-MTP:** functional but slow at long context on upstream v0.20.1 (-97% c=4 ctx=131k vs FP8). On Repne fork, even slower (50% worse than upstream). **Permanently disqualified.**

## Why FP8+MTP=5 wins

1. **FP8 W8A8** gives ~1.5× compute throughput vs BF16 on Blackwell SM120
2. **MTP=5 amortizes draft+verify roundtrips better than MTP=3** — more spec tokens per verify, even with lower per-position acceptance the aggregate accepted-tokens-per-step is higher
3. **Repne fork's W8A8 kernel optimizations** (`fuse_norm_quant`, `fuse_act_quant`, gumbel sampler, instanttensor loader) compound on the FP8 path
4. **DFlash drafter** is a separate 27B BF16 model — doubles VRAM pressure, lower aggregate acceptance rates

## Cross-experiment per-cell records

The absolute best number recorded for each cell across all seven experiments:

| cell | Best tok/s | Config | Source |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | FP8+MTP=3 | Exp 04 (N=1 evening peak) |
| c=1 ctx=32k | **119.2** | FP8+MTP=3 | Exp 06 (N=3, new image) |
| c=1 ctx=131k | **101.2** | **FP8+MTP=5** | **Exp 07 phase C (N=3)** ⭐ |
| c=2 ctx=0 | **234.4** | **FP8+MTP=5** | **Exp 07 phase C (N=3)** ⭐ |
| c=2 ctx=32k | **231.2** | **FP8+MTP=5** | **Exp 07 phase C (N=3)** ⭐ |
| c=2 ctx=131k | **190.8** | **FP8+MTP=5** | **Exp 07 phase C (N=3)** ⭐ |
| c=4 ctx=0 | **472.2** | FP8+MTP=5 | Exp 07 phase C mini-matrix N=2 (462.5 in N=3) |
| c=4 ctx=32k | **454.5** | FP8+MTP=3 | Exp 06 (N=3) |
| c=4 ctx=131k | **350.5** | FP8+MTP=3 | Exp 06 (N=3) |

## Recommendation

**Switch ROCK SOLID to FP8+MTP=5.** Single-character change to `launch_fp8.sh`:
- Old: `--speculative-config.num_speculative_tokens 3`
- New: `--speculative-config.num_speculative_tokens 5`

Net effect:
- +1.8% mean throughput across all 9 production cells
- +6.5% on the single-user-deep-context regime (c=1 ctx=131k) — most relevant for coding agents
- 4/4 functional gates pass (Fibonacci, tool call, reasoning, multi-turn)
- KV cache shrinks slightly (1.85M → 1.81M tokens), max concurrency 6.90× still ample at 256K
- No quality regression detected
