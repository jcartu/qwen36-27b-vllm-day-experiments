# State-of-the-art per regime — Qwen3.6-27B on dual RTX PRO 6000 Blackwell, TP=2

Aggregating all six experiments (4 quant configs × multiple vLLM builds × multiple speculative decoding variants).

## Production-relevant cells (image `repne/vllm:latest` = `d0a200f77546`, N=3)

| cell | SOTA tok/s | Config | Runner-up | Notes |
|---|---|---|---|---|
| c=1 ctx=0 | **117.1** | FP8+MTP=3 | DFlash=7 (92.9) | FP8 wins by 26% |
| c=1 ctx=32k | **119.2** | FP8+MTP=3 | DFlash=8 (95.2) | FP8 wins by 25% |
| c=1 ctx=131k | **95.0** | FP8+MTP=3 | DFlash=15 (87.6) | FP8 wins by 8% |
| c=2 ctx=0 | **227.2** | FP8+MTP=3 | DFlash=8 (180.7) | FP8 wins by 26% |
| c=2 ctx=32k | **227.0** | FP8+MTP=3 | DFlash=7 (184.8) | FP8 wins by 23% |
| c=2 ctx=131k | **184.9** | FP8+MTP=3 | DFlash=7 (164.8) | FP8 wins by 12% |
| c=4 ctx=0 | **449.8** | FP8+MTP=3 | DFlash=7 (344.9) | FP8 wins by 30% |
| c=4 ctx=32k | **454.5** | FP8+MTP=3 | DFlash=8 (336.1) | FP8 wins by 35% |
| c=4 ctx=131k | **350.5** | FP8+MTP=3 | DFlash=7 (299.6) | FP8 wins by 17% |

**FP8+MTP=3 dominates every cell.** No DFlash variant or quantization scheme came within striking distance.

## NVFP4-MTP — short-context only

NVFP4-MTP=3 (community sakamakismile model on upstream v0.20.1) had a few short-context wins early in the day before being disqualified for long-context degradation:

- c=1 ctx=0 single-stream: NVFP4 109.4 vs FP8 117.1 (still loses)
- c=4 ctx=0 peak throughput: NVFP4 416.6 vs FP8 449.8 (loses)

**NVFP4 was never SOTA on any cell where it didn't have a tight ctx requirement.** Final verdict: do not promote.

## Speculative decoding method comparison

| Method | Best tok/s (any cell) | Best regime | Drafter |
|---|---|---|---|
| MTP=3 (FP8) | **454.5** (c=4 ctx=32k) | All cells | None — uses MTP head built into FP8 weights |
| DFlash=7 (BF16) | 344.9 (c=4 ctx=0) | Best DFlash variant | z-lab/Qwen3.6-27B-DFlash |
| DFlash=8 (BF16) | 336.1 (c=4 ctx=32k) | Tied second | Same |
| DFlash=15 (BF16) | 313.4 (c=4 ctx=0) | Worst — too many tokens, drafter fails | Same |
| MTP=3 (BF16, day morning) | ~290 (c=4 ctx=128k) | Behind FP8+MTP=3 substantially | None |
| dflash=8 + upstream v0.20.1 | 290.9 (c=4 ctx=0) | **Collapses past 131K** to <50 tok/s | Drafter rejection rate >95% |
| MTP=3 + upstream v0.20.1 | 413.8 (c=4 ctx=0) | Comparable at peak | Same MTP path |

## Why FP8+MTP=3 wins

1. **FP8 is faster than BF16.** ~1.5× compute throughput for matmuls on Blackwell SM120.
2. **MTP draft sampling is more efficient than DFlash on this drafter.** MTP acceptance averages 50-70%; DFlash variants average 12-30%. With smaller acceptance per position, you spend more per output token on draft+verify roundtrips.
3. **The Qwen3.6-27B FP8 release ships with an MTP head trained jointly with the base model.** The DFlash drafter is a separate 27B model — it's BF16, doubles VRAM pressure, and has lower acceptance rates.
4. **Repne fork engine has fp8 kernel optimizations** (`fuse_norm_quant`, `fuse_act_quant` in compilation_config) that don't apply to BF16 weights.

## Recommendation

**Production stays on FP8+MTP=3 on the Repne fork.** This is the "27B ROCK SOLID" config in opencode. No changes.

If Repne ever stops shipping, fallback to upstream v0.20.1 with the same FP8+MTP=3 config — you lose ~5-14% short-context throughput but long-context is essentially identical (see experiment 04).

For BF16+DFlash, **upstream is non-viable** — drafter collapses past 131K (see experiment 05). And on the Repne fork, **DFlash never beats FP8+MTP=3** anyway (see experiment 06).

## Cross-experiment per-cell records

Just the absolute best number we recorded for each cell, anywhere across the six experiments:

| cell | Best tok/s | Source | Image |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | FP8+MTP=3 | Repne fork (5e7583ca, N=1 evening) |
| c=1 ctx=32k | **119.2** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=1 ctx=128k | **90.1** | BF16+DFlash=8 morning EXP-1 | Repne fork (5e7583ca) — different ctx so not directly comparable |
| c=1 ctx=131k | **99.0** | FP8+MTP=3 | Upstream v0.20.1 (one-off N=1 win) |
| c=2 ctx=0 | **227.2** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=2 ctx=32k | **227.0** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=2 ctx=131k | **184.9** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=4 ctx=0 | **454.5** | FP8+MTP=3 c=4 ctx=32k | Repne fork (d0a200f7) — single-shot variance probably |
| c=4 ctx=32k | **454.5** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=4 ctx=128k | **290.0** | BF16+DFlash=8 morning EXP-1 (N=5) | Repne fork (5e7583ca) |
| c=4 ctx=131k | **350.5** | FP8+MTP=3 | Repne fork (d0a200f7, N=3) |
| c=4 ctx=244k | **160.3** | BF16+DFlash=8 morning OLD baseline | Repne fork (f0e7574, May 3) — older image |

**FP8+MTP=3 holds the SOTA for every cell where it competes.** The only "BF16+DFlash wins" entries are at ctx values where FP8+MTP=3 wasn't directly tested (128k, 244k).
