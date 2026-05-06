# State-of-the-art per regime — Qwen3.6-27B on dual RTX PRO 6000 Blackwell, TP=2

**Updated by experiment 08 (X1+Y1 sprint) — MTP=3 is the production winner, not MTP=5.**

## Production-relevant cells — FP8+MTP=3 (CORRECTED SOTA)

After experiment 07 we briefly switched to MTP=5 based on c=1-4 N=3 data showing +1.8% mean. Experiment 08 extended the bench to c=8/16/32 (realistic production concurrency) and discovered **MTP=3 wins by +10.5% mean at high concurrency, up to +20.7% at c=32 ctx=0**. The crossover happens between c=4 and c=8.

### Low-concurrency cells (c=1-4, where MTP=5 was tested originally)

| cell | MTP=3 | MTP=5 | Winner |
|---|---|---|---|
| c=1 ctx=0 | 117.1 | **119.9** | MTP=5 +2.4% |
| c=1 ctx=131k | 95.0 | **101.2** | MTP=5 +6.5% |
| c=2 ctx=0 | 227.2 | **234.4** | MTP=5 +3.2% |
| c=4 ctx=0 | 449.8 | **462.5** | MTP=5 +2.8% |

### High-concurrency cells (c=8-32, what production actually hits)

| cell | MTP=5 | MTP=3 | Winner |
|---|---|---|---|
| c=8 ctx=0 | 865.8 | **875.0** | MTP=3 +1.1% |
| c=16 ctx=0 | 1329.7 | **1520.6** | MTP=3 **+14.4%** |
| c=32 ctx=0 | 1726.5 | **2083.7** | MTP=3 **+20.7%** ⭐ |
| c=32 ctx=16k | 1561.2 | **1892.3** | MTP=3 **+21.2%** ⭐ |

**Crossover at c=8.** Below c=8: MTP=5 wins small. At c=8+: MTP=3 wins big.

## Why MTP=3 wins at production concurrency

At low concurrency (c=1-4), spec decoding's wasted work on rejected drafts costs little because there's spare compute. At higher concurrency, every rejected position costs proportionally more. MTP=5 has 5 spec positions; positions 4-5 have lower acceptance, so MTP=5 wastes more compute. MTP=3 hits the sweet spot.

## Cross-experiment per-cell records (final)

| cell | Best tok/s | Config | Source |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | FP8+MTP=3 | Exp 04 (N=1 evening peak) |
| c=1 ctx=131k | **101.2** | FP8+MTP=5 | Exp 07 phase C (N=3) — only cell where MTP=5 holds an absolute record |
| c=2 ctx=0 | **234.4** | FP8+MTP=5 | Exp 07 phase C (N=3) |
| c=2 ctx=131k | **190.8** | FP8+MTP=5 | Exp 07 phase C (N=3) |
| c=4 ctx=0 | **472.2** | FP8+MTP=5 | Exp 07 phase C mini-matrix N=2 |
| c=4 ctx=32k | **454.5** | FP8+MTP=3 | Exp 06 (N=3) |
| c=4 ctx=131k | **350.5** | FP8+MTP=3 | Exp 06 (N=3) |
| c=8 ctx=0 | **875.0** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=8 ctx=32k | **795.4** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=8 ctx=131k | **534.4** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=16 ctx=0 | **1520.6** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=16 ctx=32k | **1186.0** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=16 ctx=64k | **1047.1** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=32 ctx=0 | **2083.7** | FP8+MTP=3 | Exp 08 X1 (N=2) ⭐ |
| c=32 ctx=16k | **1892.3** | FP8+MTP=3 | Exp 08 X1 (N=2) |
| c=32 ctx=32k | **1656.3** | FP8+MTP=3 | Exp 08 X1 (N=2) |

**Production hits 2083 tok/s at c=32 ctx=0** — that's the single highest aggregate throughput we've ever measured on this hardware.

## Quality picture (from experiment 08 phase Y1)

- **BF16 GGUF perplexity (reference): 7.620 ± 0.062**
- **Q8_0 GGUF perplexity (test): 7.623 ± 0.063**
- **Q8/BF16 ratio: 1.0004** (Q8 is 0.04% worse — within noise)
- **Mean KLD Q8 vs BF16: 0.001828** (very small)
- **Same top-p Q8 vs BF16: 97.9%** (Q8 picks same top token 97.9% of time)

Phaelon's "W8A8 lobotomization" claim doesn't manifest empirically. Q8 quality is essentially identical to BF16. By extension and per Qwen's FP8 model card claims, FP8 W8A8 should also be very close to BF16. **Phase B (exp 07) functional tests confirmed FP8 produces correct outputs on diverse prompts.**

## Recommendation (final)

**Production: FP8+MTP=3 on Repne fork.** Restored to this config at end of experiment 08.

- Holds SOTA at every production-relevant cell (c=8 through c=32 across all ctx tiers)
- 2083 tok/s peak throughput at c=32 ctx=0
- 350 tok/s at c=4 ctx=131k (single coding agent with deep file context)
- Quality verified: 8/8 functional tests pass, no W8A8 regression detected

**MTP=5** holds 4 SOTA records but only at c=1-4. If you have a workload that's *exclusively* single-user with deep context, MTP=5 gains you 2-7% there. For any bursty/multi-user workload, MTP=3 dominates.
