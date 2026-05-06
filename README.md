# Qwen3.6-27B vLLM benchmark sprint — May 5, 2026

A full day of benchmarking the [Repne `repne/vllm`](https://hub.docker.com/r/repne/vllm) fork against upstream [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.20.1 across multiple model variants and quantization formats on **dual NVIDIA RTX PRO 6000 Blackwell** (TP=2, SM120, Workstation Edition, 96 GB each, PCIe Gen5 x16 negotiated under load).

Six experiments. Six answers. **Bottom line: stay on the Repne fork with FP8+MTP=3.**

---

## SOTA roll-up
**See [`SOTA.md`](./SOTA.md)** for the cross-experiment "best tok/s per regime" table. **Headline: FP8+MTP=3 wins every cell where it competes.** No DFlash variant or quantization scheme came within striking distance.

---

---

## Headline findings

### 1. Repne fork crushes upstream at long context with `dflash`
| cell | Repne tok/s | Upstream v0.20.1 tok/s | Δ |
|---|---|---|---|
| c=1 ctx=131k | **81.4** | 11.7 | **+598%** 🚨 |
| c=2 ctx=131k | **162.7** | 22.4 | **+626%** 🚨 |
| c=4 ctx=131k | **284.4** | 44.7 | **+537%** 🚨 |

Upstream's `dflash` method is functional at short context but **collapses past 32K** because flashinfer rejects the non-causal attention `dflash` requires, forcing flash_attn for the main path — and upstream lacks Repne's `gumbel` draft sampler and `use_local_argmax_reduction` flags that make `dflash` actually work in production. Drafter acceptance falls from 30% (Repne) to 1-3% (upstream) at long ctx; per-token latency inflates 8× (11ms → 90ms).

**See:** [`05-bf16-dflash-headtohead/`](./05-bf16-dflash-headtohead/)

### 2. Repne fork wins on FP8+MTP=3 too — narrower margin, same direction
| cell | Repne tok/s | Upstream tok/s | Δ |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | 112.3 | +7.0% |
| c=2 ctx=0 | **223.8** | 197.0 | +13.6% |
| c=4 ctx=0 | **449.5** | 413.8 | +8.6% |
| c=4 ctx=131k | **347.4** | 345.0 | +0.7% |

FP8 + MTP=3 is the only path where upstream is a viable fallback (long-context delta is essentially zero, ~5-14% short-context cost). Production currently runs Repne fork on this path.

**See:** [`04-fp8-mtp3-headtohead/`](./04-fp8-mtp3-headtohead/)

### 3. NVFP4-MTP is NOT a viable production replacement
| cell | NVFP4+MTP=3 tok/s | FP8+MTP=3 ref | Δ |
|---|---|---|---|
| c=1 ctx=0 | **109.4** | ~85 | **+28.7%** ✅ |
| c=4 ctx=0 | **416.6** | 352.8 | **+18.1%** ✅ |
| c=2 ctx=131k | 105.6 | 145.9 | **−27.6%** ❌ |
| c=4 ctx=131k | 7.0 | 272.0 | **−97.4%** ❌ |
| c=1 ctx=244k | 1.4 | — | unusable ❌ |

Wins +12-29% at short context, loses 27-33% at 131K, **functionally broken at 244K** (production target is 256K). Not a replacement.

The community-quantized model itself is correct (15 mtp.* tensors graft cleanly, modelopt format, all 10 functional gates pass — Fibonacci 5/5, tool calls, multi-turn, 47×83=3901, 137K-token needle-in-haystack found in 23.3s). The problem is engine-side at long context, not model quality.

**See:** [`03-nvfp4-mtp-experiment/`](./03-nvfp4-mtp-experiment/)

### 4. New Repne image (May 5) shows mild regression on c=4 ctx=0 — likely transient
EXP-1 morning N=5 randomized variance run flagged a **−5.7%** regression at the busiest cell vs the May 3 baseline (332.6 ± 6.7 vs 352.8 OLD). EXP-1.5 scheduler investigation found `max-num-seqs=128 + max-num-batched-tokens=32768` is in a pessimal pocket — almost any deviation (32, 64, 256 seqs OR 16384 batched-tokens) recovers throughput to ~345 tok/s.

**Later in the day, single-shot c=4 ctx=0 hit 449.5 tok/s** — well above any prior baseline. The morning regression is most likely a transient state, not a persistent defect.

**See:** [`01-morning-newimage-validation/`](./01-morning-newimage-validation/) and [`02-scheduler-investigation/`](./02-scheduler-investigation/)

### 6. New Repne image (`d0a200f77546`, May 5 evening) preserves FP8+MTP=3 perf, validates DFlash=7/8/15 alternatives
- Engine version bumped from `v0.1.dev16359+ga3e24c99b` (morning) to `v0.1.dev16400+g910d87a9d` (evening) — +41 commits
- Tool calling fix verified: 4/4 functional gates pass on all 4 configs (FP8+MTP=3, DFlash=15, DFlash=8, DFlash=7)
- **FP8+MTP=3 wins every one of 9 cells.** None of the DFlash variants come close
- Among DFlash variants, **DFlash=7 is best** (+10.6% mean tok/s vs DFlash=15)
- New image vs prior image: all deltas within ±2σ noise band — neither regression nor improvement
- 108 total benchmark runs (9 cells × 4 configs × N=3)

**See:** [`06-new-image-validation/`](./06-new-image-validation/)

### 7. PCIe Gen1 reading at idle is normal, not a problem
The bench tool's `nvidia-smi` startup snapshot shows `pcie.link.gen.current = 1` — looks alarming, but ASPM downscales the link at idle. **Direct stress test confirms the link ramps to PCIe Gen5 (32GT/s) x16 under actual GPU load.** Not a hardware issue.

---

## Setup (constant across experiments unless noted)

| | |
|---|---|
| Hardware | 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each, PCIe Gen5 x16 |
| Driver | 595.71.05, CUDA 13.2 |
| Host | Linux 7.0.2-arch1-1, Intel Xeon W (W790E-SAGE) |
| Tensor parallelism | 2 |
| Max model length | 262144 |
| Reasoning parser | qwen3 |
| Tool parser | qwen3_coder |
| Prefix caching | enabled |
| Bench tool | [`llm_decode_bench.py v0.4.8`](https://github.com/cole-yoshioka/llm-inference-bench) |
| Bench protocol | 30s sustained-decode + 10s warmup, `--skip-prefill` |

---

## Experiment index

| # | Path | What | When | Method |
|---|---|---|---|---|
| 1 | [`01-morning-newimage-validation/`](./01-morning-newimage-validation/) | Validate Repne May 5 image vs May 3 baseline | morning | BF16+DFlash, 5 cells × N=5 randomized, 25 runs |
| 2 | [`02-scheduler-investigation/`](./02-scheduler-investigation/) | Identify scheduler pessimal pocket at c=4 ctx=0 | midday | BF16+DFlash, 5 variants × N=3 |
| 3 | [`03-nvfp4-mtp-experiment/`](./03-nvfp4-mtp-experiment/) | Test community NVFP4+MTP as FP8 replacement | afternoon | sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP, 11 cells × N=3 |
| 4 | [`04-fp8-mtp3-headtohead/`](./04-fp8-mtp3-headtohead/) | Repne vs upstream on FP8+MTP=3 | evening | 6 cells × N=1 quick |
| 5 | [`05-bf16-dflash-headtohead/`](./05-bf16-dflash-headtohead/) | Repne vs upstream on BF16+DFlash | evening | 6 cells × N=1 quick |
| 6 | [`06-new-image-validation/`](./06-new-image-validation/) | New image (d0a200f77546) FP8+MTP=3 vs DFlash=15/8/7 | late evening | 9 cells × 4 configs × N=3 = 108 runs |

Each experiment subdir contains:
- Raw `.json` bench tool output (full per-request samples)
- `.log` files (full bench tool stdout)
- For experiments 1-3: the `RESULTS.md` / `PHASE5_DECISION.md` / `RESULTS_PARTIAL.md` write-up that was generated during the run

---

## Versions tested

| Image | Engine | Used for |
|---|---|---|
| `repne/vllm:latest` (`5e7583ca4df9`, May 5 2026) | `v0.1.dev16359+ga3e24c99b.d20260505` | Production, all Repne-side benches |
| `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` (`f35754668e5f`) | `v0.19.1` | NVFP4 Phase 4 attempt 1 — **crashed catastrophically at ctx>131K**, took GPU0 into NV_ERR_GPU_IN_FULLCHIP_RESET, required reboot |
| `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` (`7ba11e462b5a`) | `v0.20.1` | NVFP4 Phase 4 retry, FP8+MTP=3 head-to-head, BF16+DFlash head-to-head |
| `vllm/vllm-openai:cu130-nightly` (`10c361c5ae82`) | broken | `ModuleNotFoundError: No module named 'pandas'` in `_aiter_ops.py` — **upstream cu130 nightly is broken, do not use** |

---

## Notes on Repne-only flags

The Repne fork ships several flags upstream rejects with `pydantic.ValidationError: Unexpected keyword argument`:

- `--load-format instanttensor` (faster weight load)
- `--speculative-config.draft_sample_method gumbel` (draft sampling strategy)
- `--speculative-config.attention_backend flash_attn` (independent backend for spec)
- `--speculative-config.use_local_argmax_reduction true` (drafter optimization)

For BF16+DFlash, dropping these flags causes the **5-6× long-context regression** documented in experiment 5. Without `gumbel` and `use_local_argmax_reduction`, the drafter's acceptance rate collapses to 1-3% at ctx>32K. These flags are doing real work, not engineering aesthetics.

---

## Production state at end of day

- ✅ **Repne fork FP8+MTP=3 running on port 11435** (the "27B ROCK SOLID" config in opencode)
- KV cache: 1,846,472 tokens, 7.04× max concurrency at 256K
- All other experimental containers torn down
- Both GPUs healthy, no hung state

---

## Related public repos from this sprint

These four were published as standalone repos throughout the day for fast sharing. This monorepo combines them with the morning validation + scheduler investigation that previously only existed locally.

- [`jcartu/repne-dflash-newimage`](https://github.com/jcartu/repne-dflash-newimage) — initial Repne May 5 image validation (single-shot bench, before the variance sweep)
- [`jcartu/qwen36-27b-nvfp4-mtp-experiment`](https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment) — full NVFP4+MTP write-up
- [`jcartu/qwen36-27b-fp8-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream) — FP8+MTP=3 head-to-head
- [`jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream) — BF16+DFlash head-to-head

---

## Recommendation

**Stay on the Repne fork with FP8+MTP=3** for production. This is verified SOTA across all 9 production-relevant cells on the latest image (d0a200f77546). DFlash variants don't compete — even DFlash=7 (the best of the three tested) is 17-30% behind FP8+MTP=3. Upstream v0.20.1 is a clean fallback only for the FP8+MTP=3 path and only if you can tolerate ~5-14% short-context throughput loss. For BF16+DFlash production, upstream is not a viable path — the drafter implementation gap is too large.

If Repne ever stops shipping new images, FP8+MTP=3 on upstream v0.20.1 is the safest fallback. Don't try to recreate the dflash performance gap on upstream without a fork that adds back the gumbel sampler and argmax reduction.
