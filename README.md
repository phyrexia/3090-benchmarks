# RTX 3090 inference benchmarks

Measured inference results from a home node with **two used RTX 3090s**, November 2026. These are observations from this machine, with the conditions below, rather than a general ranking of models or GPUs.

Charts are in [the author’s posts](https://x.com/phyrexia) and the measurements are in the [CSVs](data/).

The CSVs give units in column names and include `source_date` and `conditions`. Blank cells indicate missing or inapplicable values. HyperQwen TP=2 uses one serving instance across both cards, so `streams_per_instance` equals `streams_total`.

## Hardware and software

- Two NVIDIA GeForce RTX 3090 cards, **24 GB VRAM each**.
- **PCIe 3.0 x8/x8**, **no NVLink**. Tensor parallelism below uses both cards; independent instances keep a model on each card.
- **Debian 13**, NVIDIA driver **610.57.04**. Initial setup included a modification that increased FP16 compute throughput. PCIe peer-to-peer access was unavailable in the dual-card setup.
- **InklingEngine `0b7a1c5a`** for the raw speed and power experiments; **`3b9e285`** for the code-review quality experiment. Model: **Inkling 2 Mini, UD-Q4_K_XL**. KV-cache quantisation is specified separately.
- **HyperQwen `3d6c9543292e`**, reporting **vLLM 0.33.0**, for the November 19–20 measurements. Model: **Qwen3.8 27B, W4A16**, with **DFlash2** speculative decoding and **TP=2**. The later power records leave the engine revision, weight and KV quantisation, context limit and speculative decoding mode unpinned. The HyperQwen quality record also leaves the engine revision and context limit unpinned.
- **llama.cpp, Unsloth `b12588-mix-c0c3de8`**, for the same-file Inkling quality comparison, with DSpark speculative decoding.

Power targets and effective limits are **per card**. Measured total power and energy cover the GPUs only, excluding the rest of the computer and power-supply losses. Cooling and thermal state changed between experiments; compare within a labelled condition before comparing across dates.

## 1. One RTX 3090: Inkling 2 Mini on InklingEngine

### Decode and prompt reading by depth

**Conditions.** InklingEngine `0b7a1c5a`, UD-Q4_K_XL weights, KV q6, one GPU at a **275 W** cap, **256K configured context**, one stream, and built-in filler text. One raw continuation pass generated **128 tokens per depth** on **2026-11-23**. Sampling temperature does not apply to the Inkling filler-text benchmarks in sections 1 and 3. [CSV](data/inkling-single-275w.csv)

| Context depth | Prompt interval read | Prompt reading (tok/s) | Decode (tok/s) |
|---|---|---:|---:|
| 0 | — | — | 126.9 |
| 4K | 0 → 4K | 3,830 | 118.7 |
| 16K | 4K → 16K | 3,658 | 115.1 |
| 32K | 16K → 32K | 3,287 | 111.0 |
| 128K | 32K → 128K | 2,325 | 93.7 |
| Near 256K | 128K → near 256K | 1,449 | 77.3 |

Prompt reading is incremental over the indicated interval, not a fresh read of the entire prefix. Here K means 1,024 tokens: the context allocation is **262,144 tokens**, and the final fill is **262,015** so the generated tokens fit.

### Context and VRAM fit

**Conditions:** InklingEngine `0b7a1c5a`; Inkling 2 Mini UD-Q4_K_XL; **275 W per card**, one GPU, **one stream**; filler text; raw continuation, no chat reasoning mode; **2026-11-23**. Context and KV precision vary by row. [CSV](data/inkling-context-fit.csv).

| Configured context (tokens) | KV cache | VRAM used (GiB) | Reported available VRAM (GiB) |
|---:|---|---:|---:|
| 262,144 (256K) | q6 | 20.9 | 23.6 |
| 393,216 (384K) | q4 | 21.0 | 23.6 |

These are VRAM allocation measurements with filler text. Retrieval quality at these context lengths and the effect of KV precision on quality were unmeasured.

**Serving limitation:** the measured InklingEngine version serves **one request at a time per instance**. Two independent instances can use the two cards, but that is separate from tensor parallelism and from batching within one instance.

### Separate sample at a 300 W effective cap

**Conditions.** InklingEngine `0b7a1c5a`, UD-Q4_K_XL, KV q6, one better-ventilated GPU, one stream, **135,168-token context**, filler text, and normal memory and SM clocks. One raw continuation pass generated **256 tokens per depth** on **2026-11-24**. The power target was **350 W** and an automatic power-limit guard on the host capped it at **300 W**. Mean draw under load was **283 W** and peak GPU temperature was **68 °C**. [CSV](data/inkling-single-300w.csv)

| Context depth | Prompt interval read | Prompt reading (tok/s) | Decode (tok/s) |
|---|---|---:|---:|
| 0 | — | — | 126.1 |
| 4K | 0 → 4K | 3,973 | 120.1 |
| 16K | 4K → 16K | 3,802 | 116.3 |
| 32K | 16K → 32K | 3,402 | 112.1 |
| 128K | 32K → 128K | 2,391 | 95.1 |

This is a separate card, context allocation and date, not an isolated test of the effect of power.

## 2. Two RTX 3090s, TP=2: Qwen3.8 27B on HyperQwen

### Code, reasoning off, 290 W per card

**Conditions.** HyperQwen `3d6c9543292e`, vLLM 0.33.0, W4A16 weights, DFlash2, **TP=2**, **290 W per card**, **128K configured context**, code prompts, reasoning **off**, and temperature **0.6** on **2026-11-19**. Streams vary by row. The run does not independently record KV quantisation. [CSV](data/qwen-code-no-reasoning.csv)

| Simultaneous streams | Aggregate output (tok/s) |
|---:|---:|
| 1 | 180 |
| 2 | 297 |
| 4 | 431 |
| 8 | 521 |

These are the reported aggregate rates. Ambiguous per-request rates are omitted. Code is favourable to speculative decoding in these observations; do not assume the same rates for prose.

### Code writing, reasoning on, 220 W per card

**Conditions:** HyperQwen `3d6c9543292e`, vLLM 0.33.0; W4A16; int8 per-token/head KV; DFlash2; **TP=2**; **220 W per card**; **131,072-token context**; short code-writing prompts; reasoning **on**; temperature **0.6**; **1,200 output-token limit per request**; streams vary by row; **2026-11-20**. [CSV](data/qwen-code-reasoning.csv).

| Simultaneous streams | Aggregate output (tok/s) | Mean TTFT (s) |
|---:|---:|---:|
| 1 | 140.09 | 0.090 |
| 2 | Not measured | Not measured |
| 4 | 283.75 | 0.264 |
| 8 | 373.47 | 0.491 |

**All generated output in this experiment remained inside reasoning until the token limit.** These rates count reasoning tokens, not useful completed answers. TTFT is the first reasoning delta. Aggregate rates divide output tokens by the wall time of the entire step, including waiting. This is a different workload and power condition from the preceding table.

### Cold prefill

**Conditions.** HyperQwen `3d6c9543292e`, vLLM 0.33.0, Qwen3.8 27B W4A16, int8 per-token/head KV, DFlash2, **TP=2**, **220 W per card**, one stream, **131,072-token context**, and reasoning **on**. These synthetic-document requests on **2026-11-20** all had zero cached input tokens. The immediate-reread experiment used pairs of requests. The long-document batch experiment cycled through several documents and used temperature **0.6** and a higher output limit. [CSV](data/qwen-cold-prefill.csv)

| Experiment and observation | Input (tokens) | Output-token limit | Cold TTFT (s) |
|---|---:|---:|---:|
| Immediate-reread experiment, short document, initial request | 3,892 | 256 | 2.363 |
| Immediate-reread experiment, long document, initial request | 88,355 | 256 | 65.622 |
| Batch experiment, shorter-document control | 42,118 | 1,500 | 34.041 |
| Batch experiment, long document, first pass | 88,355 | 1,500 | 66.194 |
| Batch experiment, long document, second pass | 88,355 | 1,500 | 78.549 |

TTFT measures time to the first reasoning delta, including time beyond prompt processing. The three 88,355-token observations used the same input but different request sequences and, across experiments, different output limits.

## 3. Power: calibration, night profile and tuning levers

### HyperQwen calibration: 200–300 W targets

**Conditions.** HyperQwen, Qwen3.8 27B, **TP=2**, **eight concurrent streams** in a closed loop, prose, temperature **0.6**, server-default reasoning, normal memory clock, and a **1,536-token output limit** on **2026-11-24**. Each step used **60 s of warm-up and 240 s of measurement** after convergence. The run did not independently record the revision, quantisation, context limit or speculative decoding mode. [CSV](data/qwen-power-calibration.csv)

| Step | Target (W/card) | Effective limit (W/card) | Generation, total (tok/s) | Both GPUs, median (W) | GPU energy (J/token) | Hotter GPU, median / max (°C) |
|---|---:|---:|---:|---:|---:|---:|
| Initial, cooler | 275 | 275 | 346.7 | 548.0 | 1.58 | 75 / 78 |
| Thermal reduction | 300 | 270 | 336.6 | 538.0 | 1.59 | 82 / 84 |
| Hot repeat | 275 | 270 | 350.6 | 538.2 | 1.54 | 82 / 84 |
| Later step | 250 | 250 | 341.0 | 498.6 | 1.46 | 77 / 79 |
| Later step | 225 | 225 | 332.3 | 448.9 | 1.35 | 71 / 73 |
| Later step | 200 | 200 | 301.5 | 399.2 | 1.32 | 73 / 75 |

Thermal protection held both cards at **270 W** throughout the 300 W target window and for **166 of 240 seconds** in the hot 275 W repeat.

Within this sweep, the lower targets improved energy per generated token. These short, sequential windows have no independent repetitions and do not establish a universal optimum. GPU J/token uses integrated GPU energy over the token-counter window, rather than median watts divided by a separately timed rate.

### Night profile: memory at 5,001 MHz

**HyperQwen conditions.** Qwen3.8 27B, **TP=2**, and **eight requests in flight** on real code-review prompts on **2026-11-23**. The spot records do not independently identify the engine revision, quantisation, context limit, speculative decoding mode, sampling temperature or reasoning setting. [CSV](data/qwen-night-profile.csv)

| Target (W/card) | Memory clock | Reported total generation (tok/s) |
|---:|---|---:|
| 150 | Normal | 91 |
| 150 | Locked to 5,001 MHz | 186 |
| 130 | Locked to 5,001 MHz | 105 |

The rates are reported spot measurements. The records do not identify the counter or timing window used to calculate them. The normal memory clock and measured GPU draw are unavailable for these points.

**Inkling conditions:** InklingEngine `0b7a1c5a`; Inkling 2 Mini UD-Q4_K_XL; KV q6; **130 W target per card**; **two independent instances, one stream per instance**; **40K configured context**, evaluated at **32K depth**; filler text; raw continuation, no chat reasoning mode; **six repetitions per card/depth**, **256 generated tokens per depth**; **2026-11-24**. [CSV](data/inkling-night-profile.csv).

| Configuration | Memory clock | SM clock cap (MHz) | Decode at 32K, SUM (tok/s) | Prompt reading at 32K, SUM (tok/s) | Measured both GPUs (W) |
|---|---|---:|---:|---:|---:|
| Baseline | Normal | None | 51.7 | 1,122 | 290 |
| Memory lock only | 5,001 MHz | None | 75.0 | 1,541 | 244.7 |
| Memory lock only (repeat) | 5,001 MHz | None | 73.4 | — | — |
| Memory lock and SM cap | 5,001 MHz | 1,200 | 74.2 | 1,539 | 238.5 |

**SUM means the sum of two single-instance medians.** The report gives decode alone for the memory-only repeat. The combined-setting row uses the report’s summary of the controls at that setting. Baseline draw exceeded the nominal target.

### Daytime Inkling power and clock/KV levers

**Conditions:** InklingEngine `0b7a1c5a`; Inkling 2 Mini UD-Q4_K_XL; KV q6; **two independent single-GPU instances**, **one stream each**; **40K context**; filler text; raw continuation, no chat reasoning mode; normal memory, no SM clock cap; **six repetitions per card/depth**, **256 generated tokens per depth**; **2026-11-24**. Per-card targets vary by row. All speed columns are **SUMs of instance medians**; the baseline row averages controls. [CSV](data/inkling-day-power.csv).

| Target (W/card) | Condition | Decode at 0, SUM (tok/s) | Decode at 32K, SUM (tok/s) | Prompt at 32K, SUM (tok/s) | Measured both GPUs (W) |
|---:|---|---:|---:|---:|---:|
| 225 | Power step | 225.7 | 201.6 | 4,473 | 401.7 |
| 250 | Baseline controls | 238.1 | 211.4 | 5,410 | 436.4 |
| 275 | Power step | 247.8 | 219.2 | 6,267 | 457.9 |

**Lever conditions:** InklingEngine `0b7a1c5a`; Inkling 2 Mini UD-Q4_K_XL; **250 W target per card**; two independent instances, one stream each; **40K context**, evaluated at **32K depth**; filler text; raw continuation, no chat reasoning mode; six repetitions per card/depth, **256 generated tokens per depth**; **2026-11-24**. Normal memory, no SM clock cap and KV q6 except for the single change named in each row. Speeds are **SUMs of two instance medians**. Percentage changes compare each candidate’s decode rate with the mean of its own adjacent controls. [CSV](data/inkling-day-levers.csv).

| Configuration | Decode at 32K, SUM (tok/s) | Decode before / after (tok/s) | Decode change vs controls | Measured both GPUs (W) | Mean control power (W) |
|---|---:|---:|---:|---:|---:|
| SM clock capped at 1,650 MHz | 211.25 | 211.30 / 211.30 | −0.02% | 411.3 | 436.2 |
| Memory locked to 5,001 MHz | 137.75 | 211.25 / 211.45 | −34.82% | 374.8 | 434.3 |
| KV q8 | 208.70 | 211.35 / 211.10 | −1.20% | 436.1 | 437.6 |
| KV q4 | 214.80 | 211.10 / 211.10 | +1.75% | 437.0 | 437.2 |

The SM cap preserved speed while reducing draw in this comparison. The 5,001 MHz memory lock reduced daytime generation. KV q4's effect on quality was unmeasured.

## 4. Prefix-cache finding

HyperQwen could reuse a document immediately, yet lose its prefix after cycling through other long documents. Dense retention of hybrid-model states increased pressure on the shared cache. Changing the existing `PREFIX_RETENTION` setting preserved the first document in the measured rotation.

**Conditions:** HyperQwen `3d6c9543292e`, vLLM 0.33.0; Qwen3.8 27B W4A16; int8 per-token/head KV; DFlash2; **TP=2**; **220 W per card**; **one stream**; **131,072-token context**; **seven distinct synthetic documents**, then reread the first after the other six; first document **88,355 tokens**; reasoning **on**; **256 output-token limit**; **2026-11-20**. Only retention changed between the two conditions. [CSV](data/prefix-cache.csv).

| Retention | Cached tokens on first reread after rotation | TTFT (s) |
|---|---:|---:|
| `None` — dense | 0 | 81.04 |
| `PREFIX_RETENTION=0` | 88,128 | 0.44 |

Immediate rereads hit the cache under both settings. The dense run's initial read took **67.39 s**, and the **81.04 s** observation above came from rereading that document after six others, with zero cached tokens. Section 2 reports initial requests and a separate batch run. Here the finding concerns first-document prefix reuse after rotation. Sparse retention can trade away intermediate reuse points.

See the [public HyperQwen field report](https://github.com/syv-ai/HyperQwen/issues/482).

## 5. Quality on code review

The same **seven real-code chunks** were reviewed and each emitted finding was classified as real, false or doubtful by the same judge, **GPT 6.1 Sol**. Precision is **real / (real + false)**; doubtful findings are excluded. There is no complete defect inventory to establish recall.

**Conditions.** One request at a time, code-review prompts, temperature **0.6**, and a **64,000-token output limit**. KV precision is unavailable for these runs. The HyperQwen quality record does not independently pin its revision or context limit. [CSV](data/code-review-quality.csv)

| Model / engine | Weight quantisation | GPUs | W/card | Context (tokens) | Reasoning budget (tokens) | Date |
|---|---|---:|---:|---:|---|---|
| Inkling 2 Mini / InklingEngine `3b9e285` | UD-Q4_K_XL | 1 | 275 | 131,072 | High, capped at 16,384 | 2026-11-23 |
| Inkling 2 Mini / llama.cpp Unsloth `b12588-mix-c0c3de8`, DSpark | UD-Q4_K_XL | 1 | 200 | 98,304 | On, capped at 16,384 | 2026-11-22 |
| Qwen3.8 27B / HyperQwen, TP=2, DFlash2 | W4A16 | 2 | 200 | — | On, capped at 16,384 | 2026-11-22 |
| Inkling 2 Mini / InklingEngine `3b9e285` | UD-Q4_K_XL | 1 | 275 | 131,072 | Off | 2026-11-23 |

| Engine and reasoning | Real | False | Doubtful | Valid JSON chunks | Generation wall time (s) | Precision | Real findings / generation hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| InklingEngine, high | 13 | 18 | 5 | 7/7 | 1,350 | 41.9% | 34.7 |
| llama.cpp, on | 13 | 17 | 6 | 7/7 | 4,488.8 | 43.3% | 10.4 |
| HyperQwen, on | 8 | 11 | 12 | 6/7 | 1,378.8 | 42.1% | 20.9 |
| InklingEngine, off | 3 | 96 | 7 | 4/7 | 2,114 | 3.0% | 5.1 |

Failed chunks contributed no findings. Their generation time is included in the per-hour denominator, which excludes judging time. Three InklingEngine chunks with reasoning off reached the output cap without valid JSON. HyperQwen had one generation failure. Precision describes the findings the judge could assess and does not measure recall. The sample is too small to establish statistical equivalence, and the different power limits and dates limit engine-speed comparisons.

## 6. Measurement methodology

- **Change one setting at a time**, with a control before and after the candidate. Compare against both controls to detect drift rather than attributing every speed change to the setting.
- For repeated depth measurements, take **medians per GPU and depth**. When two independent instances are added, label the result **SUM of medians**. Keep it distinct from total tokens divided by a shared elapsed interval. The single-card snapshots above are single passes, not repeated medians.
- **Warm up after convergence** before a timed power window. The calibration used **60 seconds of warm-up and 240 seconds of measurement** per step. That does not make the sequential steps independent repeated trials.
- For the HyperQwen power sweep, use the delta of **`vllm:generation_tokens_total` divided by elapsed time**, summed over the relevant series. Isolate the workload: unrelated traffic would enter that counter too. Count generated tokens, including reasoning, rather than assuming they are visible answer tokens.
- For the short concurrency experiment with reasoning on, divide completed output-token counts by the **whole step's wall time**, including waiting. Keep those rates distinct from the engine's depth-specific decode report and from the power sweep's server counter.
- Measure **TTFT** separately from decode and completed-answer latency. Record whether it is the first reasoning delta. Check cached-token counts to distinguish cold input from prefix reuse.
- Record requested caps, effective limits, temperatures and clocks. Integrate sampled GPU power over the same window as the token count for **GPU joules per generated token**. Report warm/cool state and thermal reductions instead of silently treating targets as measured draw.

## 7. Upstream links

- [InklingEngine upstream](https://github.com/ManyGlue0/InklingEngine).
- [InklingEngine PR #2](https://github.com/ManyGlue0/InklingEngine/pull/2): reasoning budget and HTTP-server hardening.
- [HyperQwen upstream](https://github.com/syv-ai/HyperQwen).

## 8. Licence, citation and contact

Text and data are licensed under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**. See [LICENSE](LICENSE). Linked software and model weights retain their own licences.

Suggested citation:

> @phyrexia. *RTX 3090 inference benchmarks*. November 2026. https://github.com/phyrexia/3090-benchmarks. CC BY 4.0. Specify the CSV filename, measurement date and repository revision used.

Please credit **@phyrexia**, link to the repository and licence, and indicate changes. Questions and follow-up measurements: [@phyrexia on X](https://x.com/phyrexia).
