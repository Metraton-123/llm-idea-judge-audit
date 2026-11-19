# Findings summary

## Design

- 6 generation conditions: `vanilla`, `high_temperature`, `generic_diversity`,
  `seed_similarity`, `generated_history`, `dual_memory`.
- 5 topics x 5 seed papers x 6 conditions x 5 ideas = **750 ideas** (150 generation calls).
- The final data contains 100 non-memory calls from the original batch and 50 memory
  calls from a repaired re-run (the first memory run had an empty memory injection).
- AI judging: 3 rounds x 4 dimensions per idea = **2,250 calls / 9,000 scores**
  (qwen3.5-flash in the main study; qwen3.8-flash and GLM-5.3 for cross-model checks).
- Human evaluation: calibration 15 pairs, validation 10 pairs, formal 100 pairs;
  3 non-expert raters, blind, with a frozen rubric.

## Main results

### 1. The AI judge scale is compressed

For qwen3.5-flash novelty:

- 73.0% of round-level scores are the single value **7**;
- only **5 integer values** (4-8) ever occur;
- 61.5% of ideas receive the same novelty score in all three rounds.

Averaging over more rounds cannot recover resolution that the scale does not have.

### 2. The non-expert panel does not provide a valid criterion

On the 10 validation pairs:

- unanimous M1 agreement: **1/10**;
- Fleiss' kappa **-0.111**, Krippendorff's alpha **-0.074**;
- two-rater subsets: 6/10, 4/10, 2/10 (Cohen's kappa .298, -.111, -.096);
- M3' cross-source: 0/9;
- M4 ICC(2,3): left -0.158, right 0.418; per-rater L-R bias +0.60 / -0.40 / -0.50.

The verdict depends on which two raters are selected, not on the ideas.

### 3. The comparison is not estimable at this scale

Simulation on the observed marginal distributions shows that at n = 10-15 pairs and
3-5 ordinal levels, power to detect a true correlation of .7 is only **.46-.78**;
the observed correlation is attenuated accordingly. A null human-judge correlation is a
low-power result, not evidence that the judge is unrelated to human judgment.

### 4. Cross-model checks locate a single-judge risk

- qwen3.5 novelty: 73.0% at 7, 5 values;
- qwen3.8-flash: 40.9% at 6, 6 values;
- GLM-5.3: 42.1% at 6, 7 values;
- qwen3.8-flash vs GLM-5.3 Spearman **.822** (novelty), .793 (feasibility),
  .596 (soundness), .681 (significance);
- qwen3.5 vs either newer judge is approximately zero (-.056 to +.046).

The extreme compression is version-specific; the newer judges share a signal that the
old judge does not. This reinforces the need for multi-judge audits rather than a
single-judge conclusion.

### 5. Specificity anchor pattern replicates in direction

Correlations with the structural specificity index (standard tier) are weak but
directionally consistent across judges:

| dimension | qwen3.5 | qwen3.8-flash | GLM-5.3 |
|---|---:|---:|---:|
| Feasibility | .103 | .151 | .090 |
| Soundness | .098 | .060 | .087 |
| Novelty | -.058 | -.101 | -.070 |
| Significance | -.054 | -.063 | -.061 |

## What this does NOT show

- It does **not** show that AI scores are invalid; the human criterion failed first.
- It does **not** show that all non-expert evaluation is useless; it is scoped to
  this panel, rubric, sample size and instrument.
- It does **not** identify a causal effect of memory conditions; the memory bank was
  selected by the same judge's novelty scores and was very small.

## Practical recommendation

Use a stop-rule before collecting human validation data: if the planned panel size and
ordinal resolution cannot reach a target power, do not interpret a null as evidence of
no association. Prefer expert anchors or objective anchors when they are available.
