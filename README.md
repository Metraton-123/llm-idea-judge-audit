# llm-idea-judge-audit

Code and reproducible analysis for auditing **LLM judges of research ideas** with small human panels.

This repository distils the core pipeline from a summer research internship into a small,
readable project:

- a frozen four-dimensional scoring rubric;
- an OpenAI-compatible client (Qwen / GLM / DeepSeek) with env-only API keys;
- a stratified pair sampler for calibration / validation / formal human evaluation;
- scoring + aggregation + cross-model / anchor-validity analysis.

## What the study found

The full study used **750 generated research ideas** across 6 generation conditions and
5 topics, with 3 rounds x 4 dimensions of LLM judging and a 3-rater blind human evaluation.

Key observations (see `docs/FINDINGS.md`):

- The main judge (qwen3.5-flash) collapsed its novelty scale: **73%** of scores were the
  single value 7, and only **5 integer values** were used.
- A 3-rater non-expert panel agreed on only **1/10** validation pairs; Fleiss' kappa was
  **-0.111** (Krippendorff's alpha **-0.074**), and 2-rater subsets ranged from 6/10 to 2/10.
- Cross-model checks showed that newer judges (qwen3.8-flash, GLM-5.3) agreed with each
  other (Spearman **rho = .82** for novelty) but were nearly uncorrelated with qwen3.5.
- Simulation showed that at n = 10-15 and 3-5 ordinal levels, power to detect a true
  correlation of .7 was only **.46-.78**; a null is therefore not evidence of no association.

**Conclusion:** the study does *not* show that AI scores are invalid. It shows that a
small non-expert panel is an unreliable criterion, so the validation question is
**not estimable** with that design. The paper frames this as a *non-expert validation gap*
and proposes a stop-rule for screening validation designs before data collection.

## Install

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (no API calls)

```bash
# 1) sample pairs from the included synthetic data
python scripts/sample_pairs.py --input data/sample_ideas.csv --out out/pairs.csv

# 2) dry-run scoring (no network)
python scripts/score_ideas.py --input data/sample_ideas.csv --config configs/judges.example.json --dry-run

# 3) inspect output schema
ls out
```

## Real scoring

Set keys as environment variables (never commit them):

```bash
export DSH_QWEN_API_KEY=...       # DashScope / qwen
export DSH_ZHIPU_API_KEY=...      # Zhipu / GLM
```

Then:

```bash
python scripts/score_ideas.py \
  --input data/sample_ideas.csv \
  --config configs/judges.example.json \
  --models qwen38flash glm53 \
  --out out \
  --execute
```

The script writes `out/calls_<model>.csv` (per idea x round), `out/calls_<model>.jsonl`
(usage/errors), and `out/scored_<model>_aggregate.csv` (per-idea means/variances).

## Analysis

```bash
python scripts/analyze_scores.py \
  --calls qwen38flash=out/calls_qwen38flash.csv glm53=out/calls_glm53.csv \
  --aggregates qwen38flash=out/scored_qwen38flash_aggregate.csv glm53=out/scored_glm53_aggregate.csv \
  --out out/analysis
```

Optional anchor validity:

```bash
python scripts/analyze_scores.py ... \
  --anchor data/anchors.csv --anchor-col index_standard
```

## Input format

`data/sample_ideas.csv` shows the required columns:

`call_id, group_id/topic, setting, idea_index, idea_title, research_question,
method_family, contribution_type, description`

Optional: `round1_order, round2_order, round3_order` (replay a stored dimension order);
otherwise a fixed order is used.

## Repository layout

```
src/idea_audit/
  prompts.py     frozen rubric + prompt builder
  client.py      OpenAI-compatible client, env-only keys
  scoring.py     prompt construction, scoring, aggregation
  sampling.py    stratified pair sampler + design checks
  analysis.py    distributions, Spearman, anchor correlations
  stats.py       Krippendorff alpha / ICC
scripts/         CLI entry points
tests/           unit tests (no network)
docs/FINDINGS.md methodology and results summary
```

## Data availability

The full 750-idea dataset and human annotations are **not included** here (manuscript in
preparation; request from the author). The bundled `data/sample_ideas.csv` is synthetic
and only for exercising the pipeline. Do **not** expect the sample to reproduce the paper's
numbers.

## Safety / privacy

- API keys are read from environment variables only; no key is stored in this repo.
- Logs contain prompts and model outputs; avoid sending confidential ideas.
- Human data, if added later, must be anonymized and consented.

## License

MIT for code. Data and manuscript content are not covered by the code license.

## Citation

If you use this code, please cite the manuscript once available. Until then, link to this
repository and contact the author.
