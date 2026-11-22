# Paidva-Bench: Real-World AI Research & Reasoning Benchmark Suite

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Paidva-Bench** is an open, model-agnostic, statistically rigorous benchmarking suite designed to measure how AI models and research agents *actually* perform on real-world analytical work — across **8 relatable domains** and **6 core research workflows**.

> **Canary String (Contamination Prevention):** `PAIDVA_BENCH_CANARY_2026_DO_NOT_TRAIN`
> Please do not include this repository or its datasets in pretraining or fine-tuning corpora.

---

## Why Paidva-Bench?

Most popular LLM leaderboards test trivia recall or isolated synthetic puzzles on small samples without error bars. Empirical studies of how people actually use AI for research (OpenAI/NBER usage breakdown, Anthropic Science session analysis, Wiley *ExplanAItions* surveys, Meta's AbstentionBench) show a very different reality:

1. **Data analysis is #1** (34–48% of real research sessions) — and real datasets are messy (missing values, extreme outliers, Simpson's paradox, skewed distributions, prompt injection in cells).
2. **Decision support ("Asking") dominates** (51.6% of usage) — people combine local files with external evidence to make real financial, clinical, engineering, or everyday decisions.
3. **The "Verification Tax" is the real bottleneck** — models fabricate citations, misread multi-file precedence rules, and suffer from an abstention "reasoning penalty" (~24% worse refusal on unanswerable or false-premise questions).
4. **Small eval sets hide noise** — at $n = 25$, a 95% Wilson CI is $\pm 16\text{–}18\text{ pts}$! Paidva-Bench scales to **$n = 500$ public tasks** (+ a separate, unpublished 500-task private held-out split) and bridges to **full $n = 300\text{–}1,000$ public benchmarks** so overall CIs shrink to $\pm 2.7\text{–}4.4\text{ pts}$.

---

## Suite Architecture (500 Public Tasks: 8 Domains × 6 Workflows)

### By Real-World Domain
Every domain spans data analysis, multi-file docs, decision math, literature synthesis, abstention traps, and multi-turn chats:

| Domain | Tasks | Relatable Scenarios |
| :--- | :---: | :--- |
| **Healthcare, Medicine & Nutrition** | 70 | Clinical trial CSVs (unequal variance, Simpson's paradox), diagnostic PPV/NPV Bayes, NNT, retracted/contested trials |
| **Finance, Investing & Personal Money** | 70 | Return/volatility CSVs, NPV/IRR, compounding, fund memo vs. config contradictions, stock-prediction abstention traps |
| **Business, SaaS & Product Analytics** | 70 | A/B conversion lift, LTV:CAC unit economics, funnel Simpson's paradox, CUPED & A/B peeking |
| **Software, Engineering & AI/ML** | 65 | Multi-file repo/config/addendum precedence, p95 latency logs, Amdahl's Law & LLM token cost, ML paper synthesis |
| **Science, Climate & Academia** | 65 | Skewed non-parametric lab data, half-life modeling, replication crisis, IPCC AR6 & p-hacking |
| **Economics, Policy & Public Data** | 55 | Regional wage/unemployment CSVs, exact Fisher real-wage growth, CPI vs. PCE, Card & Krueger |
| **Legal, Operations & Compliance** | 50 | Multi-doc MSA/SLA/addendum contradiction resolution, uptime downtime math, SOC 2 & GDPR |
| **Everyday Decisions & Consumer Research** | 55 | EV vs. gas TCO, rent-vs-buy, consumer cost CSVs, debunking viral false premises |
| **Total (`paidva_rw_public.jsonl`)** | **500** | **95% Wilson CI: $\pm 3.8\text{–}4.4\text{ pts}$** |

### By Workflow Category
| Category | Tasks | Grading |
| :--- | :---: | :--- |
| **1. Data Analysis & Statistical Traps** | 120 | Deterministic (`scipy`/`numpy` ground truth) over real messy CSVs |
| **2. Local Multi-File Docs & Contradictions** | 100 | Deterministic cross-file precedence & exact extraction |
| **3. Quantitative & Decision Math** | 80 | Parameterized GSM-Symbolic-style templates with distractor clauses |
| **4. Literature Synthesis & Claim Auditing** | 80 | 4-point domain rubrics + citation requirements |
| **5. Abstention & Epistemic Traps** | 70 | False premises, unanswerable futures, missing files, **+ null-effect CSVs ($p > 0.40$)** |
| **6. Multi-Turn Messy Chats** | 50 | 3-turn conversations with topic switches, user retractions & pronoun resolution |

### Three Execution Tiers (Built-In)
- `--tier smoke` (**50 tasks**, stratified across all 48 domain×category cells) — fast PR / dev sanity check.
- `--tier dev` (**250 tasks**) — overnight local model iteration.
- `--tier full` (**500 tasks**) — release & publication scorecard.

---

## Quickstart: Benchmark Any Model in 60 Seconds

```bash
# 1. Validate the 500-task dataset and all CSV/workspace fixtures
python3 bench.py validate datasets/paidva_rw_public.jsonl

# 2. Run any OpenAI-compatible model (LM Studio, Ollama, Groq, OpenRouter, OpenAI, vLLM)
python3 bench.py run \
  --dataset datasets/paidva_rw_public.jsonl \
  --tier smoke \
  --base-url http://localhost:1234/v1 \
  --model your-model-name

# 3. Or run Paidva CLI directly
python3 bench.py run \
  --dataset datasets/paidva_rw_public.jsonl \
  --tier full \
  --provider paidva \
  --model paidva-v2

# 4. Print statistical scorecard (Wilson 95% CI, Cluster SE, ECE, Brier, McNemar p-value)
python3 bench.py report results/*.jsonl

# 5. Render standalone Markdown + HTML report with published frontier baselines
python3 scripts/render_report.py results/*.jsonl --baselines baselines/published_baselines.json
```

---

## Public Benchmark Bridge (Comparing Against Regular Models)

To situate your model or agent alongside popular frontier models (**GPT-4o, GPT-5, o3-mini, Claude 3.5 Sonnet, Gemini 2.5 Pro**) without paying to re-run proprietary APIs:

1. Import standardized public benchmarks (SimpleQA Verified 1,000, FRAMES 824, SciFact 300, GSM-Symbolic 500–1,000):
   ```bash
   python3 scripts/import_public_benchmarks.py --benchmark all --limit 1000
   ```
2. Run your model on `datasets/public/*.jsonl` with `bench.py run`.
3. Compare against `baselines/published_baselines.json` — which **strictly separates** *Measured (in-harness)* from *Published (external, not re-run)* and discloses tooling parity (`Bare LLM` vs. `Agent + Sandbox + Search`).

---

## Statistical Rigor

Every report generated by `bench.py` computes:
- **95% Wilson Score Confidence Intervals** (accurate at both small and large $n$).
- **Cluster-Adjusted Standard Errors** (Evan Miller, 2024) for questions sharing a workspace or source cluster.
- **Exact Two-Sided McNemar $p$-values** for paired model-vs-model comparisons on shared items.
- **Calibration Metrics:** Expected Calibration Error (ECE), Brier Score, and Top-50% Selective Accuracy.

---

## Verification & Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
