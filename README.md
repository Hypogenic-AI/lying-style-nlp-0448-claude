# Lying Style: Do LLMs Write Differently When They Lie?

Investigating whether the text distribution of a language model shifts detectably when it is instructed to lie (via direct instruction or roleplaying) compared to normal truthful output.

## Key Findings

- **Yes, lying text is strongly distinguishable.** A logistic regression on 21 linguistic features achieves **90.0% accuracy** (chance = 50%, permutation p = 0.005) at detecting deceptive LLM text.
- **Lies are shorter and more assertive.** Truthful responses average 42 words with hedging and caveats; lies average 24-30 words with high certainty language and no hedging.
- **Different lying methods produce different styles.** Roleplay lies are theatrical and over-confident (exclamation marks, superlatives); direct lies are subtler. A 3-class classifier achieves 84.4% accuracy (chance = 33%).
- **Modern models differ from GPT-2-era findings.** Schuster et al. (2019) found no stylometric differences in GPT-2; we show modern RLHF-tuned models have distinct truthful vs. deceptive writing modes.

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas scikit-learn scipy matplotlib seaborn datasets textstat

# Collect responses (requires OPENAI_API_KEY)
python src/collect_responses.py

# Run analysis
python src/analyze_responses.py
```

## File Structure

```
├── REPORT.md                   # Full research report with results
├── planning.md                 # Research plan and methodology
├── src/
│   ├── collect_responses.py    # GPT-4.1 data collection (3 conditions × 150 questions)
│   └── analyze_responses.py    # Feature extraction, stats, classification, plots
├── results/
│   ├── raw_responses.json      # 450 raw LLM responses
│   ├── features.csv            # Extracted features for all responses
│   ├── statistical_tests.csv   # Mann-Whitney U tests with Bonferroni correction
│   ├── classification_results.json  # Classifier accuracy, AUC, permutation test
│   └── plots/                  # 6 publication-quality visualizations
├── datasets/                   # TruthfulQA, Geometry of Truth (pre-downloaded)
├── papers/                     # 23 related research papers
└── literature_review.md        # Synthesized literature review
```

See [REPORT.md](REPORT.md) for the full analysis.
