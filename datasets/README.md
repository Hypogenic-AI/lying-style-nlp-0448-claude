# Downloaded Datasets

This directory contains datasets for the lying style research project. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: TruthfulQA

### Overview
- **Source**: [HuggingFace: truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- **Size**: 817 questions (generation + multiple choice configs)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Factual QA truthfulness evaluation
- **Splits**: validation only (817 examples)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds.save_to_disk("datasets/truthfulqa/generation")
ds_mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
ds_mc.save_to_disk("datasets/truthfulqa/multiple_choice")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/truthfulqa/generation")
```

### Sample Data
See `truthfulqa/samples.json` for examples. Each record has: question, best_answer, correct_answers, incorrect_answers, category.

### Notes
- 38 categories covering health, law, finance, politics, etc.
- Questions are adversarially designed so models tend to produce misconceptions
- Primary dataset for our research: prompt LLMs to answer truthfully vs. lie using provided answers as ground truth

---

## Dataset 2: Geometry of Truth Datasets

### Overview
- **Source**: [github.com/saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth)
- **Size**: ~3,000+ true/false statements across multiple topics
- **Format**: CSV files
- **Task**: Truth/falsehood probing
- **License**: MIT (from repo)

### Download Instructions
```bash
git clone https://github.com/saprmarks/geometry-of-truth.git
cp -r geometry-of-truth/datasets/* datasets/geometry_of_truth/
```

### Key Files
- `cities.csv` — statements about cities (95 KB)
- `neg_cities.csv` — negated city statements (101 KB)
- `sp_en_trans.csv` — Spanish-English translation statements (15 KB)
- `companies_true_false.csv` — company fact statements (91 KB)
- `common_claim_true_false.csv` — common claims (292 KB)
- `counterfact_true_false.csv` — counterfactual statements (3.1 MB)

### Notes
- Clean, balanced true/false statements
- Includes negated variants (important for testing polarity effects)
- Used in Marks & Tegmark (2023) and Burger et al. (2024)

---

## Dataset 3: Truth is Universal Datasets

### Overview
- **Source**: [github.com/sciai-lab/Truth_is_Universal](https://github.com/sciai-lab/Truth_is_Universal)
- **Size**: ~45,000 statements across 6 topics × 6 variants
- **Format**: CSV files
- **Task**: Cross-topic, cross-lingual truth probing
- **License**: MIT (from repo)

### Download Instructions
```bash
git clone https://github.com/sciai-lab/Truth_is_Universal.git
cp -r Truth_is_Universal/datasets/* datasets/truth_is_universal/
```

### Notes
- Superset of Geometry of Truth datasets with expanded variants
- Includes: affirmative, negated, conjunction, disjunction, German, negated German
- Designed explicitly for lie detection experiments

---

## Dataset 4: LLM-LieDetector QA Datasets

### Overview
- **Source**: [github.com/LoryPack/LLM-LieDetector](https://github.com/LoryPack/LLM-LieDetector)
- **Size**: >20,000 questions across 11 sub-datasets
- **Format**: JSON files
- **Task**: Question answering for lie elicitation
- **License**: MIT (from repo)

### Download Instructions
```bash
git clone https://github.com/LoryPack/LLM-LieDetector.git
# Data is in LLM-LieDetector/data/
```

### Key Sub-datasets
- `wikidata.json` — WikiData general knowledge (19 MB)
- `sciq.json` — Science questions (8.7 MB)
- `commonsense_QA_v2_dev.json` — Common sense (5 MB)
- `tatoeba-eng-fra.json` — English-French translation (14 MB)
- `mathematical_problems.json` — Math (2.9 MB)
- `synthetic_facts_all.json` — Synthetic/unknowable facts (1 MB)
- Anthropic self-awareness datasets (3 files, ~1.5 MB each)

### Notes
- Questions designed so models can be prompted to lie or tell truth
- Includes 48 "elicitation questions" (probes.csv) for behavioral lie detection
- The most comprehensive QA dataset for LLM lie experiments
