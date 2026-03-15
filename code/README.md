# Cloned Repositories

## 1. Truth_is_Universal
- **URL**: https://github.com/sciai-lab/Truth_is_Universal
- **Paper**: "Truth is Universal: Robust Detection of Lies in LLMs" (NeurIPS 2024)
- **Location**: `code/Truth_is_Universal/`
- **Purpose**: 2D truth-sensitive probe for deception detection across models
- **Key files**: Jupyter notebooks, classifiers (TTPD, LR, CCS), datasets
- **Dependencies**: PyTorch, transformers, scikit-learn
- **Notes**: Contains datasets of true/false statements with expanded variants. Requires downloading LLaMA/Gemma weights separately for activation extraction.

## 2. LLM-LieDetector
- **URL**: https://github.com/LoryPack/LLM-LieDetector
- **Paper**: "How to Catch an AI Liar" (ICLR 2024)
- **Location**: `code/LLM-LieDetector/`
- **Purpose**: Black-box lie detection through behavioral probing
- **Key files**:
  - `data/probes.csv` — 48 elicitation questions
  - `data/processed_questions/` — QA datasets
  - Notebooks for running lie detection experiments
- **Dependencies**: OpenAI API, transformers
- **Notes**: Contains extensive QA datasets and lie-generation prompts. Can be used to generate lying vs. truthful text from multiple models.

## 3. geometry-of-truth
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Paper**: "The Geometry of Truth" (Marks & Tegmark, 2023)
- **Location**: `code/geometry-of-truth/`
- **Purpose**: Probing linear truth structure in LLM representations
- **Key files**:
  - `datasets/` — True/false statement datasets (cities, companies, etc.)
  - `acts/` — Pre-computed LLaMA activations
  - Probe training and visualization scripts
- **Dependencies**: PyTorch, transformers
- **Notes**: Foundational codebase for truth direction probing. Datasets are directly usable for our experiments.

## 4. representation-engineering
- **URL**: https://github.com/andyzoujm/representation-engineering
- **Paper**: "Representation Engineering" (Zou et al., 2023)
- **Location**: `code/representation-engineering/`
- **Purpose**: Top-down approach to analyzing and steering LLM representations
- **Key files**: RepReading and RepControl pipelines, evaluation framework
- **Dependencies**: PyTorch, transformers
- **Notes**: Most comprehensive toolkit for representation-level analysis. Can be used to extract and compare truthful vs. deceptive representations.
