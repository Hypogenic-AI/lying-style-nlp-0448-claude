# Research Plan: Lying Style in Language Models

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding whether LLMs produce detectably different text when lying has direct implications for AI safety, misinformation detection, and trust in AI systems. If lying creates a distinct "stylistic fingerprint" in surface-level text, this enables black-box detection of deceptive outputs without requiring access to model internals — making it deployable in real-world settings where users interact with closed-source models.

### Gap in Existing Work
Prior work has extensively studied **internal representations** of truth/lies (Azaria 2023, Marks 2023, Burger 2024) showing >90% detection from activations. Behavioral probing (Pacchiardi 2024) detects lies via follow-up questions. However, **surface-level distributional analysis of lying text from modern instruction-tuned models is largely unstudied**. Schuster (2019) found no stylometric differences — but used GPT-2-era models. Modern RLHF-tuned models, trained extensively on truthfulness, may produce detectable artifacts when forced to lie. No study has systematically compared multiple lying induction methods (direct instruction, roleplaying, jailbreaking) and analyzed the resulting text distributions.

### Our Novel Contribution
We conduct the first systematic, surface-level distributional analysis of text produced by modern LLMs under truthful vs. multiple deceptive conditions (direct lying instruction, roleplaying as a liar, jailbreak-style prompts). We measure whether the text distribution shifts are statistically detectable using lexical, syntactic, and semantic features — without requiring model internals.

### Experiment Justification
- **Experiment 1 (Data Collection):** Prompt a modern LLM with factual questions under 3 conditions (truthful, direct lie instruction, roleplaying liar). Needed to create matched truthful/deceptive text pairs.
- **Experiment 2 (Feature Analysis):** Extract linguistic features (sentence length, vocabulary diversity, hedging markers, certainty language, function words) and test for distributional differences. Needed to determine *which* features differ.
- **Experiment 3 (Classifier-based Detection):** Train a simple classifier to distinguish truthful from deceptive text. Needed to measure *how detectable* the difference is overall.
- **Experiment 4 (Cross-condition Analysis):** Compare distributions across lying methods. Needed to determine if different deception methods produce different stylistic signatures.

## Research Question
When a language model is instructed to lie (via direct instruction, roleplaying, or jailbreak prompts), does the resulting text distribution differ measurably from truthful text in terms of surface-level linguistic features?

## Hypothesis Decomposition
- **H1:** LLM-generated deceptive text differs from truthful text in lexical features (vocabulary diversity, word frequency distributions).
- **H2:** Deceptive text shows different syntactic patterns (sentence length, complexity).
- **H3:** Deceptive text contains more hedging language, qualifiers, or uncertainty markers.
- **H4:** Different lying induction methods produce distinguishable text distributions from each other.
- **H5:** A simple classifier can distinguish truthful from deceptive LLM text above chance.

## Proposed Methodology

### Approach
Use the OpenAI API (GPT-4.1) to generate responses to 150 factual questions from TruthfulQA and Geometry of Truth datasets under 3 conditions:
1. **Truthful**: Standard prompt asking for factual answer
2. **Direct lie**: Explicitly instructed to give an incorrect/false answer
3. **Roleplay lie**: Prompted to roleplay as a character who always lies

For each condition, collect the full text response. Then extract linguistic features and perform statistical comparisons.

### Why GPT-4.1?
- State-of-the-art instruction-following model
- Will actually follow lying instructions (unlike heavily safety-filtered models that refuse)
- Represents the class of models users actually interact with
- API access enables large-scale data collection

### Experimental Steps
1. Curate 150 factual questions (100 from TruthfulQA, 50 from Geometry of Truth)
2. Design 3 prompt templates (truthful, direct lie, roleplay lie)
3. Collect responses (150 questions × 3 conditions = 450 responses)
4. Extract features: sentence length, word count, type-token ratio, hedging markers, certainty markers, function word ratios, sentiment scores
5. Statistical tests: Mann-Whitney U, permutation tests, effect sizes (Cohen's d)
6. Train logistic regression classifier on feature vectors
7. Visualize distributions and differences

### Baselines
- Random classifier (50% accuracy)
- Majority class classifier
- Single-feature classifiers (each feature alone)

### Evaluation Metrics
- **Distributional:** Jensen-Shannon divergence, KL divergence between feature distributions
- **Statistical:** p-values, effect sizes (Cohen's d), confidence intervals
- **Classification:** Accuracy, F1, AUC-ROC for truthful vs. deceptive detection

### Statistical Analysis Plan
- Non-parametric tests (Mann-Whitney U) since distributions may be non-normal
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals for effect sizes
- Permutation tests for classifier performance
- Significance level: α = 0.05 (Bonferroni-corrected)

## Expected Outcomes
- **If hypothesis supported:** Deceptive text will show measurably different feature distributions (e.g., longer sentences, more hedging, lower vocabulary diversity), and a classifier will achieve >60% accuracy.
- **If hypothesis refuted:** Feature distributions will be statistically indistinguishable, classifier near chance level, supporting Schuster (2019)'s finding that LMs don't change style when lying.
- **Partial support:** Some features differ but others don't; lying method matters.

## Timeline and Milestones
1. Planning: 15 min ✓
2. Environment setup: 10 min
3. Data collection (API calls): 30 min
4. Feature extraction: 20 min
5. Statistical analysis: 20 min
6. Visualization: 15 min
7. Documentation: 20 min

## Potential Challenges
- Model may refuse to lie under some prompts → use multiple induction methods
- API rate limits → batch with delays
- Small sample sizes → use non-parametric tests, bootstrap CIs
- Content confound (different facts stated) → analyze style features independent of content

## Success Criteria
1. Successfully collect 400+ responses across conditions
2. Complete statistical comparison of feature distributions
3. Report effect sizes and significance for each feature
4. Train and evaluate classifier
5. Clear answer to whether lying produces detectable distributional shift
