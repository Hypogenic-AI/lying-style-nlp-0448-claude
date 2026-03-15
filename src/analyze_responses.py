"""
Feature extraction and statistical analysis of truthful vs deceptive LLM responses.
Extracts lexical, syntactic, and semantic features, then compares distributions.
"""

import json
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import textstat

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Hedging and certainty markers ---
HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "may", "likely",
    "unlikely", "probably", "somewhat", "arguably", "supposedly",
    "allegedly", "apparently", "seemingly", "roughly", "approximately",
    "generally", "typically", "usually", "often", "sometimes",
    "it seems", "it appears", "in some cases", "to some extent",
}

CERTAINTY_WORDS = {
    "certainly", "definitely", "absolutely", "clearly", "obviously",
    "undoubtedly", "without doubt", "surely", "indeed", "truly",
    "always", "never", "every", "all", "none", "must", "proven",
    "confirmed", "established", "known", "fact", "actually",
    "specifically", "exactly", "precisely",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nothing", "nowhere", "n't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't", "shouldn't"}


def extract_features(text: str) -> dict:
    """Extract linguistic features from a single response text."""
    if not text or not text.strip():
        return None

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return None

    words_lower = [w.lower().strip(".,!?;:\"'()[]") for w in words]
    words_clean = [w for w in words_lower if w]
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = max(len(sentences), 1)

    # Lexical features
    unique_words = set(words_clean)
    type_token_ratio = len(unique_words) / max(len(words_clean), 1)

    # Word length distribution
    word_lengths = [len(w) for w in words_clean if w]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0

    # Sentence features
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = np.mean(sent_lengths) if sent_lengths else 0
    std_sentence_length = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # Hedging and certainty
    text_lower = text.lower()
    hedging_count = sum(1 for h in HEDGING_WORDS if h in text_lower)
    certainty_count = sum(1 for c in CERTAINTY_WORDS if c in text_lower)

    # Negation
    negation_count = sum(1 for w in words_clean if w in NEGATION_WORDS)

    # Function words (approximation)
    function_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "of", "in", "to", "for", "with", "on", "at", "from", "by",
                      "as", "it", "that", "this", "which", "who", "whom", "what",
                      "and", "or", "but", "if", "then", "than", "when", "where"}
    function_word_ratio = sum(1 for w in words_clean if w in function_words) / max(len(words_clean), 1)

    # Punctuation
    comma_count = text.count(",")
    exclamation_count = text.count("!")
    question_mark_count = text.count("?")
    parenthesis_count = text.count("(") + text.count(")")
    quote_count = text.count('"') + text.count("'") // 2  # rough

    # Readability
    try:
        flesch_reading = textstat.flesch_reading_ease(text)
    except:
        flesch_reading = 50.0

    # First person pronouns (may indicate different framing)
    first_person = sum(1 for w in words_clean if w in {"i", "me", "my", "mine", "we", "us", "our", "ours"})

    # Superlatives and intensifiers
    intensifiers = {"very", "extremely", "incredibly", "remarkably", "particularly",
                    "especially", "exceptionally", "highly", "deeply", "strongly",
                    "most", "worst", "best", "greatest", "largest", "smallest",
                    "single", "entire", "whole", "complete", "total", "utter",
                    "every", "all", "famous", "famously", "known", "renowned",
                    "notorious"}
    intensifier_count = sum(1 for w in words_clean if w in intensifiers)

    return {
        "word_count": word_count,
        "num_sentences": num_sentences,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "type_token_ratio": type_token_ratio,
        "avg_word_length": avg_word_length,
        "hedging_count": hedging_count,
        "hedging_rate": hedging_count / num_sentences,
        "certainty_count": certainty_count,
        "certainty_rate": certainty_count / num_sentences,
        "negation_count": negation_count,
        "negation_rate": negation_count / max(word_count, 1),
        "function_word_ratio": function_word_ratio,
        "comma_rate": comma_count / max(word_count, 1),
        "exclamation_count": exclamation_count,
        "question_mark_count": question_mark_count,
        "parenthesis_count": parenthesis_count,
        "flesch_reading_ease": flesch_reading,
        "first_person_count": first_person,
        "first_person_rate": first_person / max(word_count, 1),
        "intensifier_count": intensifier_count,
        "intensifier_rate": intensifier_count / max(word_count, 1),
    }


def load_and_extract():
    """Load responses and extract features for all."""
    with open(RESULTS_DIR / "raw_responses.json") as f:
        responses = json.load(f)

    records = []
    for r in responses:
        feats = extract_features(r["response"])
        if feats is None:
            continue
        feats["question_id"] = r["question_id"]
        feats["condition"] = r["condition"]
        feats["source"] = r["source"]
        feats["category"] = r["category"]
        feats["response"] = r["response"]
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"Extracted features for {len(df)} responses")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    return df


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compare feature distributions between truthful and each lying condition."""
    feature_cols = [c for c in df.columns if c not in
                    {"question_id", "condition", "source", "category", "response"}]

    results = []
    truthful = df[df["condition"] == "truthful"]

    for lie_cond in ["direct_lie", "roleplay_lie"]:
        lying = df[df["condition"] == lie_cond]

        for feat in feature_cols:
            truth_vals = truthful[feat].values
            lie_vals = lying[feat].values

            # Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(truth_vals, lie_vals, alternative="two-sided")
            except:
                u_stat, p_value = np.nan, 1.0

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(truth_vals)**2 + np.std(lie_vals)**2) / 2)
            cohens_d = (np.mean(lie_vals) - np.mean(truth_vals)) / pooled_std if pooled_std > 0 else 0

            results.append({
                "feature": feat,
                "comparison": f"truthful_vs_{lie_cond}",
                "truth_mean": np.mean(truth_vals),
                "truth_std": np.std(truth_vals),
                "lie_mean": np.mean(lie_vals),
                "lie_std": np.std(lie_vals),
                "cohens_d": cohens_d,
                "mann_whitney_u": u_stat,
                "p_value": p_value,
            })

    # Also compare direct_lie vs roleplay_lie
    direct = df[df["condition"] == "direct_lie"]
    roleplay = df[df["condition"] == "roleplay_lie"]
    for feat in feature_cols:
        d_vals = direct[feat].values
        r_vals = roleplay[feat].values
        try:
            u_stat, p_value = stats.mannwhitneyu(d_vals, r_vals, alternative="two-sided")
        except:
            u_stat, p_value = np.nan, 1.0
        pooled_std = np.sqrt((np.std(d_vals)**2 + np.std(r_vals)**2) / 2)
        cohens_d = (np.mean(r_vals) - np.mean(d_vals)) / pooled_std if pooled_std > 0 else 0
        results.append({
            "feature": feat,
            "comparison": "direct_lie_vs_roleplay_lie",
            "truth_mean": np.mean(d_vals),
            "truth_std": np.std(d_vals),
            "lie_mean": np.mean(r_vals),
            "lie_std": np.std(r_vals),
            "cohens_d": cohens_d,
            "mann_whitney_u": u_stat,
            "p_value": p_value,
        })

    results_df = pd.DataFrame(results)

    # Bonferroni correction
    n_tests = len(results_df)
    results_df["p_bonferroni"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
    results_df["significant_bonf"] = results_df["p_bonferroni"] < 0.05

    return results_df


def classification_experiment(df: pd.DataFrame) -> dict:
    """Train classifiers to distinguish truthful from lying responses."""
    feature_cols = [c for c in df.columns if c not in
                    {"question_id", "condition", "source", "category", "response"}]

    results = {}

    # Binary: truthful vs all lies
    df_binary = df.copy()
    df_binary["label"] = (df_binary["condition"] != "truthful").astype(int)

    X = df_binary[feature_cols].values
    y = df_binary["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")
    lr_f1 = cross_val_score(lr, X_scaled, y, cv=cv, scoring="f1")
    lr_auc = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")

    results["truthful_vs_all_lies"] = {
        "logistic_regression": {
            "accuracy": f"{lr_scores.mean():.3f} ± {lr_scores.std():.3f}",
            "f1": f"{lr_f1.mean():.3f} ± {lr_f1.std():.3f}",
            "auc_roc": f"{lr_auc.mean():.3f} ± {lr_auc.std():.3f}",
            "accuracy_raw": lr_scores.mean(),
        }
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    rf_f1 = cross_val_score(rf, X_scaled, y, cv=cv, scoring="f1")
    rf_auc = cross_val_score(rf, X_scaled, y, cv=cv, scoring="roc_auc")

    results["truthful_vs_all_lies"]["random_forest"] = {
        "accuracy": f"{rf_scores.mean():.3f} ± {rf_scores.std():.3f}",
        "f1": f"{rf_f1.mean():.3f} ± {rf_f1.std():.3f}",
        "auc_roc": f"{rf_auc.mean():.3f} ± {rf_auc.std():.3f}",
        "accuracy_raw": rf_scores.mean(),
    }

    # Feature importance from full-data LR
    lr.fit(X_scaled, y)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    results["feature_importance_lr"] = importance.to_dict("records")

    # RF feature importance
    rf.fit(X_scaled, y)
    rf_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    results["feature_importance_rf"] = rf_importance.to_dict("records")

    # 3-class: truthful vs direct_lie vs roleplay_lie
    df_3class = df.copy()
    label_map = {"truthful": 0, "direct_lie": 1, "roleplay_lie": 2}
    df_3class["label"] = df_3class["condition"].map(label_map)
    X3 = df_3class[feature_cols].values
    y3 = df_3class["label"].values
    X3_scaled = scaler.fit_transform(X3)

    lr3 = LogisticRegression(max_iter=1000, random_state=42)
    lr3_scores = cross_val_score(lr3, X3_scaled, y3, cv=cv, scoring="accuracy")
    results["three_class"] = {
        "logistic_regression_accuracy": f"{lr3_scores.mean():.3f} ± {lr3_scores.std():.3f}",
        "accuracy_raw": lr3_scores.mean(),
        "chance_level": "0.333",
    }

    # Pairwise: truthful vs direct_lie only
    df_td = df[df["condition"].isin(["truthful", "direct_lie"])].copy()
    df_td["label"] = (df_td["condition"] == "direct_lie").astype(int)
    X_td = scaler.fit_transform(df_td[feature_cols].values)
    y_td = df_td["label"].values
    lr_td = LogisticRegression(max_iter=1000, random_state=42)
    td_scores = cross_val_score(lr_td, X_td, y_td, cv=cv, scoring="accuracy")
    td_auc = cross_val_score(lr_td, X_td, y_td, cv=cv, scoring="roc_auc")
    results["truthful_vs_direct_lie"] = {
        "accuracy": f"{td_scores.mean():.3f} ± {td_scores.std():.3f}",
        "auc_roc": f"{td_auc.mean():.3f} ± {td_auc.std():.3f}",
    }

    # Pairwise: truthful vs roleplay_lie
    df_tr = df[df["condition"].isin(["truthful", "roleplay_lie"])].copy()
    df_tr["label"] = (df_tr["condition"] == "roleplay_lie").astype(int)
    X_tr = scaler.fit_transform(df_tr[feature_cols].values)
    y_tr = df_tr["label"].values
    lr_tr = LogisticRegression(max_iter=1000, random_state=42)
    tr_scores = cross_val_score(lr_tr, X_tr, y_tr, cv=cv, scoring="accuracy")
    tr_auc = cross_val_score(lr_tr, X_tr, y_tr, cv=cv, scoring="roc_auc")
    results["truthful_vs_roleplay_lie"] = {
        "accuracy": f"{tr_scores.mean():.3f} ± {tr_scores.std():.3f}",
        "auc_roc": f"{tr_auc.mean():.3f} ± {tr_auc.std():.3f}",
    }

    # Permutation test for significance of truthful vs all lies
    from sklearn.model_selection import permutation_test_score
    lr_perm = LogisticRegression(max_iter=1000, random_state=42)
    df_binary2 = df.copy()
    df_binary2["label"] = (df_binary2["condition"] != "truthful").astype(int)
    X_perm = scaler.fit_transform(df_binary2[feature_cols].values)
    y_perm = df_binary2["label"].values
    score, perm_scores, perm_p = permutation_test_score(
        lr_perm, X_perm, y_perm, cv=cv, n_permutations=200, random_state=42, scoring="accuracy"
    )
    results["permutation_test"] = {
        "observed_accuracy": f"{score:.3f}",
        "permutation_p_value": f"{perm_p:.4f}",
        "mean_null_accuracy": f"{np.mean(perm_scores):.3f}",
    }

    return results


def create_visualizations(df: pd.DataFrame, stats_df: pd.DataFrame, clf_results: dict):
    """Create publication-quality visualizations."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    colors = {"truthful": "#2ecc71", "direct_lie": "#e74c3c", "roleplay_lie": "#9b59b6"}

    # 1. Key feature distributions
    key_features = ["word_count", "avg_sentence_length", "hedging_rate",
                    "certainty_rate", "negation_rate", "intensifier_rate",
                    "type_token_ratio", "flesch_reading_ease"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 4, idx % 4]
        for cond in ["truthful", "direct_lie", "roleplay_lie"]:
            data = df[df["condition"] == cond][feat]
            ax.hist(data, bins=20, alpha=0.5, label=cond, color=colors[cond], density=True)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Feature Distributions: Truthful vs. Lying Conditions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Effect sizes heatmap
    pivot_features = ["word_count", "avg_sentence_length", "hedging_rate", "certainty_rate",
                      "negation_rate", "intensifier_rate", "type_token_ratio",
                      "function_word_ratio", "flesch_reading_ease", "first_person_rate",
                      "comma_rate", "exclamation_count"]

    effect_data = stats_df[stats_df["feature"].isin(pivot_features)].copy()
    pivot = effect_data.pivot_table(index="feature", columns="comparison", values="cohens_d")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Cohen's d Effect Sizes by Feature and Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "effect_sizes_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Significant features bar chart
    sig_df = stats_df[stats_df["comparison"].str.startswith("truthful")].copy()
    sig_df["abs_d"] = sig_df["cohens_d"].abs()
    sig_df = sig_df.sort_values("abs_d", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in sig_df["p_bonferroni"]]
    ax.barh(range(len(sig_df)), sig_df["abs_d"], color=bar_colors)
    ax.set_yticks(range(len(sig_df)))
    ax.set_yticklabels([f"{r['feature']} ({r['comparison'].split('_vs_')[1]})"
                        for _, r in sig_df.iterrows()], fontsize=8)
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5, label="Small effect (0.2)")
    ax.axvline(x=0.5, color="gray", linestyle="-.", alpha=0.5, label="Medium effect (0.5)")
    ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="Large effect (0.8)")
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Effect Sizes: Truthful vs. Lying (Red = Bonferroni significant)", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "effect_sizes_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Box plots for key features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    box_features = ["word_count", "hedging_rate", "certainty_rate",
                    "negation_rate", "intensifier_rate", "avg_sentence_length"]
    for idx, feat in enumerate(box_features):
        ax = axes[idx // 3, idx % 3]
        data_to_plot = [df[df["condition"] == c][feat].values for c in ["truthful", "direct_lie", "roleplay_lie"]]
        bp = ax.boxplot(data_to_plot, labels=["Truthful", "Direct Lie", "Roleplay"],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], [colors["truthful"], colors["direct_lie"], colors["roleplay_lie"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel(feat)
    plt.suptitle("Feature Distributions by Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Feature importance plot
    if "feature_importance_lr" in clf_results:
        imp = pd.DataFrame(clf_results["feature_importance_lr"])
        imp = imp.sort_values("abs_coefficient", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ["#e74c3c" if c > 0 else "#3498db" for c in imp["coefficient"]]
        ax.barh(imp["feature"], imp["coefficient"], color=bar_colors)
        ax.set_xlabel("Logistic Regression Coefficient")
        ax.set_title("Top Features for Detecting Deceptive Text (LR)", fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Violin plots for certainty vs hedging
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, feat in enumerate(["certainty_rate", "hedging_rate"]):
        ax = axes[idx]
        for cond in ["truthful", "direct_lie", "roleplay_lie"]:
            data = df[df["condition"] == cond][feat]
            parts = ax.violinplot([data.values], positions=[list(colors.keys()).index(cond)],
                                  showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[cond])
                pc.set_alpha(0.6)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Truthful", "Direct Lie", "Roleplay"])
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel(feat)
    plt.suptitle("Certainty vs Hedging by Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "certainty_vs_hedging.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {len(list(PLOTS_DIR.glob('*.png')))} plots to {PLOTS_DIR}")


def main():
    print("=" * 60)
    print("LYING STYLE ANALYSIS")
    print("=" * 60)

    # 1. Load and extract features
    print("\n--- Feature Extraction ---")
    df = load_and_extract()

    # Save features
    df.to_csv(RESULTS_DIR / "features.csv", index=False)

    # 2. Descriptive statistics
    print("\n--- Descriptive Statistics ---")
    feature_cols = [c for c in df.columns if c not in
                    {"question_id", "condition", "source", "category", "response"}]
    desc = df.groupby("condition")[feature_cols].agg(["mean", "std"]).round(3)
    print(desc.T.to_string())

    # 3. Statistical tests
    print("\n--- Statistical Tests ---")
    stats_df = statistical_tests(df)
    stats_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    # Show significant results
    sig = stats_df[stats_df["significant_bonf"]]
    print(f"\nSignificant features (Bonferroni-corrected p < 0.05): {len(sig)}")
    if len(sig) > 0:
        for _, row in sig.iterrows():
            print(f"  {row['comparison']}: {row['feature']} "
                  f"(d={row['cohens_d']:.3f}, p_bonf={row['p_bonferroni']:.4f})")

    # Show largest effect sizes
    print("\nTop 10 effect sizes (|Cohen's d|):")
    top_effects = stats_df.copy()
    top_effects["abs_d"] = top_effects["cohens_d"].abs()
    top_effects = top_effects.sort_values("abs_d", ascending=False).head(10)
    for _, row in top_effects.iterrows():
        sig_marker = "*" if row["p_bonferroni"] < 0.05 else ""
        print(f"  {row['comparison']}: {row['feature']} "
              f"d={row['cohens_d']:.3f} p_bonf={row['p_bonferroni']:.4f}{sig_marker}")

    # 4. Classification
    print("\n--- Classification Experiments ---")
    clf_results = classification_experiment(df)

    print(f"\nTruthful vs All Lies (binary):")
    for model_name, model_res in clf_results["truthful_vs_all_lies"].items():
        print(f"  {model_name}: acc={model_res['accuracy']}, f1={model_res['f1']}, auc={model_res['auc_roc']}")

    print(f"\n3-class (truthful/direct/roleplay):")
    print(f"  LR accuracy: {clf_results['three_class']['logistic_regression_accuracy']} (chance=0.333)")

    print(f"\nPairwise:")
    print(f"  Truthful vs Direct Lie: acc={clf_results['truthful_vs_direct_lie']['accuracy']}, "
          f"auc={clf_results['truthful_vs_direct_lie']['auc_roc']}")
    print(f"  Truthful vs Roleplay:   acc={clf_results['truthful_vs_roleplay_lie']['accuracy']}, "
          f"auc={clf_results['truthful_vs_roleplay_lie']['auc_roc']}")

    print(f"\nPermutation test:")
    print(f"  Observed: {clf_results['permutation_test']['observed_accuracy']}, "
          f"p={clf_results['permutation_test']['permutation_p_value']}, "
          f"null mean={clf_results['permutation_test']['mean_null_accuracy']}")

    # Save all results
    with open(RESULTS_DIR / "classification_results.json", "w") as f:
        json.dump(clf_results, f, indent=2, default=str)

    # 5. Visualizations
    print("\n--- Creating Visualizations ---")
    create_visualizations(df, stats_df, clf_results)

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    binary_acc = clf_results["truthful_vs_all_lies"]["logistic_regression"]["accuracy_raw"]
    print(f"Binary classification accuracy: {binary_acc:.3f}")
    print(f"Chance level: 0.500")
    n_sig = len(stats_df[stats_df["significant_bonf"]])
    n_total = len(stats_df)
    print(f"Significant features: {n_sig}/{n_total} (Bonferroni-corrected)")

    perm_p = float(clf_results["permutation_test"]["permutation_p_value"])
    if binary_acc > 0.55 and perm_p < 0.05:
        print("\n=> CONCLUSION: Lying text IS distributionally different from truthful text.")
    elif binary_acc > 0.55:
        print("\n=> CONCLUSION: Some distributional differences detected, but statistical significance is marginal.")
    else:
        print("\n=> CONCLUSION: Lying text is NOT clearly distinguishable from truthful text at the surface level.")

    print(f"\nPermutation test p-value: {perm_p}")


if __name__ == "__main__":
    main()
