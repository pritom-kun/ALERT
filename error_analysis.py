"""
TTP Error Analysis & Near-Miss Confusion Analysis
====================================================
Analyzes fine-tuned SciBERT model failures on MITRE ATT&CK TTP classification,
identifies near-miss confusions, worst-performing categories, cross-sentence
reference issues, and concrete failure examples. Optionally evaluates
state-of-the-art LLMs on the failure cases.

Usage:
------
# Part 1: Analyze fine-tuned model errors (local, no API)
python error_analysis.py --analyze

# Part 2: Test LLMs on failure cases (requires API keys)
python error_analysis.py --llm_eval --llm_model gpt

# Both together
python error_analysis.py --analyze --llm_eval --llm_model gpt

# Customize model checkpoint
python error_analysis.py --analyze --al_type margin --train_len 600
"""

import argparse
import json
import os
import pickle
import re
import textwrap
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.nn import functional as F
from torch.utils import data

from data.tram import create_tram_dataset
from metrics.classification_metrics import get_logits_labels
from net.bert import modernbert, roberta, scibert

# ============================================================================
# Constants
# ============================================================================
OUTPUT_DIR = "./results/error_analysis"
NEAR_MISS_MARGIN_THRESHOLD = 0.1  # softmax margin below this = near-miss
TOP_N_CONFUSED_PAIRS = 15
BOTTOM_K_CLASSES = 10
EXAMPLES_PER_CATEGORY = 5

def _load_technique_names(csv_path: str = "./data/cti/cti2mitre.csv") -> Dict[str, str]:
    """
    Load MITRE ATT&CK technique names from cti2mitre.csv.
    The CSV has columns: label_tec, label_subtec, tec_name.
    Sub-technique IDs (e.g. T1003.001) take priority over parent IDs.
    Falls back to an empty dict if the file is not found.
    """
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    # Parent technique names (label_tec → tec_name)
    mapping = df.groupby("label_tec")["tec_name"].first().to_dict()
    # Sub-technique names override (label_subtec → tec_name)
    sub_mapping = df.groupby("label_subtec")["tec_name"].first().to_dict()
    mapping.update(sub_mapping)
    return mapping


TECHNIQUE_NAMES: Dict[str, str] = _load_technique_names()



def get_technique_label(tid: str) -> str:
    """Return 'T1027 (Obfuscated Files)' style label."""
    name = TECHNIQUE_NAMES.get(tid, "")
    short = name.split(":")[0] if ":" in name else name
    # Truncate long names
    if len(short) > 30:
        short = short[:27] + "..."
    return f"{tid} ({short})" if short else tid


# ============================================================================
# Part 1: Fine-Tuned Model Error Analysis
# ============================================================================

def load_model_and_predict(
    model_name: str = "scibert",
    al_type: str = "margin",
    train_len: int = 600,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the fine-tuned model and generate predictions with full softmax
    probabilities on the test set.

    Returns:
        y_true_int, y_pred_int, softmax_probs, logits_np, class_names, raw_texts
    """
    tokenizer_name_map = {
        "scibert": "allenai/scibert_scivocab_uncased",
        "roberta": "roberta-base",
        "modernbert": "answerdotai/ModernBERT-base",
    }
    model_fn_map = {"scibert": scibert, "roberta": roberta, "modernbert": modernbert}

    tokenizer_name = tokenizer_name_map[model_name]
    model_fn = model_fn_map[model_name]

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(seed)

    num_classes = 50

    train_dataset, test_dataset, tokenizer = create_tram_dataset(
        data_path="./data/cti/tram.json",
        tokenizer_name=tokenizer_name,
        seed=seed,
        transform=False,
    )

    kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

    ckpt = f"checkpoints/al_type_{al_type}_{train_len}_samples.pt"
    model = model_fn(tokenizer, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Get logits and labels
    logits, labels = get_logits_labels(model, test_loader, device)
    softmax_probs = F.softmax(logits, dim=1)

    y_true_int = labels.cpu().numpy()
    y_pred_int = torch.argmax(softmax_probs, dim=1).cpu().numpy()
    softmax_np = softmax_probs.cpu().numpy()
    logits_np = logits.cpu().numpy()

    # Load label encoder for class names
    with open("./saves/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    class_names = encoder.classes_.tolist()

    # Load raw texts for the test set
    with open("./data/cti/tram.json") as f:
        data_json = json.load(f)
    all_texts = [
        row["text"]
        for row in data_json["sentences"]
        if len(row["mappings"]) > 0
    ]
    with open(f"./saves/splits_seed_{seed}.json") as f:
        splits = json.load(f)
    raw_texts = [all_texts[i] for i in splits["test_idx"]]

    return y_true_int, y_pred_int, softmax_np, logits_np, class_names, raw_texts


def analyze_per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Compute per-class precision, recall, F1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    df = pd.DataFrame({
        "technique_id": class_names,
        "technique_name": [TECHNIQUE_NAMES.get(c, "") for c in class_names],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    df = df.sort_values("f1", ascending=True).reset_index(drop=True)
    return df


def find_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = TOP_N_CONFUSED_PAIRS,
) -> List[Dict]:
    """
    Extract the top-N most confused technique pairs from the confusion matrix.
    Returns pairs sorted by confusion count (off-diagonal).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_id": class_names[i],
                    "pred_id": class_names[j],
                    "true_name": TECHNIQUE_NAMES.get(class_names[i], ""),
                    "pred_name": TECHNIQUE_NAMES.get(class_names[j], ""),
                    "count": int(cm[i, j]),
                    "true_support": int(cm[i].sum()),
                    "confusion_rate": float(cm[i, j] / cm[i].sum()) if cm[i].sum() > 0 else 0,
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def detect_near_misses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    softmax_probs: np.ndarray,
    class_names: List[str],
    raw_texts: List[str],
    margin_threshold: float = NEAR_MISS_MARGIN_THRESHOLD,
) -> List[Dict]:
    """
    Find near-miss misclassifications where the softmax margin between
    the top prediction and the correct class is small.

    A 'near miss' means the model almost got it right — the correct class
    had high probability but was narrowly beaten by the wrong class.
    """
    near_misses = []
    wrong_mask = y_true != y_pred

    for idx in np.where(wrong_mask)[0]:
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        probs = softmax_probs[idx]

        pred_prob = probs[pred_label]
        true_prob = probs[true_label]
        margin = pred_prob - true_prob

        # Top-3 predictions
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(class_names[k], float(probs[k])) for k in top3_idx]

        if margin < margin_threshold:
            near_misses.append({
                "sample_idx": int(idx),
                "text": raw_texts[idx],
                "true_id": class_names[true_label],
                "pred_id": class_names[pred_label],
                "true_name": TECHNIQUE_NAMES.get(class_names[true_label], ""),
                "pred_name": TECHNIQUE_NAMES.get(class_names[pred_label], ""),
                "pred_confidence": float(pred_prob),
                "true_class_prob": float(true_prob),
                "margin": float(margin),
                "top3_predictions": top3,
            })

    near_misses.sort(key=lambda x: x["margin"])
    return near_misses


def detect_cross_sentence_references(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_texts: List[str],
    class_names: List[str],
    softmax_probs: np.ndarray,
) -> List[Dict]:
    """
    Identify misclassified samples whose text contains indicators of
    cross-sentence or external context dependency:
      - Citation markers: (Citation: ...)
      - Pronouns suggesting referencing another sentence: 'it', 'they', 'the group'
      - Leading lowercase or continuation markers
    """
    # Patterns indicating potential cross-sentence dependency
    citation_pattern = re.compile(r"\(Citation:[^)]+\)", re.IGNORECASE)
    pronoun_start = re.compile(
        r"^(it |its |they |the group |the malware |the tool |the threat |the actor |"
        r"this |these |those |has been |was also |can also |also |may also |"
        r"additionally |furthermore |moreover )",
        re.IGNORECASE,
    )
    lowercase_start = re.compile(r"^[a-z]")

    cross_ref_samples = []
    wrong_mask = y_true != y_pred

    for idx in np.where(wrong_mask)[0]:
        text = raw_texts[idx]
        reasons = []

        if citation_pattern.search(text):
            reasons.append("contains_citation")
        if pronoun_start.match(text.strip()):
            reasons.append("pronoun/anaphoric_start")
        if lowercase_start.match(text.strip()):
            reasons.append("lowercase_start (mid-sentence fragment)")

        if reasons:
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            probs = softmax_probs[idx]

            cross_ref_samples.append({
                "sample_idx": int(idx),
                "text": text,
                "true_id": class_names[true_label],
                "pred_id": class_names[pred_label],
                "true_name": TECHNIQUE_NAMES.get(class_names[true_label], ""),
                "pred_name": TECHNIQUE_NAMES.get(class_names[pred_label], ""),
                "pred_confidence": float(probs[pred_label]),
                "true_class_prob": float(probs[true_label]),
                "reference_indicators": reasons,
            })

    return cross_ref_samples


def extract_failure_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    softmax_probs: np.ndarray,
    class_names: List[str],
    raw_texts: List[str],
    confused_pairs: List[Dict],
    worst_classes_df: pd.DataFrame,
    n_examples: int = EXAMPLES_PER_CATEGORY,
) -> Dict:
    """
    Extract concrete failure examples for:
      1. Each of the top confused pairs
      2. Each of the worst-performing classes
    """
    examples = {"by_confused_pair": {}, "by_worst_class": {}}

    # --- Examples for top confused pairs ---
    for pair in confused_pairs:
        true_id = pair["true_id"]
        pred_id = pair["pred_id"]
        pair_key = f"{true_id}_as_{pred_id}"
        pair_examples = []

        true_cls = class_names.index(true_id)
        pred_cls = class_names.index(pred_id)

        # Find samples where true=true_cls and pred=pred_cls
        for idx in range(len(y_true)):
            if y_true[idx] == true_cls and y_pred[idx] == pred_cls:
                probs = softmax_probs[idx]
                pair_examples.append({
                    "text": raw_texts[idx],
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "true_name": TECHNIQUE_NAMES.get(true_id, ""),
                    "pred_name": TECHNIQUE_NAMES.get(pred_id, ""),
                    "pred_confidence": float(probs[pred_cls]),
                    "true_class_prob": float(probs[true_cls]),
                    "margin": float(probs[pred_cls] - probs[true_cls]),
                })
                if len(pair_examples) >= n_examples:
                    break

        examples["by_confused_pair"][pair_key] = pair_examples

    # --- Examples for worst-performing classes ---
    worst_classes = worst_classes_df.head(BOTTOM_K_CLASSES)
    for _, row in worst_classes.iterrows():
        tid = row["technique_id"]
        cls_idx = class_names.index(tid)
        cls_examples = []

        for idx in range(len(y_true)):
            if y_true[idx] == cls_idx and y_pred[idx] != cls_idx:
                probs = softmax_probs[idx]
                pred_cls = y_pred[idx]
                cls_examples.append({
                    "text": raw_texts[idx],
                    "true_id": tid,
                    "pred_id": class_names[pred_cls],
                    "true_name": TECHNIQUE_NAMES.get(tid, ""),
                    "pred_name": TECHNIQUE_NAMES.get(class_names[pred_cls], ""),
                    "pred_confidence": float(probs[pred_cls]),
                    "true_class_prob": float(probs[cls_idx]),
                    "margin": float(probs[pred_cls] - probs[cls_idx]),
                })
                if len(cls_examples) >= n_examples:
                    break

        examples["by_worst_class"][tid] = cls_examples

    return examples


def save_all_failure_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    softmax_probs: np.ndarray,
    class_names: List[str],
    raw_texts: List[str],
    output_path: str,
) -> List[Dict]:
    """Save all misclassified samples for LLM evaluation in Part 2."""
    failures = []
    wrong_mask = y_true != y_pred

    for idx in np.where(wrong_mask)[0]:
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        probs = softmax_probs[idx]

        failures.append({
            "sample_idx": int(idx),
            "text": raw_texts[idx],
            "true_id": class_names[true_label],
            "pred_id": class_names[pred_label],
            "true_name": TECHNIQUE_NAMES.get(class_names[true_label], ""),
            "pred_name": TECHNIQUE_NAMES.get(class_names[pred_label], ""),
            "pred_confidence": float(probs[pred_label]),
            "true_class_prob": float(probs[true_label]),
            "margin": float(probs[pred_label] - probs[true_label]),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_failures": len(failures),
            "n_total": len(y_true),
            "error_rate": len(failures) / len(y_true),
            "class_names": class_names,
            "failures": failures,
        }, f, indent=2)

    return failures


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    confused_pairs: List[Dict],
    output_path: str,
):
    """Plot a confusion heatmap focused on the most confused techniques."""
    # Get the unique techniques involved in the top confused pairs
    involved = set()
    for pair in confused_pairs:
        involved.add(pair["true_id"])
        involved.add(pair["pred_id"])
    involved = sorted(involved)

    # Map to indices
    idx_map = {c: i for i, c in enumerate(class_names)}
    sub_indices = [idx_map[t] for t in involved]

    # Build sub-confusion matrix
    cm_full = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_sub = cm_full[np.ix_(sub_indices, sub_indices)]

    # Normalize by row (true class)
    row_sums = cm_sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_sub / row_sums

    # Labels
    labels = [get_technique_label(t) for t in involved]

    fig, ax = plt.subplots(figsize=(max(12, len(involved) * 0.8), max(10, len(involved) * 0.7)))
    sns.heatmap(
        cm_norm,
        annot=cm_sub,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Confusion Rate (row-normalized)"},
    )
    ax.set_xlabel("Predicted Technique", fontsize=12)
    ax.set_ylabel("True Technique", fontsize=12)
    ax.set_title("Confusion Heatmap: Most Confused TTP Pairs", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion heatmap to {output_path}")


def plot_per_class_f1(per_class_df: pd.DataFrame, output_path: str):
    """Bar chart of per-class F1 scores (bottom classes highlighted)."""
    df = per_class_df.copy()
    df["label"] = df["technique_id"].apply(get_technique_label)
    df = df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.25)))
    colors = ["#e74c3c" if f < 0.3 else "#f39c12" if f < 0.5 else "#2ecc71" for f in df["f1"]]
    ax.barh(df["label"], df["f1"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores (Worst → Best)", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="F1 = 0.50")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-class F1 chart to {output_path}")


# ============================================================================
# Pretty Printing
# ============================================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_per_class_table(df: pd.DataFrame, title: str, n: int = None):
    show = df.head(n) if n else df
    print(f"\n  {title}")
    print(f"  {'Technique ID':<14} {'Name':<40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*80}")
    for _, r in show.iterrows():
        name = r["technique_name"][:38] if r["technique_name"] else ""
        print(f"  {r['technique_id']:<14} {name:<40} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['support']:>6}")


def print_confused_pairs(pairs: List[Dict]):
    print(f"\n  {'True → Predicted':<35} {'Count':>6} {'Rate':>8} {'True Name':<30} → {'Pred Name':<30}")
    print(f"  {'-'*115}")
    for p in pairs:
        pair_str = f"{p['true_id']} → {p['pred_id']}"
        tn = p["true_name"][:28] if p["true_name"] else ""
        pn = p["pred_name"][:28] if p["pred_name"] else ""
        print(f"  {pair_str:<35} {p['count']:>6} {p['confusion_rate']:>7.1%} {tn:<30} → {pn:<30}")


def print_failure_examples(examples: List[Dict], header: str, max_text_len: int = 120):
    if not examples:
        print(f"\n  {header}: (no examples)")
        return
    print(f"\n  {header}")
    print(f"  {'-'*80}")
    for i, ex in enumerate(examples, 1):
        text = ex["text"][:max_text_len] + ("..." if len(ex["text"]) > max_text_len else "")
        print(f"  Example {i}:")
        print(f"    Text: \"{text}\"")
        print(f"    True: {ex['true_id']} ({ex.get('true_name', '')})")
        pred_extra = ""
        if 'pred_confidence' in ex:
            pred_extra = (f"  [conf={ex['pred_confidence']:.3f}, "
                          f"true_prob={ex.get('true_class_prob', 0):.3f}, "
                          f"margin={ex.get('margin', 0):.3f}]")
        if ex.get('reference_indicators'):
            pred_extra += f"  indicators={ex['reference_indicators']}"
        print(f"    Pred: {ex['pred_id']} ({ex.get('pred_name', '')}){pred_extra}")


# ============================================================================
# Part 1 Main: run_analysis
# ============================================================================

def run_analysis(
    model_name: str = "scibert",
    al_type: str = "margin",
    train_len: int = 600,
    seed: int = 1,
    margin_threshold: float = NEAR_MISS_MARGIN_THRESHOLD,
    top_n_pairs: int = TOP_N_CONFUSED_PAIRS,
) -> Dict:
    """Run the full error analysis pipeline (Part 1)."""

    print_section("LOADING MODEL & GENERATING PREDICTIONS")
    print(f"  Model: {model_name} | AL: {al_type} | Train samples: {train_len}")

    y_true, y_pred, softmax_probs, logits_np, class_names, raw_texts = (
        load_model_and_predict(model_name, al_type, train_len, seed)
    )

    n_total = len(y_true)
    n_correct = int((y_true == y_pred).sum())
    n_wrong = n_total - n_correct

    print(f"\n  Test set: {n_total} samples")
    print(f"  Correct:  {n_correct} ({n_correct/n_total:.1%})")
    print(f"  Wrong:    {n_wrong} ({n_wrong/n_total:.1%})")

    # ---- Per-class performance ----
    print_section("PER-CLASS PERFORMANCE (Bottom {})".format(BOTTOM_K_CLASSES))

    per_class_df = analyze_per_class_performance(y_true, y_pred, class_names)
    print_per_class_table(per_class_df, f"Bottom {BOTTOM_K_CLASSES} classes by F1:", BOTTOM_K_CLASSES)

    # Overall F1 (macro only)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  Overall F1 (macro): {f1_mac:.4f}")

    # ---- Top confused pairs ----
    print_section(f"TOP {top_n_pairs} MOST CONFUSED TTP PAIRS")
    confused_pairs = find_top_confused_pairs(y_true, y_pred, class_names, top_n_pairs)
    print_confused_pairs(confused_pairs)

    # ---- Near-miss analysis ----
    print_section("NEAR-MISS CONFUSIONS (margin < {:.2f})".format(margin_threshold))
    near_misses = detect_near_misses(
        y_true, y_pred, softmax_probs, class_names, raw_texts, margin_threshold
    )
    print(f"  Total misclassifications: {n_wrong}")
    print(f"  Near-misses (margin < {margin_threshold}): {len(near_misses)} "
          f"({len(near_misses)/max(n_wrong,1):.1%} of errors)")

    # Distribution of near-miss pairs
    nm_pair_counts = Counter(
        (nm["true_id"], nm["pred_id"]) for nm in near_misses
    )
    print(f"\n  Top near-miss confusion pairs:")
    for (tid, pid), count in nm_pair_counts.most_common(10):
        tn = TECHNIQUE_NAMES.get(tid, "")[:25]
        pn = TECHNIQUE_NAMES.get(pid, "")[:25]
        print(f"    {tid} → {pid}  (×{count})  {tn} → {pn}")

    # Print some near-miss examples
    print_failure_examples(near_misses[:5], "Sample near-miss examples (smallest margins)")

    # ---- Cross-sentence references ----
    print_section("CROSS-SENTENCE REFERENCE FAILURES")
    cross_ref = detect_cross_sentence_references(
        y_true, y_pred, raw_texts, class_names, softmax_probs
    )
    print(f"  Misclassifications with cross-sentence indicators: {len(cross_ref)} "
          f"({len(cross_ref)/max(n_wrong,1):.1%} of errors)")

    # Breakdown by indicator type
    indicator_counts = Counter()
    for cr in cross_ref:
        for ind in cr["reference_indicators"]:
            indicator_counts[ind] += 1
    for ind, count in indicator_counts.most_common():
        print(f"    {ind}: {count}")

    print_failure_examples(cross_ref[:5], "Sample cross-sentence reference failures")

    # ---- Concrete failure examples ----
    print_section("CONCRETE FAILURE EXAMPLES BY CONFUSED PAIR")
    failure_examples = extract_failure_examples(
        y_true, y_pred, softmax_probs, class_names, raw_texts,
        confused_pairs, per_class_df, EXAMPLES_PER_CATEGORY,
    )

    for pair_key, exs in list(failure_examples["by_confused_pair"].items())[:5]:
        print_failure_examples(exs, f"Pair: {pair_key}")

    print_section("CONCRETE FAILURE EXAMPLES BY WORST CLASS")
    for cls_key, exs in list(failure_examples["by_worst_class"].items())[:5]:
        tname = TECHNIQUE_NAMES.get(cls_key, "")
        print_failure_examples(exs, f"Worst class: {cls_key} ({tname})")

    # ---- Save outputs ----
    print_section("SAVING RESULTS")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save failure cases for Part 2
    failure_path = os.path.join(OUTPUT_DIR, "failure_cases.json")
    all_failures = save_all_failure_cases(
        y_true, y_pred, softmax_probs, class_names, raw_texts, failure_path
    )
    print(f"  Saved {len(all_failures)} failure cases to {failure_path}")

    # Save per-class metrics
    per_class_path = os.path.join(OUTPUT_DIR, "per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print(f"  Saved per-class metrics to {per_class_path}")

    # Save confused pairs
    pairs_path = os.path.join(OUTPUT_DIR, "confused_pairs.json")
    with open(pairs_path, "w") as f:
        json.dump(confused_pairs, f, indent=2)
    print(f"  Saved confused pairs to {pairs_path}")

    # Save near-misses
    nm_path = os.path.join(OUTPUT_DIR, "near_misses.json")
    with open(nm_path, "w") as f:
        json.dump(near_misses[:100], f, indent=2)  # Top 100
    print(f"  Saved near-misses to {nm_path}")

    # Save cross-sentence reference failures
    cr_path = os.path.join(OUTPUT_DIR, "cross_sentence_failures.json")
    with open(cr_path, "w") as f:
        json.dump(cross_ref, f, indent=2)
    print(f"  Saved cross-sentence failures to {cr_path}")

    # Save concrete examples
    ex_path = os.path.join(OUTPUT_DIR, "failure_examples.json")
    with open(ex_path, "w") as f:
        json.dump(failure_examples, f, indent=2)
    print(f"  Saved failure examples to {ex_path}")

    # Plots
    plot_confusion_heatmap(
        y_true, y_pred, class_names, confused_pairs,
        os.path.join(OUTPUT_DIR, "confusion_heatmap.png"),
    )
    plot_per_class_f1(
        per_class_df,
        os.path.join(OUTPUT_DIR, "per_class_f1.png"),
    )

    # ---- LaTeX tables ----
    print_section("LATEX TABLES")

    # Per-class bottom-K table
    bottom = per_class_df.head(BOTTOM_K_CLASSES)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Bottom " + str(BOTTOM_K_CLASSES) + r" worst-performing TTP categories by F1 score.}")
    print(r"\label{tab:worst_ttp}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Technique ID & Name & Precision & Recall & F1 & Support \\")
    print(r"\midrule")
    for _, r in bottom.iterrows():
        name = r["technique_name"].replace("&", r"\&")[:35]
        print(f"{r['technique_id']} & {name} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['support']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Confused pairs table
    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Top " + str(min(10, len(confused_pairs))) + r" most confused TTP pairs.}")
    print(r"\label{tab:confused_pairs}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"True & Predicted & Count & Rate & Near-Miss \\")
    print(r"\midrule")
    nm_pair_set = {(nm["true_id"], nm["pred_id"]) for nm in near_misses}
    for p in confused_pairs[:10]:
        is_nm = "\\checkmark" if (p["true_id"], p["pred_id"]) in nm_pair_set else ""
        print(f"{p['true_id']} & {p['pred_id']} & {p['count']} & {p['confusion_rate']:.1%} & {is_nm} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save LaTeX to file
    latex_path = os.path.join(OUTPUT_DIR, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by error_analysis.py\n")
        f.write("% Bottom-K worst-performing TTP categories\n")
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Bottom " + str(BOTTOM_K_CLASSES) + r" worst-performing TTP categories by F1 score.}" + "\n")
        f.write(r"\label{tab:worst_ttp}" + "\n")
        f.write(r"\begin{tabular}{llcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Technique ID & Name & Precision & Recall & F1 & Support \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, r in bottom.iterrows():
            name = r["technique_name"].replace("&", r"\&")[:35]
            f.write(f"{r['technique_id']} & {name} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['support']} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        f.write("% Top confused pairs\n")
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Top " + str(min(10, len(confused_pairs))) + r" most confused TTP pairs.}" + "\n")
        f.write(r"\label{tab:confused_pairs}" + "\n")
        f.write(r"\begin{tabular}{llccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"True & Predicted & Count & Rate & Near-Miss \\" + "\n")
        f.write(r"\midrule" + "\n")
        for p in confused_pairs[:10]:
            is_nm = r"\checkmark" if (p["true_id"], p["pred_id"]) in nm_pair_set else ""
            f.write(f"{p['true_id']} & {p['pred_id']} & {p['count']} & {p['confusion_rate']:.1%} & {is_nm} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"\n  Saved LaTeX tables to {latex_path}")

    return {
        "n_total": n_total,
        "n_wrong": n_wrong,
        "per_class_df": per_class_df,
        "confused_pairs": confused_pairs,
        "near_misses": near_misses,
        "cross_ref_failures": cross_ref,
        "failure_examples": failure_examples,
        "all_failures": all_failures,
        "class_names": class_names,
    }


# ============================================================================
# Part 2: LLM Evaluation on Failure Cases
# ============================================================================

def run_llm_eval(
    llm_model: str = "gpt",
    zeroshot: bool = True,
    rag: bool = False,
    failure_path: str = None,
) -> Dict:
    """
    Evaluate a state-of-the-art LLM on the failure cases identified by Part 1.
    Uses the LLMClassifier from llm_api_inference.py.
    """
    from llm_api_inference import LLMClassifier

    # Load failure cases
    if failure_path is None:
        failure_path = os.path.join(OUTPUT_DIR, "failure_cases.json")

    if not os.path.exists(failure_path):
        print(f"\n  ERROR: Failure cases file not found at {failure_path}")
        print("  Run --analyze first to generate failure cases.")
        return {}

    with open(failure_path) as f:
        failure_data = json.load(f)

    failures = failure_data["failures"]
    class_names = failure_data["class_names"]
    n_failures = len(failures)

    print_section(f"LLM EVALUATION ON {n_failures} FAILURE CASES")
    fshot = "zero-shot" if zeroshot else "few-shot"
    print(f"  LLM: {llm_model} | Mode: {fshot} | RAG: {rag}")
    print(f"  Failure cases: {n_failures} (from {failure_data['n_total']} total test samples)")

    # Initialize classifier
    classifier = LLMClassifier(llm_model, rag=rag)

    # Get texts and true labels
    texts = [f["text"] for f in failures]
    true_ids = [f["true_id"] for f in failures]

    # Run LLM classification
    print(f"\n  Running {fshot} classification on {n_failures} failure cases...")
    llm_predictions = classifier.classify_batch(texts, class_names, zeroshot)

    # Compute metrics
    n_llm_correct = sum(1 for pred, true in zip(llm_predictions, true_ids) if pred == true)
    llm_accuracy = n_llm_correct / n_failures

    print(f"\n  LLM resolved {n_llm_correct}/{n_failures} failure cases ({llm_accuracy:.1%})")

    # Per-category analysis: which confusion pairs does the LLM fix?
    fixed_pairs = Counter()
    still_wrong_pairs = Counter()
    llm_wrong_different = Counter()  # LLM wrong but with a different wrong answer

    for fail, llm_pred in zip(failures, llm_predictions):
        true_id = fail["true_id"]
        finetuned_pred = fail["pred_id"]
        pair = (true_id, finetuned_pred)

        if llm_pred == true_id:
            fixed_pairs[pair] += 1
        elif llm_pred == finetuned_pred:
            still_wrong_pairs[pair] += 1
        else:
            llm_wrong_different[pair] += 1

    print_section("LLM FIX RATE BY CONFUSION PAIR")
    # Aggregate by pair
    all_pairs_set = set(list(fixed_pairs.keys()) + list(still_wrong_pairs.keys()) + list(llm_wrong_different.keys()))
    pair_results = []
    for pair in all_pairs_set:
        fixed = fixed_pairs.get(pair, 0)
        same_wrong = still_wrong_pairs.get(pair, 0)
        diff_wrong = llm_wrong_different.get(pair, 0)
        total = fixed + same_wrong + diff_wrong
        pair_results.append({
            "true_id": pair[0],
            "pred_id": pair[1],
            "total_failures": total,
            "llm_fixed": fixed,
            "llm_same_error": same_wrong,
            "llm_different_error": diff_wrong,
            "fix_rate": fixed / total if total > 0 else 0,
        })
    pair_results.sort(key=lambda x: x["total_failures"], reverse=True)

    print(f"  {'True → Pred':<30} {'Total':>6} {'Fixed':>6} {'Same':>6} {'Diff':>6} {'Fix%':>7}")
    print(f"  {'-'*75}")
    for pr in pair_results[:20]:
        ps = f"{pr['true_id']} → {pr['pred_id']}"
        print(f"  {ps:<30} {pr['total_failures']:>6} {pr['llm_fixed']:>6} "
              f"{pr['llm_same_error']:>6} {pr['llm_different_error']:>6} {pr['fix_rate']:>6.1%}")

    # Per-class LLM accuracy (on failure subset)
    per_class_fix = defaultdict(lambda: {"total": 0, "fixed": 0})
    for fail, llm_pred in zip(failures, llm_predictions):
        tid = fail["true_id"]
        per_class_fix[tid]["total"] += 1
        if llm_pred == tid:
            per_class_fix[tid]["fixed"] += 1

    print_section("LLM FIX RATE BY TRUE CLASS (on failure cases)")
    class_fix_list = [
        {"true_id": tid, "total": v["total"], "fixed": v["fixed"],
         "fix_rate": v["fixed"]/v["total"] if v["total"] > 0 else 0,
         "name": TECHNIQUE_NAMES.get(tid, "")}
        for tid, v in per_class_fix.items()
    ]
    class_fix_list.sort(key=lambda x: x["fix_rate"])

    print(f"  {'Technique':<14} {'Name':<35} {'Total':>6} {'Fixed':>6} {'Fix%':>7}")
    print(f"  {'-'*70}")
    for cf in class_fix_list:
        name = cf["name"][:33]
        print(f"  {cf['true_id']:<14} {name:<35} {cf['total']:>6} {cf['fixed']:>6} {cf['fix_rate']:>6.1%}")

    # Concrete examples: cases LLM fixed vs cases LLM also failed
    print_section("EXAMPLES: CASES LLM FIXED")
    llm_fixed_examples = []
    for fail, llm_pred in zip(failures, llm_predictions):
        if llm_pred == fail["true_id"]:
            llm_fixed_examples.append({
                "text": fail["text"],
                "true_id": fail["true_id"],
                "pred_id": fail["pred_id"],  # fine-tuned model's wrong answer
                "llm_pred": llm_pred,
                "true_name": fail.get("true_name", ""),
                "pred_name": fail.get("pred_name", ""),
                "pred_confidence": fail.get("pred_confidence", 0),
                "true_class_prob": fail.get("true_class_prob", 0),
                "margin": fail.get("margin", 0),
            })
    for i, ex in enumerate(llm_fixed_examples[:5], 1):
        text = ex["text"][:120] + ("..." if len(ex["text"]) > 120 else "")
        print(f"\n  Example {i}:")
        print(f"    Text: \"{text}\"")
        print(f"    True: {ex['true_id']} ({ex['true_name']})")
        print(f"    Fine-tuned pred: {ex['pred_id']} ({ex['pred_name']})")
        print(f"    LLM pred: {ex['llm_pred']} ✓")

    print_section("EXAMPLES: CASES LLM ALSO FAILED")
    llm_still_wrong = []
    for fail, llm_pred in zip(failures, llm_predictions):
        if llm_pred != fail["true_id"]:
            llm_still_wrong.append({
                "text": fail["text"],
                "true_id": fail["true_id"],
                "finetuned_pred": fail["pred_id"],
                "llm_pred": llm_pred,
                "true_name": fail.get("true_name", ""),
                "pred_name": fail.get("pred_name", ""),
                "pred_confidence": fail.get("pred_confidence", 0),
                "true_class_prob": fail.get("true_class_prob", 0),
                "margin": fail.get("margin", 0),
            })
    for i, ex in enumerate(llm_still_wrong[:5], 1):
        text = ex["text"][:120] + ("..." if len(ex["text"]) > 120 else "")
        llm_pred_name = TECHNIQUE_NAMES.get(ex["llm_pred"], "")
        print(f"\n  Example {i}:")
        print(f"    Text: \"{text}\"")
        print(f"    True: {ex['true_id']} ({ex['true_name']})")
        print(f"    Fine-tuned pred: {ex['finetuned_pred']} ({ex['pred_name']})")
        print(f"    LLM pred: {ex['llm_pred']} ({llm_pred_name}) ✗")

    # Save LLM results
    llm_results = {
        "llm_model": llm_model,
        "mode": fshot,
        "rag": rag,
        "n_failures": n_failures,
        "n_llm_correct": n_llm_correct,
        "llm_accuracy_on_failures": llm_accuracy,
        "pair_results": pair_results,
        "per_class_fix": class_fix_list,
        "n_fixed_examples": len(llm_fixed_examples),
        "n_still_wrong": len(llm_still_wrong),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rag_suffix = "_rag" if rag else ""
    shot_suffix = "_zeroshot" if zeroshot else "_fewshot"
    results_path = os.path.join(OUTPUT_DIR, f"llm_eval_{llm_model}{shot_suffix}{rag_suffix}.json")
    with open(results_path, "w") as f:
        json.dump(llm_results, f, indent=2)
    print(f"\n  Saved LLM evaluation results to {results_path}")

    # Save detailed per-sample results for reproducibility
    detailed = []
    for fail, llm_pred in zip(failures, llm_predictions):
        detailed.append({
            "text": fail["text"],
            "true_id": fail["true_id"],
            "finetuned_pred": fail["pred_id"],
            "llm_pred": llm_pred,
            "llm_correct": llm_pred == fail["true_id"],
        })
    detailed_path = os.path.join(OUTPUT_DIR, f"llm_predictions_{llm_model}{shot_suffix}{rag_suffix}.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"  Saved detailed LLM predictions to {detailed_path}")

    return llm_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TTP Error Analysis & Near-Miss Confusion Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Actions
    parser.add_argument("--analyze", action="store_true",
                        help="Run Part 1: fine-tuned model error analysis")
    parser.add_argument("--llm_eval", action="store_true",
                        help="Run Part 2: LLM evaluation on failure cases")

    # Model config (Part 1)
    parser.add_argument("--model_name", type=str, default="scibert",
                        choices=["scibert", "roberta", "modernbert"],
                        help="Base model architecture")
    parser.add_argument("--al_type", type=str, default="margin",
                        help="Active learning strategy for checkpoint")
    parser.add_argument("--train_len", type=int, default=600,
                        help="Training set size for checkpoint")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")

    # LLM config (Part 2)
    parser.add_argument("--llm_model", type=str, default="gpt",
                        help="LLM to evaluate (gpt, claude, gemini, deepseek, llama, kimi)")
    parser.add_argument("--zeroshot", action="store_true", default=True,
                        help="Zero-shot mode for LLM (default: True)")
    parser.add_argument("--fewshot", action="store_true", default=False,
                        help="Few-shot mode for LLM")
    parser.add_argument("--rag", action="store_true", default=False,
                        help="Enable RAG/search tools for LLM")

    # Thresholds
    parser.add_argument("--margin_threshold", type=float, default=NEAR_MISS_MARGIN_THRESHOLD,
                        help="Near-miss margin threshold")
    parser.add_argument("--top_n_pairs", type=int, default=TOP_N_CONFUSED_PAIRS,
                        help="Number of top confused pairs to report")

    args = parser.parse_args()

    if not args.analyze and not args.llm_eval:
        parser.error("At least one of --analyze or --llm_eval is required.")

    # Override module-level defaults with CLI args
    margin_threshold = args.margin_threshold
    top_n_pairs = args.top_n_pairs

    zeroshot = not args.fewshot  # --fewshot overrides default True

    if args.analyze:
        results = run_analysis(
            model_name=args.model_name,
            al_type=args.al_type,
            train_len=args.train_len,
            seed=args.seed,
            margin_threshold=margin_threshold,
            top_n_pairs=top_n_pairs,
        )

    if args.llm_eval:
        llm_results = run_llm_eval(
            llm_model=args.llm_model,
            zeroshot=zeroshot,
            rag=args.rag,
        )

    print_section("DONE")
    print("  All results saved to: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()
