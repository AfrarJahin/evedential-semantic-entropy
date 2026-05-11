"""Compute AUROC and AUARC from a local offline wandb run directory."""
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics as sk_metrics


def auroc(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
    return sk_metrics.auc(fpr, tpr)


def auarc(accuracies, uncertainties):
    accuracies = np.array(accuracies)
    uncertainties = np.array(uncertainties)
    quantiles = np.linspace(0.1, 1, 20)
    accs = []
    for q in quantiles:
        cutoff = np.quantile(uncertainties, q)
        sel = uncertainties <= cutoff
        accs.append(np.mean(accuracies[sel]) if sel.any() else float('nan'))
    dx = quantiles[1] - quantiles[0]
    return float(np.nansum(np.array(accs) * dx))


def load_pkl(run_dir):
    path = Path(run_dir) / 'files' / 'uncertainty_measures.pkl'
    if not path.exists():
        # try without 'files' subdir
        path = Path(run_dir) / 'uncertainty_measures.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute(run_dir):
    data = load_pkl(run_dir)

    is_false = np.array(data['validation_is_false'])
    accuracy = 1 - is_false
    measures = data['uncertainty_measures']

    print(f"\nRun: {run_dir}")
    print(f"Samples: {len(is_false)}  |  Accuracy: {accuracy.mean():.3f}\n")
    print(f"{'Measure':<35} {'AUROC':>8} {'AUARC':>8} {'Mean Unc':>10}")
    print('-' * 65)

    for name, vals in measures.items():
        vals = np.array(vals)
        if len(vals) != len(is_false):
            vals = vals[:len(is_false)]
        auc = auroc(is_false, vals)
        arc = auarc(accuracy, vals)
        print(f"{name:<35} {auc:>8.4f} {arc:>8.4f} {vals.mean():>10.4f}")

    if len(is_false) < 20:
        print("\nWARNING: Only", len(is_false), "samples — AUROC/AUARC are not reliable below ~100 samples.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # auto-find most recent offline run
        import glob, os
        runs = sorted(glob.glob('EXP/wandb/offline-run-*'), key=os.path.getmtime, reverse=True)
        if not runs:
            print("Usage: python compute_scores.py <run_dir>")
            sys.exit(1)
        run_dir = runs[0]
        print(f"Auto-selected most recent run: {run_dir}")
    else:
        run_dir = sys.argv[1]

    compute(run_dir)
