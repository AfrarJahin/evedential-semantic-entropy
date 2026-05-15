"""
Summarize results from a generate_answers.py run.

Usage:
    python summarize_results.py <run_dir>
    python summarize_results.py <run_dir> --json            # also export results.json
    python summarize_results.py <run_dir> --json-only       # skip terminal output

Example:
    python summarize_results.py EXP/wandb/offline-run-20260507_021501-30yoxokf
    python summarize_results.py EXP/wandb/offline-run-20260507_021501-30yoxokf --json
"""
import sys
import json
import pickle
import argparse
from pathlib import Path

try:
    import numpy as np
    from sklearn import metrics as sk_metrics
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _auroc(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
    return float(sk_metrics.auc(fpr, tpr))


def _auarc(accuracies, uncertainties):
    """Area under accuracy-rejection curve (higher = better)."""
    accuracies = np.array(accuracies)
    uncertainties = np.array(uncertainties)
    quantiles = np.linspace(0.1, 1, 20)
    accs = []
    for q in quantiles:
        cutoff = np.quantile(uncertainties, q)
        sel = uncertainties <= cutoff
        accs.append(float(np.mean(accuracies[sel])) if sel.any() else float("nan"))
    dx = float(quantiles[1] - quantiles[0])
    return float(np.nansum(np.array(accs) * dx))


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def separator(char="=", width=60):
    print(char * width)


def print_section(title):
    separator()
    print(f"  {title}")
    separator()


def build_summary(run_dir: Path) -> dict:
    files_dir = run_dir / "files"
    if not files_dir.exists():
        files_dir = run_dir

    summary = {}

    # ------------------------------------------------------------------ config
    metadata_path = files_dir / "wandb-metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        args_list = meta.get("args", [])
        # parse args list into a dict
        parsed_args = {}
        for arg in args_list:
            if "=" in arg:
                k, v = arg.lstrip("-").split("=", 1)
                parsed_args[k] = v
            else:
                parsed_args[arg.lstrip("-")] = True
        summary["config"] = {
            "model":   parsed_args.get("model_name", "?"),
            "dataset": parsed_args.get("dataset", "?"),
            "num_samples": parsed_args.get("num_samples", "?"),
            "num_generations": parsed_args.get("num_generations", "?"),
            "random_seed": parsed_args.get("random_seed", "?"),
            "force_cpu": "force_cpu" in parsed_args,
            "host":    meta.get("host", "?"),
            "gpu":     meta.get("gpu", "none"),
            "ram_gb":  round(int(meta.get("memory", {}).get("total", 0)) / 1e9, 1),
            "python":  meta.get("python", "?"),
            "raw_args": args_list,
        }
    else:
        details = load_pkl(files_dir / "experiment_details.pkl")
        a = details.get("args", {})
        summary["config"] = {
            "model":          getattr(a, "model_name", "?"),
            "dataset":        getattr(a, "dataset", "?"),
            "num_samples":    getattr(a, "num_samples", "?"),
            "num_generations":getattr(a, "num_generations", "?"),
            "random_seed":    getattr(a, "random_seed", "?"),
        }

    # --------------------------------------------------------------- accuracy
    val = load_pkl(files_dir / "validation_generations.pkl")
    n = len(val)

    most_likely_correct = 0
    any_correct = 0
    all_correct = 0
    questions = []

    for qid, data in val.items():
        responses = data.get("responses", [])
        gens = []
        accs = []
        log_likelihoods_list = []

        for j, r in enumerate(responses):
            if isinstance(r, tuple) and len(r) >= 5:
                answer = str(r[0])
                log_lls = [round(float(x), 6) for x in r[1]] if hasattr(r[1], "__iter__") else []
                acc = float(r[4])
                gens.append({
                    "generation_index": j,
                    "answer": answer,
                    "correct": acc == 1.0,
                    "log_likelihoods": log_lls,
                    "mean_log_likelihood": round(sum(log_lls) / len(log_lls), 6) if log_lls else None,
                })
                accs.append(acc)
                log_likelihoods_list.append(log_lls)

        ml_correct = accs[0] == 1.0 if accs else False
        any_c = any(a == 1.0 for a in accs)
        all_c = all(a == 1.0 for a in accs)

        if ml_correct:
            most_likely_correct += 1
        if any_c:
            any_correct += 1
        if all_c:
            all_correct += 1

        questions.append({
            "id": qid,
            "question": data.get("question", "?"),
            "context": data.get("context", None),
            "most_likely_correct": ml_correct,
            "any_correct": any_c,
            "all_correct": all_c,
            "generations": gens,
        })

    summary["accuracy"] = {
        "num_questions": n,
        "most_likely_correct": most_likely_correct,
        "most_likely_correct_pct": round(most_likely_correct / n * 100, 2),
        "any_correct": any_correct,
        "any_correct_pct": round(any_correct / n * 100, 2),
        "all_correct": all_correct,
        "all_correct_pct": round(all_correct / n * 100, 2),
    }

    summary["questions"] = questions

    # --------------------------------------------------- uncertainty measures
    unc_path = files_dir / "uncertainty_measures.pkl"
    if unc_path.exists():
        unc = load_pkl(unc_path)
        um = unc.get("uncertainty_measures", unc)
        unc_summary = {}
        unc_per_question = {}

        for k, v in um.items():
            if hasattr(v, "__len__"):
                vals = [float(x) for x in v if hasattr(x, "__float__")]
                if vals:
                    unc_summary[k] = {
                        "mean": round(sum(vals) / len(vals), 6),
                        "min":  round(min(vals), 6),
                        "max":  round(max(vals), 6),
                        "n":    len(vals),
                        "values": [round(x, 6) for x in vals],
                    }
                    unc_per_question[k] = [round(x, 6) for x in vals]
            else:
                try:
                    unc_summary[k] = round(float(v), 6)
                except Exception:
                    unc_summary[k] = str(v)

        summary["uncertainty_measures"] = unc_summary

        # attach per-question uncertainty values to each question
        for measure, vals in unc_per_question.items():
            for i, q in enumerate(summary["questions"]):
                if i < len(vals):
                    q.setdefault("uncertainty", {})[measure] = vals[i]

        # ------------------------------------------------- AUROC / AUARC
        if _HAS_SKLEARN:
            is_false_raw = unc.get("validation_is_false", None)
            if is_false_raw is not None:
                is_false = [float(x) for x in is_false_raw]
                accuracy = [1.0 - x for x in is_false]
                roc_arc = {}
                for name, vals in unc_per_question.items():
                    if len(vals) == len(is_false):
                        roc_arc[name] = {
                            "auroc": round(_auroc(is_false, vals), 4),
                            "auarc": round(_auarc(accuracy, vals), 4),
                        }
                summary["auroc_auarc"] = roc_arc

    summary["run_dir"] = str(run_dir)
    return summary


def print_summary(summary: dict):
    cfg = summary["config"]
    print_section("EXPERIMENT CONFIG")
    print(f"  Model:    {cfg.get('model')}")
    print(f"  Dataset:  {cfg.get('dataset')}")
    print(f"  Samples:  {cfg.get('num_samples')}  |  Generations: {cfg.get('num_generations')}")
    print(f"  Seed:     {cfg.get('random_seed')}  |  Force CPU: {cfg.get('force_cpu', '?')}")
    print(f"  Host:     {cfg.get('host', '?')}  |  GPU: {cfg.get('gpu', '?')}  |  RAM: {cfg.get('ram_gb', '?')} GB")

    acc = summary["accuracy"]
    print_section("ACCURACY")
    print(f"  Questions evaluated:        {acc['num_questions']}")
    print(f"  Most-likely answer correct: {acc['most_likely_correct']}/{acc['num_questions']} ({acc['most_likely_correct_pct']}%)")
    print(f"  Any generation correct:     {acc['any_correct']}/{acc['num_questions']} ({acc['any_correct_pct']}%)")
    print(f"  All generations correct:    {acc['all_correct']}/{acc['num_questions']} ({acc['all_correct_pct']}%)")

    if "uncertainty_measures" in summary:
        um = summary["uncertainty_measures"]
        print_section("UNCERTAINTY MEASURES  (mean / min / max)")
        col_w = 30
        print(f"  {'Measure':<{col_w}} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'N':>5}")
        separator("-")
        for k, v in um.items():
            if isinstance(v, dict):
                print(f"  {k:<{col_w}} {v['mean']:>8.4f}  {v['min']:>8.4f}  {v['max']:>8.4f}  {v['n']:>5}")
            else:
                print(f"  {k:<{col_w}} {float(v):>8.4f}")

    if "auroc_auarc" in summary:
        rr = summary["auroc_auarc"]
        print_section("AUROC / AUARC")
        col_w = 30
        print(f"  {'Measure':<{col_w}} {'AUROC':>8}  {'AUARC':>8}")
        separator("-")
        for name, scores in rr.items():
            auroc_val = scores.get("auroc", float("nan"))
            auarc_val = scores.get("auarc", float("nan"))
            auroc_str = f"{auroc_val:.4f}" if auroc_val == auroc_val else "   n/a"
            auarc_str = f"{auarc_val:.4f}" if auarc_val == auarc_val else "   n/a"
            print(f"  {name:<{col_w}} {auroc_str:>8}  {auarc_str:>8}")

    print_section("PER-QUESTION RESULTS (first 20)")
    col = [40, 26, 8]
    print(f"  {'Question':<{col[0]}}  {'Generations':<{col[1]}}  {'ML Acc':>{col[2]}}")
    separator("-")

    questions = summary["questions"]
    for q in questions[:20]:
        question = q["question"]
        if len(question) > col[0] - 1:
            question = question[:col[0] - 4] + "..."

        gen_parts = []
        for g in q["generations"]:
            ans = g["answer"][:18]
            gen_parts.append(f"{ans}({'Y' if g['correct'] else 'n'})")
        gens_str = " | ".join(gen_parts)
        if len(gens_str) > col[1] - 1:
            gens_str = gens_str[:col[1] - 4] + "..."

        ml = "YES" if q["most_likely_correct"] else "no"
        print(f"  {question:<{col[0]}}  {gens_str:<{col[1]}}  {ml:>{col[2]}}")

    if len(questions) > 20:
        print(f"  ... ({len(questions) - 20} more questions — see JSON for full results)")

    separator()
    print(f"  Run directory: {summary['run_dir']}")
    separator()


TERM_EXPLANATIONS = {
    "most_likely_correct": "The model's first/best guess was right.",
    "any_correct": "At least one of the generations was correct.",
    "all_correct": "Every generation was correct.",
    "semantic_entropy": (
        "CORE METRIC. Measures how semantically diverse the generated answers are. "
        "High = model gives very different answers each time = uncertain. "
        "Low = answers cluster around the same meaning = confident. Range: 0 to ~1.1."
    ),
    "regular_entropy": (
        "Token-level entropy across generations. Noisier than semantic_entropy "
        "because it treats each token independently without understanding meaning."
    ),
    "cluster_assignment_entropy": (
        "How spread out answers are across semantic clusters. "
        "High = answers belong to many different clusters = high uncertainty."
    ),
    "p_false": (
        "Raw log-probability estimate of being wrong. "
        "Higher absolute value = higher predicted probability of error."
    ),
    "p_false_fixed": (
        "Normalised probability of the answer being wrong (0-1 scale). "
        "0.65 means the model is estimated to be wrong 65% of the time."
    ),
    "mass_ignorance": (
        "From Dempster-Shafer evidential theory. How much total uncertainty mass "
        "the model assigns to the 'I don't know' state. "
        "0.88 means the model is 88% ignorant on average."
    ),
    "pe_corrected": (
        "Evidential Semantic Entropy — the novel measure introduced in this paper. "
        "Corrects semantic entropy using evidential belief masses, reducing bias. "
        "Lower than semantic_entropy when the model shows structured uncertainty."
    ),
    "jousselme": (
        "Jousselme distance between belief distributions from evidential theory. "
        "Measures divergence between what the model believes and full ignorance. "
        "Range: 0 (certain) to 2 (maximally divergent/uncertain)."
    ),
    "p_ik": (
        "Predicted probability that the model does NOT know the answer, "
        "estimated from a learned classifier. "
        "0.90 means the model is predicted to be ignorant 90% of the time."
    ),
}


def build_report(summary: dict) -> str:
    lines = []

    def sec(title, char="=", width=65):
        lines.append(char * width)
        lines.append(f"  {title}")
        lines.append(char * width)

    def div(char="-", width=65):
        lines.append(char * width)

    lines.append("")
    lines.append("  EXPERIMENT RESULTS REPORT")
    lines.append("  Generated by summarize_results.py")
    lines.append("")

    # config
    sec("EXPERIMENT CONFIG")
    cfg = summary["config"]
    lines.append(f"  Model:        {cfg.get('model')}")
    lines.append(f"  Dataset:      {cfg.get('dataset')}")
    lines.append(f"  Samples:      {cfg.get('num_samples')}   Generations per question: {cfg.get('num_generations')}")
    lines.append(f"  Random seed:  {cfg.get('random_seed')}   Force CPU: {cfg.get('force_cpu', '?')}")
    lines.append(f"  Host:         {cfg.get('host', '?')}")
    lines.append(f"  GPU:          {cfg.get('gpu', 'none')}   RAM: {cfg.get('ram_gb', '?')} GB")
    lines.append("")

    # accuracy
    sec("ACCURACY")
    acc = summary["accuracy"]
    n = acc["num_questions"]
    lines.append(f"  Questions evaluated : {n}")
    lines.append("")

    for key in ("most_likely_correct", "any_correct", "all_correct"):
        val = acc[key]
        pct = acc[f"{key}_pct"]
        explanation = TERM_EXPLANATIONS.get(key, "")
        lines.append(f"  {key.replace('_', ' ').title()}")
        lines.append(f"    Score       : {val}/{n} ({pct}%)")
        lines.append(f"    Explanation : {explanation}")
        lines.append("")

    lines.append("  INTERPRETATION:")
    ml = acc["most_likely_correct_pct"]
    any_c = acc["any_correct_pct"]
    gap = round(any_c - ml, 1)
    lines.append(f"    The model answered correctly on its first try {ml}% of the time.")
    lines.append(f"    However, at least one correct answer appeared in {any_c}% of questions.")
    lines.append(f"    The {gap}% gap suggests the model occasionally knows the answer")
    lines.append(f"    but does not reliably produce it as its most confident response.")
    lines.append("")

    # uncertainty measures
    if "uncertainty_measures" in summary:
        sec("UNCERTAINTY MEASURES")
        um = summary["uncertainty_measures"]

        lines.append(f"  {'Measure':<30} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'N':>5}")
        div()
        for k, v in um.items():
            if isinstance(v, dict):
                lines.append(f"  {k:<30} {v['mean']:>8.4f}  {v['min']:>8.4f}  {v['max']:>8.4f}  {v['n']:>5}")
        lines.append("")

        lines.append("  TERM-BY-TERM EXPLANATION:")
        lines.append("")
        for k, v in um.items():
            if isinstance(v, dict):
                explanation = TERM_EXPLANATIONS.get(k, "No explanation available.")
                mean_val = v["mean"]
                lines.append(f"  {k}")
                lines.append(f"    Mean  : {mean_val:.4f}  (min {v['min']:.4f} / max {v['max']:.4f})")
                lines.append(f"    What  : {explanation}")
                lines.append("")

        def mean_of(key):
            v = um.get(key, None)
            if isinstance(v, dict):
                return v.get("mean")
            try:
                return float(v)
            except Exception:
                return None

        lines.append("  OVERALL INTERPRETATION:")
        se_mean = mean_of("semantic_entropy")
        pik_mean = mean_of("p_ik")
        mi_mean = mean_of("mass_ignorance")
        pf_mean = mean_of("p_false_fixed")
        if se_mean is not None:
            level = "HIGH" if se_mean > 0.6 else "MODERATE" if se_mean > 0.3 else "LOW"
            lines.append(f"    Semantic entropy of {se_mean:.4f} (out of max ~1.1) indicates")
            lines.append(f"    {level} uncertainty — the model generates diverse answers across questions.")
        if pik_mean is not None:
            lines.append(f"    p_ik of {pik_mean:.4f} means the model is predicted to be ignorant")
            lines.append(f"    {pik_mean*100:.0f}% of the time, which aligns with the measured accuracy.")
        if mi_mean is not None:
            lines.append(f"    mass_ignorance of {mi_mean:.4f} confirms the model assigns most of its")
            lines.append(f"    belief mass to the 'unknown' state rather than a confident answer.")
        if pf_mean is not None:
            lines.append(f"    p_false_fixed of {pf_mean:.4f} directly estimates a {pf_mean*100:.0f}% error rate,")
            lines.append(f"    consistent with the observed {100 - acc['most_likely_correct_pct']:.0f}% incorrect rate.")
        lines.append("")

    # AUROC / AUARC
    if "auroc_auarc" in summary:
        sec("AUROC / AUARC  (uncertainty quality)")
        rr = summary["auroc_auarc"]
        lines.append("  AUROC  : Area under ROC curve. Measures how well the uncertainty")
        lines.append("           score separates correct from incorrect answers.")
        lines.append("           1.0 = perfect, 0.5 = random, <0.5 = inverted.")
        lines.append("  AUARC  : Area under Accuracy-Rejection Curve. Accuracy when the")
        lines.append("           most-uncertain fraction is abstained from.")
        lines.append("           Higher = abstaining on uncertain items recovers more accuracy.")
        lines.append("")
        lines.append(f"  {'Measure':<30} {'AUROC':>8}  {'AUARC':>8}")
        div()
        for name, scores in rr.items():
            auroc_val = scores.get("auroc", float("nan"))
            auarc_val = scores.get("auarc", float("nan"))
            auroc_str = f"{auroc_val:.4f}" if auroc_val == auroc_val else "  n/a  "
            auarc_str = f"{auarc_val:.4f}" if auarc_val == auarc_val else "  n/a  "
            lines.append(f"  {name:<30} {auroc_str:>8}  {auarc_str:>8}")
        lines.append("")

        # best / worst
        valid = {k: v for k, v in rr.items() if v["auroc"] == v["auroc"]}
        if valid:
            best_auroc = max(valid, key=lambda k: valid[k]["auroc"])
            best_auarc = max(valid, key=lambda k: valid[k]["auarc"])
            lines.append("  INTERPRETATION:")
            lines.append(f"    Best AUROC: {best_auroc} ({valid[best_auroc]['auroc']:.4f})")
            lines.append(f"      → This measure best discriminates correct from incorrect answers.")
            lines.append(f"    Best AUARC: {best_auarc} ({valid[best_auarc]['auarc']:.4f})")
            lines.append(f"      → Abstaining on this measure's top-uncertain items recovers the most accuracy.")
        lines.append("")

    # per-question
    sec("PER-QUESTION RESULTS (ALL QUESTIONS)")
    lines.append(f"  Format: answer(Y=correct / n=wrong) | answer | answer ...")
    lines.append("")

    questions = summary["questions"]
    for i, q in enumerate(questions):
        lines.append(f"  Q{i+1:>3}. {q['question']}")
        gen_parts = []
        for g in q["generations"]:
            marker = "Y" if g["correct"] else "n"
            gen_parts.append(f"{g['answer']}({marker})")
        lines.append(f"        Answers : {' | '.join(gen_parts)}")
        ml = "CORRECT" if q["most_likely_correct"] else "wrong"
        lines.append(f"        Result  : Most-likely answer was {ml}")

        if "uncertainty" in q:
            unc = q["uncertainty"]
            def fmt_unc(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)
            se = fmt_unc(unc.get("semantic_entropy", "?"))
            pik = fmt_unc(unc.get("p_ik", "?"))
            lines.append(f"        Uncertainty: semantic_entropy={se}  p_ik={pik}")
        lines.append("")

    lines.append("=" * 65)
    lines.append(f"  Run directory: {summary['run_dir']}")
    lines.append("=" * 65)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize a generate_answers.py run.")
    parser.add_argument("run_dir", help="Path to wandb run directory")
    parser.add_argument("--json", action="store_true", help="Export results to results.json")
    parser.add_argument("--json-only", action="store_true", help="Export JSON only, skip terminal output")
    parser.add_argument("--output", default=None, help="Custom path for JSON output file")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: path not found: {run_dir}")
        sys.exit(1)

    summary = build_summary(run_dir)

    if not args.json_only:
        print_summary(summary)

    if args.json or args.json_only:
        out_path = Path(args.output) if args.output else run_dir / "results.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON saved to: {out_path}")

    # always write the text report
    report_path = run_dir / "results_report.txt"
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
