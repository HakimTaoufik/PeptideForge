#!/usr/bin/env python3
"""Audit saved CSVs without training; optionally summarize trusted stored scores.

Base report uses only the standard library. --figures needs matplotlib;
--stored-scores needs pandas and deserializes the repository's pickle.
Run from any directory. All source artifacts are read-only.
"""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gpt_2": "GPT-2", "transformer": "Transformer", "lstm": "LSTM",
    "cvae": "CVAE", "diffusion": "Diffusion", "maskgit": "MaskGIT",
    "ebm": "EBM", "jepa": "JEPA",
}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def properties(sequence, scale):
    return (
        len(sequence),
        sequence.count("K") + sequence.count("R")
        - sequence.count("D") - sequence.count("E"),
        math.fsum(scale[a] for a in sequence) / len(sequence),
    )


def tag_values(length, charge, hydro):
    # Length/charge boundaries inferred from the rich CSV labels, NOT the
    # notebook's conflicting half-open bins. Hydro is an unrounded mean.
    return (
        "<LEN_SHORT>" if length <= 20 else "<LEN_MED>" if length <= 35 else "<LEN_LONG>",
        "<CHARGE_LOW>" if charge <= 1 else "<CHARGE_MED>" if charge <= 4 else "<CHARGE_HIGH>",
        "<HYDRO_LOW>" if hydro < -1 else "<HYDRO_MED>" if hydro < 1 else "<HYDRO_HIGH>",
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "peptideforge", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    x = list(range(len(rows)))
    for offset, (key, label, color) in enumerate([
        ("length_match", "Length", "#2563eb"),
        ("charge_match", "Charge", "#0f766e"),
        ("hydro_match", "Hydrophobicity", "#b45309"),
    ]):
        axes[0].bar([i + (offset - 1) * .24 for i in x],
                    [r[key] / r["samples"] for r in rows], .24, label=label, color=color)
    for offset, (key, label, color) in enumerate([
        ("unique", "Unique / samples", "#475569"),
        ("novel_vs_train", "Absent from train / samples", "#7c3aed"),
    ]):
        axes[1].bar([i + (offset - .5) * .34 for i in x],
                    [r[key] / r["samples"] for r in rows], .34, label=label, color=color)
    for ax, title in zip(axes, ["Requested property matches", "Exact sequence statistics"]):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, [r["model"] for r in rows], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="lower left", fontsize=8, framealpha=.95)
    fig.suptitle("Saved generation artifacts · 270 samples per model", fontsize=14, fontweight="bold")
    fig.supxlabel("CSV label boundaries; exact-match novelty only. No biological activity measured.", fontsize=9)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    if path.suffix == ".svg":
        # Matplotlib emits trailing spaces inside multiline path attributes.
        # Newlines still separate coordinates after stripping line-end spaces.
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets",
                        help="Directory for derived CSV/JSON reports (default: docs/assets)")
    parser.add_argument("--figures", action="store_true", help="Also create saved_samples.svg (requires matplotlib)")
    parser.add_argument("--stored-scores", action="store_true",
                        help="Deserialize the trusted repository pickle with pandas; export historical classifier summaries")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"scripts/research_report.py"}

    def load(relative):
        sources.add(relative)
        return read_csv(ROOT / relative)

    feature_source = "training/utils/feature_extraction.py"
    sources.add(feature_source)
    module = ast.parse((ROOT / feature_source).read_text())
    scale = next(ast.literal_eval(n.value) for n in module.body
                 if isinstance(n, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "KYTE_DOOLITTLE" for t in n.targets))
    datasets = {}
    sets = {}
    for group, pattern in [
        ("amp", "data/generator/{}_generator_amp_only.csv"),
        ("scorer", "data/scorer/{}_scorer.csv"),
    ]:
        for split in ("train", "val", "test"):
            data = load(pattern.format(split))
            sequences = {r["sequence"] for r in data}
            sets[group, split] = sequences
            entry = {
                "rows": len(data), "unique": len(sequences),
                "labels": dict(sorted(Counter(r["label"] for r in data).items())),
                "min_length": min(map(len, sequences)), "max_length": max(map(len, sequences)),
                "valid_aa": sum(bool(r["sequence"]) and set(r["sequence"]) <= AA for r in data),
            }
            if group == "amp":
                relative = f"data/generator/{split}_core.csv"
                sources.add(relative)
                lines = (ROOT / relative).read_text().splitlines()
                parsed = ["".join(p for p in line.split() if not p.startswith("<")) for line in lines]
                entry.update(core_lines=len(lines), loader_rows=len(lines) - 1,
                             core_matches_rich_order=parsed == [r["sequence"] for r in data])
                entry["observed_label_ranges"] = {
                    prop: {
                        tag: [min(float(r[prop]) for r in data if r[column] == tag),
                              max(float(r[prop]) for r in data if r[column] == tag)]
                        for tag in sorted({r[column] for r in data})
                    } for prop, column in [("length", "length_bin"), ("charge", "charge_bin"),
                                            ("hydrophobicity", "hydro_bin")]
                }
            datasets[f"{group}_{split}"] = entry

    rows = []
    raw = {}
    for key, name in MODELS.items():
        data = load(f"logs/generation_{key}.csv")
        raw[key] = data
        valid = [r for r in data if r["sequence"] and set(r["sequence"]) <= AA]
        matches = Counter()
        for r in valid:
            tags = tag_values(*properties(r["sequence"], scale))
            for tag, col in zip(tags, ["length_tag", "charge_tag", "hydro_tag"]):
                matches[col] += tag == r[col]
        combos = Counter(tuple(r[c] for c in ["length_tag", "charge_tag", "hydro_tag"]) for r in data)
        rows.append({
            "model": name, "samples": len(data), "valid_aa": len(valid),
            "unique": len({r["sequence"] for r in data}),
            "novel_vs_train": sum(r["sequence"] not in sets["amp", "train"] for r in data),
            "length_match": matches["length_tag"], "charge_match": matches["charge_tag"],
            "hydro_match": matches["hydro_tag"], "condition_combinations": len(combos),
            "min_samples_per_combination": min(combos.values()),
            "max_samples_per_combination": max(combos.values()),
            "min_length": min(len(r["sequence"]) for r in data),
            "max_length": max(len(r["sequence"]) for r in data),
        })
    write_csv(args.output_dir / "generated_metrics.csv", rows)
    if args.stored_scores:
        import pandas as pd
        relative = "logs/scores_all_models.pkl"
        sources.add(relative)
        scores = pd.read_pickle(ROOT / relative)
        summaries = []
        for key, name in MODELS.items():
            df = scores[key]
            if Counter(df.sequence) != Counter(r["sequence"] for r in raw[key]):
                raise ValueError(f"{name}: score archive and generation CSV sequences differ")
            # Tags in the pickle were merged by deduplicated sequence, so never
            # use its tags to compute adherence for repeated sequences.
            summaries.append({
                "model": name, "samples": len(df),
                "stored_predicted_amp": int(df.amp_prediction.sum()),
                "stored_mean_probability": float(df.amp_probability.mean()),
            })
        write_csv(args.output_dir / "stored_score_metrics.csv", summaries)
    report = {
        "definitions": {
            "novelty": "Number of sample rows absent by exact sequence match from the full rich AMP training CSV; duplicates count per row.",
            "charge": "count(K)+count(R)-count(D)-count(E)",
            "hydrophobicity": "Unrounded math.fsum Kyte-Doolittle values / length; floating-point boundary cases can differ from NumPy.",
            "length_bins": "<=20; 21..35; >=36 (inferred from rich CSV labels)",
            "charge_bins": "<=1; 2..4; >=5 (inferred from rich CSV labels)",
            "hydro_bins": "<-1; [-1,1); >=1",
        },
        "datasets": datasets,
        "exact_split_overlaps": {
            f"{group}_{a}_{b}": len(sets[group, a] & sets[group, b])
            for group in ("amp", "scorer")
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        },
        "amp_subset_of_scorer": {split: sets["amp", split] <= sets["scorer", split]
                                 for split in ("train", "val", "test")},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(sources)},
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.figures:
        plot_metrics(rows, args.output_dir / "saved_samples.svg")
    print(f"Audited {len(rows)} models / {sum(r['samples'] for r in rows)} saved samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
