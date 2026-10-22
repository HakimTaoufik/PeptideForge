# Experiments and evidence

[Landing page](../README.md) · [Research](RESEARCH.md) · [Reproducibility](REPRODUCIBILITY.md)

## Pipeline and artifact map

1. Rich AMP CSVs store sequences, labels, properties, and bins. Headerless core CSVs export three conditioning tags plus sequence and EOS.
2. Each model has a separate training CLI in `generator/`; there is no shared experiment config. CLI defaults, shell wrappers, and checkpoint `config` dictionaries are distinct evidence sources.
3. Training scripts select and save `best.pt` using their own validation objective. Seven checkpoints are tracked under `models/generator/`. Their contents are `model_state` and `config`; they do not store epochs, losses, optimizer states, seeds, or data hashes.
4. Generation artifacts contain 27 length × charge × hydrophobicity combinations with 10 samples each: **270 per model; 2,160 total**. [generate.sh](../generate.sh) describes such a sweep, but its current GPT-2 invocation is broken and its summary list includes only GPT-2. It is not a verified reconstruction of the historical run.
5. [scorer.py](../scorer/scorer.py) computes 35 composition/physicochemical features and applies a random forest at threshold 0.5. Metadata and stored predictions survive; its `.joblib` weights are absent.
6. [metrics_notebook.ipynb](../metrics_notebook.ipynb) contains historical plots and tables. The new [report utility](../scripts/research_report.py) recomputes descriptive CSV statistics without that scorer or notebook state.

The preserved [summary](../logs/summary.txt) marks all eight methods `OK`, with reported times from 49 seconds to 3m47s. It lacks a timestamp, device, commands, or run identifiers. Treat these as historical status records, not training durations or comparable speed measurements. Detailed `.log` files are not tracked in this checkout.

Checkpoint `config` inspection confirms the following saved sizes (all sequence limits are 64 tokens/positions):

| Checkpoint family | Saved architecture configuration |
|---|---|
| Transformer | Width 128, 4 heads, 4 layers, feed-forward width 512, vocabulary 31 |
| LSTM | Embedding 64, hidden width 192, 2 layers, dropout 0.35, concatenated tags |
| CVAE | Embedding 64, hidden width 256, latent width 32, 1 recurrent layer |
| Diffusion | Width 128, 4 heads, 4 layers, feed-forward width 512; 100 cosine-schedule steps |
| MaskGIT | Width 128, 4 heads, 4 layers, feed-forward width 512 |
| EBM | Width 128, 4 heads, 4 layers, feed-forward width 512; 2 coupling layers of width 32 |
| JEPA | Width 128, 4 heads, 4 encoder layers, 2 predictor layers; EMA decay 0.996 |

The six tag-vector models have vocabulary size 23 and nine possible tag values, with three active tags per example. No GPT-2 checkpoint configuration is available locally.

## Recoverable sample statistics

[generated_metrics.csv](assets/generated_metrics.csv) is computed directly from `logs/generation_*.csv`. Every denominator below is 270, including repeats and any property mismatches; all 2,160 saved sequences have nonempty standard-amino-acid strings.

| Model | Unique sequences | Absent from AMP train | Length match | Charge match | Hydro match |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 267 | 233 | 263 | 213 | 222 |
| Transformer | 257 | 214 | 256 | 226 | 207 |
| LSTM | 268 | 251 | 233 | 227 | 215 |
| CVAE | 270 | 270 | 204 | 200 | 192 |
| Diffusion | 270 | 270 | 145 | 135 | 144 |
| MaskGIT | 270 | 270 | 173 | 197 | 188 |
| EBM | 225 | 225 | 90 | 213 | 210 |
| JEPA | 267 | 261 | 90 | 192 | 194 |

Matches use the **rich CSV label convention**, not the notebook's conflicting length/charge boundaries. Novelty counts sample rows absent by exact sequence match from all 23,791 rich training records, including the first record omitted by current core loaders. It says nothing about sequence similarity or activity. Some saved sequences fall outside the training length range: GPT-2 9–50, LSTM 9–54, CVAE 10–53. Full ranges are in the CSV.

![Property matches and exact sequence statistics for the saved samples](assets/saved_samples.svg)

Figure source: generation CSVs and the Kyte–Doolittle scale in feature extraction. No reranking, random subsampling, or new model inference is used. Source SHA-256 hashes and split checks are in [audit.json](assets/audit.json).

## Stored classifier outputs

[stored_score_metrics.csv](assets/stored_score_metrics.csv) summarizes [scores_all_models.pkl](../logs/scores_all_models.pkl). The report checks that each archived sequence multiset matches its generation CSV. Probabilities and labels are **stored predictions**, not recomputed inference. Counts and displayed means:

| Model | Stored positive predictions / 270 | Mean stored P(AMP) |
|---|---:|---:|
| GPT-2 | 204 | 0.739290 |
| Transformer | 226 | 0.810664 |
| LSTM | 210 | 0.744801 |
| CVAE | 189 | 0.660698 |
| Diffusion | 196 | 0.674343 |
| MaskGIT | 217 | 0.729839 |
| EBM | 267 | 0.901681 |
| JEPA | 255 | 0.872934 |

These are classifier acceptance counts, not experimentally confirmed AMP hit rates. The scorer weights, calibration, and predictions cannot be regenerated from the current tree. The score notebook joins tags using `drop_duplicates(subset='sequence')`, assigning the first tag combination to repeated sequences. **Use original CSV rows for adherence**, not archived tags.

## Historical scorer validation

An earlier [scorer notebook at commit 6739d5d](https://github.com/HakimTaoufik/PeptideForge/blob/6739d5d16faf81b46dba2ffe46676a0457440212/noteboooks/scorer.ipynb) provides more evidence than the current tree alone. Its code cell at zero-based index 12 evaluates `rf_clf` on the validation split at threshold 0.5; the saved output reports:

| Accuracy | Precision | Recall | F1 | ROC AUC | Average precision (`pr_auc` in notebook) |
|---:|---:|---:|---:|---:|---:|
| 0.974992 | 0.934981 | 0.928379 | 0.931669 | 0.993901 | 0.981051 |

The notebook configures 300 trees, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, and `random_state=42`, and exports `rf_clf` with threshold 0.5. These are historical validation outputs, not newly reproduced or held-out test results, and there is no weight hash tying them conclusively to the current score archive. The validation sweep also considers other thresholds; no claim of independent test performance is made.

## Notebook caveats and missing results

- [evaluation_notebook.ipynb](../evaluation_notebook.ipynb) has only its imports/style cells executed. Later cells contain five manually entered samples per model, handwritten loss curves and scorecard numbers, and a manually selected “best” model. Those values are not promoted as measured results.
- `metrics_notebook.ipynb` has genuine saved outputs, but cells are stored out of execution order. Plot cells precede data initialization, and composite rankings predate later metric-cell executions. Its header says five samples per combination although the artifacts contain ten. It is not currently restart-and-run-all reproducible.
- Notebook “edit distance” is padded positional mismatch, not Levenshtein distance; diversity subsampling uses an unseeded random choice. Its composite weights are subjective. Neither is used for a new ranking here.
- No generator training curves with machine-readable run provenance, confirmed checkpoint validation losses, multi-seed runs, test-set generator likelihoods, or biological measurements were recovered. Checkpoint configs verify architecture, not the historical training invocation.

For future controlled runs, fill this record for **each** model rather than mixing objectives or copying example numbers:

| Model | Git/data/checkpoint hashes | Seed + exact CLI | Device/runtime | Validation objective + value | Held-out test metric + value | Repeated runs/uncertainty |
|---|---|---|---|---|---|---|
| Each of the eight methods | Not recorded | Not recorded | Not recorded | Not recovered | Not recovered | Not recorded |

This documentation pass performs no retraining and leaves all historical artifacts intact.
