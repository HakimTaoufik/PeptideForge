# Reproducibility

[Landing page](../README.md) · [Research](RESEARCH.md) · [Experiments](EXPERIMENTS.md)

## Start with saved evidence

From the repository root, with Python 3.14 as declared in [pyproject.toml](../pyproject.toml) and [.python-version](../.python-version):

```bash
python3 scripts/research_report.py --help
python3 scripts/research_report.py --output-dir /tmp/peptideforge-audit
```

This path needs only Python's standard library. It reads source CSVs and writes derived CSV/JSON reports to the specified directory. It does not load checkpoints or the score pickle. The script also works when launched outside the repository root.

## Minimal CPU environment and inference

With `uv` installed, these commands create an isolated environment and request specific versions without changing `pyproject.toml` or `uv.lock`. Network access and compatible Python 3.14 wheels are required. CPU setup is a minimal inference/report environment, not an installation of every notebook or GPT-2 dependency.

```bash
uv venv --python 3.14 .venv-cpu
uv --no-config pip install --python .venv-cpu/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv --no-config pip install --python .venv-cpu/bin/python numpy==2.4.3 pandas==3.0.1 matplotlib==3.10.8
source .venv-cpu/bin/activate
python generator/lstm/generate_lstm.py --help
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python generator/lstm/generate_lstm.py \
  --model_path models/generator/lstm/best.pt \
  --length_tag '<LEN_MED>' --charge_tag '<CHARGE_MED>' --hydro_tag '<HYDRO_MED>' --n 1
```

The LSTM command reads a tracked checkpoint and prints a stochastic sequence; it does not overwrite saved samples. Use `python`, not `uv run`, inside this environment to avoid automatic project synchronization. Inspect [PyTorch's installation selector](https://docs.pytorch.org/get-started/locally/) for other platforms and GPU builds. The project scripts select `cuda` when available and otherwise CPU; they do not select Apple MPS. ROCm PyTorch also uses the `torch.cuda` API.

The CPU pins are a tested subset based on existing dependency versions (the lock has `torch==2.10.0+rocm7.1`). They pin direct requirements rather than the full dependency graph. The [resolved smoke-test environment](assets/cpu-smoke-environment.txt) records all installed versions; see the validation record below for platform details.

To regenerate all committed reports and the figure:

```bash
MPLCONFIGDIR=/tmp/peptideforge-mpl python scripts/research_report.py --figures --stored-scores
```

`--stored-scores` explicitly opts into deserializing the repository's trusted pandas pickle; use it only with a trusted checkout. It requires pandas, verifies sequence multisets, and exports existing predictions without the missing scorer weights. `--figures` requires matplotlib. The two CSV exports retain exact counts/full numeric means; the figure shows proportions. [audit.json](assets/audit.json) records source hashes. Without these flags, the base command emits only raw CSV statistics and the split/hash audit.

## Other existing generation entry points

These commands use tracked checkpoints and were smoke-tested on CPU with one sequence. Sampling is unseeded. The EBM example deliberately uses 10 updates for a cheap execution check rather than its 500-step default.

```bash
python generator/transformers/generate_transformer.py --model_path models/generator/transformers/best.pt --n 1
python generator/vae/generate_vae.py --model_path models/generator/vae/best.pt --mode sample --n 1
python generator/diffusion/generate_diffusion.py --model_path models/generator/diffusion/best.pt --mode ancestral --n 1
python generator/maskgit/generate_maskgit.py --model_path models/generator/maskgit/best.pt --mode iterative --n 1
python generator/energy/generate_ebm.py --model_path models/generator/energy/best.pt --mode anneal --n_steps 10 --n 1
python generator/jepa/generate_jepa.py --model_path models/generator/jepa/best.pt --mode refine --n 1
```

Each uses medium tags by default; all accept `--length_tag`, `--charge_tag`, and `--hydro_tag`. Quote angle-bracket tags in the shell. Fixed-grid samplers also accept `--seq_len`; inspect their `--help` before comparing length control.

## Training and unavailable components

No training is needed to inspect the portfolio. The model table links every train/generate script. As a concrete command template for a **future** LSTM training run:

```bash
python generator/lstm/train_lstm.py --help
python generator/lstm/train_lstm.py \
  --train_csv data/generator/train_core.csv --val_csv data/generator/val_core.csv \
  --output_dir outputs/lstm-new-run --epochs 500 --batch_size 64 --lr 1e-3
```

The CLI was checked; training was not run. A GPU is recommended for full training. This command intentionally retains the existing core loader behavior, which treats the first data line as a header and uses **23,790 train / 2,973 validation samples**. Fixing that loader would change the experiment and is outside this presentation pass. Rich AMP CSVs are not drop-in replacements: trainers read the first column, which is `id` in those files. New outputs go under ignored `outputs/` rather than replacing tracked checkpoints.

GPT-2 training/generation additionally requires Transformers and, for training, Accelerate plus network/cache access for `gpt2`. Its actual CLI argument is `--model_path`; `generate.sh` passes the unsupported `--model_dir`. There is no `models/generator/gpt2/` directory in this checkout. Its saved CSVs are inspectable, but full generation is blocked until compatible fine-tuned weights are supplied. [train.py](../generator/gpt2/train.py) and [generate.py](../generator/gpt2/generate.py) also have `/Data/protein/…` defaults; override paths rather than assuming that location exists.

The scorer requires scikit-learn/joblib and `rf_amp_scorer_v1.joblib`, which is not tracked. Its defaults start with `../models/` and depend on the working directory. Existing predictions remain available; neither scorer inference nor the [Marimo demo](../demo.py), which relies on it, was validated end to end. The natural-language parser also requires a downloaded MiniLM model. Do not present these as working offline quickstarts.

## Existing environment and hardware assumptions

[uv.lock](../uv.lock) is preserved unchanged. It selects PyTorch 2.10.0+ROCm7.1 and resolves several other packages through the ROCm index. `pyproject.toml` enables prereleases and `unsafe-best-match`; the `amd` and `nvidia` extras do not make this lock a portable CPU/CUDA installation. Plain `uv sync` can install the locked ROCm build via transitive dependencies, even without an extra. The historical package name `jepa-protein` is retained to avoid unnecessary lock changes; public-facing documentation uses PeptideForge.

No original training GPU model, memory budget, or trustworthy full-run duration was recovered. The ROCm dependency choice indicates intended AMD support, not verified training hardware. Seven checkpoints are small enough for the CPU smoke tests; this does not establish full training resource requirements.

## Validation record

Presentation audit baseline: `f5a7662445f890496563eacec8dfb99f447723c0`. Initial validation used the existing host environment: Python 3.14.8, PyTorch 2.14.1 (ROCm 7.2 build, `torch.cuda.is_available() == False`), NumPy 2.5.4, and pandas 2.3.3. The exact quickstart installation then succeeded in a separate `.venv-cpu` on Linux x86_64: Python 3.14.8, PyTorch 2.10.0+cpu, NumPy 2.4.3, pandas 3.0.1, and matplotlib 3.10.8. All seven generation smoke tests and full report regeneration also passed there. This tests CPU inference and reporting, not the original training environment.

- All 14 non-GPT-2 train/generate `--help` commands succeeded. All seven tracked checkpoints loaded; one-sample generation succeeded for each. EBM used 10 steps. PyTorch emitted mask-type/nested-tensor warnings for Transformer-based methods; execution still succeeded.
- The standard-library audit, full CSV/pickle summaries, and figure generation ran successfully. Source artifacts were read only; no training, scoring inference, notebook execution, or model downloads occurred.
- `uv --no-cache lock --check --offline` passed after the metadata-only edit. No dependency upgrades were performed.
- GPT-2 help could not execute in the host environment because Transformers is unavailable; both GPT-2 CLIs were parsed and their arguments inspected statically. Training CLI inspection does not establish compatibility of the installed Transformers version.
- The documentation's relative file/directory links and Markdown heading anchors were checked, and the generated SVG was inspected visually and parsed as XML. README/detailed result tables were checked against the exported CSVs. The report also ran from outside the repository root; all source hashes and split/count invariants passed.
- `git diff --check` and shell syntax checks (`bash -n generate.sh run_all.sh`) passed. Model/data/checkpoint/log/notebook files and `uv.lock` have no diff from the audit baseline. No files were deleted. The documentation adds a Mermaid architecture/pipeline diagram and one plotted figure; no training curves or rankings were invented.

## Artifact policy

Keep datasets, notebooks including outputs, generation logs, score archives, and existing checkpoints as historical evidence. No existing paths were relocated and no artifacts were deleted in this pass; no `.DS_Store` file was found. `.gitignore` now excludes OS metadata, the CPU virtual environment, and new `outputs/`/`runs/` directories while allowing intentional documentation figures.

For future large artifacts, use a release, Git LFS, or external storage with explicit provenance, license, immutable version, size, and SHA-256 manifest. Do not migrate or remove currently tracked files without a separate agreed change. Avoid committing new binary runs to ordinary Git or rewriting the existing history.
