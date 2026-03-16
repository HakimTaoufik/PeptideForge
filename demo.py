import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Peptide Generator")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import subprocess
    import os
    import re
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from io import BytesIO
    from collections import Counter
    from math import log, pi, cos, sin
    import base64
    import pandas as pd
    from scorer.scorer import main_scorer
    return (
        mo, np, subprocess, os, re, itertools, matplotlib, plt,
        Patch, Line2D, BytesIO, Counter, log, pi, cos, sin, base64, pd, main_scorer,
    )


@app.cell
def _(mo):
    mo.md("""
# Conditional Peptide Generator

Select a **model**, **property tags**, then generate.  
10 candidates are produced, scored by the AMP classifier, and the best is analyzed.
    """)
    return


@app.cell
def _(np):
    BASE_DIR = "./models/generator"
    SCRIPT_DIR = "./generator"

    MODEL_CONFIG = {
        "GPT-2": {
            "script": f"{SCRIPT_DIR}/gpt2/generate.py",
            "model_path": f"{BASE_DIR}/gpt2/",
            "extra_args": [],
        },
        "Transformer": {
            "script": f"{SCRIPT_DIR}/transformers/generate_transformer.py",
            "model_path": f"{BASE_DIR}/transformers/best.pt",
            "extra_args": [],
        },
        "LSTM": {
            "script": f"{SCRIPT_DIR}/lstm/generate_lstm.py",
            "model_path": f"{BASE_DIR}/lstm/best.pt",
            "extra_args": ["--condition_mode", "concat"],
        },
        "CVAE": {
            "script": f"{SCRIPT_DIR}/vae/generate_vae.py",
            "model_path": f"{BASE_DIR}/vae/best.pt",
            "extra_args": ["--mode", "sample"],
        },
        "Diffusion": {
            "script": f"{SCRIPT_DIR}/diffusion/generate_diffusion.py",
            "model_path": f"{BASE_DIR}/diffusion/best.pt",
            "extra_args": ["--mode", "ancestral", "--temperature", "1.2"],
        },
        "MaskGIT": {
            "script": f"{SCRIPT_DIR}/maskgit/generate_maskgit.py",
            "model_path": f"{BASE_DIR}/maskgit/best.pt",
            "extra_args": ["--mode", "iterative", "--temperature", "1.6"],
        },
        "EBM": {
            "script": f"{SCRIPT_DIR}/energy/generate_ebm.py",
            "model_path": f"{BASE_DIR}/energy/best.pt",
            "extra_args": ["--mode", "anneal", "--n_steps", "500"],
        },
        "JEPA": {
            "script": f"{SCRIPT_DIR}/jepa/generate_jepa.py",
            "model_path": f"{BASE_DIR}/jepa/best.pt",
            "extra_args": ["--mode", "refine", "--temperature", "1.1"],
        },
    }

    TAG_BIN_MAP = {
        "length": {
            "<LEN_SHORT>": (-np.inf, 20),
            "<LEN_MED>":   (20, 35),
            "<LEN_LONG>":  (35, np.inf),
        },
        "charge": {
            "<CHARGE_LOW>":  (-np.inf, 1),
            "<CHARGE_MED>":  (1, 4),
            "<CHARGE_HIGH>": (4, np.inf),
        },
        "hydrophobicity": {
            "<HYDRO_LOW>":  (-np.inf, -1.0),
            "<HYDRO_MED>":  (-1.0, 1.0),
            "<HYDRO_HIGH>": (1.0, np.inf),
        },
    }

    return BASE_DIR, SCRIPT_DIR, MODEL_CONFIG, TAG_BIN_MAP


@app.cell
def _(mo, MODEL_CONFIG):
    model_dropdown = mo.ui.dropdown(
        options=list(MODEL_CONFIG.keys()), value="Transformer", label="Model",
    )
    length_tag = mo.ui.dropdown(
        options=["<LEN_SHORT>", "<LEN_MED>", "<LEN_LONG>"], value="<LEN_MED>", label="Length",
    )
    charge_tag = mo.ui.dropdown(
        options=["<CHARGE_LOW>", "<CHARGE_MED>", "<CHARGE_HIGH>"], value="<CHARGE_MED>", label="Charge",
    )
    hydro_tag = mo.ui.dropdown(
        options=["<HYDRO_LOW>", "<HYDRO_MED>", "<HYDRO_HIGH>"], value="<HYDRO_MED>", label="Hydrophobicity",
    )
    generate_btn = mo.ui.run_button(label="Generate 10 peptides")

    mo.hstack([model_dropdown, length_tag, charge_tag, hydro_tag, generate_btn], gap=1, justify="start")
    return model_dropdown, length_tag, charge_tag, hydro_tag, generate_btn


@app.cell
def _():
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    KD_SCALE = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }

    CHARGE_MAP = {_a: 0 for _a in AMINO_ACIDS}
    CHARGE_MAP.update({"R": 1, "K": 1, "H": 0.1, "D": -1, "E": -1})

    AA_GROUPS = {
        "Hydrophobic": list("AILMFVW"), "Polar": list("STNQ"),
        "Positive": list("RKH"), "Negative": list("DE"), "Special": list("CGPY"),
    }
    GRP_COLORS = {
        "Hydrophobic": "#D97706", "Polar": "#059669",
        "Positive": "#2563EB", "Negative": "#DC2626", "Special": "#7C3AED",
    }

    AA_TO_COLOR = {}
    for _gn, _aas in AA_GROUPS.items():
        for _a in _aas:
            AA_TO_COLOR[_a] = GRP_COLORS[_gn]

    STYLE = {
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E5E7EB", "axes.labelcolor": "#111827",
        "text.color": "#111827", "xtick.color": "#6B7280", "ytick.color": "#6B7280",
        "grid.color": "#F3F4F6", "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#E5E7EB", "font.size": 10, "figure.dpi": 150,
    }

    return AMINO_ACIDS, KD_SCALE, CHARGE_MAP, AA_GROUPS, GRP_COLORS, AA_TO_COLOR, STYLE


@app.cell
def _(mo, generate_btn, model_dropdown, length_tag, charge_tag, hydro_tag,
      MODEL_CONFIG, TAG_BIN_MAP, subprocess, re, np, pd, main_scorer,
      AMINO_ACIDS, itertools, log):
    generate_btn

    _model = model_dropdown.value
    _cfg = MODEL_CONFIG[_model]
    _lt = length_tag.value
    _ct = charge_tag.value
    _ht = hydro_tag.value

    _cmd = [
        "uv", "run", _cfg["script"],
        "--model_path", _cfg["model_path"],
        "--length_tag", _lt, "--charge_tag", _ct, "--hydro_tag", _ht,
        "--n", "10",
    ] + _cfg["extra_args"]

    try:
        _r = subprocess.run(_cmd, capture_output=True, text=True, timeout=180)
        gen_output = _r.stdout + _r.stderr
    except subprocess.TimeoutExpired:
        gen_output = "TIMEOUT — generation exceeded 180s"
    except FileNotFoundError:
        gen_output = "[Demo mode — model script not found]\n"
        _rng = np.random.default_rng(42)
        for _idx in range(10):
            _ln = _rng.integers(12, 35)
            _sq = "".join(_rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=_ln))
            gen_output += f"  {_idx+1}. {_sq} (len={_ln})\n"

    _raw = []
    for _line in gen_output.split("\n"):
        _m = re.search(r"\d+\.\s+([A-Z]{5,})", _line.strip())
        if _m:
            _sq = re.sub(r"\s.*", "", _m.group(1))
            if all(_c in AMINO_ACIDS for _c in _sq):
                _raw.append(_sq)
    if not _raw:
        for _line in gen_output.split("\n"):
            for _w in _line.split():
                _w = _w.strip("().,;:")
                if len(_w) >= 5 and all(_c in AMINO_ACIDS for _c in _w):
                    _raw.append(_w)

    _seen = set()
    _unique = []
    for _sq in _raw:
        if _sq not in _seen:
            _seen.add(_sq)
            _unique.append(_sq)
    _unique = _unique[:10]

    # ── Score with main_scorer ──
    if _unique:
        scored_df = main_scorer(_unique)
    else:
        scored_df = pd.DataFrame(columns=[
            "sequence", "length", "charge", "hydrophobicity",
            "frac_basic", "frac_acidic", "cysteine_count",
            "amp_probability", "amp_prediction",
        ])

    # ── Compute extra metrics ──
    def _tag_match(_row):
        _ok = 0
        _lo, _hi = TAG_BIN_MAP["length"][_lt]
        if _lo <= _row["length"] < _hi: _ok += 1
        _lo, _hi = TAG_BIN_MAP["charge"][_ct]
        if _lo <= _row["charge"] < _hi: _ok += 1
        _lo, _hi = TAG_BIN_MAP["hydrophobicity"][_ht]
        if _lo <= _row["hydrophobicity"] < _hi: _ok += 1
        return _ok / 3

    def _seq_entropy(_seq):
        _cnt = {}
        for _c in _seq: _cnt[_c] = _cnt.get(_c, 0) + 1
        _tot = len(_seq)
        return -sum((_v/_tot)*log(_v/_tot+1e-10) for _v in _cnt.values()) / log(20)

    def _max_repeat(_seq):
        return max(len(list(_g)) for _, _g in itertools.groupby(_seq)) if _seq else 0

    if len(scored_df) > 0:
        scored_df = scored_df.copy()
        scored_df["tag_adherence"] = scored_df.apply(_tag_match, axis=1)
        scored_df["aa_entropy"] = scored_df["sequence"].apply(_seq_entropy)
        scored_df["unique_aas"] = scored_df["sequence"].apply(lambda _s: len(set(_s)))
        scored_df["max_repeat"] = scored_df["sequence"].apply(_max_repeat)
        scored_df["repeat_pen"] = scored_df.apply(
            lambda _r: max(0, 1 - (_r["max_repeat"] - 3) / _r["length"]) if _r["length"] > 0 else 0, axis=1)
        scored_df["composite"] = (
            0.30 * scored_df["amp_probability"]
            + 0.30 * scored_df["tag_adherence"]
            + 0.15 * scored_df["aa_entropy"]
            + 0.10 * (scored_df["unique_aas"] / 20)
            + 0.10 * scored_df["repeat_pen"]
            + 0.05 * scored_df["amp_prediction"].astype(float)
        )
        scored_df = scored_df.sort_values("composite", ascending=False).reset_index(drop=True)

    best_seq = scored_df.iloc[0]["sequence"] if len(scored_df) > 0 else ""
    chosen_model = _model
    chosen_tags = (_lt, _ct, _ht)

    return gen_output, scored_df, best_seq, chosen_model, chosen_tags


@app.cell
def _(mo, scored_df, best_seq, chosen_model, chosen_tags):
    if len(scored_df) == 0:
        mo.stop(True, mo.md("No sequences generated. Check model paths."))

    _lt, _ct, _ht = chosen_tags
    _rows = []
    for _i, _row in scored_df.iterrows():
        _sq = _row["sequence"]
        _rows.append({
            "#": _i + 1,
            "Sequence": _sq[:35] + ("..." if len(_sq) > 35 else ""),
            "Len": int(_row["length"]),
            "Charge": round(float(_row["charge"]), 1),
            "Hydro": round(float(_row["hydrophobicity"]), 2),
            "P(AMP)": round(float(_row["amp_probability"]), 3),
            "AMP": "yes" if _row["amp_prediction"] == 1 else "no",
            "Tags": f"{_row['tag_adherence']:.0%}",
            "Entropy": round(float(_row["aa_entropy"]), 3),
            "Score": round(float(_row["composite"]), 4),
            "": "best" if _sq == best_seq else "",
        })

    mo.md(f"""
## {chosen_model} — `{_lt}` `{_ct}` `{_ht}`

{mo.ui.table(_rows, selection=None)}
    """)
    return


@app.cell
def _(mo, best_seq, scored_df, AA_TO_COLOR):
    if not best_seq:
        mo.stop(True, mo.md(""))

    best_row = scored_df.iloc[0]

    _colored = "".join(
        f'<span style="display:inline-block;width:24px;height:28px;line-height:28px;text-align:center;'
        f'margin:1px;border-radius:4px;font-family:monospace;font-size:14px;font-weight:700;'
        f'background:{AA_TO_COLOR.get(_c, "#9CA3AF")}18;color:{AA_TO_COLOR.get(_c, "#6B7280")};'
        f'border:1px solid {AA_TO_COLOR.get(_c, "#D1D5DB")}44">{_c}</span>'
        for _c in best_seq
    )

    _amp_ok = best_row["amp_prediction"] == 1
    _amp_color = "#059669" if _amp_ok else "#DC2626"
    _amp_text = f"Predicted AMP (P={best_row['amp_probability']:.3f})" if _amp_ok else f"Not AMP (P={best_row['amp_probability']:.3f})"

    mo.md(f"""
### Best candidate

<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:12px;padding:24px;margin:12px 0">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
<span style="font-size:12px;color:#6B7280;text-transform:uppercase;letter-spacing:1px">Sequence — {int(best_row['length'])} residues</span>
<span style="background:{_amp_color}15;color:{_amp_color};font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px;border:1px solid {_amp_color}33">{_amp_text}</span>
</div>
<div style="line-height:2.2;overflow-x:auto">{_colored}</div>
<div style="margin-top:16px;display:flex;gap:16px;font-size:11px;color:#9CA3AF">
<span><span style="color:#D97706">&#9632;</span> Hydrophobic</span>
<span><span style="color:#059669">&#9632;</span> Polar</span>
<span><span style="color:#2563EB">&#9632;</span> Positive</span>
<span><span style="color:#DC2626">&#9632;</span> Negative</span>
<span><span style="color:#7C3AED">&#9632;</span> Special</span>
</div>
</div>

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0">
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px">
<div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.5px">Net charge</div>
<div style="font-size:22px;font-weight:700;color:#111827">{best_row['charge']:+.1f}</div>
</div>
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px">
<div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.5px">Hydrophobicity</div>
<div style="font-size:22px;font-weight:700;color:#111827">{best_row['hydrophobicity']:.3f}</div>
</div>
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px">
<div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.5px">Frac basic / acidic</div>
<div style="font-size:22px;font-weight:700;color:#111827">{best_row['frac_basic']:.2f}<span style="color:#9CA3AF;font-size:14px"> / </span>{best_row['frac_acidic']:.2f}</div>
</div>
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px">
<div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.5px">Tag adherence</div>
<div style="font-size:22px;font-weight:700;color:#111827">{best_row['tag_adherence']:.0%}</div>
</div>
</div>
    """)
    return best_row,


@app.cell
def _(mo, best_seq, best_row, np, Counter, plt, BytesIO, base64,
      AMINO_ACIDS, AA_GROUPS, GRP_COLORS, AA_TO_COLOR, KD_SCALE, CHARGE_MAP,
      Patch, STYLE, log, pi, cos, sin):
    if not best_seq:
        mo.stop(True, mo.md(""))

    plt.rcParams.update(STYLE)

    def _img(_fig):
        _buf = BytesIO()
        _fig.savefig(_buf, format="png", bbox_inches="tight", facecolor="#FFF", dpi=160, pad_inches=0.12)
        plt.close(_fig)
        return f'<img src="data:image/png;base64,{base64.b64encode(_buf.getvalue()).decode()}" style="width:100%;border-radius:8px;border:1px solid #E5E7EB"/>'

    _n = len(best_seq)
    _pos = np.arange(1, _n + 1)

    # ── Composition donut ──
    _f1, _a1 = plt.subplots(figsize=(4.2, 4.2))
    _gc = {_gn: sum(1 for _c in best_seq if _c in _aas) for _gn, _aas in AA_GROUPS.items()}
    _nz = {_k: _v for _k, _v in _gc.items() if _v > 0}
    _, _tx, _at = _a1.pie(
        _nz.values(), labels=[f"{_k}\n{_v}" for _k, _v in _nz.items()],
        colors=[GRP_COLORS[_k] for _k in _nz], autopct="%1.0f%%",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5))
    for _x in _at: _x.set_fontsize(8); _x.set_fontweight("bold"); _x.set_color("#374151")
    for _x in _tx: _x.set_fontsize(7.5); _x.set_color("#6B7280")
    _a1.set_title("Composition", fontsize=11, fontweight="600", pad=10)
    _h1 = _img(_f1)

    # ── Helical wheel ──
    _f2, _a2 = plt.subplots(figsize=(4.2, 4.2))
    _a2.set_aspect("equal")
    _nr = min(_n, 18)
    for _j in range(_nr):
        _rad = (_j * 100 - 90) * pi / 180
        _xp, _yp = cos(_rad), sin(_rad)
        _c = best_seq[_j]; _h = KD_SCALE.get(_c, 0)
        if _h > 1: _fc, _ec = "#FEF3C7", "#D97706"
        elif _h < -1: _fc, _ec = "#DBEAFE", "#2563EB"
        else: _fc, _ec = "#F3F4F6", "#9CA3AF"
        _a2.add_patch(plt.Circle((_xp, _yp), 0.19, fc=_fc, ec=_ec, lw=1.5, zorder=3))
        _a2.text(_xp, _yp, _c, ha="center", va="center", fontsize=8,
                fontfamily="monospace", fontweight="bold", color=_ec, zorder=4)
        if _j > 0:
            _pr = ((_j-1)*100-90)*pi/180
            _a2.plot([cos(_pr), _xp], [sin(_pr), _yp], color="#E5E7EB", lw=0.7, zorder=1)
    _a2.set_xlim(-1.55, 1.55); _a2.set_ylim(-1.55, 1.55); _a2.axis("off")
    _a2.set_title(f"Helical wheel ({_nr} res)", fontsize=11, fontweight="600", pad=10)
    _a2.legend(handles=[
        Patch(fc="#FEF3C7", ec="#D97706", label="Hydrophobic"),
        Patch(fc="#DBEAFE", ec="#2563EB", label="Hydrophilic"),
        Patch(fc="#F3F4F6", ec="#9CA3AF", label="Neutral"),
    ], loc="lower center", fontsize=7, ncol=3, framealpha=0.9, handlelength=1.2)
    _h2 = _img(_f2)

    # ── Property radar ──
    _f3, _a3 = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    _cats = ["P(AMP)", "Tag\nmatch", "AA\nentropy", "Diversity", "Length\nnorm", "Charge\nnorm"]
    _vals = [
        float(best_row["amp_probability"]),
        float(best_row["tag_adherence"]),
        float(best_row["aa_entropy"]),
        len(set(best_seq)) / 20,
        min(1, best_row["length"] / 50),
        min(1, max(0, best_row["charge"] / 10)),
    ]
    _rn = len(_cats)
    _rang = [_j / _rn * 2 * pi for _j in range(_rn)]
    _rvc = _vals + _vals[:1]; _rac = _rang + _rang[:1]
    _a3.plot(_rac, _rvc, color="#2563EB", lw=2)
    _a3.fill(_rac, _rvc, color="#2563EB", alpha=0.08)
    _a3.scatter(_rang, _vals, s=25, c="#2563EB", zorder=3)
    _a3.set_xticks(_rang); _a3.set_xticklabels(_cats, fontsize=7.5, color="#374151")
    _a3.set_ylim(0, 1); _a3.set_yticks([0.25, 0.5, 0.75, 1.0])
    _a3.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7, color="#9CA3AF")
    _a3.set_title("Quality radar", fontsize=11, fontweight="600", pad=18)
    _a3.grid(color="#E5E7EB"); _a3.spines["polar"].set_color("#E5E7EB")
    _h3 = _img(_f3)

    # ── Hydrophobicity profile ──
    _f4, _a4 = plt.subplots(figsize=(14, 2.8))
    _hp = [KD_SCALE.get(_c, 0) for _c in best_seq]
    _bc = [AA_TO_COLOR.get(_c, "#9CA3AF") for _c in best_seq]
    _a4.bar(_pos, _hp, color=_bc, alpha=0.7, width=0.75, edgecolor="white", linewidth=0.3)
    _a4.axhline(0, color="#D1D5DB", lw=0.7)
    if _n >= 5:
        _wn = np.convolve(_hp, np.ones(5)/5, mode="valid")
        _a4.plot(np.arange(3, 3+len(_wn)), _wn, color="#111827", lw=1.8, alpha=0.6, label="5-res avg")
        _a4.legend(fontsize=8, framealpha=0.9)
    _a4.set_xlabel("Position", fontsize=9); _a4.set_ylabel("KD", fontsize=9)
    _a4.set_title("Hydrophobicity profile", fontsize=11, fontweight="600")
    _a4.set_xlim(0.25, _n + 0.75); _a4.grid(axis="y", alpha=0.3)
    _a4.spines["top"].set_visible(False); _a4.spines["right"].set_visible(False)
    _h4 = _img(_f4)

    # ── Cumulative charge ──
    _f5, _a5 = plt.subplots(figsize=(14, 2.8))
    _cp = np.array([CHARGE_MAP.get(_c, 0) for _c in best_seq])
    _cc = np.cumsum(_cp)
    _a5.fill_between(_pos, 0, _cc, alpha=0.1, color="#2563EB")
    _a5.plot(_pos, _cc, color="#2563EB", lw=2)
    _mask_pos = _cp > 0; _mask_neg = _cp < 0
    if _mask_pos.any():
        _a5.scatter(_pos[_mask_pos], _cc[_mask_pos], s=18, c="#2563EB", zorder=3, label="+ residue")
    if _mask_neg.any():
        _a5.scatter(_pos[_mask_neg], _cc[_mask_neg], s=18, c="#DC2626", zorder=3, label="- residue")
    _a5.axhline(0, color="#D1D5DB", lw=0.7)
    _a5.set_xlabel("Position", fontsize=9); _a5.set_ylabel("Cumulative charge", fontsize=9)
    _a5.set_title("Charge profile", fontsize=11, fontweight="600")
    _a5.set_xlim(0.25, _n + 0.75); _a5.legend(fontsize=8, framealpha=0.9)
    _a5.grid(axis="y", alpha=0.3)
    _a5.spines["top"].set_visible(False); _a5.spines["right"].set_visible(False)
    _h5 = _img(_f5)

    # ── AA frequency ──
    _f6, _a6 = plt.subplots(figsize=(14, 2.8))
    _cnt = Counter(best_seq)
    _fr = [_cnt.get(_a, 0) / _n for _a in AMINO_ACIDS]
    _brc = [AA_TO_COLOR.get(_a, "#9CA3AF") for _a in AMINO_ACIDS]
    _a6.bar(AMINO_ACIDS, _fr, color=_brc, alpha=0.8, edgecolor="white", linewidth=0.3)
    _a6.set_ylabel("Frequency", fontsize=9)
    _ent = -sum((_v/_n)*log(_v/_n+1e-10) for _v in _cnt.values()) / log(20) if _n > 0 else 0
    _a6.set_title(f"Amino acid frequency (entropy = {_ent:.3f})", fontsize=11, fontweight="600")
    _a6.grid(axis="y", alpha=0.3)
    _a6.spines["top"].set_visible(False); _a6.spines["right"].set_visible(False)
    for _tk in _a6.get_xticklabels(): _tk.set_fontfamily("monospace"); _tk.set_fontweight("bold"); _tk.set_fontsize(9)
    _h6 = _img(_f6)

    # ── Layout: 3 square plots on top, 3 wide plots below ──
    mo.md(f"""
## Peptide analysis

{mo.hstack([mo.Html(_h1), mo.Html(_h2), mo.Html(_h3)], gap=0.5, widths=[1, 1, 1])}

{mo.Html(_h4)}

{mo.Html(_h5)}

{mo.Html(_h6)}
    """)
    return


@app.cell
def _(mo, scored_df, best_seq, plt, BytesIO, base64, STYLE, np):
    if len(scored_df) < 2:
        mo.stop(True, mo.md(""))

    plt.rcParams.update(STYLE)

    def _img2(_fig):
        _buf = BytesIO()
        _fig.savefig(_buf, format="png", bbox_inches="tight", facecolor="#FFF", dpi=160, pad_inches=0.12)
        plt.close(_fig)
        return f'<img src="data:image/png;base64,{base64.b64encode(_buf.getvalue()).decode()}" style="width:100%;border-radius:8px;border:1px solid #E5E7EB"/>'

    _df = scored_df
    _lbl = [f"#{_j+1}" for _j in range(len(_df))]
    _col = ["#2563EB" if _df.iloc[_j]["sequence"] == best_seq else "#D1D5DB" for _j in range(len(_df))]
    _ec = ["#1D4ED8" if _df.iloc[_j]["sequence"] == best_seq else "#9CA3AF" for _j in range(len(_df))]

    _fig, _axes = plt.subplots(1, 4, figsize=(18, 3.8), gridspec_kw={"wspace": 0.4})

    _axes[0].barh(_lbl, _df["composite"], color=_col, edgecolor=_ec, linewidth=0.5, height=0.55)
    _axes[0].set_xlabel("Score", fontsize=9); _axes[0].set_title("Composite", fontsize=10, fontweight="600")
    _axes[0].invert_yaxis(); _axes[0].spines["top"].set_visible(False); _axes[0].spines["right"].set_visible(False)

    _axes[1].barh(_lbl, _df["amp_probability"], color=_col, edgecolor=_ec, linewidth=0.5, height=0.55)
    _axes[1].axvline(0.5, color="#DC2626", lw=0.7, ls="--", alpha=0.4)
    _axes[1].set_xlabel("P(AMP)", fontsize=9); _axes[1].set_title("AMP probability", fontsize=10, fontweight="600")
    _axes[1].invert_yaxis(); _axes[1].spines["top"].set_visible(False); _axes[1].spines["right"].set_visible(False)

    for _j in range(len(_df)):
        _sz = 90 if _df.iloc[_j]["sequence"] == best_seq else 30
        _axes[2].scatter(_df.iloc[_j]["length"], _df.iloc[_j]["charge"],
                        s=_sz, c=_col[_j], edgecolors=_ec[_j], linewidths=0.5, zorder=3)
        _axes[2].annotate(f"#{_j+1}", (_df.iloc[_j]["length"], _df.iloc[_j]["charge"]),
                         fontsize=6, ha="center", va="bottom", textcoords="offset points",
                         xytext=(0, 4), color="#6B7280")
    _axes[2].set_xlabel("Length", fontsize=9); _axes[2].set_ylabel("Charge", fontsize=9)
    _axes[2].set_title("Length vs charge", fontsize=10, fontweight="600")
    _axes[2].grid(alpha=0.12); _axes[2].spines["top"].set_visible(False); _axes[2].spines["right"].set_visible(False)

    _axes[3].barh(_lbl, _df["tag_adherence"], color=_col, edgecolor=_ec, linewidth=0.5, height=0.55)
    _axes[3].axvline(1.0, color="#059669", lw=0.7, ls="--", alpha=0.4)
    _axes[3].set_xlabel("Adherence", fontsize=9); _axes[3].set_xlim(0, 1.1)
    _axes[3].set_title("Tag faithfulness", fontsize=10, fontweight="600")
    _axes[3].invert_yaxis(); _axes[3].spines["top"].set_visible(False); _axes[3].spines["right"].set_visible(False)

    _fig.suptitle("All candidates", fontsize=12, fontweight="600", y=1.02)
    plt.tight_layout()

    mo.md(f"""
## Candidate comparison

{mo.Html(_img2(_fig))}
    """)
    return


@app.cell
def _(mo, gen_output):
    mo.md(f"""
<details>
<summary style="cursor:pointer;color:#6B7280;font-size:12px">Raw output</summary>

```
{gen_output[:4000]}
```

</details>
    """)
    return


if __name__ == "__main__":
    app.run()
