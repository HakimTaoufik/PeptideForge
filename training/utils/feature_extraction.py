import numpy as np
from collections import Counter


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

STANDARD_AAS = set(AA_LIST)


def validate_sequence(seq: str):
    seq = seq.strip().upper()
    if len(seq) == 0:
        raise ValueError("Sequence is empty.")
    bad_chars = sorted(set(seq) - STANDARD_AAS)
    if bad_chars:
        raise ValueError(f"Sequence contains non-standard amino acids: {bad_chars}")
    return seq


def approximate_net_charge(seq: str) -> int:
    return seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")


def mean_hydrophobicity(seq: str) -> float:
    return float(np.mean([KYTE_DOOLITTLE[aa] for aa in seq]))


def aa_fraction_features(seq: str):
    counts = Counter(seq)
    n = len(seq)
    return {f"frac_{aa}": counts.get(aa, 0) / n for aa in AA_LIST}


def extra_sequence_features(seq: str):
    counts = Counter(seq)
    n = len(seq)

    basic = counts.get("K", 0) + counts.get("R", 0)
    acidic = counts.get("D", 0) + counts.get("E", 0)
    aromatic = counts.get("F", 0) + counts.get("W", 0) + counts.get("Y", 0)
    aliphatic = (
        counts.get("A", 0) + counts.get("V", 0) + counts.get("I", 0)
        + counts.get("L", 0) + counts.get("M", 0)
    )
    polar = counts.get("S", 0) + counts.get("T", 0) + counts.get("N", 0) + counts.get("Q", 0)

    return {
        "length": n,
        "charge": approximate_net_charge(seq),
        "hydrophobicity": mean_hydrophobicity(seq),
        "frac_basic": basic / n,
        "frac_acidic": acidic / n,
        "cysteine_count": counts.get("C", 0),
        "basic_count": basic,
        "acidic_count": acidic,
        "aromatic_frac": aromatic / n,
        "aliphatic_frac": aliphatic / n,
        "polar_frac": polar / n,
        "proline_frac": counts.get("P", 0) / n,
        "glycine_frac": counts.get("G", 0) / n,
        "cysteine_frac": counts.get("C", 0) / n,
        "lys_arg_ratio": counts.get("K", 0) / max(counts.get("R", 0) + 1, 1),
    }


def compute_features_for_sequence(seq: str):
    seq = validate_sequence(seq)

    feats = {}
    feats.update(extra_sequence_features(seq))
    feats.update(aa_fraction_features(seq))
    return feats