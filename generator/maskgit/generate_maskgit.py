import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
from train_maskgit import (
    PeptideMaskGIT, TAG2ID, NUM_TAG_VALUES, VOCAB_SIZE,
    TOKEN2ID, ID2TOKEN, AMINO_ACIDS,
    PAD_ID, EOS_ID, MASK_ID,
)


def make_tag_vec(length_tag, charge_tag, hydro_tag):
    vec = torch.zeros(NUM_TAG_VALUES)
    for t in [length_tag, charge_tag, hydro_tag]:
        if t in TAG2ID:
            vec[TAG2ID[t]] = 1.0
    return vec


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PeptideMaskGIT(
        cfg["vocab_size"], cfg["num_tags"],
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ── Mask schedule for generation ────────────────────────────────────────

def cosine_unmask_schedule(num_steps, seq_len):
    """
    Returns how many tokens to unmask at each step.
    Cosine schedule: slow start (few tokens), accelerating reveal.
    """
    ratios = []
    for i in range(num_steps):
        t_prev = i / num_steps
        t_curr = (i + 1) / num_steps
        r_prev = math.cos(t_prev * math.pi / 2)
        r_curr = math.cos(t_curr * math.pi / 2)
        ratios.append(r_prev - r_curr)

    # convert to token counts
    counts = [max(1, int(r * seq_len)) for r in ratios]

    # adjust to sum to seq_len
    total = sum(counts)
    if total < seq_len:
        counts[-1] += seq_len - total
    elif total > seq_len:
        for i in range(len(counts) - 1, -1, -1):
            excess = total - seq_len
            reduce = min(counts[i] - 1, excess)
            counts[i] -= reduce
            total -= reduce
            if total == seq_len:
                break

    return counts


# ── Sampling strategies ─────────────────────────────────────────────────

@torch.no_grad()
def sample_iterative(model, tags, seq_len, device, num_steps=20,
                     temperature=1.0, top_k=0):
    """
    MaskGIT iterative parallel decoding:
    1. Start fully masked
    2. At each step: predict all, sample, keep most confident, re-mask rest
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    pad_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

    unmask_counts = cosine_unmask_schedule(num_steps, seq_len)

    for step, n_unmask in enumerate(unmask_counts):
        n_remaining = is_masked.sum().item()
        if n_remaining == 0:
            break

        n_unmask = min(n_unmask, n_remaining)

        # temperature annealing: high early (diversity), low late (refinement)
        progress = step / max(len(unmask_counts) - 1, 1)
        step_temp = temperature * (1.5 - 0.7 * progress)

        logits = model(x, tags_batch, pad_mask)
        logits = logits / step_temp

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < topk_vals[..., -1:]] = -float("inf")

        probs = F.softmax(logits, dim=-1)

        # sample tokens for all positions
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)
        confidence = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)

        # zero out confidence for non-masked positions
        confidence[~is_masked] = -1.0

        # Gumbel noise on confidence to break mode collapse (decays over steps)
        gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-8) + 1e-8)
        noisy_conf = confidence + 0.5 * gumbel * (1.0 - progress)
        noisy_conf[~is_masked] = -1.0

        # unmask top-n_unmask most confident (with noise)
        _, topk_idx = noisy_conf.view(-1).topk(n_unmask)

        flat_x = x.view(-1)
        flat_x[topk_idx] = sampled.view(-1)[topk_idx]
        x = flat_x.view(B, seq_len)

        flat_mask = is_masked.view(-1)
        flat_mask[topk_idx] = False
        is_masked = flat_mask.view(B, seq_len)

    # fill any stragglers
    if is_masked.any():
        logits = model(x, tags_batch, pad_mask)
        probs = F.softmax(logits / temperature, dim=-1)
        fill = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)
        x[is_masked] = fill[is_masked]

    return x.squeeze(0)


@torch.no_grad()
def sample_with_refinement(model, tags, seq_len, device, num_steps=20,
                           refine_steps=3, temperature=1.0):
    """
    Generate iteratively, then do additional refinement passes:
    re-mask the least confident positions and re-predict them.
    """
    # initial generation
    x = sample_iterative(model, tags, seq_len, device,
                         num_steps=num_steps, temperature=temperature)

    B = 1
    tags_batch = tags.unsqueeze(0).to(device)
    x = x.unsqueeze(0)
    pad_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

    for r in range(refine_steps):
        logits = model(x, tags_batch, pad_mask)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.gather(2, x.unsqueeze(-1)).squeeze(-1)

        # re-mask bottom 20% confidence positions
        n_remask = max(1, seq_len // 5)
        _, worst_idx = confidence.view(-1).topk(n_remask, largest=False)

        flat_x = x.view(-1)
        flat_x[worst_idx] = MASK_ID
        x = flat_x.view(B, seq_len)

        # re-predict
        logits = model(x, tags_batch, pad_mask)
        probs = F.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)

        is_mask = (x == MASK_ID)
        x[is_mask] = sampled[is_mask]

    return x.squeeze(0)


@torch.no_grad()
def sample_with_trajectory(model, tags, seq_len, device, num_steps=20,
                           temperature=1.0):
    """
    Iterative decoding that records the unmasking progression.
    Returns (final_ids, trajectory) where trajectory is list of (step, state_string).
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    pad_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

    unmask_counts = cosine_unmask_schedule(num_steps, seq_len)
    trajectory = [(0, format_state(x.squeeze(0)))]

    for step, n_unmask in enumerate(unmask_counts, 1):
        n_remaining = is_masked.sum().item()
        if n_remaining == 0:
            break
        n_unmask = min(n_unmask, n_remaining)

        logits = model(x, tags_batch, pad_mask)
        progress = (step - 1) / max(len(unmask_counts) - 1, 1)
        step_temp = temperature * (1.5 - 0.7 * progress)
        logits = logits / step_temp
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)
        confidence = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)
        confidence[~is_masked] = -1.0

        gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-8) + 1e-8)
        noisy_conf = confidence + 0.3 * gumbel * (1.0 - progress)
        noisy_conf[~is_masked] = -1.0

        _, topk_idx = noisy_conf.view(-1).topk(n_unmask)
        flat_x = x.view(-1)
        flat_x[topk_idx] = sampled.view(-1)[topk_idx]
        x = flat_x.view(B, seq_len)

        flat_mask = is_masked.view(-1)
        flat_mask[topk_idx] = False
        is_masked = flat_mask.view(B, seq_len)

        trajectory.append((step, format_state(x.squeeze(0))))

    return x.squeeze(0), trajectory


# ── Helpers ─────────────────────────────────────────────────────────────

def ids_to_peptide(ids):
    aa = [ID2TOKEN[i.item()] for i in ids
          if ID2TOKEN.get(i.item(), "") in AMINO_ACIDS]
    return "".join(aa)


def format_state(ids):
    chars = []
    for i in ids:
        tok = ID2TOKEN.get(i.item(), "?")
        if tok == "<MASK>":
            chars.append("·")
        elif tok in AMINO_ACIDS:
            chars.append(tok)
        elif tok == "<EOS>":
            chars.append("|")
        else:
            chars.append("_")
    return "".join(chars)


LENGTH_HINTS = {
    "<LEN_SHORT>": 12,
    "<LEN_MED>":   20,
    "<LEN_LONG>":  30,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_maskgit/best.pt")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--mode", default="iterative",
                        choices=["iterative", "refine", "trajectory"])
    parser.add_argument("--refine_steps", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_path, device)

    tags_list = [args.length_tag, args.charge_tag, args.hydro_tag]
    tag_vec = make_tag_vec(*tags_list)

    if args.seq_len > 0:
        seq_len = args.seq_len
    else:
        base = LENGTH_HINTS.get(args.length_tag, 20)
        seq_len = base + 1

    print(f"\nMode: {args.mode} | Tags: {' '.join(tags_list)} | "
          f"Seq len: {seq_len} | Steps: {args.num_steps}\n")

    if args.mode == "trajectory":
        ids, traj = sample_with_trajectory(
            model, tag_vec, seq_len, device,
            num_steps=args.num_steps, temperature=args.temperature,
        )
        print("Unmasking progression:")
        for step, state in traj:
            n_revealed = sum(1 for c in state if c != '·')
            pct = 100 * n_revealed / seq_len
            print(f"  step {step:3d} | {state} ({pct:.0f}%)")
        peptide = ids_to_peptide(ids)
        print(f"\n  Final: {peptide} (len={len(peptide)})")

    else:
        sample_fn = sample_with_refinement if args.mode == "refine" else sample_iterative
        print("Generated peptides:\n")
        for i in range(args.n):
            if args.mode == "refine":
                ids = sample_fn(model, tag_vec, seq_len, device,
                                num_steps=args.num_steps,
                                refine_steps=args.refine_steps,
                                temperature=args.temperature)
            else:
                ids = sample_fn(model, tag_vec, seq_len, device,
                                num_steps=args.num_steps,
                                temperature=args.temperature)
            peptide = ids_to_peptide(ids)
            print(f"  {i+1}. {peptide} (len={len(peptide)})")


if __name__ == "__main__":
    main()
