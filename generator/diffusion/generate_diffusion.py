import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
from train_diffusion import (
    PeptideDenoiser, AbsorbingSchedule,
    TAG2ID, NUM_TAG_VALUES, VOCAB_SIZE,
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
    model = PeptideDenoiser(
        cfg["vocab_size"], cfg["num_tags"],
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    schedule = AbsorbingSchedule(cfg["diffusion_steps"], cfg["noise_schedule"])
    return model, cfg, schedule


# ── Sampling strategies ─────────────────────────────────────────────────

@torch.no_grad()
def sample_ancestral(model, schedule, tags, seq_len, device, temperature=1.0):
    """
    Standard ancestral sampling: reverse the diffusion step by step.
    At each step t → t-1, predict x_0, then re-corrupt to t-1.
    """
    B = 1
    T = schedule.T
    tags_batch = tags.unsqueeze(0).to(device)

    # start fully masked
    x_t = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    seq_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    pad_mask = ~seq_mask

    for step in range(T, 0, -1):
        t_batch = torch.tensor([step], device=device)
        logits = model(x_t, t_batch, tags_batch, pad_mask)
        logits = logits / temperature

        # predict x_0
        probs = F.softmax(logits, dim=-1)
        x_0_pred = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)

        if step == 1:
            x_t = x_0_pred
        else:
            # re-corrupt to t-1
            gamma_prev = schedule.gamma[step - 1].item()
            mask_again = torch.rand(B, seq_len, device=device) < gamma_prev
            x_t = x_0_pred.clone()
            x_t[mask_again & seq_mask] = MASK_ID

    return x_t.squeeze(0)


@torch.no_grad()
def sample_confidence(model, schedule, tags, seq_len, device,
                      temperature=1.0, num_steps=50):
    """
    Confidence-based unmasking (MaskGIT-style):
    Start fully masked, at each step unmask the most confident positions.
    Faster than full T-step ancestral sampling.
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    seq_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    pad_mask = ~seq_mask
    is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)

    tokens_per_step = max(1, seq_len // num_steps)

    for step in range(num_steps):
        n_masked = is_masked.sum().item()
        if n_masked == 0:
            break

        # use a timestep proportional to remaining masked fraction
        frac = n_masked / seq_len
        t_val = max(1, int(frac * schedule.T))
        t_batch = torch.tensor([t_val], device=device)

        # temperature annealing: high early (diversity), low late (refinement)
        progress = step / max(num_steps - 1, 1)
        step_temp = temperature * (1.5 - 0.7 * progress)

        logits = model(x, t_batch, tags_batch, pad_mask)
        logits = logits / step_temp
        probs = F.softmax(logits, dim=-1)

        # sample tokens and get confidence
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)
        confidence = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # only consider masked positions
        confidence[~is_masked] = -1.0

        # Gumbel noise on confidence to break mode collapse (decays over steps)
        gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-8) + 1e-8)
        noisy_conf = confidence + 0.3 * gumbel * (1.0 - progress)
        noisy_conf[~is_masked] = -1.0

        # unmask top-k most confident (with noise)
        n_unmask = min(tokens_per_step, n_masked)
        _, topk_idx = noisy_conf.view(-1).topk(n_unmask)
        flat_x = x.view(-1)
        flat_sampled = sampled.view(-1)
        flat_x[topk_idx] = flat_sampled[topk_idx]
        x = flat_x.view(B, seq_len)

        flat_mask = is_masked.view(-1)
        flat_mask[topk_idx] = False
        is_masked = flat_mask.view(B, seq_len)

    # fill any remaining masked tokens
    if is_masked.any():
        t_batch = torch.tensor([1], device=device)
        logits = model(x, t_batch, tags_batch, pad_mask)
        probs = F.softmax(logits / temperature, dim=-1)
        fill = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)
        x[is_masked] = fill[is_masked]

    return x.squeeze(0)


@torch.no_grad()
def sample_with_trajectory(model, schedule, tags, seq_len, device,
                           temperature=1.0, record_every=10):
    """
    Ancestral sampling that records intermediate states for visualization.
    Returns (final_sequence, list_of_intermediate_states).
    """
    B = 1
    T = schedule.T
    tags_batch = tags.unsqueeze(0).to(device)

    x_t = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    seq_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    pad_mask = ~seq_mask

    trajectory = []

    for step in range(T, 0, -1):
        t_batch = torch.tensor([step], device=device)
        logits = model(x_t, t_batch, tags_batch, pad_mask)
        probs = F.softmax(logits / temperature, dim=-1)
        x_0_pred = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, seq_len)

        if step % record_every == 0 or step == T:
            state = x_t.squeeze(0).clone()
            trajectory.append((step, state))

        if step == 1:
            x_t = x_0_pred
        else:
            gamma_prev = schedule.gamma[step - 1].item()
            mask_again = torch.rand(B, seq_len, device=device) < gamma_prev
            x_t = x_0_pred.clone()
            x_t[mask_again & seq_mask] = MASK_ID

    trajectory.append((0, x_t.squeeze(0).clone()))
    return x_t.squeeze(0), trajectory


# ── Helpers ─────────────────────────────────────────────────────────────

def ids_to_peptide(ids):
    aa = [ID2TOKEN[i.item()] for i in ids
          if ID2TOKEN.get(i.item(), "") in AMINO_ACIDS]
    return "".join(aa)


def format_state(ids):
    """Format a noisy state for display: AAs shown, masks as dots."""
    chars = []
    for i in ids:
        tok = ID2TOKEN.get(i.item(), "?")
        if tok == "<MASK>":
            chars.append("·")
        elif tok in AMINO_ACIDS:
            chars.append(tok)
        elif tok == "<EOS>":
            break
        else:
            chars.append("_")
    return "".join(chars)


# ── Length heuristic ────────────────────────────────────────────────────

LENGTH_HINTS = {
    "<LEN_SHORT>": 12,
    "<LEN_MED>":   20,
    "<LEN_LONG>":  30,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_diffusion/best.pt")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=0,
                        help="Sequence length to generate (0 = auto from length tag)")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--mode", default="ancestral",
                        choices=["ancestral", "confidence", "trajectory"])
    parser.add_argument("--confidence_steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, schedule = load_model(args.model_path, device)

    tags_list = [args.length_tag, args.charge_tag, args.hydro_tag]
    tag_vec = make_tag_vec(*tags_list)

    # determine sequence length
    if args.seq_len > 0:
        seq_len = args.seq_len
    else:
        base = LENGTH_HINTS.get(args.length_tag, 20)
        seq_len = base + 1  # +1 for EOS

    print(f"\nMode: {args.mode} | Tags: {' '.join(tags_list)} | Seq len: {seq_len}\n")

    if args.mode == "trajectory":
        seq_ids, traj = sample_with_trajectory(
            model, schedule, tag_vec, seq_len, device,
            temperature=args.temperature,
        )
        print("Denoising trajectory:")
        for step, state in traj:
            print(f"  t={step:3d} | {format_state(state)}")
        peptide = ids_to_peptide(seq_ids)
        print(f"\n  Final: {peptide} (len={len(peptide)})")

    else:
        sample_fn = (sample_ancestral if args.mode == "ancestral"
                     else sample_confidence)
        print("Generated peptides:\n")
        for i in range(args.n):
            if args.mode == "ancestral":
                seq_ids = sample_fn(model, schedule, tag_vec, seq_len, device,
                                    temperature=args.temperature)
            else:
                seq_ids = sample_fn(model, schedule, tag_vec, seq_len, device,
                                    temperature=args.temperature,
                                    num_steps=args.confidence_steps)
            peptide = ids_to_peptide(seq_ids)
            print(f"  {i+1}. {peptide} (len={len(peptide)})")


if __name__ == "__main__":
    main()
