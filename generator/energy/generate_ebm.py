import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
from train_ebm import (
    PseudoLikelihoodEBM, TAG2ID, NUM_TAG_VALUES, VOCAB_SIZE,
    TOKEN2ID, ID2TOKEN, AMINO_ACIDS, AA_IDS,
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
    model = PseudoLikelihoodEBM(
        cfg["vocab_size"], cfg["num_tags"],
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        num_coupling_layers=cfg.get("num_coupling_layers", 2),
        coupling_dim=cfg.get("coupling_dim", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ── Gibbs Sampling ──────────────────────────────────────────────────────

@torch.no_grad()
def gibbs_sample(model, tags, seq_len, max_len, device,
                 n_steps=200, temperature=1.0):
    """
    Standard Gibbs sampling using learned conditionals.
    At each step: pick random position, sample from p(x_i | x_{-i}, tags).
    Much more efficient than the old EBM — no need to evaluate all 20 AAs
    with separate forward passes; the model gives the full distribution.
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    # initialize from uniform random AAs
    aa_tensor = torch.tensor(AA_IDS, device=device)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    x[:, :seq_len] = aa_tensor[torch.randint(len(AA_IDS), (B, seq_len), device=device)]
    pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    pad_mask[:, :seq_len] = False

    for step in range(n_steps):
        pos = torch.randint(seq_len, (1,)).item()
        probs = model.conditional_at_position(x, tags_batch, pos, pad_mask, temperature)
        x[0, pos] = torch.multinomial(probs[0], 1).item()

    return x[0, :seq_len]


# ── Gibbs with Annealing ───────────────────────────────────────────────

@torch.no_grad()
def gibbs_anneal(model, tags, seq_len, max_len, device,
                 n_steps=500, t_start=2.0, t_end=0.3):
    """
    Simulated annealing Gibbs: high temperature → low temperature.
    Multiple positions per step early on for faster exploration.
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    aa_tensor = torch.tensor(AA_IDS, device=device)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    x[:, :seq_len] = aa_tensor[torch.randint(len(AA_IDS), (B, seq_len), device=device)]
    pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    pad_mask[:, :seq_len] = False

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t_end + 0.5 * (t_start - t_end) * (1 + math.cos(frac * math.pi))

        # more mutations early, fewer late
        n_mutations = max(1, int((1 - frac) * 3))
        for _ in range(n_mutations):
            pos = torch.randint(seq_len, (1,)).item()
            probs = model.conditional_at_position(x, tags_batch, pos, pad_mask, temp)
            x[0, pos] = torch.multinomial(probs[0], 1).item()

    return x[0, :seq_len]


# ── Parallel Tempering ─────────────────────────────────────────────────

@torch.no_grad()
def parallel_tempering(model, tags, seq_len, max_len, device,
                       n_steps=300, temperatures=None, swap_every=10):
    """
    Run multiple Gibbs chains at different temperatures simultaneously.
    Periodically attempt to swap states between adjacent chains.
    The lowest-temperature chain produces the final sample.
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.8, 1.2, 2.0]

    n_chains = len(temperatures)
    B = n_chains
    tags_batch = tags.unsqueeze(0).expand(B, -1).to(device)

    aa_tensor = torch.tensor(AA_IDS, device=device)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    for c in range(B):
        x[c, :seq_len] = aa_tensor[torch.randint(len(AA_IDS), (seq_len,), device=device)]
    pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    pad_mask[:, :seq_len] = False

    for step in range(n_steps):
        # Gibbs update each chain at its own temperature
        pos = torch.randint(seq_len, (1,)).item()
        for c in range(n_chains):
            x_single = x[c:c+1]
            tags_single = tags_batch[c:c+1]
            pad_single = pad_mask[c:c+1]
            probs = model.conditional_at_position(
                x_single, tags_single, pos, pad_single, temperatures[c],
            )
            x[c, pos] = torch.multinomial(probs[0], 1).item()

        # swap adjacent chains (Metropolis criterion)
        if step % swap_every == 0 and step > 0:
            for c in range(n_chains - 1):
                # compute energy for both chains
                seq_mask = ~pad_mask
                e_c = model.pseudo_likelihood_energy(
                    x[c:c+1], tags_batch[c:c+1], seq_mask[c:c+1],
                ).item()
                e_c1 = model.pseudo_likelihood_energy(
                    x[c+1:c+2], tags_batch[c+1:c+2], seq_mask[c+1:c+2],
                ).item()

                beta_c = 1.0 / temperatures[c]
                beta_c1 = 1.0 / temperatures[c + 1]
                delta = (beta_c - beta_c1) * (e_c1 - e_c)

                if delta < 0 or torch.rand(1).item() < math.exp(-delta):
                    x_temp = x[c].clone()
                    x[c] = x[c + 1]
                    x[c + 1] = x_temp

    # return the coldest chain
    return x[0, :seq_len]


# ── Block Gibbs (resample multiple positions at once) ──────────────────

@torch.no_grad()
def block_gibbs(model, tags, seq_len, max_len, device,
                n_steps=200, block_size=3, temperature=0.8):
    """
    Block Gibbs: mask a block of positions, predict all at once,
    then sample. Faster mixing than single-site Gibbs.
    """
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)

    aa_tensor = torch.tensor(AA_IDS, device=device)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    x[:, :seq_len] = aa_tensor[torch.randint(len(AA_IDS), (B, seq_len), device=device)]
    pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    pad_mask[:, :seq_len] = False

    for step in range(n_steps):
        # pick a random contiguous block
        start = torch.randint(max(1, seq_len - block_size), (1,)).item()
        end = min(start + block_size, seq_len)

        # mask the block
        x_masked = x.clone()
        x_masked[0, start:end] = MASK_ID

        # get logits for the whole sequence
        h = model.encode(x_masked, tags_batch, pad_mask)
        logits = model.head(h)

        # sample each position in the block from its conditional
        for pos in range(start, end):
            probs = F.softmax(logits[0, pos, :] / temperature, dim=-1)
            x[0, pos] = torch.multinomial(probs, 1).item()

    return x[0, :seq_len]


# ── Trajectory recording ───────────────────────────────────────────────

@torch.no_grad()
def gibbs_trajectory(model, tags, seq_len, max_len, device,
                     n_steps=200, temperature=0.8, record_every=20):
    B = 1
    tags_batch = tags.unsqueeze(0).to(device)
    seq_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    seq_mask[:, :seq_len] = True

    aa_tensor = torch.tensor(AA_IDS, device=device)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    x[:, :seq_len] = aa_tensor[torch.randint(len(AA_IDS), (B, seq_len), device=device)]
    pad_mask = ~seq_mask

    trajectory = []

    for step in range(n_steps + 1):
        if step % record_every == 0:
            e = model.pseudo_likelihood_energy(x, tags_batch, seq_mask).item()
            seq_str = ids_to_peptide(x[0, :seq_len])
            trajectory.append((step, e, seq_str))

        if step == n_steps:
            break

        # anneal temperature
        frac = step / max(n_steps - 1, 1)
        temp = 0.3 + 0.5 * (temperature - 0.3) * (1 + math.cos(frac * math.pi))

        pos = torch.randint(seq_len, (1,)).item()
        probs = model.conditional_at_position(x, tags_batch, pos, pad_mask, temp)
        x[0, pos] = torch.multinomial(probs[0], 1).item()

    return x[0, :seq_len], trajectory


# ── Helpers ─────────────────────────────────────────────────────────────

def ids_to_peptide(ids):
    aa = [ID2TOKEN[i.item()] for i in ids
          if ID2TOKEN.get(i.item(), "") in AMINO_ACIDS]
    return "".join(aa)


LENGTH_HINTS = {
    "<LEN_SHORT>": 12,
    "<LEN_MED>":   20,
    "<LEN_LONG>":  30,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_ebm/best.pt")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--mode", default="anneal",
                        choices=["gibbs", "anneal", "tempering", "block", "trajectory"])
    parser.add_argument("--n_steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_path, device)
    max_len = cfg["max_len"]

    tags_list = [args.length_tag, args.charge_tag, args.hydro_tag]
    tag_vec = make_tag_vec(*tags_list)

    if args.seq_len > 0:
        seq_len = args.seq_len
    else:
        seq_len = LENGTH_HINTS.get(args.length_tag, 20)

    print(f"\nMode: {args.mode} | Tags: {' '.join(tags_list)} | "
          f"Seq len: {seq_len} | Steps: {args.n_steps}\n")

    if args.mode == "trajectory":
        ids, traj = gibbs_trajectory(
            model, tag_vec, seq_len, max_len, device,
            n_steps=args.n_steps, temperature=args.temperature,
        )
        print("Energy minimization trajectory:")
        for step, energy, seq in traj:
            print(f"  step {step:4d} | E={energy:8.3f} | {seq}")
        peptide = ids_to_peptide(ids)
        print(f"\n  Final: {peptide} (len={len(peptide)})")

    else:
        sample_fns = {
            "gibbs": lambda: gibbs_sample(
                model, tag_vec, seq_len, max_len, device,
                n_steps=args.n_steps, temperature=args.temperature,
            ),
            "anneal": lambda: gibbs_anneal(
                model, tag_vec, seq_len, max_len, device,
                n_steps=args.n_steps,
            ),
            "tempering": lambda: parallel_tempering(
                model, tag_vec, seq_len, max_len, device,
                n_steps=args.n_steps,
            ),
            "block": lambda: block_gibbs(
                model, tag_vec, seq_len, max_len, device,
                n_steps=args.n_steps, temperature=args.temperature,
            ),
        }

        sample_fn = sample_fns[args.mode]
        print(f"Generated peptides ({args.mode}):\n")

        for i in range(args.n):
            ids = sample_fn()
            peptide = ids_to_peptide(ids)

            # compute energy
            x_full = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
            x_full[0, :seq_len] = ids
            sm = torch.zeros(1, max_len, dtype=torch.bool, device=device)
            sm[0, :seq_len] = True
            e = model.pseudo_likelihood_energy(
                x_full, tag_vec.unsqueeze(0).to(device), sm,
            ).item()
            print(f"  {i+1}. {peptide} (len={len(peptide)}, E={e:.3f})")


if __name__ == "__main__":
    main()
