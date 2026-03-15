import argparse
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Vocabulary ──────────────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<MASK>"]

TAG_TOKENS = [
    "<LEN_SHORT>", "<LEN_MED>", "<LEN_LONG>",
    "<CHARGE_LOW>", "<CHARGE_MED>", "<CHARGE_HIGH>",
    "<HYDRO_LOW>", "<HYDRO_MED>", "<HYDRO_HIGH>",
]

ALL_TOKENS = SPECIAL_TOKENS + AMINO_ACIDS
TOKEN2ID = {t: i for i, t in enumerate(ALL_TOKENS)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)
PAD_ID = TOKEN2ID["<PAD>"]
EOS_ID = TOKEN2ID["<EOS>"]
MASK_ID = TOKEN2ID["<MASK>"]
NUM_AA = len(AMINO_ACIDS)
AA_IDS = [TOKEN2ID[aa] for aa in AMINO_ACIDS]

TAG2ID = {t: i for i, t in enumerate(TAG_TOKENS)}
NUM_TAG_VALUES = len(TAG_TOKENS)


def parse_line(text):
    parts = str(text).strip().split()
    tags = []
    seq = []
    for p in parts:
        if p in TAG2ID:
            tags.append(TAG2ID[p])
        elif p == "<EOS>":
            continue
        elif not p.startswith("<"):
            seq.extend(TOKEN2ID[ch] for ch in p if ch in TOKEN2ID)
    return tags, seq


# ── Dataset ─────────────────────────────────────────────────────────────

class PeptideDataset(Dataset):
    """
    Pseudo-likelihood training: randomly mask a fraction of positions
    and predict them from context + tags. Same as MLM but the model
    is used as an energy function at generation time.
    """
    def __init__(self, csv_path, max_length=64, mask_fraction=0.15):
        df = pd.read_csv(csv_path)
        col = df.columns[0]
        self.samples = []
        for text in df[col]:
            tags, seq_ids = parse_line(text)
            seq_ids = seq_ids + [EOS_ID]
            if len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]
            self.samples.append((tags, seq_ids))
        self.max_length = max_length
        self.mask_fraction = mask_fraction

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tags, seq_ids = self.samples[idx]
        L = len(seq_ids)
        pad_len = self.max_length - L

        tag_vec = torch.zeros(NUM_TAG_VALUES)
        for t in tags:
            tag_vec[t] = 1.0

        x_clean = torch.tensor(seq_ids + [PAD_ID] * pad_len, dtype=torch.long)
        seq_mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)

        # random masking for pseudo-likelihood
        mask_prob = self.mask_fraction + 0.3 * torch.rand(1).item()
        rand_vals = torch.rand(self.max_length)
        do_mask = (rand_vals < mask_prob) & seq_mask

        if do_mask.sum() == 0:
            real_pos = seq_mask.nonzero(as_tuple=True)[0]
            do_mask[real_pos[torch.randint(len(real_pos), (1,))]] = True

        x_masked = x_clean.clone()
        x_masked[do_mask] = MASK_ID

        targets = x_clean.clone()
        targets[~do_mask] = -100

        return {
            "tags": tag_vec,
            "x_clean": x_clean,
            "x_masked": x_masked,
            "targets": targets,
            "seq_mask": seq_mask,
        }


# ── Model ───────────────────────────────────────────────────────────────

class PairwiseCouplingLayer(nn.Module):
    """
    Explicit pairwise interaction between all position pairs.
    Inspired by Potts model couplings J_ij(x_i, x_j).
    Projects to a coupling space and computes bilinear interactions.
    """
    def __init__(self, d_model, coupling_dim=32):
        super().__init__()
        self.query = nn.Linear(d_model, coupling_dim)
        self.key = nn.Linear(d_model, coupling_dim)
        self.value = nn.Linear(d_model, d_model)
        self.scale = coupling_dim ** -0.5

    def forward(self, h, pad_mask=None):
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)
        attn = (Q @ K.transpose(-1, -2)) * self.scale
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e9)
        attn = F.softmax(attn, dim=-1)
        return attn @ V


class PseudoLikelihoodEBM(nn.Module):
    """
    Pseudo-Likelihood Energy Model.

    Energy is defined as: E(x, tags) = -sum_i log p(x_i | x_{-i}, tags)

    Architecture:
    1. Bidirectional Transformer encodes context (masked positions get <MASK>)
    2. Pairwise coupling layers capture explicit position-position interactions
    3. Per-position head predicts AA distribution
    4. Trained via pseudo-likelihood (masked token prediction)
    5. Generates via Gibbs sampling using the learned conditionals
    """
    def __init__(self, vocab_size, num_tags, d_model=128, nhead=4,
                 num_layers=4, dim_ff=512, max_len=64, dropout=0.1,
                 num_coupling_layers=2, coupling_dim=32):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.tag_proj = nn.Linear(num_tags, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.coupling_layers = nn.ModuleList([
            PairwiseCouplingLayer(d_model, coupling_dim)
            for _ in range(num_coupling_layers)
        ])
        self.coupling_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_coupling_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x, tags, pad_mask=None):
        B, L = x.shape
        device = x.device
        positions = torch.arange(L, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions)
        tag_emb = self.tag_proj(tags).unsqueeze(1)
        h = self.drop(h + tag_emb)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        for coupling, norm in zip(self.coupling_layers, self.coupling_norms):
            h = h + coupling(norm(h), pad_mask)

        h = self.ln_f(h)
        return h

    def forward(self, x_masked, tags, targets=None, pad_mask=None):
        h = self.encode(x_masked, tags, pad_mask)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def pseudo_likelihood_energy(self, x, tags, seq_mask):
        """
        Full pseudo-likelihood: E(x) = -sum_i log p(x_i | x_{-i}, tags)
        Masks each position one at a time.
        """
        B, L = x.shape
        device = x.device
        pad_mask = ~seq_mask
        total_nll = torch.zeros(B, device=device)
        n_positions = seq_mask.sum(dim=1).float()

        for i in range(L):
            if not seq_mask[:, i].any():
                continue
            x_masked = x.clone()
            x_masked[:, i] = MASK_ID
            h = self.encode(x_masked, tags, pad_mask)
            logits_i = self.head(h[:, i, :])
            log_probs = F.log_softmax(logits_i, dim=-1)
            nll_i = -log_probs.gather(1, x[:, i:i+1]).squeeze(1)
            total_nll += nll_i * seq_mask[:, i].float()

        return total_nll / n_positions.clamp(min=1)

    @torch.no_grad()
    def conditional_at_position(self, x, tags, pos, pad_mask=None, temperature=1.0):
        """p(x_pos | x_{-pos}, tags) — the Gibbs sampling primitive."""
        x_masked = x.clone()
        x_masked[:, pos] = MASK_ID
        h = self.encode(x_masked, tags, pad_mask)
        logits = self.head(h[:, pos, :]) / temperature
        return F.softmax(logits, dim=-1)


# ── Training ────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x_masked"].to(device)
            tags = batch["tags"].to(device)
            targets = batch["targets"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            pad_mask = ~seq_mask
            _, loss = model(x, tags, targets, pad_mask)
            n = (targets != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_ebm")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_coupling_layers", type=int, default=2)
    parser.add_argument("--coupling_dim", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PseudoLikelihoodEBM(
        VOCAB_SIZE, NUM_TAG_VALUES,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        max_len=args.max_length, dropout=args.dropout,
        num_coupling_layers=args.num_coupling_layers,
        coupling_dim=args.coupling_dim,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for batch in train_loader:
            x = batch["x_masked"].to(device)
            tags = batch["tags"].to(device)
            targets = batch["targets"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            pad_mask = ~seq_mask

            _, loss = model(x, tags, targets, pad_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = (targets != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

        scheduler.step()
        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": VOCAB_SIZE, "num_tags": NUM_TAG_VALUES,
                    "d_model": args.d_model, "nhead": args.nhead,
                    "num_layers": args.num_layers, "dim_ff": args.dim_ff,
                    "max_len": args.max_length, "dropout": args.dropout,
                    "num_coupling_layers": args.num_coupling_layers,
                    "coupling_dim": args.coupling_dim,
                },
            }, f"{args.output_dir}/best.pt")
            print(f"  ↑ saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
