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
    def __init__(self, csv_path, max_length=64):
        df = pd.read_csv(csv_path)
        col = df.columns[0]
        self.samples = []
        for text in df[col]:
            tags, seq_ids = parse_line(text)
            seq_ids = seq_ids + [EOS_ID]  # append EOS as part of the sequence
            if len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]
            self.samples.append((tags, seq_ids))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tags, seq_ids = self.samples[idx]
        L = len(seq_ids)
        pad_len = self.max_length - L

        tag_vec = torch.zeros(NUM_TAG_VALUES)
        for t in tags:
            tag_vec[t] = 1.0

        x_0 = torch.tensor(seq_ids + [PAD_ID] * pad_len, dtype=torch.long)
        mask = torch.tensor([1] * L + [0] * pad_len, dtype=torch.bool)

        return {"tags": tag_vec, "x_0": x_0, "seq_mask": mask}


# ── Noise Schedule ──────────────────────────────────────────────────────

class AbsorbingSchedule:
    """
    Absorbing-state discrete diffusion.
    At timestep t, each token is independently replaced with <MASK>
    with probability gamma(t), where gamma goes from 0 to ~1.
    """
    def __init__(self, num_steps=100, schedule="cosine"):
        self.T = num_steps
        if schedule == "cosine":
            # cosine schedule: slow start, accelerating corruption
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alpha = torch.cos((steps / num_steps) * (math.pi / 2)).clamp(min=1e-4)
            self.gamma = 1.0 - alpha  # mask probability at each step
        elif schedule == "linear":
            self.gamma = torch.linspace(0.0, 1.0, num_steps + 1)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample_t(self, batch_size):
        return torch.randint(1, self.T + 1, (batch_size,))

    def corrupt(self, x_0, t, seq_mask):
        """
        x_0:      (B, L) clean token ids
        t:        (B,)   timestep per sample
        seq_mask: (B, L) True for real tokens, False for padding
        Returns x_t with some tokens replaced by MASK_ID
        """
        B, L = x_0.shape
        gamma_t = self.gamma.to(t.device)[t].unsqueeze(1)  # (B, 1)
        mask_probs = torch.rand(B, L, device=x_0.device)
        do_mask = (mask_probs < gamma_t) & seq_mask

        x_t = x_0.clone()
        x_t[do_mask] = MASK_ID
        return x_t, do_mask


# ── Denoiser Model ──────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class PeptideDenoiser(nn.Module):
    """
    Bidirectional Transformer that predicts x_0 from x_t.
    Conditioned on: timestep t, tag vector, noisy sequence x_t.
    """
    def __init__(self, vocab_size, num_tags, d_model=128, nhead=4,
                 num_layers=4, dim_ff=512, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # tag conditioning
        self.tag_proj = nn.Linear(num_tags, d_model)

        # bidirectional Transformer encoder (NOT causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_t, t, tags, pad_mask):
        """
        x_t:      (B, L) noisy token ids
        t:        (B,)   timestep
        tags:     (B, num_tags) multi-hot
        pad_mask: (B, L) True for padding positions
        Returns logits (B, L, vocab_size)
        """
        B, L = x_t.shape
        device = x_t.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_emb(x_t) * math.sqrt(self.d_model) + self.pos_emb(positions)

        # add timestep embedding (broadcast to all positions)
        t_emb = self.time_emb(t)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # add tag embedding (broadcast to all positions)
        tag_emb = self.tag_proj(tags)  # (B, d_model)
        x = x + tag_emb.unsqueeze(1)

        # bidirectional self-attention (no causal mask)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, L, vocab_size)
        return logits


# ── Training ────────────────────────────────────────────────────────────

def evaluate(model, schedule, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_0 = batch["x_0"].to(device)
            tags = batch["tags"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            B = x_0.size(0)

            t = schedule.sample_t(B).to(device)
            x_t, corrupted = schedule.corrupt(x_0, t, seq_mask)
            pad_mask = ~seq_mask

            logits = model(x_t, t, tags, pad_mask)

            # loss on corrupted positions only
            loss_mask = corrupted
            if loss_mask.sum() == 0:
                continue

            logits_flat = logits[loss_mask]
            targets_flat = x_0[loss_mask]
            loss = F.cross_entropy(logits_flat, targets_flat)

            n = loss_mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_diffusion")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--noise_schedule", default="cosine", choices=["cosine", "linear"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = AbsorbingSchedule(args.diffusion_steps, args.noise_schedule)

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PeptideDenoiser(
        VOCAB_SIZE, NUM_TAG_VALUES,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        max_len=args.max_length, dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}  Diffusion steps: {args.diffusion_steps}")

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
            x_0 = batch["x_0"].to(device)
            tags = batch["tags"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            B = x_0.size(0)

            # sample random timestep per sample
            t = schedule.sample_t(B).to(device)

            # corrupt
            x_t, corrupted = schedule.corrupt(x_0, t, seq_mask)
            pad_mask = ~seq_mask

            # predict x_0 from x_t
            logits = model(x_t, t, tags, pad_mask)

            # loss only on positions that were masked
            loss_mask = corrupted
            if loss_mask.sum() == 0:
                continue

            logits_flat = logits[loss_mask]
            targets_flat = x_0[loss_mask]
            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = loss_mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n

        scheduler.step()
        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate(model, schedule, val_loader, device)

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
                    "diffusion_steps": args.diffusion_steps,
                    "noise_schedule": args.noise_schedule,
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
