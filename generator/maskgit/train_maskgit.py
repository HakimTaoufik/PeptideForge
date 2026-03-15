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
    """
    During training, a random fraction of tokens are masked.
    The mask ratio is sampled uniformly from [0.1, 1.0] per sample,
    following the MaskGIT schedule.
    """
    def __init__(self, csv_path, max_length=64):
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

        # random mask ratio per sample (cosine schedule favors higher ratios)
        u = torch.rand(1).item()
        mask_ratio = math.cos(u * math.pi / 2)  # maps [0,1] → [1,0], biased toward more masking
        mask_ratio = max(0.1, mask_ratio)

        # decide which positions to mask
        rand_vals = torch.rand(self.max_length)
        do_mask = (rand_vals < mask_ratio) & seq_mask

        # ensure at least one position is masked
        if do_mask.sum() == 0:
            real_positions = seq_mask.nonzero(as_tuple=True)[0]
            do_mask[real_positions[torch.randint(len(real_positions), (1,))]] = True

        x_masked = x_clean.clone()
        x_masked[do_mask] = MASK_ID

        # targets: only compute loss on masked positions
        targets = x_clean.clone()
        targets[~do_mask] = -100

        return {
            "tags": tag_vec,
            "x_masked": x_masked,
            "targets": targets,
            "seq_mask": seq_mask,
            "mask_ratio": torch.tensor(mask_ratio),
        }


# ── Model ───────────────────────────────────────────────────────────────

class PeptideMaskGIT(nn.Module):
    """
    Bidirectional Transformer that predicts masked tokens.
    No timestep embedding (unlike diffusion) — the model infers
    corruption level from the input itself.
    Conditioned on tag vector added to every position.
    """
    def __init__(self, vocab_size, num_tags, d_model=128, nhead=4,
                 num_layers=4, dim_ff=512, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.tag_proj = nn.Linear(num_tags, d_model)
        self.drop = nn.Dropout(dropout)

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

    def forward(self, x, tags, pad_mask=None):
        """
        x:        (B, L) token ids (with some MASK_ID)
        tags:     (B, num_tags) multi-hot
        pad_mask: (B, L) True for padding
        Returns logits (B, L, vocab_size)
        """
        B, L = x.shape
        device = x.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions)

        tag_emb = self.tag_proj(tags).unsqueeze(1)  # (B, 1, d_model)
        h = h + tag_emb

        h = self.drop(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


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

            logits = model(x, tags, pad_mask)
            loss_mask = targets != -100
            if loss_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                logits[loss_mask], targets[loss_mask],
            )
            n = loss_mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_maskgit")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PeptideMaskGIT(
        VOCAB_SIZE, NUM_TAG_VALUES,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        max_len=args.max_length, dropout=args.dropout,
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

            logits = model(x, tags, pad_mask)
            loss_mask = targets != -100
            if loss_mask.sum() == 0:
                continue

            loss = F.cross_entropy(logits[loss_mask], targets[loss_mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = loss_mask.sum().item()
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
