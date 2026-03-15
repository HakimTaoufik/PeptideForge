import argparse
import os
import copy
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

        x = torch.tensor(seq_ids + [PAD_ID] * pad_len, dtype=torch.long)
        seq_mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)

        # ── JEPA masking: mask a contiguous block (I-JEPA style) ──
        # Target block: 50-80% of the sequence (harder prediction task)
        target_ratio = 0.5 + 0.3 * torch.rand(1).item()
        block_len = max(2, int(target_ratio * L))
        max_start = max(0, L - block_len)
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = min(start + block_len, L)

        target_mask = torch.zeros(self.max_length, dtype=torch.bool)
        target_mask[start:end] = True
        target_mask = target_mask & seq_mask  # don't mask padding

        context_mask = seq_mask & ~target_mask  # visible positions

        # context input: real tokens where visible, MASK where target
        x_context = x.clone()
        x_context[target_mask] = MASK_ID

        return {
            "tags": tag_vec,
            "x_clean": x,
            "x_context": x_context,
            "seq_mask": seq_mask,
            "target_mask": target_mask,
            "context_mask": context_mask,
        }


# ── Encoder ─────────────────────────────────────────────────────────────

class JEPAEncoder(nn.Module):
    """
    Bidirectional Transformer encoder.
    Used for both context encoder and target encoder (EMA copy).
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

    def forward(self, x, tags, pad_mask=None):
        B, L = x.shape
        device = x.device
        positions = torch.arange(L, device=device).unsqueeze(0)

        h = self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions)
        tag_emb = self.tag_proj(tags).unsqueeze(1)
        h = self.drop(h + tag_emb)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)
        return h  # (B, L, d_model)


# ── Predictor ───────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    Lightweight Transformer that takes context encoder output
    and predicts target encoder representations at masked positions.
    Operates in latent space — never sees raw tokens.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2,
                 dim_ff=256, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # mask token embedding (learnable query for target positions)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, context_repr, target_mask, pad_mask=None):
        """
        context_repr: (B, L, d_model) from context encoder
        target_mask:  (B, L) True for positions to predict
        Returns predictions at ALL positions (B, L, d_model)
        """
        B, L, D = context_repr.shape
        device = context_repr.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        pos_enc = self.pos_emb(positions)

        # at target positions: replace context repr with learnable mask embedding
        h = context_repr.clone()
        mask_expanded = target_mask.unsqueeze(-1).expand(-1, -1, D)
        mask_token = self.mask_emb.expand(B, L, -1)
        h = torch.where(mask_expanded, mask_token + pos_enc, h + pos_enc)

        h = self.predictor(h, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)
        return h


# ── Token Decoder (for generation) ─────────────────────────────────────

class TokenDecoder(nn.Module):
    """
    Maps latent representations back to token logits.
    Trained alongside JEPA (Stage 1+2 combined).
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, h):
        return self.head(h)


# ── Full JEPA Model ────────────────────────────────────────────────────

class PeptideJEPA(nn.Module):
    def __init__(self, vocab_size, num_tags, d_model=128, nhead=4,
                 enc_layers=4, pred_layers=2, dim_ff=512, pred_ff=256,
                 max_len=64, dropout=0.1, ema_decay=0.996):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.ema_decay = ema_decay

        # context encoder (trained via gradients)
        self.context_encoder = JEPAEncoder(
            vocab_size, num_tags, d_model, nhead, enc_layers,
            dim_ff, max_len, dropout,
        )

        # target encoder (EMA of context encoder — no gradients)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # predictor (trained via gradients)
        self.predictor = JEPAPredictor(
            d_model, nhead, pred_layers, pred_ff, max_len, dropout,
        )

        # token decoder (trained via CE loss)
        self.token_decoder = TokenDecoder(d_model, vocab_size)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: target_encoder ← decay * target + (1-decay) * context"""
        for t_param, c_param in zip(self.target_encoder.parameters(),
                                     self.context_encoder.parameters()):
            t_param.data.mul_(self.ema_decay).add_(c_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, x_context, x_clean, tags, seq_mask, target_mask):
        """
        Returns:
            jepa_loss: MSE between predicted and target representations (latent space)
            token_loss: CE between decoded tokens and ground truth (token space)
            pred_repr: predicted representations at target positions
        """
        pad_mask = ~seq_mask

        # context encoder: sees masked input
        context_repr = self.context_encoder(x_context, tags, pad_mask)

        # target encoder (EMA): sees clean input, produces targets
        with torch.no_grad():
            target_repr = self.target_encoder(x_clean, tags, pad_mask)

        # predictor: predict target representations from context
        pred_repr = self.predictor(context_repr, target_mask, pad_mask)

        # ── JEPA loss: MSE in latent space at target positions ──
        target_positions = target_mask.unsqueeze(-1).expand_as(pred_repr)
        pred_at_target = pred_repr[target_positions].view(-1, self.d_model)
        true_at_target = target_repr[target_positions].view(-1, self.d_model)

        # normalize before MSE (VICReg-style, prevents collapse)
        pred_norm = F.normalize(pred_at_target, dim=-1)
        true_norm = F.normalize(true_at_target, dim=-1)
        jepa_loss = F.mse_loss(pred_norm, true_norm)

        # ── Token decoding loss: CE at target positions ──
        token_logits = self.token_decoder(pred_repr)  # (B, L, V)
        token_logits_flat = token_logits[target_mask]  # (N_masked, V)
        token_targets = x_clean[target_mask]  # (N_masked,)
        token_loss = F.cross_entropy(token_logits_flat, token_targets)

        return jepa_loss, token_loss, pred_repr

    @torch.no_grad()
    def generate_step(self, x, tags, is_masked, pad_mask):
        """
        One refinement step:
        1. Encode current (partially masked) sequence
        2. Predict representations at masked positions
        3. Decode to token logits
        Returns logits at all positions
        """
        self.eval()
        context_repr = self.context_encoder(x, tags, pad_mask)
        pred_repr = self.predictor(context_repr, is_masked, pad_mask)
        logits = self.token_decoder(pred_repr)
        return logits


# ── Variance-Covariance regularization (prevents collapse) ─────────────

def vicreg_regularization(representations, target_mask, d_model=128):
    """
    Full VICReg regularization on predicted representations.
    - Variance: std of each dimension should stay above 1
    - Covariance: off-diagonal covariance should be near 0 (decorrelation)
    """
    reps = representations[target_mask]  # (N, d_model)
    if reps.size(0) < 4:
        return torch.tensor(0.0, device=reps.device)

    # center
    reps = reps - reps.mean(dim=0)

    # variance loss: std should be >= 1
    std = reps.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    # covariance loss: decorrelate dimensions
    N = reps.size(0)
    cov = (reps.T @ reps) / max(N - 1, 1)
    # zero out diagonal, penalize off-diagonal
    cov.fill_diagonal_(0.0)
    cov_loss = (cov.pow(2)).mean()

    return var_loss + 0.5 * cov_loss


# ── Training ────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_jepa, total_token, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_ctx = batch["x_context"].to(device)
            x_clean = batch["x_clean"].to(device)
            tags = batch["tags"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            B = x_ctx.size(0)

            jepa_loss, token_loss, _ = model(x_ctx, x_clean, tags, seq_mask, target_mask)
            total_jepa += jepa_loss.item() * B
            total_token += token_loss.item() * B
            count += B

    return total_jepa / count, total_token / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_jepa")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--pred_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--jepa_weight", type=float, default=5.0)
    parser.add_argument("--token_weight", type=float, default=1.0)
    parser.add_argument("--vicreg_weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PeptideJEPA(
        VOCAB_SIZE, NUM_TAG_VALUES,
        d_model=args.d_model, nhead=args.nhead,
        enc_layers=args.enc_layers, pred_layers=args.pred_layers,
        dim_ff=args.dim_ff, max_len=args.max_length,
        dropout=args.dropout, ema_decay=args.ema_decay,
    ).to(device)

    # only train context_encoder, predictor, token_decoder (not target_encoder)
    trainable = list(model.context_encoder.parameters()) + \
                list(model.predictor.parameters()) + \
                list(model.token_decoder.parameters())
    param_count = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {param_count:,} / Total: {total_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_jepa, total_token, total_var, count = 0.0, 0.0, 0.0, 0

        # EMA warmup: start with lower decay (faster update) then increase
        ema_decay = min(args.ema_decay, 1 - (1 - args.ema_decay) * (1 - epoch / 50))
        model.ema_decay = ema_decay

        for batch in train_loader:
            x_ctx = batch["x_context"].to(device)
            x_clean = batch["x_clean"].to(device)
            tags = batch["tags"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            B = x_ctx.size(0)

            jepa_loss, token_loss, pred_repr = model(
                x_ctx, x_clean, tags, seq_mask, target_mask,
            )

            var_loss = vicreg_regularization(pred_repr, target_mask, args.d_model)

            loss = (args.jepa_weight * jepa_loss +
                    args.token_weight * token_loss +
                    args.vicreg_weight * var_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # EMA update target encoder
            model.update_target_encoder()

            total_jepa += jepa_loss.item() * B
            total_token += token_loss.item() * B
            total_var += var_loss.item() * B
            count += B

        scheduler.step()

        t_jepa = total_jepa / count
        t_token = total_token / count
        t_var = total_var / count
        v_jepa, v_token = evaluate(model, val_loader, device)
        val_combined = v_jepa + v_token

        print(
            f"Epoch {epoch:3d} | "
            f"jepa {t_jepa:.4f} | token {t_token:.4f} | var {t_var:.4f} | "
            f"val_jepa {v_jepa:.4f} | val_token {v_token:.4f} | "
            f"ema {ema_decay:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": VOCAB_SIZE, "num_tags": NUM_TAG_VALUES,
                    "d_model": args.d_model, "nhead": args.nhead,
                    "enc_layers": args.enc_layers, "pred_layers": args.pred_layers,
                    "dim_ff": args.dim_ff, "max_len": args.max_length,
                    "dropout": args.dropout, "ema_decay": args.ema_decay,
                },
            }, f"{args.output_dir}/best.pt")
            print(f"  ↑ saved best (val={val_combined:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
