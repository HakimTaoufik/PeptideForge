import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Vocabulary ──────────────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<SOS>"]

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
SOS_ID = TOKEN2ID["<SOS>"]

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
            if len(seq_ids) > max_length - 1:
                seq_ids = seq_ids[:max_length - 1]
            self.samples.append((tags, seq_ids))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tags, seq_ids = self.samples[idx]
        L = len(seq_ids)

        # encoder input: AA1 AA2 ... AAn <EOS>
        enc_in = seq_ids + [EOS_ID]
        # decoder input:  <SOS> AA1 AA2 ... AAn
        dec_in = [SOS_ID] + seq_ids
        # decoder target: AA1 AA2 ... AAn <EOS>
        dec_tgt = seq_ids + [EOS_ID]

        seq_len = len(enc_in)
        pad_len = self.max_length - seq_len

        tag_vec = torch.zeros(NUM_TAG_VALUES)
        for t in tags:
            tag_vec[t] = 1.0

        enc_padded = torch.tensor(enc_in + [PAD_ID] * pad_len, dtype=torch.long)
        dec_in_padded = torch.tensor(dec_in + [PAD_ID] * pad_len, dtype=torch.long)
        dec_tgt_padded = torch.tensor(dec_tgt + [-100] * pad_len, dtype=torch.long)
        lengths = torch.tensor(seq_len, dtype=torch.long)

        return {
            "tags": tag_vec,
            "enc_input": enc_padded,
            "dec_input": dec_in_padded,
            "dec_target": dec_tgt_padded,
            "length": lengths,
        }


# ── Model ───────────────────────────────────────────────────────────────

class PeptideCVAE(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=64, hidden_dim=256,
                 latent_dim=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.tag_proj = nn.Linear(num_tags, emb_dim)

        # ── Encoder ──
        self.encoder = nn.GRU(
            emb_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # bidirectional → 2 * hidden_dim
        enc_out_dim = hidden_dim * 2 + emb_dim  # concat with tag embedding
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # ── Decoder ──
        # decoder input = emb_dim + tag_emb + latent
        dec_input_dim = emb_dim + emb_dim + latent_dim
        self.decoder = nn.GRU(
            dec_input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

        # latent + tag → initial decoder hidden
        self.latent_to_hidden = nn.Linear(latent_dim + emb_dim, hidden_dim * num_layers)

    def encode(self, enc_input, tags, lengths):
        B = enc_input.size(0)
        x = self.embedding(enc_input)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        _, h = self.encoder(packed)
        # h: (num_layers * 2, B, hidden_dim) → take last layer fwd + bwd
        h_fwd = h[-2]  # (B, hidden_dim)
        h_bwd = h[-1]  # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim*2)

        tag_emb = self.tag_proj(tags)  # (B, emb_dim)
        enc_out = torch.cat([h_cat, tag_emb], dim=-1)  # (B, hidden_dim*2 + emb_dim)

        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        return mu, logvar, tag_emb

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, dec_input, z, tag_emb, hidden=None):
        B, T = dec_input.shape
        x = self.embedding(dec_input)  # (B, T, emb_dim)

        # concat z and tag_emb at every timestep
        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        tag_exp = tag_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, z_exp, tag_exp], dim=-1)

        if hidden is None:
            h_in = torch.cat([z, tag_emb], dim=-1)
            hidden = self.latent_to_hidden(h_in)
            hidden = hidden.view(B, self.num_layers, self.hidden_dim)
            hidden = hidden.permute(1, 0, 2).contiguous()
            hidden = torch.tanh(hidden)

        out, hidden = self.decoder(x, hidden)
        out = self.drop(out)
        logits = self.output_head(out)
        return logits, hidden

    def forward(self, enc_input, dec_input, tags, lengths, dec_target=None):
        mu, logvar, tag_emb = self.encode(enc_input, tags, lengths)
        z = self.reparameterize(mu, logvar)
        logits, _ = self.decode(dec_input, z, tag_emb)

        recon_loss = None
        if dec_target is not None:
            recon_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                dec_target.reshape(-1),
                ignore_index=-100,
            )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return logits, recon_loss, kl_loss, mu, logvar

    @torch.no_grad()
    def generate(self, tag_vec, max_len=64, temperature=1.0, top_k=0, top_p=0.9, z=None):
        self.eval()
        device = next(self.parameters()).device
        tags = tag_vec.unsqueeze(0).to(device)
        tag_emb = self.tag_proj(tags)

        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)

        # init hidden
        h_in = torch.cat([z, tag_emb], dim=-1)
        hidden = self.latent_to_hidden(h_in)
        hidden = hidden.view(1, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)

        token = torch.tensor([[SOS_ID]], device=device)
        result = []

        for _ in range(max_len):
            x = self.embedding(token)
            z_exp = z.unsqueeze(1)
            tag_exp = tag_emb.unsqueeze(1)
            x = torch.cat([x, z_exp, tag_exp], dim=-1)

            out, hidden = self.decoder(x, hidden)
            logits = self.output_head(out[:, -1, :]) / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = -float("inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            token = next_id

            if next_id.item() == EOS_ID:
                break
            if ID2TOKEN.get(next_id.item(), "") in AMINO_ACIDS:
                result.append(ID2TOKEN[next_id.item()])

        return "".join(result)


# ── KL Annealing ────────────────────────────────────────────────────────

def kl_weight_schedule(epoch, warmup_epochs=10, max_weight=1.0):
    """Linear KL annealing: 0 → max_weight over warmup_epochs."""
    if epoch < warmup_epochs:
        return max_weight * (epoch / warmup_epochs)
    return max_weight


# ── Training ────────────────────────────────────────────────────────────

def evaluate(model, loader, device, kl_w):
    model.eval()
    total_recon, total_kl, total_tokens = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            enc = batch["enc_input"].to(device)
            dec_in = batch["dec_input"].to(device)
            tags = batch["tags"].to(device)
            tgt = batch["dec_target"].to(device)
            lengths = batch["length"].to(device)

            _, recon, kl, _, _ = model(enc, dec_in, tags, lengths, tgt)
            n = (tgt != -100).sum().item()
            total_recon += recon.item() * n
            total_kl += kl.item() * n
            total_tokens += n

    avg_recon = total_recon / max(total_tokens, 1)
    avg_kl = total_kl / max(total_tokens, 1)
    return avg_recon, avg_kl, avg_recon + kl_w * avg_kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_vae")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kl_warmup", type=int, default=50)
    parser.add_argument("--kl_max", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PeptideCVAE(
        VOCAB_SIZE, NUM_TAG_VALUES,
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}  Latent dim: {args.latent_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        kl_w = kl_weight_schedule(epoch, args.kl_warmup, args.kl_max)
        total_recon, total_kl, total_tokens = 0.0, 0.0, 0

        for batch in train_loader:
            enc = batch["enc_input"].to(device)
            dec_in = batch["dec_input"].to(device)
            tags = batch["tags"].to(device)
            tgt = batch["dec_target"].to(device)
            lengths = batch["length"].to(device)

            _, recon_loss, kl_loss, _, _ = model(enc, dec_in, tags, lengths, tgt)
            loss = recon_loss + kl_w * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = (tgt != -100).sum().item()
            total_recon += recon_loss.item() * n
            total_kl += kl_loss.item() * n
            total_tokens += n

        scheduler.step()
        train_recon = total_recon / max(total_tokens, 1)
        train_kl = total_kl / max(total_tokens, 1)
        val_recon, val_kl, val_total = evaluate(model, val_loader, device, kl_w)
        # fixed metric using kl_max so it's comparable across all epochs
        val_metric = val_recon + args.kl_max * val_kl

        print(
            f"Epoch {epoch:3d} | "
            f"train_recon {train_recon:.4f} | train_kl {train_kl:.4f} | "
            f"val_recon {val_recon:.4f} | val_kl {val_kl:.4f} | "
            f"kl_w {kl_w:.3f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        # only start checkpointing/patience after warmup is complete
        if epoch >= args.kl_warmup:
            if val_metric < best_val_loss:
                best_val_loss = val_metric
                patience_counter = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "num_tags": NUM_TAG_VALUES,
                        "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
                        "latent_dim": args.latent_dim, "num_layers": args.num_layers,
                        "dropout": args.dropout, "max_length": args.max_length,
                    },
                }, f"{args.output_dir}/best.pt")
                print(f"  ↑ saved best (val_metric={val_metric:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
