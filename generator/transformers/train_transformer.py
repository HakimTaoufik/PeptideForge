import argparse
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── Vocabulary ──────────────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

SPECIAL_TOKENS = [
    "<PAD>", "<EOS>",
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
NUM_TAGS = 3


def tokenize(text):
    parts = text.strip().split()
    tokens = []
    for part in parts:
        if part in TOKEN2ID:
            tokens.append(TOKEN2ID[part])
        else:
            tokens.extend(TOKEN2ID[ch] for ch in part if ch in TOKEN2ID)
    return tokens


# ── Dataset ─────────────────────────────────────────────────────────────

class PeptideDataset(Dataset):
    def __init__(self, csv_path, max_length=64):
        df = pd.read_csv(csv_path)
        col = df.columns[0]
        self.sequences = []
        for text in df[col]:
            ids = tokenize(str(text))
            if len(ids) > max_length:
                ids = ids[:max_length]
            self.sequences.append(ids)
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx]
        padded = ids + [PAD_ID] * (self.max_length - len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == PAD_ID] = -100
        labels[:NUM_TAGS] = -100  # mask conditioning tags

        return {"input_ids": input_ids, "labels": labels}


# ── Model ───────────────────────────────────────────────────────────────

class PeptideTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_ff=512, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) * math.sqrt(self.d_model) + self.pos_emb(positions))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        pad_mask = (input_ids == PAD_ID)

        memory = torch.zeros(B, 1, self.d_model, device=device)
        x = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, tag_ids, max_len=64, temperature=1.0, top_k=0, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([tag_ids], dtype=torch.long, device=device)

        for _ in range(max_len - len(tag_ids)):
            logits, _ = self.forward(ids)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = -float("inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

        return ids.squeeze(0).tolist()


# ── Training loop ───────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(ids, labels)
            n_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_transformer")
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
    print(f"Vocab size: {VOCAB_SIZE}  Device: {device}")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = PeptideTransformer(
        VOCAB_SIZE, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        max_len=args.max_length, dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50  # early stopping patience
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(ids, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        scheduler.step()
        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": VOCAB_SIZE, "d_model": args.d_model,
                    "nhead": args.nhead, "num_layers": args.num_layers,
                    "dim_ff": args.dim_ff, "max_len": args.max_length,
                    "dropout": args.dropout,
                },
            }, f"{args.output_dir}/best.pt")
            print(f"  ↑ saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
