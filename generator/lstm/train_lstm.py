import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

# Tags are encoded as a separate conditioning vector, not as sequence tokens
TAG2ID = {t: i for i, t in enumerate(TAG_TOKENS)}
NUM_TAG_VALUES = len(TAG_TOKENS)  # 9 total


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
            # input:  <SOS> AA1 AA2 ... AAn
            # target: AA1 AA2 ... AAn <EOS>
            inp = [SOS_ID] + seq_ids
            tgt = seq_ids + [EOS_ID]
            if len(inp) > max_length:
                inp = inp[:max_length]
                tgt = tgt[:max_length]
            self.samples.append((tags, inp, tgt))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tags, inp, tgt = self.samples[idx]
        L = len(inp)
        pad_len = self.max_length - L

        tag_vec = torch.zeros(NUM_TAG_VALUES)
        for t in tags:
            tag_vec[t] = 1.0

        inp_padded = torch.tensor(inp + [PAD_ID] * pad_len, dtype=torch.long)
        tgt_padded = torch.tensor(tgt + [-100] * pad_len, dtype=torch.long)
        mask = torch.tensor([1] * L + [0] * pad_len, dtype=torch.long)

        return {
            "tags": tag_vec,
            "input_ids": inp_padded,
            "targets": tgt_padded,
            "mask": mask,
        }


# ── Model ───────────────────────────────────────────────────────────────

class PeptideLSTM(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=64, hidden_dim=256,
                 num_layers=2, dropout=0.2, condition_mode="concat"):
        """
        condition_mode:
            'init'   — tags set initial hidden state
            'concat' — tags concatenated to embedding at every step
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_mode = condition_mode

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)

        # tag projection
        if condition_mode == "init":
            self.tag_to_h = nn.Linear(num_tags, hidden_dim * num_layers)
            self.tag_to_c = nn.Linear(num_tags, hidden_dim * num_layers)
            lstm_input_dim = emb_dim
        else:  # concat
            self.tag_proj = nn.Linear(num_tags, emb_dim)
            lstm_input_dim = emb_dim * 2

        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, tags, batch_size, device):
        if self.condition_mode == "init":
            h = self.tag_to_h(tags).view(batch_size, self.num_layers, self.hidden_dim)
            c = self.tag_to_c(tags).view(batch_size, self.num_layers, self.hidden_dim)
            h = h.permute(1, 0, 2).contiguous()
            c = c.permute(1, 0, 2).contiguous()
            return (torch.tanh(h), torch.tanh(c))
        else:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)

    def forward(self, input_ids, tags, targets=None):
        B, T = input_ids.shape
        device = input_ids.device
        hidden = self._init_hidden(tags, B, device)

        x = self.embedding(input_ids)  # (B, T, emb_dim)

        if self.condition_mode == "concat":
            tag_emb = self.tag_proj(tags).unsqueeze(1).expand(-1, T, -1)  # (B, T, emb_dim)
            x = torch.cat([x, tag_emb], dim=-1)  # (B, T, emb_dim*2)

        out, _ = self.lstm(x, hidden)  # (B, T, hidden_dim)
        out = self.drop(out)
        logits = self.head(out)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, tag_vec, max_len=64, temperature=1.0, top_k=0, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        tags = tag_vec.unsqueeze(0).to(device)
        hidden = self._init_hidden(tags, 1, device)

        token = torch.tensor([[SOS_ID]], device=device)
        result = []

        for _ in range(max_len):
            x = self.embedding(token)
            if self.condition_mode == "concat":
                tag_emb = self.tag_proj(tags).unsqueeze(1)
                x = torch.cat([x, tag_emb], dim=-1)

            out, hidden = self.lstm(x, hidden)
            logits = self.head(out[:, -1, :]) / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = -float("inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            token = next_id

            if next_id.item() == EOS_ID:
                break
            if ID2TOKEN.get(next_id.item(), "") in AMINO_ACIDS:
                result.append(ID2TOKEN[next_id.item()])

        return "".join(result)


# ── Training ────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            targets = batch["targets"].to(device)
            _, loss = model(ids, tags, targets)
            n = (targets != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train_generator_amp_only.csv")
    parser.add_argument("--val_csv", default="val_generator_amp_only.csv")
    parser.add_argument("--output_dir", default="./peptide_lstm")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--condition_mode", default="concat", choices=["init", "concat"])
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PeptideDataset(args.train_csv, args.max_length)
    val_ds = PeptideDataset(args.val_csv, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = PeptideLSTM(
        VOCAB_SIZE, NUM_TAG_VALUES,
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
        condition_mode=args.condition_mode,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}  Condition mode: {args.condition_mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            targets = batch["targets"].to(device)

            _, loss = model(ids, tags, targets)
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

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": VOCAB_SIZE, "num_tags": NUM_TAG_VALUES,
                    "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers, "dropout": args.dropout,
                    "condition_mode": args.condition_mode,
                    "max_length": args.max_length,
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
