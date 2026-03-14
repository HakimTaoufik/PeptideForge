import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

TAG_TOKENS = [
    "<LEN_SHORT>", "<LEN_MED>", "<LEN_LONG>",
    "<CHARGE_LOW>", "<CHARGE_MED>", "<CHARGE_HIGH>",
    "<HYDRO_LOW>", "<HYDRO_MED>", "<HYDRO_HIGH>",
    "<EOS>",
]

NUM_CONDITIONING_TAGS = 3


def space_separate_aa(text):
    """'<LEN_LONG> <CHARGE_MED> <HYDRO_LOW> RKLLR <EOS>'
    → '<LEN_LONG> <CHARGE_MED> <HYDRO_LOW> R K L L R <EOS>'
    """
    parts = text.strip().split()
    result = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            result.append(part)
        else:
            result.append(" ".join(list(part)))
    return " ".join(result)


class PeptideDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        df = pd.read_csv(csv_path)
        col = df.columns[0]

        self.input_ids = []
        self.attention_masks = []
        self.labels_list = []

        for text in df[col]:
            text = space_separate_aa(str(text))
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)

            labels = ids.clone()
            labels[mask == 0] = -100
            # mask the 3 conditioning tags so loss only covers the sequence + <EOS>
            labels[:NUM_CONDITIONING_TAGS] = -100

            self.input_ids.append(ids)
            self.attention_masks.append(mask)
            self.labels_list.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels_list[idx],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="/Data/protein/data/generator/train_core.csv")
    parser.add_argument("--val_csv", default="/Data/protein/data/generator/val_core.csv")
    parser.add_argument("--output_dir", default="/Data/protein/outputs/gpt/peptide_gpt2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    # --- tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": TAG_TOKENS,
        "pad_token": "<PAD>",
    })

    # --- model ---
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # --- datasets ---
    train_dataset = PeptideDataset(args.train_csv, tokenizer, args.max_length)
    val_dataset = PeptideDataset(args.val_csv, tokenizer, args.max_length)
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # --- training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    save_path = f"{args.output_dir}/best"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
