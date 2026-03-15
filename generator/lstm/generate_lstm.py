import argparse
import torch
from train_lstm import (
    PeptideLSTM, TAG2ID, TAG_TOKENS, NUM_TAG_VALUES,
    AMINO_ACIDS, ID2TOKEN,
)


def make_tag_vec(length_tag, charge_tag, hydro_tag):
    vec = torch.zeros(NUM_TAG_VALUES)
    for t in [length_tag, charge_tag, hydro_tag]:
        if t in TAG2ID:
            vec[TAG2ID[t]] = 1.0
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_lstm/best.pt")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = PeptideLSTM(
        cfg["vocab_size"], cfg["num_tags"],
        emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        condition_mode=cfg["condition_mode"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tag_vec = make_tag_vec(args.length_tag, args.charge_tag, args.hydro_tag)
    tags = [args.length_tag, args.charge_tag, args.hydro_tag]

    print(f"\nGenerated peptides for {' '.join(tags)}:\n")
    for i in range(args.n):
        seq = model.generate(tag_vec, temperature=args.temperature, top_p=args.top_p)
        print(f"  {i+1}. {seq} (len={len(seq)})")


if __name__ == "__main__":
    main()
