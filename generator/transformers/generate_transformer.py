import argparse
import torch
from train_transformer import PeptideTransformer, TOKEN2ID, ID2TOKEN, EOS_ID, AMINO_ACIDS


def generate_peptides(model, tags, num_sequences=5, max_len=64,
                      temperature=1.0, top_k=0, top_p=0.9):
    tag_ids = [TOKEN2ID[t] for t in tags]
    sequences = []
    for _ in range(num_sequences):
        ids = model.generate(tag_ids, max_len=max_len,
                             temperature=temperature, top_k=top_k, top_p=top_p)
        aa = [ID2TOKEN[i] for i in ids if ID2TOKEN[i] in AMINO_ACIDS]
        sequences.append("".join(aa))
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_transformer/best.pt")
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

    model = PeptideTransformer(
        cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tags = [args.length_tag, args.charge_tag, args.hydro_tag]
    sequences = generate_peptides(model, tags, num_sequences=args.n,
                                  temperature=args.temperature, top_p=args.top_p)

    print(f"\nGenerated peptides for {' '.join(tags)}:\n")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq} (len={len(seq)})")


if __name__ == "__main__":
    main()
