import argparse
import torch
import numpy as np
from train_vae import (
    PeptideCVAE, TAG2ID, NUM_TAG_VALUES,
    AMINO_ACIDS, ID2TOKEN,
)


def make_tag_vec(length_tag, charge_tag, hydro_tag):
    vec = torch.zeros(NUM_TAG_VALUES)
    for t in [length_tag, charge_tag, hydro_tag]:
        if t in TAG2ID:
            vec[TAG2ID[t]] = 1.0
    return vec


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PeptideCVAE(
        cfg["vocab_size"], cfg["num_tags"],
        emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./peptide_vae/best.pt")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mode", default="sample", choices=["sample", "interpolate", "sweep"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_path, device)
    latent_dim = cfg["latent_dim"]

    tags = [args.length_tag, args.charge_tag, args.hydro_tag]
    tag_vec = make_tag_vec(*tags)

    if args.mode == "sample":
        print(f"\nGenerated peptides for {' '.join(tags)}:\n")
        for i in range(args.n):
            seq = model.generate(tag_vec, temperature=args.temperature, top_p=args.top_p)
            print(f"  {i+1}. {seq} (len={len(seq)})")

    elif args.mode == "interpolate":
        print(f"\nLatent interpolation with tags {' '.join(tags)}:\n")
        z_start = torch.randn(1, latent_dim, device=device)
        z_end = torch.randn(1, latent_dim, device=device)
        steps = args.n
        for i in range(steps):
            alpha = i / max(steps - 1, 1)
            z = (1 - alpha) * z_start + alpha * z_end
            seq = model.generate(tag_vec, temperature=args.temperature, top_p=args.top_p, z=z)
            print(f"  α={alpha:.2f} | {seq} (len={len(seq)})")

    elif args.mode == "sweep":
        print(f"\nLatent sweep (fixed z, varying tags):\n")
        z_fixed = torch.randn(1, latent_dim, device=device)
        tag_combos = [
            ("<LEN_SHORT>", "<CHARGE_LOW>",  "<HYDRO_LOW>"),
            ("<LEN_SHORT>", "<CHARGE_MED>",  "<HYDRO_MED>"),
            ("<LEN_SHORT>", "<CHARGE_HIGH>", "<HYDRO_HIGH>"),
            ("<LEN_MED>",   "<CHARGE_LOW>",  "<HYDRO_LOW>"),
            ("<LEN_MED>",   "<CHARGE_MED>",  "<HYDRO_MED>"),
            ("<LEN_MED>",   "<CHARGE_HIGH>", "<HYDRO_HIGH>"),
            ("<LEN_LONG>",  "<CHARGE_LOW>",  "<HYDRO_LOW>"),
            ("<LEN_LONG>",  "<CHARGE_MED>",  "<HYDRO_MED>"),
            ("<LEN_LONG>",  "<CHARGE_HIGH>", "<HYDRO_HIGH>"),
        ]
        for combo in tag_combos:
            tv = make_tag_vec(*combo)
            seq = model.generate(tv, temperature=args.temperature, top_p=args.top_p, z=z_fixed)
            label = " ".join(combo)
            print(f"  {label:55s} → {seq} (len={len(seq)})")


if __name__ == "__main__":
    main()
