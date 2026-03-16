import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_peptides(model, tokenizer, tags, num_sequences=5,
                      max_length=64, temperature=1.0, top_k=0, top_p=0.9):
    prompt = " ".join(tags)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    sequences = []
    for output in outputs:
        tokens = tokenizer.convert_ids_to_tokens(output)
        aa_tokens = [t.replace("Ġ", "") for t in tokens
                     if not (t.startswith("<") and t.endswith(">"))
                     and t.replace("Ġ", "").isalpha()
                     and len(t.replace("Ġ", "")) == 1]
        sequences.append("".join(aa_tokens))

    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/Data/protein/outputs/gpt/peptide_gpt2/best")
    parser.add_argument("--length_tag", default="<LEN_MED>")
    parser.add_argument("--charge_tag", default="<CHARGE_MED>")
    parser.add_argument("--hydro_tag", default="<HYDRO_MED>")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    TAG_TOKENS = [
        "<LEN_SHORT>", "<LEN_MED>", "<LEN_LONG>",
        "<CHARGE_LOW>", "<CHARGE_MED>", "<CHARGE_HIGH>",
        "<HYDRO_LOW>", "<HYDRO_MED>", "<HYDRO_HIGH>",
        "<EOS>",
    ]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": TAG_TOKENS,
        "pad_token": "<PAD>",
    })

    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    tags = [args.length_tag, args.charge_tag, args.hydro_tag]
    sequences = generate_peptides(
        model, tokenizer, tags,
        num_sequences=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"\nGenerated peptides for {' '.join(tags)}:\n")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq} (len={len(seq)})")


if __name__ == "__main__":
    main()
