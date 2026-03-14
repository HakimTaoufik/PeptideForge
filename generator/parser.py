from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    device="cpu",
)

# Define one label description per value per tag
LABELS = {
    "length":        ["short peptide", "medium length peptide", "long peptide"],
    "charge":        ["low charge", "medium charge", "high charge"],
    "hydrophobicity":["low hydrophobicity, hydrophilic", "medium hydrophobicity", "high hydrophobicity, hydrophobic"],
    "type":          ["antimicrobial peptide AMP", "non-antimicrobial"],
}

TAG_VALUES = {
    "length":         ["<LEN_SHORT>", "<LEN_MED>", "<LEN_LONG>"],
    "charge":         ["<CHARGE_LOW>", "<CHARGE_MED>", "<CHARGE_HIGH>"],
    "hydrophobicity": ["<HYDRO_LOW>", "<HYDRO_MED>", "<HYDRO_HIGH>"],
    "type":           ["<AMP>", "<NON_AMP>"],
}

# Pre-encode all labels once at startup
label_embeddings = {
    dim: model.encode(descs, convert_to_tensor=True)
    for dim, descs in LABELS.items()
}

def parse_tags(user_input: str) -> str:
    query_emb = model.encode(user_input, convert_to_tensor=True)
    tags = []
    for dim, embs in label_embeddings.items():
        scores = util.cos_sim(query_emb, embs)[0]
        best_idx = scores.argmax().item()
        tags.append(TAG_VALUES[dim][best_idx])
    return " ".join(tags)