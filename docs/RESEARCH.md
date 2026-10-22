# Research scope

[Landing page](../README.md) · [Experiments](EXPERIMENTS.md) · [Reproducibility](REPRODUCIBILITY.md)

## Question and representation

How do different generative objectives behave when asked to produce peptide sequences with coarse, interpretable physicochemical properties? PeptideForge implements eight approaches to this question. It provides implementations, checkpoints for seven approaches, and saved samples for all eight; it does not establish a biologically validated generator or a controlled architecture benchmark.

Peptides are ordered strings over the 20 standard amino acids. Position, residue identity, and sequence length matter. The training examples contain three property tags followed by residues and an end token, for example:

```text
<LEN_SHORT> <CHARGE_MED> <HYDRO_MED> QKKIRARLSA <EOS>
```

This is the second line of [train_core.csv](../data/generator/train_core.csv). The generator task uses AMP-labelled examples. Although the richer CSVs contain `<AMP>`/`<NONAMP>` prefixes, the core inputs contain only length, charge, and hydrophobicity tags. AMP status is not a fourth trained control in these generator scripts. The optional [natural-language parser](../generator/parser.py) predicts four tags using MiniLM similarity, including `<NON_AMP>`; this differs from the stored `<NONAMP>` spelling and is not a validated conditioning interface.

## Data and provenance

| Asset | Train | Validation | Test | Content |
|---|---:|---:|---:|---|
| `data/generator/*_generator_amp_only.csv` | 23,791 | 2,974 | 2,974 | Rich records, all `label=1` |
| `data/generator/*_core.csv` | 23,791 lines | 2,974 lines | 2,974 lines | Headerless three-tag sequence records |
| `data/scorer/*_scorer.csv` | 129,556 | 16,195 | 16,195 | AMP and negative examples |

Scorer positives number 23,791 / 2,974 / 2,974; negatives number 105,765 / 13,221 / 13,221. The corresponding `data/other/{train,val,test}.csv` files are byte-identical to the scorer CSVs. Other text exports are retained as historical representations.

All six rich splits contain unique sequences of length 10–50 using standard amino acids. Within each task, the three splits have **zero exact sequence overlap**. AMP records are subsets of the corresponding scorer splits. Core lines match the rich AMP sequence order exactly. These checks do not establish separation by sequence homology, biological family, or source database. Counts and hashes are in [audit.json](assets/audit.json).

The current tree has no raw-data acquisition or split-construction program. Git history retains raw FASTAs, a file-size manifest, and an incomplete dataset notebook in [commit ca178d6](https://github.com/HakimTaoufik/PeptideForge/tree/ca178d679b2c8684d3538201c9c3883b3a0efb60/noteboooks). The notebook sets seed 42 but stops before the complete split pipeline; importing `train_test_split` does not prove how these splits were made. Names such as `amp_…`, `Fabaceae_eliminated_…`, and `Viridiplantae_eliminated_…` are provenance clues, not sufficient attribution. No upstream dataset DOI, release, collection protocol, or license was recovered. Do not attribute these data to a particular AMP database without further evidence.

## Conditional constraints

| Property | Definition | Bins observed in rich CSV labels | Notebook/demo evaluation bins |
|---|---|---|---|
| Length | Number of residues | Short ≤20; medium 21–35; long ≥36 | Short <20; medium [20,35); long ≥35 |
| Charge | `count(K)+count(R)-count(D)-count(E)` | Low ≤1; medium 2–4; high ≥5 | Low <1; medium [1,4); high ≥4 |
| Hydrophobicity | Mean Kyte–Doolittle value | Low <−1; medium [−1,1); high ≥1 | Same nominal boundaries |

The first two columns follow [feature extraction](../training/utils/feature_extraction.py) and inspection of saved labels. Charge is a composition proxy, not a full pH-dependent electrostatic calculation. The older evaluation notebook adds `0.1 × count(H)` to its charge proxy, unlike the scorer and rich CSVs. Floating-point means close to hydrophobicity boundaries can change bins; the new utility uses an unrounded `math.fsum` mean and records this choice.

Tags specify a requested bin, not a guarantee. Diffusion, MaskGIT, EBM, and JEPA generation scripts use length hints 12/20/30; these differ from the dataset's medium/long ranges. Their `--seq_len` option can override the hint. Autoregressive and CVAE decoders can stop at EOS. These sampling choices confound comparisons of architecture alone.

## Implemented architectures

See the [complete source/entry-point table](../README.md#implemented-models). GPT-2 loads pretrained `gpt2` weights and adds tags plus a pad token; residues are space-separated for the GPT-2 tokenizer. The scratch Transformer uses an amino-acid vocabulary, learned positional embeddings, and causal Transformer decoder layers with dummy zero memory. Both train next-token objectives and mask padding plus the first three label positions.

The LSTM uses residue embeddings and either projected tags concatenated at each step or tags initializing hidden/cell states. The saved configuration uses concatenation. The CVAE uses a bidirectional GRU encoder, Gaussian latent mean/log-variance, and a conditioned GRU decoder; its objective combines reconstruction and warmed-up KL loss.

Discrete diffusion uses an absorbing mask process and a bidirectional Transformer denoiser conditioned on tags and timestep. MaskGIT uses variable-ratio masking and a bidirectional Transformer without diffusion timestep embeddings, with iterative confidence-based decoding. The EBM is specifically a **pseudo-likelihood** model: a bidirectional Transformer plus learned pairwise coupling layers predicts masked residues, with Gibbs/annealed/block sampling; it is not evidence of partition-function or contrastive-divergence training. Its energy helper returns mean conditional negative log likelihood over positions.

JEPA combines a context encoder, an exponential-moving-average target encoder, a representation predictor, and a token decoder. Its training objective includes normalized latent MSE, token cross-entropy, and variance/covariance regularization. Generation uses token decoding and iterative refinement; this is a peptide adaptation with a generative head, not an exact reproduction of image I-JEPA.

## Interpretation and limitations

Saved scores measure agreement with a composition-based random forest. They are neither antimicrobial assays nor independent assessments of toxicity, selectivity, stability, structure, or synthesis feasibility. Conditioning and scoring share length/charge/hydrophobicity features; the scorer can reward those proxies. The archive baseline assigns `amp_probability=1` to labelled training records rather than predicting their probabilities.

No multi-seed estimates, confidence intervals, matched compute budgets, homology-aware novelty evaluation, or wet-lab measurements were recovered. Exact-match novelty can coexist with near-duplicate biological sequences. Objectives and stochastic validation losses differ across families. The available artifacts therefore support implementation and sample-level analysis, not a general best-model ranking.

## References and rights

Conceptual references, rather than claims of exact reproduction:

- Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Sohn et al. (2015), [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html).
- Austin et al. (2021), [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006).
- Chang et al. (2022), [MaskGIT](https://arxiv.org/abs/2202.04200).
- Assran et al. (2023), [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243).

The repository has no root LICENSE file. This documentation does not grant a new license or resolve rights for the datasets/checkpoints. Preserve attribution and applicable terms for upstream data, pretrained GPT-2/MiniLM assets, and software dependencies; dataset attribution remains incomplete.
