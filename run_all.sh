#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Master training & generation script for all peptide generation models
#  Runs: Transformer → VAE → LSTM → Diffusion → MaskGIT → EBM
#  Handles errors gracefully — each model runs independently
# ═══════════════════════════════════════════════════════════════════════

set -o pipefail

# ── Configuration ───────────────────────────────────────────────────────
TRAIN_CSV="${TRAIN_CSV:-/Data/protein/data/generator/train_core.csv}"
VAL_CSV="${VAL_CSV:-/Data/protein/data/generator/val_core.csv}"
SCRIPT_DIR="${SCRIPT_DIR:-./generator/}"
LOG_DIR="${LOG_DIR:-/Data/protein/logs}"

# generation test tags
LEN_TAG="<LEN_MED>"
CHARGE_TAG="<CHARGE_MED>"
HYDRO_TAG="<HYDRO_MED>"
N_GEN=5

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Setup ───────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary.txt"
> "$SUMMARY_FILE"

TOTAL_START=$(date +%s)

declare -A MODEL_STATUS
declare -A MODEL_TIME

# ── Helper functions ────────────────────────────────────────────────────

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

log_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

log_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

log_info() {
    echo -e "  ${YELLOW}→${NC} $1"
}

run_model() {
    local name="$1"
    local train_cmd="$2"
    local gen_cmd="$3"
    local log_file="$LOG_DIR/${name}.log"

    log_header "$name"

    local start=$(date +%s)

    # ── Train ──
    log_info "Training..."
    echo "=== TRAINING ===" > "$log_file"
    echo "Command: $train_cmd" >> "$log_file"
    echo "" >> "$log_file"

    eval "$train_cmd" >> "$log_file" 2>&1
    local train_exit=$?

    if [ $train_exit -ne 0 ]; then
        log_fail "Training FAILED (exit code $train_exit)"
        log_fail "See $log_file for details"
        echo "  Last 5 lines of log:"
        tail -5 "$log_file" | sed 's/^/    /'
        MODEL_STATUS[$name]="TRAIN_FAILED"
        local end=$(date +%s)
        MODEL_TIME[$name]=$(( end - start ))
        return 1
    fi

    log_success "Training completed"

    # ── Generate ──
    log_info "Generating $N_GEN test sequences..."
    echo "" >> "$log_file"
    echo "=== GENERATION ===" >> "$log_file"
    echo "Command: $gen_cmd" >> "$log_file"
    echo "" >> "$log_file"

    local gen_output
    gen_output=$(eval "$gen_cmd" 2>&1)
    local gen_exit=$?

    echo "$gen_output" >> "$log_file"

    if [ $gen_exit -ne 0 ]; then
        log_fail "Generation FAILED (exit code $gen_exit)"
        echo "  Last 5 lines:"
        echo "$gen_output" | tail -5 | sed 's/^/    /'
        MODEL_STATUS[$name]="GEN_FAILED"
    else
        log_success "Generation completed"
        echo "$gen_output" | grep -E "^\s+[0-9]+\." | head -$N_GEN | sed 's/^/    /'
        MODEL_STATUS[$name]="OK"
    fi

    local end=$(date +%s)
    MODEL_TIME[$name]=$(( end - start ))
    local elapsed=${MODEL_TIME[$name]}
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    log_info "Time: ${mins}m ${secs}s"

    return 0
}

# ── Check prerequisites ────────────────────────────────────────────────

log_header "PREFLIGHT CHECKS"

if [ ! -f "$TRAIN_CSV" ]; then
    echo -e "${RED}ERROR: Training CSV not found: $TRAIN_CSV${NC}"
    echo "Set TRAIN_CSV environment variable or place file in current directory"
    exit 1
fi
log_success "Train CSV: $TRAIN_CSV"

if [ ! -f "$VAL_CSV" ]; then
    echo -e "${RED}ERROR: Validation CSV not found: $VAL_CSV${NC}"
    exit 1
fi
log_success "Val CSV:   $VAL_CSV"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: PyTorch not available${NC}"
    exit 1
fi
log_success "PyTorch OK"

GPU_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    print(f'{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('CPU only')
" 2>/dev/null)
log_info "Device: $GPU_INFO"

TRAIN_SIZE=$(wc -l < "$TRAIN_CSV")
log_info "Train samples: ~$((TRAIN_SIZE - 1))"

echo ""
echo -e "${BOLD}Running 7 models sequentially...${NC}"


# ═══════════════════════════════════════════════════════════════════════
#  1. TRANSFORMER (from scratch)
# ═══════════════════════════════════════════════════════════════════════

#run_model "Transformer" \
#    "uv run ${SCRIPT_DIR}/transformers/train_transformer.py \
#        --train_csv $TRAIN_CSV \
#        --val_csv $VAL_CSV \
#        --output_dir /Data/protein/outputs/transformers/peptide_transformer \
#        --epochs 500 \
#        --batch_size 64 \
#        --lr 3e-4" \
#    "uv run ${SCRIPT_DIR}/transformers/generate_transformer.py \
#        --model_path /Data/protein/outputs/transformers/peptide_transformer/best.pt \
#        --length_tag '$LEN_TAG' \
#        --charge_tag '$CHARGE_TAG' \
#        --hydro_tag '$HYDRO_TAG' \
#        --n $N_GEN"


# ═══════════════════════════════════════════════════════════════════════
#  2. CVAE
# ═══════════════════════════════════════════════════════════════════════

run_model "CVAE" \
    "uv run ${SCRIPT_DIR}/vae/train_vae.py \
        --train_csv $TRAIN_CSV \
        --val_csv $VAL_CSV \
        --output_dir /Data/protein/outputs/vae/peptide_vae \
        --epochs 500 \
        --batch_size 64 \
        --lr 1e-3 \
        --latent_dim 32 \
        --kl_warmup 50 \
        --kl_max 0.5" \
    "uv run ${SCRIPT_DIR}/vae/generate_vae.py \
        --model_path /Data/protein/outputs/vae/peptide_vae/best.pt \
        --length_tag '$LEN_TAG' \
        --charge_tag '$CHARGE_TAG' \
        --hydro_tag '$HYDRO_TAG' \
        --n $N_GEN \
        --mode sample"


# ═══════════════════════════════════════════════════════════════════════
#  3. LSTM
# ═══════════════════════════════════════════════════════════════════════

#run_model "LSTM" \
#    "uv run ${SCRIPT_DIR}/lstm/train_lstm.py \
#        --train_csv $TRAIN_CSV \
#        --val_csv $VAL_CSV \
#        --output_dir /Data/protein/outputs/lstm/peptide_lstm \
#        --epochs 500 \
#        --batch_size 64 \
#        --lr 1e-3 \
#        --hidden_dim 192 \
#        --dropout 0.35 \
#        --condition_mode concat" \
#    "uv run ${SCRIPT_DIR}/lstm/generate_lstm.py \
#        --model_path /Data/protein/outputs/lstm/peptide_lstm/best.pt \
#        --length_tag '$LEN_TAG' \
#        --charge_tag '$CHARGE_TAG' \
#        --hydro_tag '$HYDRO_TAG' \
#        --n $N_GEN"


# ═══════════════════════════════════════════════════════════════════════
#  4. DISCRETE DIFFUSION
# ═══════════════════════════════════════════════════════════════════════

run_model "Diffusion" \
    "uv run ${SCRIPT_DIR}/diffusion/train_diffusion.py \
        --train_csv $TRAIN_CSV \
        --val_csv $VAL_CSV \
        --output_dir /Data/protein/outputs/diffusion/peptide_diffusion \
        --epochs 500 \
        --batch_size 64 \
        --lr 3e-4 \
        --diffusion_steps 100 \
        --noise_schedule cosine" \
    "uv run ${SCRIPT_DIR}/diffusion/generate_diffusion.py \
        --model_path /Data/protein/outputs/diffusion/peptide_diffusion/best.pt \
        --length_tag '$LEN_TAG' \
        --charge_tag '$CHARGE_TAG' \
        --hydro_tag '$HYDRO_TAG' \
        --n $N_GEN \
        --mode ancestral \
        --temperature 1.2"


# ═══════════════════════════════════════════════════════════════════════
#  5. MASKGIT
# ═══════════════════════════════════════════════════════════════════════

#run_model "MaskGIT" \
#    "uv run ${SCRIPT_DIR}/maskgit/train_maskgit.py \
#       --train_csv $TRAIN_CSV \
#        --val_csv $VAL_CSV \
#        --output_dir /Data/protein/outputs/maskgit/peptide_maskgit \
#       --epochs 500 \
#       --batch_size 64 \
#        --lr 3e-4" \
#    "uv run ${SCRIPT_DIR}/maskgit/generate_maskgit.py \
#        --model_path /Data/protein/outputs/maskgit/peptide_maskgit/best.pt \
#        --length_tag '$LEN_TAG' \
#        --charge_tag '$CHARGE_TAG' \
#        --hydro_tag '$HYDRO_TAG' \
#        --n $N_GEN \
#        --mode iterative \
#        --temperature 1.2"


# ═══════════════════════════════════════════════════════════════════════
#  6. ENERGY-BASED MODEL
# ═══════════════════════════════════════════════════════════════════════

run_model "EBM" \
    "uv run ${SCRIPT_DIR}/energy/train_ebm.py \
        --train_csv $TRAIN_CSV \
        --val_csv $VAL_CSV \
        --output_dir /Data/protein/outputs/energy/peptide_ebm \
        --epochs 400 \
        --batch_size 64 \
        --lr 3e-4" \
    "uv run ${SCRIPT_DIR}/energy/generate_ebm.py \
        --model_path /Data/protein/outputs/energy/peptide_ebm/best.pt \
        --length_tag '$LEN_TAG' \
        --charge_tag '$CHARGE_TAG' \
        --hydro_tag '$HYDRO_TAG' \
        --n $N_GEN \
        --mode anneal \
        --n_steps 500"


# ═══════════════════════════════════════════════════════════════════════
#  7. JEPA (Joint Embedding Predictive Architecture)
# ═══════════════════════════════════════════════════════════════════════

#run_model "JEPA" \
#    "uv run ${SCRIPT_DIR}/jepa/train_jepa.py \
#        --train_csv $TRAIN_CSV \
#       --val_csv $VAL_CSV \
#        --output_dir /Data/protein/outputs/jepa/peptide_jepa \
#        --epochs 500 \
#        --batch_size 64 \
#        --lr 3e-4 \
#        --ema_decay 0.996 \
#        --jepa_weight 5.0 \
#        --token_weight 1.0 \
#        --vicreg_weight 1.0" \
#    "uv run ${SCRIPT_DIR}/jepa/generate_jepa.py \
#        --model_path /Data/protein/outputs/jepa/peptide_jepa/best.pt \
#        --length_tag '$LEN_TAG' \
#        --charge_tag '$CHARGE_TAG' \
#        --hydro_tag '$HYDRO_TAG' \
#        --n $N_GEN \
#        --mode refine \
#        --temperature 1.1"


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

log_header "SUMMARY"

echo -e "  ${BOLD}Model           Status          Time${NC}"
echo "  ─────────────────────────────────────────"

#MODELS_ORDER=("Transformer" "CVAE" "LSTM" "Diffusion" "MaskGIT" "EBM" "JEPA")
MODELS_ORDER=("CVAE" "Diffusion" "EBM")
PASS=0
FAIL=0

for model in "${MODELS_ORDER[@]}"; do
    status="${MODEL_STATUS[$model]:-NOT_RUN}"
    elapsed="${MODEL_TIME[$model]:-0}"
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))
    time_str="${mins}m ${secs}s"

    if [ "$status" = "OK" ]; then
        echo -e "  ${GREEN}✓${NC} ${model}$(printf '%*s' $((16 - ${#model})) '')${GREEN}OK${NC}$(printf '%*s' $((16 - 2)) '')${time_str}"
        ((PASS++))
    elif [ "$status" = "NOT_RUN" ]; then
        echo -e "  ${YELLOW}–${NC} ${model}$(printf '%*s' $((16 - ${#model})) '')${YELLOW}SKIPPED${NC}"
    else
        echo -e "  ${RED}✗${NC} ${model}$(printf '%*s' $((16 - ${#model})) '')${RED}${status}${NC}$(printf '%*s' $((16 - ${#status})) '')${time_str}"
        ((FAIL++))
    fi
done

echo ""
echo -e "  ${BOLD}Total: ${PASS} passed, ${FAIL} failed, ${TOTAL_MINS}m ${TOTAL_SECS}s${NC}"
echo ""

# write summary to file
{
    echo "Peptide Generation Model Training Summary"
    echo "========================================="
    echo "Date: $(date)"
    echo "Train CSV: $TRAIN_CSV"
    echo "Val CSV: $VAL_CSV"
    echo "Device: $GPU_INFO"
    echo ""
    for model in "${MODELS_ORDER[@]}"; do
        status="${MODEL_STATUS[$model]:-NOT_RUN}"
        elapsed="${MODEL_TIME[$model]:-0}"
        echo "$model: $status (${elapsed}s)"
    done
    echo ""
    echo "Total: ${PASS} passed, ${FAIL} failed, ${TOTAL_ELAPSED}s"
} > "$SUMMARY_FILE"

log_info "Summary saved to $SUMMARY_FILE"
log_info "Full logs in $LOG_DIR/"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
