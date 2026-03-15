#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Master training & generation script for all peptide generation models
#  Runs: Transformer → VAE → LSTM → Diffusion → MaskGIT → EBM
#  Handles errors gracefully — each model runs independently
# ═══════════════════════════════════════════════════════════════════════

set -o pipefail

# ── Configuration ───────────────────────────────────────────────────────
TRAIN_CSV="${TRAIN_CSV:-./data/generator/train_core.csv}"
VAL_CSV="${VAL_CSV:-./data/generator/val_core.csv}"
SCRIPT_DIR="${SCRIPT_DIR:-./generator/}"
LOG_DIR="${LOG_DIR:-./logs}"

# generation test tags
LEN_TAG=("<LEN_SHORT>" "<LEN_MED>" "<LEN_LONG>")
CHARGE_TAG=("<CHARGE_LOW>" "<CHARGE_MED>" "<CHARGE_HIGH>")
HYDRO_TAG=("<HYDRO_LOW>" "<HYDRO_MED>" "<HYDRO_HIGH>")
N_GEN=10

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
    local gen_cmd="$2"
    local log_file="$LOG_DIR/${name}.log"
    local safe_name
    safe_name=$(printf "%s" "$name" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')
    local csv_file="$LOG_DIR/generation_${safe_name}.csv"

    log_header "$name"

    local start=$(date +%s)

    # ── Generate ──
    log_info "Generating..."
    echo "=== GENERATION ===" > "$log_file"
    echo "Command: $gen_cmd" >> "$log_file"
    echo "" >> "$log_file"

    echo "sequence,length_tag,charge_tag,hydro_tag" > "$csv_file"

    local gen_output
    gen_output=$(eval "$gen_cmd" 2>&1)
    local gen_exit=$?

    echo "$gen_output" >> "$log_file"

    echo "$gen_output" | awk -v OFS=',' '
        /^\[GEN\][[:space:]]+/ {
            len_tag=$2
            charge_tag=$3
            hydro_tag=$4
            next
        }
        match($0, /^[[:space:]]*[0-9]+\. ([A-Za-z]+)/, m) {
            if (len_tag != "" && charge_tag != "" && hydro_tag != "") {
                print m[1], len_tag, charge_tag, hydro_tag
            }
        }
    ' >> "$csv_file"

    if [ $gen_exit -ne 0 ]; then
        log_fail "Generation FAILED (exit code $gen_exit)"
        log_fail "See $log_file for details"
        echo "  Last 5 lines of log:"
        tail -5 "$log_file" | sed 's/^/    /'
        log_info "Partial CSV: $csv_file"
        MODEL_STATUS[$name]="FAILED"
        local end=$(date +%s)
        MODEL_TIME[$name]=$(( end - start ))
        return 1
    fi

    log_success "Generation completed"
    log_info "Saved CSV: $csv_file"
    MODEL_STATUS[$name]="OK"

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
echo -e "${BOLD}Running 8 models...${NC}"

# ── Run models ─────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════
#  1. TRANSFORMER (from scratch)
# ═══════════════════════════════════════════════════════════════════════
run_model "Transformer" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/transformers/generate_transformer.py \
                    --model_path models/generator/transformers/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  2. CVAE
# ═══════════════════════════════════════════════════════════════════════

run_model "CVAE" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/vae/generate_vae.py \
                    --model_path models/generator/vae/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN \
                    --mode sample || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  3. LSTM
# ═══════════════════════════════════════════════════════════════════════

run_model "LSTM" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/lstm/generate_lstm.py \
                    --model_path models/generator/lstm/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  4. DISCRETE DIFFUSION
# ═══════════════════════════════════════════════════════════════════════

run_model "Diffusion" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/diffusion/generate_diffusion.py \
                    --model_path models/generator/diffusion/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN \
                    --mode ancestral \
                    --temperature 1.3 || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  5. MASKGIT
# ═══════════════════════════════════════════════════════════════════════

run_model "MaskGIT" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/maskgit/generate_maskgit.py \
                    --model_path models/generator/maskgit/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN \
                    --mode iterative \
                    --temperature 1.6 || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  6. ENERGY-BASED MODEL
# ═══════════════════════════════════════════════════════════════════════

run_model "EBM" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/energy/generate_ebm.py \
                    --model_path models/generator/energy/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN \
                    --mode anneal \
                    --n_steps 500 || exit \$?
            done
        done
    done"


# ═══════════════════════════════════════════════════════════════════════
#  7. JEPA (Joint Embedding Predictive Architecture)
# ═══════════════════════════════════════════════════════════════════════

run_model "JEPA" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/jepa/generate_jepa.py \
                    --model_path models/generator/jepa/best.pt \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN \
                    --mode refine \
                    --temperature 1.3 || exit \$?
            done
        done
    done"

# ═══════════════════════════════════════════════════════════════════════
#  8. GPT-2
# ═══════════════════════════════════════════════════════════════════════

run_model "GPT-2" \
    "for len_tag in \"\${LEN_TAG[@]}\"; do
        for charge_tag in \"\${CHARGE_TAG[@]}\"; do
            for hydro_tag in \"\${HYDRO_TAG[@]}\"; do
                echo \"[GEN] \$len_tag \$charge_tag \$hydro_tag\"
                uv run ${SCRIPT_DIR}/gpt2/generate.py \
                    --model_dir models/generator/gpt2/ \
                    --length_tag \"\$len_tag\" \
                    --charge_tag \"\$charge_tag\" \
                    --hydro_tag \"\$hydro_tag\" \
                    --n $N_GEN || exit \$?
            done
        done
    done"

# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

log_header "GENERATION SUMMARY"

echo -e "  ${BOLD}Model           Status          Time${NC}"
echo "  ─────────────────────────────────────────"

#MODELS_ORDER=("Transformer" "CVAE" "LSTM" "Diffusion" "MaskGIT" "EBM" "JEPA" "GPT-2")
MODELS_ORDER=("GPT-2")
PASS=0
FAIL=0
SKIP=0

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
        ((SKIP++))
    else
        echo -e "  ${RED}✗${NC} ${model}$(printf '%*s' $((16 - ${#model})) '')${RED}${status}${NC}$(printf '%*s' $((16 - ${#status})) '')${time_str}"
        ((FAIL++))
    fi
done

echo ""
echo -e "  ${BOLD}Total: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped, ${TOTAL_MINS}m ${TOTAL_SECS}s${NC}"
echo ""

# write summary to file
{
    echo "Peptide Generation Summary"
    echo "=========================="
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
    echo "Total: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped, ${TOTAL_ELAPSED}s"
} > "$SUMMARY_FILE"

log_info "Summary saved to $SUMMARY_FILE"
log_info "Full logs in $LOG_DIR/"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0