#!/usr/bin/env bash
#SBATCH --job-name=spfcl-phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# A task contains 2--4 sequential stages. Incomplete stages restart from their
# weights-only parent, while completed stages are hash-verified and reused.
#SBATCH --time=48:00:00

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array job}"
: "${SPFCL_REPO_ROOT:?launch_phase1_slurm.sh did not export SPFCL_REPO_ROOT}"
: "${SPFCL_CONFIG:?launch_phase1_slurm.sh did not export SPFCL_CONFIG}"
: "${DATAPATH:?DATAPATH is required}"
: "${SPFCL_CACHE:?SPFCL_CACHE is required}"
: "${SPFCL_OUTPUT:?SPFCL_OUTPUT is required}"

if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || ((SLURM_ARRAY_TASK_ID > 11)); then
  echo "SLURM_ARRAY_TASK_ID must be in 0..11; got $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

conditions=(a_to_b_naive b_to_a_naive offline_joint a_to_b_replay_1pct)
seeds=(17 29 43)
condition_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
condition="${conditions[$condition_index]}"
seed="${seeds[$seed_index]}"

venv_path="${SPFCL_VENV:-${SPFCL_REPO_ROOT}/.venv}"
[[ -x "${venv_path}/bin/python" ]] || {
  echo "Remote environment missing at $venv_path; run scripts/bootstrap_remote.sh." >&2
  exit 1
}
[[ -r "$SPFCL_CONFIG" ]] || { echo "Cannot read config: $SPFCL_CONFIG" >&2; exit 1; }

# shellcheck disable=SC1091
source "${venv_path}/bin/activate"
command -v spfcl >/dev/null 2>&1 || {
  echo "spfcl console entry point is missing from $venv_path" >&2
  exit 1
}

export SAVEPATH="${SAVEPATH:-$SPFCL_OUTPUT}"
export HF_HOME="${HF_HOME:-${SPFCL_CACHE}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${SPFCL_CACHE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SPFCL_CACHE}/xdg}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
fi
mkdir -p "$SPFCL_CACHE" "$SPFCL_OUTPUT"

if [[ "${SPFCL_TRAIN_DRY_RUN:-0}" != "1" ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available inside the allocation. Check the GPU resource request, "
        "driver-compatible torch wheel, and cluster module setup."
    )
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
fi

train_command=(
  spfcl train
  --config "$SPFCL_CONFIG"
  --condition "$condition"
  --seed "$seed"
)
if [[ "${SPFCL_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  train_command+=(--dry-run)
fi

echo "Starting condition=$condition seed=$seed task_id=$SLURM_ARRAY_TASK_ID"
srun --unbuffered "${train_command[@]}"
