#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/launch_phase1_slurm.sh [options]

Submit the 9-run core Phase-1 matrix as a Slurm array. Add --include-replay
to submit all 12 runs. DATAPATH, SPFCL_CACHE, and SPFCL_OUTPUT must be exported.

Options:
  --config PATH       Training YAML (default: configs/training/phase1_causal12.yaml)
  --include-replay    Submit task IDs 0-11 instead of the core 0-8
  --single-task ID    Submit one task ID (0-11), useful for the first GPU trial
  --partition NAME    Override $SLURM_PARTITION
  --constraint NAME   Override $SLURM_CONSTRAINT
  --account NAME      Slurm account
  --dry-run           Print the quoted sbatch command without submitting
  -h, --help          Show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
config="${repo_root}/configs/training/phase1_causal12.yaml"
# Limit concurrent jobs so the shared immutable V-JEPA cache is populated once
# without a nine-process stampede on the same files.
array_range="0-8%3"
partition="${SLURM_PARTITION:-}"
constraint="${SLURM_CONSTRAINT:-}"
account=""
dry_run=0
single_task=""

while (($#)); do
  case "$1" in
    --config)
      [[ -n "${2:-}" ]] || { echo "--config requires a path" >&2; exit 2; }
      config="$2"
      shift 2
      ;;
    --include-replay)
      array_range="0-11%3"
      shift
      ;;
    --single-task)
      [[ "${2:-}" =~ ^[0-9]+$ ]] && ((10#${2} <= 11)) || {
        echo "--single-task requires an integer in 0..11" >&2
        exit 2
      }
      single_task="$2"
      shift 2
      ;;
    --partition)
      [[ -n "${2:-}" ]] || { echo "--partition requires a value" >&2; exit 2; }
      partition="$2"
      shift 2
      ;;
    --constraint)
      [[ -n "${2:-}" ]] || { echo "--constraint requires a value" >&2; exit 2; }
      constraint="$2"
      shift 2
      ;;
    --account)
      [[ -n "${2:-}" ]] || { echo "--account requires a value" >&2; exit 2; }
      account="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -z "$single_task" ]] || array_range="$single_task"

[[ -f "$config" ]] || { echo "Training config not found: $config" >&2; exit 1; }
: "${DATAPATH:?Export DATAPATH before submitting}"
: "${SPFCL_CACHE:?Export SPFCL_CACHE before submitting}"
: "${SPFCL_OUTPUT:?Export SPFCL_OUTPUT before submitting}"

worker="${script_dir}/slurm_phase1_worker.sh"
[[ -f "$worker" ]] || { echo "Slurm worker not found: $worker" >&2; exit 1; }
if ((dry_run == 0)); then
  command -v sbatch >/dev/null 2>&1 || {
    echo "sbatch is not available; run this from a Slurm login node." >&2
    exit 1
  }
fi

mkdir -p "${SPFCL_OUTPUT}/slurm"
export SPFCL_REPO_ROOT="$repo_root"
export SPFCL_CONFIG="$(cd -- "$(dirname -- "$config")" && pwd -P)/$(basename -- "$config")"
export SAVEPATH="${SAVEPATH:-$SPFCL_OUTPUT}"
# Training reads installed videos and never needs the archive password. Do not
# copy that secret into Slurm's exported job environment if it is still set.
unset BMD_STIMULI_PASSWORD

submit=(
  sbatch
  "--array=${array_range}"
  "--output=${SPFCL_OUTPUT}/slurm/%x-%A_%a.out"
  "--error=${SPFCL_OUTPUT}/slurm/%x-%A_%a.err"
  --export=ALL
)
[[ -z "$partition" ]] || submit+=(--partition "$partition")
[[ -z "$constraint" ]] || submit+=(--constraint "$constraint")
[[ -z "$account" ]] || submit+=(--account "$account")
submit+=("$worker")

if ((dry_run)); then
  printf 'Environment: SPFCL_REPO_ROOT=%q SPFCL_CONFIG=%q\n' "$SPFCL_REPO_ROOT" "$SPFCL_CONFIG"
  printf 'Command:'
  printf ' %q' "${submit[@]}"
  printf '\n'
else
  "${submit[@]}"
fi
