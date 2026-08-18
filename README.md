# SPF-CL / TRIBE-lite remote experiment kit

This project contains a pinned, unmodified copy of official TRIBE v2 plus a
remote-first implementation of the four-subject BOLD Moments mechanism pilot.
The laptop never needs to hold fMRI, videos, V-JEPA weights, or checkpoints.

Included:

- official TRIBE v2 commit `af58661791a351a448a489042a28f6c37e1c14b7` under
  `vendor/tribev2`;
- immutable OpenNeuro `ds005165` snapshot `1.0.4` inventory/download with hash checks;
- selective MNI continuous-BOLD download compatible with the official loader;
- separately licensed stimulus download and video-only extraction;
- offline Glasser-360 operator and nuisance spatial basis preparation;
- deterministic B0/Bλ low-frequency nuisance with three preregistered doses;
- video-only TRIBE-lite, known-subject/group-head hooks, and weights-only stage forks;
- core 9-job and optional 12-job Slurm arrays;
- offline tests that do not download models or data.

## 1. Sync and install on the server

```bash
rsync -az \
  --exclude '/.venv/' --exclude '/data/' --exclude '/cache/' \
  --exclude '/outputs/' --exclude '/.env' --exclude '/.pytest_cache/' \
  --exclude '/.ruff_cache/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '*.egg-info/' \
  ./ user@server:/work/spfcl/

ssh user@server
cd /work/spfcl
cp .env.example .env
# Edit the storage paths and CUDA wheel channel in .env.
set -a
source .env
set +a
mkdir -p "$DATAPATH" "$SAVEPATH" "$SPFCL_CACHE" "$SPFCL_OUTPUT" "$HF_HOME"

bash scripts/bootstrap_remote.sh
source .venv/bin/activate
spfcl doctor

# Run this on a login/transfer node with internet access. It downloads only
# config, processor, and 4.14-GB safetensors—not the 16.5-GB original checkpoint.
spfcl download-vjepa \
  --config configs/training/phase1_causal12.yaml --dry-run
spfcl download-vjepa \
  --config configs/training/phase1_causal12.yaml
```

The bootstrap uses Python 3.11, installs the pinned official source from
`vendor/tribev2`, constrains Torch/Transformers/neuralset versions, and records a
complete `pip freeze`. It does not download the dataset.
The V-JEPA command resolves the local cache only from Hub commit
`875c192b7b704b87d1e1d99345769632dd5f739a` and verifies the weight SHA-256
`f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`.
Compute jobs never need Hub network access.

## 2. Download the low-compute BOLD Moments subset

First inspect the exact immutable snapshot plan:

```bash
spfcl download-bmd \
  --data-root "$DATAPATH" \
  --subjects 1 2 3 4 \
  --train-sessions 2 3 \
  --test-sessions 2 3 \
  --localizer-sessions 1 \
  --space mni \
  --snapshot 1.0.4 \
  --dry-run
```

Then remove `--dry-run`. This selection contains 124 runs / 508 objects and is
about 29.225 GiB: sessions 2–3 train and repeated test plus the session-1 visual
localizer. It fetches only MNI preprocessed BOLD, brain masks, confounds, events,
and required metadata—never raw anatomy, other subjects, or GLM betas.

The default backend obtains versioned S3 URLs from the OpenNeuro GraphQL snapshot,
stores a frozen selection-aware inventory, verifies annex SHA-256 or Git-blob SHA-1,
writes through `.part`, and resumes safely. `--backend s3-latest` is inspection-only;
the Phase-1 preflight rejects it because it cannot identify an immutable experiment.

For the complete session 2–5 repeated test set without downloading sessions 4–5
training runs:

```bash
spfcl download-bmd \
  --data-root "$DATAPATH" \
  --subjects 1 2 3 4 \
  --train-sessions 2 3 \
  --test-sessions 2 3 4 5 \
  --localizer-sessions 1 \
  --space mni --snapshot 1.0.4
```

That option is about 35.327 GiB. Reserve 65–70 GiB for data, videos, atlas,
V-JEPA cache, and checkpoints.

## 3. Download and install stimulus videos on the server

The videos are not in OpenNeuro and have separate non-commercial research terms.
Read the [official stimulus terms](https://boldmomentsdataset.csail.mit.edu/stimuli_metadata/stimuli_access.txt),
then explicitly accept them:

```bash
spfcl download-stimuli \
  --data-root "$DATAPATH" \
  --accept-stimulus-terms

read -s BMD_STIMULI_PASSWORD
export BMD_STIMULI_PASSWORD
spfcl install-stimuli \
  --data-root "$DATAPATH" \
  --archive "$DATAPATH/Lahner2024Bold/stimuli/stimulus_set.zip" \
  --accept-stimulus-terms
unset BMD_STIMULI_PASSWORD
```

The installer extracts only `train`, `test`, and `localizer` videos. It skips the
roughly 94k frame images, never prints the password, and refuses installation unless
the terms flag is present. Do not commit, redistribute, publicly upload, or put these
videos in a container image.

## 4. Build atlas assets, validate, and freeze the manifest

Read the [HCP-MMP atlas terms](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-data-use-terms),
then compile the two hemisphere annotations into a
360×20,484 offline operator. Training never calls the large MNE sample downloader.

```bash
spfcl prepare-atlas \
  --output-dir "$SPFCL_CACHE/atlas" \
  --accept-hcp-license

spfcl validate-data \
  --data-root "$DATAPATH" \
  --subjects 1 2 3 4 \
  --space mni --deep

mkdir -p "$SPFCL_OUTPUT/manifests"
spfcl make-manifest \
  --data-root "$DATAPATH" \
  --output "$SPFCL_OUTPUT/manifests/bmd4_manifest.jsonl" \
  --subjects 1 2 3 4 \
  --sessions 1 2 3 \
  --space mni

spfcl audit-pseudostudies \
  --data-root "$DATAPATH" \
  --subjects 1 2 3 4 --sessions 1 2 3 --space mni
```

The released events contain 750 training stimuli per session; sessions 2 and 3
share 500 and together cover all 1,000. B0 and Bλ are exact paired transformations
of the same session-3 windows. The audit checks cross-subject consistency and that
train stimuli do not enter the fixed test/localizer evaluation roles.

## 5. Smoke tests and launch

```bash
pytest -q
spfcl nuisance-preview \
  --config configs/nuisance_drift.yaml \
  --output "$SPFCL_OUTPUT/nuisance_preview.npz"
spfcl model-smoke --config configs/model_tribe_lite.yaml

spfcl train \
  --config configs/training/phase1_causal12.yaml \
  --condition a_to_b_naive --seed 17 --dry-run

bash scripts/launch_phase1_slurm.sh --dry-run
# Recommended first allocation: run only A→B, seed 17, and inspect its log/artifacts.
bash scripts/launch_phase1_slurm.sh --single-task 0
# After that task succeeds, submit the full core matrix.
bash scripts/launch_phase1_slurm.sh
# Optional second-priority 1% replay block:
# bash scripts/launch_phase1_slurm.sh --include-replay
```

The core array is three conditions × three seeds: A→B naive, B→A naive, and
offline joint. Each job records paired B0/Bλ branches; every non-overlapping
100-second Bλ model window, aligned to the official +5-second fMRI offset,
contains balanced, position-identifiable dose strata
`[0.10, 0.23, 0.36]`. The optional fourth
condition adds deterministic 1% old-segment replay.

Every continual boundary exports `spfcl.weights`: model tensors and provenance only.
The next stage creates a fresh optimizer, scheduler, epoch, and global step. Lightning
full resume is disabled, and sibling B0/Bλ forks must share the same starting hash.
Training uses the fixed 15-epoch budget and final-epoch tensors—there is deliberately
no internal validation or checkpoint selection. BOLD Moments randomizes training
stimuli into different run numbers for each subject, so a run-number validation split
would leak stimuli. The untouched official repeated-test task remains `clean_test`.

Four subjects × sessions 2–3 are exactly 8/40 = 20% of the full 10-subject ×
4-session training acquisition, matching the low-compute pilot boundary. V-JEPA,
Glasser projections, and localizer features are persistent caches on the server. The
Slurm launcher limits the array to three concurrent jobs and gives every multi-stage
task 48 hours.

After every successfully completed stage the runner atomically publishes:

- `stage.weights.pt`, with tensor/provenance hashes and no trainer state;
- `clean_test_predictions.npz`, for known-subject and group heads;
- `localizer_predictions.npz`, for all five localizer conditions and both heads;
- `spfcl_stage_audit.json`, with clean-test and primary-contrast summaries.

A stage is reusable only when all four artifacts validate. Offline analysis APIs also
cover grouped conditional nuisance probes, projection strength, paired swaps,
orthogonal subspace ablation, and dose-response slopes. ROI/lateralization analyses
require separately preregistered masks.

## Scientific boundary

- Main training is video-only and starts the trainable TRIBE layers randomly; the
  public joint checkpoint is forbidden as initialization.
- The primary Phase-1 scientific probe is the same-subject BOLD Moments silent-video
  localizer (faces/bodies/scenes/objects/scrambled), not IBC auditory/language maps.
- Audio/text projectors are untrained in this setup, so auditory/language fidelity is
  not a valid Phase-1 ground truth.
- The injected signal is an input-identifiable, +5-second-offset-aligned positive
  control, not a claim about arbitrary scanner drift.
- Four subjects × three seeds is still `n=4`, not `n=12`; the smallest exact
  two-sided subject-level sign-flip p-value is 0.125. This phase supports mechanism
  screening and effect sizes, not a definitive equivalence claim.

TRIBE v2 is CC BY-NC 4.0. See [vendor/UPSTREAM.md](vendor/UPSTREAM.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse.
