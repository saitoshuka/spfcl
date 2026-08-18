# SPFCL repository instructions

## Start here

- This repository is a remote-first Phase-1 research kit for studying scientific-probe
  forgetting and nuisance consolidation with TRIBE v2.
- Before changing scientific assumptions, data views, training stages, probes, or
  statistics, read `docs/RESEARCH_PLAN.md` completely.
- Before downloading data or launching jobs, read `README.md` completely.
- Before redistributing code, models, stimuli, or atlas assets, read
  `THIRD_PARTY_NOTICES.md` and `vendor/UPSTREAM.md`.

## Non-negotiable scientific constraints

- Keep the main experiment video-only. Audio and text projectors are not trained and
  their IBC auditory/language maps are not valid Phase-1 fidelity ground truth.
- Use the same-subject BOLD Moments silent-video localizer as the primary scientific
  probe. The primary contrasts are faces > bodies and scenes > objects.
- Freeze V-JEPA and randomly initialize the trainable TRIBE layers. Never initialize
  the main experiment from the public joint TRIBE checkpoint.
- Evaluate both the known-subject head and the group/unseen-subject head.
- Keep B0 and B_lambda branches paired: identical parent weights, stimulus windows,
  ordering, subjects, sessions, and random initialization. Only nuisance injection may
  differ.
- Never reintroduce run-number validation. BOLD Moments randomizes training stimuli
  across runs by subject, which leaks stimuli across subjects. Training uses a fixed
  epoch budget and final-epoch weights; the official repeated test task is post-stage
  `clean_test` only.
- Continual stage boundaries are weights-only forks. Do not use Lightning full resume,
  because it restores optimizer, scheduler, epoch, and global-step state.
- Treat 4 subjects x 3 seeds as subject n=4, not n=12. Phase 1 is mechanism screening
  and effect-size estimation; it cannot support a definitive equivalence claim.
- Do not claim nuisance consolidation from study-ID decoding alone. The required chain
  is acquisition, persistence/protection, and causal use.

## Repository boundaries

- `vendor/tribev2` is the pinned, unmodified upstream snapshot. Do not edit it. Put all
  adaptations in `src/spfcl`.
- Keep data, stimulus videos, model weights, checkpoints, caches, generated atlas files,
  manifests with local paths, outputs, `.env`, and credentials out of Git.
- The adapter code currently has no separately assigned redistribution license. Do not
  make the repository public or assign a license without the owner's decision.
- Prefer immutable OpenNeuro snapshot `ds005165` version `1.0.4`. The `s3-latest`
  backend is inspection-only and must not be used for production Phase-1 runs.
- Use the pinned V-JEPA revision and file hashes in
  `configs/training/phase1_causal12.yaml`; compute jobs should not contact the Hub.

## Implementation expectations

- Preserve Glasser parcel order: left 180 followed by right 180, for 360 outputs.
- Preserve 100-second non-overlapping model windows, 2 Hz stimulus features, 1 Hz fMRI
  targets, and the +5-second nuisance/extractor phase alignment.
- Keep nuisance doses `[0.10, 0.23, 0.36]` deterministic and balanced within the
  B_lambda training view.
- Any completed stage must atomically provide all four artifacts:
  `stage.weights.pt`, `clean_test_predictions.npz`, `localizer_predictions.npz`, and
  `spfcl_stage_audit.json`.
- Reuse cached stages only after provenance, tensor hash, configuration fingerprint,
  and prediction archive validation succeeds.
- Shared fMRI cache identities must include source, atlas-operator, and basis hashes.

## Validation

After changing Python, configuration, or runner logic, run:

```bash
pytest -q
ruff check src tests
bash -n scripts/bootstrap_remote.sh \
  scripts/launch_phase1_slurm.sh \
  scripts/slurm_phase1_worker.sh
```

For model/runtime changes, also run:

```bash
spfcl model-smoke --config configs/model_tribe_lite.yaml
spfcl train \
  --config configs/training/phase1_causal12.yaml \
  --condition a_to_b_naive --seed 17 --dry-run
```

Do not launch the full Slurm array as a first runtime test. Use
`bash scripts/launch_phase1_slurm.sh --single-task 0` and inspect its artifacts before
submitting the core matrix.

## Known incomplete research work

- A/B semantic balance using V-JEPA embeddings is not yet automated. Session 2 and
  session 3 are the current pseudo-study proxy; B0/B_lambda pairing is exact, but A/B
  semantic matching must not be claimed without a balance report.
- Causal array APIs exist, but automatic representation-tap export, nuisance swap,
  targeted ablation, and cross-stage/seed Go-No-Go aggregation are not fully wired.
- Real BOLD Moments data, the complete 4.14-GB V-JEPA snapshot, and a CUDA end-to-end
  run remain server-side validation steps.
