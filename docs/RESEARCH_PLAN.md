# Research plan and handoff

Status: Phase-1 remote kit implemented; full-data GPU validation and causal-analysis
integration remain open. Last consolidated: 2026-08-18.

## 1. Working title and thesis

Working title:

> Remembering the Study, Forgetting the Science: Scientific Forgetting and Nuisance
> Consolidation in Continual Brain Encoding

The project separates two failure modes that standard continual-learning evaluation
usually conflates:

1. **Standard encoding forgetting**: performance on an earlier study's held-out fMRI
   time series declines.
2. **Scientific-Probe Forgetting (SPF)**: a fixed, independent neuroscience probe no
   longer supports the correct scientific conclusion, even when encoding performance
   remains acceptable.

Scientific fidelity includes contrast maps, ROI selectivity, effect size,
lateralization, localization, and representational geometry. A model can preserve
Pearson correlation while corrupting one of these conclusions, or lose some encoding
score while preserving the conclusion.

The second thesis is narrower than “continual learning learns nuisance”:

> A stability or knowledge-protection mechanism may preserve a predictable acquisition
> or preprocessing artifact and treat it as transferable brain knowledge.

The project calls this **nuisance consolidation** only when all three links are shown:

1. **Acquisition**: after controlling stimulus, subject, and modality, nuisance enters
   the shared representation or mapping.
2. **Persistence/protection**: it remains after subsequent learning or is preferentially
   protected by replay/EWC-like mechanisms.
3. **Causal use**: nuisance swap, removal, or targeted subspace ablation changes the
   prediction, and removing nuisance restores scientific fidelity.

Study-ID decoding, recency bias, or ordinary old-task score decline is insufficient.

## 2. TRIBE v2 experimental boundary

The Phase-1 model uses the official TRIBE v2 stack with a small trainable configuration:

- freeze the general-purpose V-JEPA encoder;
- use video features only;
- randomly initialize modality projection, temporal/shared trunk, shared/group mapping,
  subject-specific heads, and group/unseen-subject head;
- forbid the public joint TRIBE checkpoint as main-experiment initialization because it
  has already seen all four source studies;
- evaluate both subject-ID-known and no-study-ID/group modes;
- keep cortex only and reduce the target to 360 Glasser parcels.

The public joint checkpoint may later be used only as an offline oracle/reference.
Parameter-isolated frozen heads can be reported as a control, but “zero forgetting” from
head isolation is not a continual-learning contribution.

The implementation is configured in:

- `configs/model_tribe_lite.yaml`;
- `configs/training/phase1_causal12.yaml`;
- `src/spfcl/train/runner.py`;
- `src/spfcl/train/tribe_experiment.py`.

## 3. Phase-1 data and pseudo-studies

Phase 1 uses only BOLD Moments/OpenNeuro `ds005165` snapshot `1.0.4`:

- subjects 1-4;
- training sessions 2 and 3;
- session-1 visual localizer;
- MNI continuous fMRI, projected to fsaverage5 and then Glasser-360;
- official repeated test stimuli remain clean post-stage evaluation data.

Four subjects x two sessions are 8/40 = 20% of the full 10-subject x four-session
training acquisition. The default selective download is approximately 29.225 GiB;
the session 2-5 repeated-test option is approximately 35.327 GiB.

Current executable pseudo-study assignment:

- **Study A**: session 2, clean/canonical.
- **Study B0**: session 3, clean paired control.
- **Study B_lambda**: the identical session-3 windows with injected nuisance.

Session 2 and session 3 each contain 750 training stimuli, share 500, and jointly cover
1,000. This is a practical low-compute proxy, not completed V-JEPA semantic matching.
Before claiming matched A/B distributions, add a frozen-embedding balance report with
strata, MMD or an equivalent preregistered measure. B0 and B_lambda are already exact
paired variants and are the valid nuisance causal control.

## 4. First nuisance: identifiable low-frequency drift

The first round injects one nuisance only: deterministic low-frequency temporal drift.
Multiple nuisance families must not be mixed until the mechanism is understood.

Current positive-control design:

- 100-second, non-overlapping model windows;
- 1 Hz fMRI output;
- one or two cycles per window, corresponding to 0.01/0.02 Hz;
- zero linear trend and normalized RMS before amplitude scaling;
- +5-second phase alignment with the official fMRI extractor offset;
- three balanced in-training doses: `0.10`, `0.23`, and `0.36`;
- deterministic sample/event keys and fixed nuisance seed;
- nuisance only in session-3 training targets, never in clean test/localizer data.

The position/dose schedule makes the positive control learnable from run phase/window
position. A drift that is unobservable from the model input would be ignored by the MSE
optimum and could not support a meaningful “failed consolidation” conclusion.

Later nuisance families should be introduced one at a time:

- HRF delay;
- temporal resampling/TR shift;
- spatial smoothing or template interpolation;
- motion/SNR, coverage mask, or scanner/preprocessing signatures.

## 5. Minimal experiment matrix

Preregistered seeds are `17`, `29`, and `43`.

Core nine Slurm jobs:

1. A -> B naive sequential fine-tuning, three seeds;
2. B -> A naive sequential fine-tuning, three seeds;
3. A union B offline joint training, three seeds.

Second-priority block:

4. A -> B with deterministic 1% old-segment replay, three seeds.

Each job may contain paired B0/B_lambda branches and therefore multiple training stages.
The executable stage DAG lives in `matrix.conditions.*.stages` inside
`configs/training/phase1_causal12.yaml`; do not replace it with a hard-coded runner plan.

Every continual boundary is a weights-only fork with a fresh optimizer, OneCycle
scheduler, epoch counter, and global step. Sibling B0/B_lambda branches must have the
same initial-state hash. Training uses 15 fixed epochs per stage and final-epoch weights.

## 6. Evaluation at every completed stage

### 6.1 Standard encoding

On the untouched clean repeated-test task, compute:

- parcel-wise Pearson correlation across time;
- Fisher-z aggregation with subjects weighted equally;
- MSE and R-squared so scale errors are not hidden by Pearson correlation;
- both known-subject and group-head predictions.

Conceptual forgetting for study `i`:

`F_enc(i) = max_t E(t, i) - E(T, i)`

The stage runner writes `clean_test_predictions.npz` and a summary into
`spfcl_stage_audit.json`.

### 6.2 Scientific probes

The primary Phase-1 probe is the same four subjects' BOLD Moments silent-video
localizer, not IBC auditory/language maps. The scan presents silent video and contains
faces, bodies, scenes, objects, and scrambled conditions.

Primary contrasts:

- faces > bodies;
- scenes > objects.

Planned fidelity measures include:

- empirical-predicted contrast-map agreement;
- preregistered ROI signed effect and effect-size error;
- lateralization error;
- localization overlap or centroid error;
- RDM/crossnobis geometry;
- counterfactual specificity.

The current automatic stage export covers condition-aggregated predictions and both
named contrast summaries for both heads. ROI, lateralization, localization, and geometry
require preregistered masks/estimators before being treated as confirmatory.

Conceptual SPF for query `q`:

`F_SPF(q) = max_t S(t, q) - S(T, q)`

Do not claim separation merely because the correlation between encoding forgetting and
SPF is non-significant. Stronger evidence is one of:

- encoding change lies inside a preregistered equivalence interval while SPF declines;
- method ranking reverses between encoding preservation and scientific fidelity;
- nuisance removal or targeted ablation restores scientific fidelity.

### 6.3 Nuisance mechanism

Probe nuisance separately in:

- shared transformer/trunk;
- subject-specific mapping/head;
- group/unseen-subject head;
- gradients or functional output subspaces.

Offline NumPy APIs already exist for grouped conditional nuisance probes, nuisance
projection strength, paired nuisance swap, orthogonal subspace ablation, and
dose-response slopes in `src/spfcl/eval/causal.py`.

Still required for a complete result pipeline:

- export the correct representation taps and nuisance ground truth from each stage;
- execute swap/removal/ablation interventions against stage models;
- aggregate subject-equal results across directions and seeds;
- create one auditable F_enc/F_SPF/nuisance Go-No-Go report.

## 7. Go/No-Go decision

Continue to real-study validation only when all three are supported:

1. encoding/SPF separation is reproducible across directions and seeds;
2. nuisance dose has a reproducible monotonic relationship with scientific degradation;
3. nuisance swap/removal or targeted ablation causally changes predictions and restores
   scientific fidelity.

No-Go or redesign if the result is limited to study-ID decoding, group-head recency bias,
or ordinary naive-fine-tuning score decline.

With four subjects, the inferential unit remains subject. Three seeds do not create 12
independent observations. The smallest exact two-sided sign-flip p-value at n=4 is
0.125, so Phase 1 supports mechanism screening and effect sizes, not a definitive
equivalence test. A confirmatory phase must add subjects.

## 8. Phase 2 and possible method contribution

Only after the Phase-1 Go decision, run a small real-shift validation:

- Wen2017, approximately 10 hours;
- BOLD Moments, approximately 10 hours;
- video-only on both sides;
- Wen -> BOLD, BOLD -> Wen, joint, naive, and 1% replay.

The purpose is to test whether the semi-synthetic single-study mechanism transfers to a
real study shift, not to reopen broad mechanism exploration.

A later method contribution could be **Scientific-Query Replay**:

- replay a small set of canonical stimuli;
- constrain contrast maps or representational geometry;
- isolate study-specific artifact in a separate adapter;
- evaluate on independent held-out scientific probes.

A particularly important comparison is whether replay/EWC-like protection preserves
nuisance more strongly than it preserves the scientific conclusion.

## 9. Artifact and reproducibility contract

Every successful stage must publish atomically:

1. `stage.weights.pt` — model tensors and provenance, no trainer state;
2. `clean_test_predictions.npz` — empirical, known-head, and group-head arrays;
3. `localizer_predictions.npz` — five conditions and both heads;
4. `spfcl_stage_audit.json` — hashes, lineage, and summaries.

Stage reuse must verify the experiment/stage fingerprint, parent hash, final tensor hash,
and prediction shapes. Fingerprints cover training/model/nuisance configs, source tree,
dependency constraints, OpenNeuro inventory/provenance, atlas operator/basis, V-JEPA
provenance, and frozen manifest.

Data, videos, model weights, credentials, caches, and outputs must stay outside Git.
Stimulus access is non-commercial/research-only and forbids redistribution. TRIBE v2 is
CC BY-NC 4.0. The original adapter code currently has no separate publication license.

## 10. Server handoff checklist

From the repository root:

1. Read `README.md`, copy `.env.example` to `.env`, and create all configured storage
   roots.
2. Run `bash scripts/bootstrap_remote.sh`, activate `.venv`, then run `spfcl doctor`.
3. Download and verify the pinned V-JEPA snapshot on an internet-enabled login node.
4. Dry-run and then download the immutable BOLD Moments subset.
5. Accept stimulus and HCP terms, install videos, and prepare atlas assets.
6. Validate data, create the frozen manifest, and audit pseudo-studies.
7. Run tests, nuisance preview, model smoke, and one training dry-run.
8. Submit only `--single-task 0` first. Inspect all four stage artifacts.
9. Submit the core array only after the cold-cache GPU smoke succeeds.

Useful commands and exact arguments are maintained in `README.md`. When implementation
and this document disagree, do not silently choose one: inspect Git history and update
the documentation and executable configuration together.

## 11. Related work supplied with the project brief

- TRIBE v2: <https://arxiv.org/html/2605.04326>
- Continual representation forgetting: <https://arxiv.org/abs/2203.13381>
- Continual learning with spurious correlations:
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/a46adbe2f0ca0e16ef8857e188991ad7-Abstract-Conference.html>
- Continual confounding: <https://proceedings.mlr.press/v267/busch25a.html>
- Counterfactual validation of brain representations:
  <https://arxiv.org/abs/2605.23895>
