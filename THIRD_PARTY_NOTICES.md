# Third-party notices

## TRIBE v2

The complete, unmodified upstream source snapshot is stored in `vendor/tribev2`.
It is Copyright Meta Platforms, Inc. and affiliates and licensed CC BY-NC 4.0.
The exact repository, commit, archive hash, and license location are recorded in
`vendor/UPSTREAM.md`.

## BOLD Moments

The fMRI dataset is OpenNeuro `ds005165`, snapshot `1.0.4` (CC0). Stimulus videos
are not included. They have separate non-commercial/research access terms at the
official BOLD Moments portal and may not be redistributed or uploaded publicly.

## V-JEPA2 feature encoder

The server fetches `facebook/vjepa2-vitg-fpc64-256` from the Hugging Face Hub at
commit `875c192b7b704b87d1e1d99345769632dd5f739a`. Its model card declares the
Apache-2.0 license. The runner verifies the safetensors SHA-256 and does not vendor
the 4.14-GB weights in this repository.

## HCP-MMP1.0 atlas

The Glasser/HCP-MMP1.0 annotations are downloaded only after explicit acceptance
and compiled into local analysis assets. Use and publication remain subject to the
[Human Connectome Project data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-data-use-terms)
and applicable citation requirements. Atlas files are not vendored here.

## Adapter code

No separate license has been assigned to the original `src/spfcl` adapter code in
this workspace. Do not assume redistribution rights beyond the third-party terms
above; choose or obtain an explicit license before publishing the repository.
