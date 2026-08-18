#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_remote.sh [options]

Create an isolated Python environment and install the vendored TRIBE v2 plus
SPF-CL. This script never downloads the fMRI dataset or restricted stimuli.

Options:
  --venv PATH             Environment path (default: <repo>/.venv)
  --python COMMAND        Python >=3.11 executable (default: python3.11)
  --torch-index-url URL   PyTorch wheel index (default: $TORCH_INDEX_URL or cu124)
  --cpu                   Use the official CPU PyTorch wheel index
  --skip-torch            Keep a site/module-provided torch installation
  -h, --help              Show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
venv_path="${SPFCL_VENV:-${repo_root}/.venv}"
python_command="${SPFCL_PYTHON:-python3.11}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
skip_torch=0

while (($#)); do
  case "$1" in
    --venv)
      [[ -n "${2:-}" ]] || { echo "--venv requires a path" >&2; exit 2; }
      venv_path="$2"
      shift 2
      ;;
    --python)
      [[ -n "${2:-}" ]] || { echo "--python requires a command" >&2; exit 2; }
      python_command="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ -n "${2:-}" ]] || { echo "--torch-index-url requires a URL" >&2; exit 2; }
      torch_index_url="$2"
      shift 2
      ;;
    --cpu)
      torch_index_url="https://download.pytorch.org/whl/cpu"
      shift
      ;;
    --skip-torch)
      skip_torch=1
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

command -v "$python_command" >/dev/null 2>&1 || {
  echo "Python executable not found: $python_command" >&2
  echo "Load a Python 3.11 module or pass --python /absolute/path/to/python." >&2
  exit 1
}

"$python_command" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >=3.11 is required; found {sys.version.split()[0]}")
print(f"Bootstrap interpreter: Python {sys.version.split()[0]}")
PY

if [[ ! -x "${venv_path}/bin/python" ]]; then
  if ((skip_torch)); then
    "$python_command" -m venv --system-site-packages "$venv_path"
  else
    "$python_command" -m venv "$venv_path"
  fi
fi

venv_python="${venv_path}/bin/python"
constraints="${repo_root}/configs/training/constraints_remote.txt"
[[ -f "$constraints" ]] || { echo "Missing constraints file: $constraints" >&2; exit 1; }
[[ -f "${repo_root}/vendor/tribev2/pyproject.toml" ]] || {
  echo "Pinned vendor/tribev2 snapshot is missing; sync the complete repository." >&2
  exit 1
}
grep -q "af58661791a351a448a489042a28f6c37e1c14b7" \
  "${repo_root}/vendor/UPSTREAM.md" || {
  echo "vendor/UPSTREAM.md does not identify the expected TRIBE v2 commit." >&2
  exit 1
}
grep -q "Attribution-NonCommercial 4.0 International" \
  "${repo_root}/vendor/tribev2/LICENSE" || {
  echo "The vendored TRIBE v2 CC BY-NC 4.0 license file is missing or changed." >&2
  exit 1
}

export PIP_DISABLE_PIP_VERSION_CHECK=1
"$venv_python" -m pip install --upgrade "pip>=24.3" "setuptools>=69" wheel

# The pinned upstream snapshot requires exca==0.5.20.  Some institutional
# Python mirrors lag the public PyPI simple index even though that immutable
# wheel is published, so install the exact PyPI artifact by URL and hash before
# resolving the remaining dependencies.
exca_wheel_url="https://files.pythonhosted.org/packages/3c/89/9a012f7080a0d7bdba63b68a934e9852f4c01dd040610e16c3f98448b3ef/exca-0.5.20-py3-none-any.whl"
exca_wheel_sha256="463cc23cb629b03fe8b560396f86df8af1678bfe924a3d2f760cf9128fe2e79d"
"$venv_python" -m pip install \
  --constraint "$constraints" \
  "exca @ ${exca_wheel_url}#sha256=${exca_wheel_sha256}"

if ((skip_torch)); then
  "$venv_python" - <<'PY'
try:
    import torch
    import torchvision
except ImportError as exc:
    raise SystemExit(
        "--skip-torch was used, but torch/torchvision are not importable in this venv"
    ) from exc
torch_version = torch.__version__.split("+", 1)[0]
vision_version = torchvision.__version__.split("+", 1)[0]
if (torch_version, vision_version) != ("2.6.0", "0.21.0"):
    raise SystemExit(
        "--skip-torch requires the constrained pair torch==2.6.0 and "
        f"torchvision==0.21.0; found {torch.__version__} / {torchvision.__version__}"
    )
print(f"Keeping existing torch {torch.__version__} / torchvision {torchvision.__version__}")
PY
else
  "$venv_python" -m pip install \
    --constraint "$constraints" \
    --index-url "$torch_index_url" \
    torch torchvision
fi

# Install the pinned, unmodified upstream source and this adapter from the
# synced checkout. Explicit requests closes an undeclared upstream import.
"$venv_python" -m pip install \
  --constraint "$constraints" \
  --editable "${repo_root}/vendor/tribev2[training]" \
  "requests>=2.32,<3"
"$venv_python" -m pip install \
  --constraint "$constraints" \
  --editable "${repo_root}[data,dev]"
"$venv_python" -m pip check
"$venv_python" -m pip freeze --all >"${venv_path}/spfcl-installed-packages.txt"

"$venv_python" - <<'PY'
from importlib import metadata

import torch
from tribev2.model import FmriEncoder
from spfcl.train import fork_stage, save_weights_checkpoint

del FmriEncoder, fork_stage, save_weights_checkpoint
print(f"torch={torch.__version__}")
print(f"tribev2={metadata.version('tribev2')}")
print(f"spfcl={metadata.version('spfcl')}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg=$(command -v ffmpeg)"
else
  echo "WARNING: ffmpeg is not on PATH; load/install it before V-JEPA video extraction." >&2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NOTE: nvidia-smi is absent on this node; this is normal on some Slurm login nodes." >&2
fi

cat <<EOF

Environment ready: ${venv_path}
Activate with: source ${venv_path}/bin/activate
Resolved package record: ${venv_path}/spfcl-installed-packages.txt

TRIBE v2 is vendored at commit af58661791a351a448a489042a28f6c37e1c14b7
under CC BY-NC 4.0. Dataset and stimulus acquisition remain separate because
the stimulus archive has its own access terms.
EOF
