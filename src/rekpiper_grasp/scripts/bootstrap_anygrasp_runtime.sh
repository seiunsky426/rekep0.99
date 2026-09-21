#!/usr/bin/env bash
set -euo pipefail

runtime_root="${REKPIPER_RUNTIME_ROOT:-${REKPIPER_ROOT:?source setup.bash first}/runtime}"
runtime_path="${1:-${runtime_root}}"
vendor_root="${REKPIPER_VENDOR_ROOT:-${runtime_root}/vendor}"
sdk_path="${vendor_root}/anygrasp_sdk"
minkowski_path="${vendor_root}/MinkowskiEngine"
graspnet_api_path="${vendor_root}/graspnetAPI"
runtime_python="${runtime_path}/bin/python"

if [ ! -x "${runtime_python}" ]; then
  echo "BLOCKED: complete M0.1 and install requirements.models.lock first."
  exit 2
fi

# Use the formal environment and its locked dependencies; never create a
# second system-site-packages environment or install training-only API extras.
"${runtime_python}" -c 'import sys, torch, grasp_nms, open3d, transforms3d; assert sys.version_info[:2] == (3, 8); assert torch.cuda.is_available(), "CUDA unavailable"'
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-2}"
platform_tag="$("${runtime_python}" -c 'import sysconfig; print(sysconfig.get_platform())')"
build_name="lib.${platform_tag}-3.8"

(
  cd "${sdk_path}/pointnet2"
  "${runtime_python}" setup.py build --build-lib "build/${build_name}"
)
(
  cd "${minkowski_path}"
  "${runtime_python}" setup.py build --build-lib "build/${build_name}" \
    --force_cuda --blas=blas \
    --blas_include_dirs=/usr/include/x86_64-linux-gnu \
    --blas_library_dirs=/usr/lib/x86_64-linux-gnu
)

"${runtime_python}" - "${graspnet_api_path}" \
  "${sdk_path}/pointnet2/build/${build_name}" \
  "${minkowski_path}/build/${build_name}" <<'PYTHON'
from pathlib import Path
import sys
import sysconfig
paths = [str(Path(path).resolve()) for path in sys.argv[1:]]
Path(sysconfig.get_path("purelib"), "rekpiper_anygrasp.pth").write_text(
    "\n".join(paths) + "\n", encoding="utf-8")
PYTHON

echo "Local CUDA extensions rebuilt in the formal runtime. Run"
echo "python tools/check_m0_models.py anygrasp to verify licensed inference."
