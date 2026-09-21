#!/usr/bin/env bash
# Source-only environment setup.  It never starts ROS nodes or touches CAN.

_REKPIPER_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REKPIPER_ROOT="${REKPIPER_ROOT:-${_REKPIPER_SETUP_DIR}}"
export REKPIPER_RUNTIME_ROOT="${REKPIPER_RUNTIME_ROOT:-${REKPIPER_ROOT}/runtime}"
export REKEP_OFFICIAL_ROOT="${REKEP_OFFICIAL_ROOT:-${REKPIPER_ROOT}/third_party/ReKep}"
# Deprecated compatibility alias. Production launch files resolve package-local
# fail-closed defaults and do not assume a monolithic directory beneath here.
export REKPIPER_CONFIG_ROOT="${REKPIPER_CONFIG_ROOT:-${REKPIPER_ROOT}/src/rekpiper_bringup/config}"
export REKPIPER_SITE_CONFIG_ROOT="${REKPIPER_SITE_CONFIG_ROOT:-${REKPIPER_RUNTIME_ROOT}/site-config}"
export REKPIPER_TRUST_ROOT="${REKPIPER_TRUST_ROOT:-${REKPIPER_RUNTIME_ROOT}/trust}"
export REKPIPER_MODEL_ROOT="${REKPIPER_MODEL_ROOT:-${REKPIPER_RUNTIME_ROOT}/models}"
export REKPIPER_VENDOR_ROOT="${REKPIPER_VENDOR_ROOT:-${REKPIPER_RUNTIME_ROOT}/vendor}"
export REKPIPER_DATA_ROOT="${REKPIPER_DATA_ROOT:-${REKPIPER_RUNTIME_ROOT}/data}"

if [[ -f "${REKPIPER_RUNTIME_ROOT}/bin/activate" ]]; then
  source "${REKPIPER_RUNTIME_ROOT}/bin/activate"
  export PYTHONNOUSERSITE=1
fi
if [[ -f "${REKPIPER_SITE_CONFIG_ROOT}/environment.sh" ]]; then
  source "${REKPIPER_SITE_CONFIG_ROOT}/environment.sh"
fi
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "${REKPIPER_ROOT}/devel/setup.bash" ]]; then
  source "${REKPIPER_ROOT}/devel/setup.bash"
fi
if [[ -d "${REKPIPER_ROOT}/src" ]]; then
  case ":${ROS_PACKAGE_PATH:-}:" in
    *":${REKPIPER_ROOT}/src:"*) ;;
    *) export ROS_PACKAGE_PATH="${REKPIPER_ROOT}/src${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}" ;;
  esac
fi
unset _REKPIPER_SETUP_DIR
