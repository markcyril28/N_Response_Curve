#!/usr/bin/env bash

# Compatibility shim; sourcing it must not change the caller's shell options.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=modules/n_response_curve/logging/run_logging.sh
source "${SCRIPT_DIR}/n_response_curve/logging/run_logging.sh"
