#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_CONFIG_PATH="configs/train.config.yaml"
readonly DEFAULT_GPU_IDS="2,3"

get_config_path() {
  local config_path="${1:-$DEFAULT_CONFIG_PATH}"
  echo "${config_path}"
}

get_gpu_ids() {
  local gpu_ids="${1:-$DEFAULT_GPU_IDS}"
  echo "${gpu_ids}"
}

main() {
  local config_path
  config_path="$(get_config_path "${1:-}")"

  local gpu_ids
  gpu_ids="$(get_gpu_ids "${2:-}")"

  echo "Launching accelerate FSDP training with ${gpu_ids=} and ${config_path=}"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" uv run accelerate launch \
    --config_file configs/fsdp_accelerate.yaml \
    -m ast_skills.train.train train_from_config \
    --config_path "${config_path}"
}

main "$@"
