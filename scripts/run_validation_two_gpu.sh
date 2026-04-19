#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_DENSE_MODELS="Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-4B,Qwen/Qwen3-Embedding-8B,vespa-engine/colbert,sentence-transformers/bert-large-nli-mean-tokens"
readonly GPU_IDS=(2 3)

get_dense_models_csv() {
  local models_csv="${1:-$DEFAULT_DENSE_MODELS}"
  if [[ -z "${models_csv}" ]]; then
    echo "${DEFAULT_DENSE_MODELS}"
    return
  fi
  echo "${models_csv}"
}

run_bm25_eval() {
  local validation_parquet="$1"
  local wandb_project="$2"
  local wandb_entity="$3"

  echo "Running BM25 validation evaluation first..."
  uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_validation_bm25_parquet \
    --validation_parquet "${validation_parquet}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --run_name "validation-bm25"
}

launch_dense_eval() {
  local model_name="$1"
  local gpu_id="$2"
  local validation_parquet="$3"
  local wandb_project="$4"
  local wandb_entity="$5"

  local run_name="validation-dense-hybrid-${model_name//\//_}"
  echo "Launching dense+hybrid evaluation: model=${model_name} gpu=${gpu_id} run_name=${run_name}" >&2
  CUDA_VISIBLE_DEVICES="${gpu_id}" uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_validation_parquet \
    --validation_parquet "${validation_parquet}" \
    --retrieval_model "${model_name}" \
    --vllm_gpu_device 0 \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --run_name "${run_name}" &
  echo $!
}

parse_models() {
  local models_csv="$1"
  local -n out_models_ref=$2
  IFS=',' read -r -a out_models_ref <<< "${models_csv}"
}

main() {
  local models_csv
  models_csv="$(get_dense_models_csv "${1:-}")"
  local validation_parquet="${2:-artifacts/val.parquet}"
  local wandb_project="${3:-ast-skills-retriever}"
  local wandb_entity="${4:-}"

  run_bm25_eval "${validation_parquet}" "${wandb_project}" "${wandb_entity}"

  local models=()
  parse_models "${models_csv}" models

  local -A pid_to_gpu=()
  local -A gpu_busy=()
  local model_index=0
  local active_jobs=0

  while [[ ${model_index} -lt ${#models[@]} || ${active_jobs} -gt 0 ]]; do
    for gpu_id in "${GPU_IDS[@]}"; do
      if [[ ${model_index} -ge ${#models[@]} ]]; then
        break
      fi
      if [[ -n "${gpu_busy[${gpu_id}]:-}" ]]; then
        continue
      fi
      local model_name="${models[${model_index}]}"
      local pid
      pid="$(launch_dense_eval "${model_name}" "${gpu_id}" "${validation_parquet}" "${wandb_project}" "${wandb_entity}")"
      pid_to_gpu["${pid}"]="${gpu_id}"
      gpu_busy["${gpu_id}"]="${pid}"
      model_index=$((model_index + 1))
      active_jobs=$((active_jobs + 1))
    done

    if [[ ${active_jobs} -eq 0 ]]; then
      continue
    fi

    local finished_pid
    wait -n -p finished_pid

    local released_gpu="${pid_to_gpu[${finished_pid}]}"
    unset "pid_to_gpu[${finished_pid}]"
    unset "gpu_busy[${released_gpu}]"
    active_jobs=$((active_jobs - 1))
    echo "Completed model on gpu=${released_gpu} pid=${finished_pid}"
  done

  echo "All evaluations completed."
}

main "$@"
