#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

shutdown_after=false
if [[ "${1:-}" == "--shutdown" ]]; then
  shutdown_after=true
fi

export LLM_TYPE=ollama
export LLM_INPUT_UNIT=template
export RAG_CONTEXT_MODE=history_only

datasets=("Spirit" "BGL")

run_dataset() {
  local dataset="$1"
  echo "=============================="
  echo "Running dataset: ${dataset}"
  echo "=============================="

  python src/LLMs/main.py \
    --dataset "${dataset}" \
    --scope high_uncertain \
    --llm-input-unit template

  python src/LLMs/main.py \
    --dataset "${dataset}" \
    --scope full_test \
    --llm-input-unit template

  echo "Finished dataset: ${dataset}"
}

for dataset in "${datasets[@]}"; do
  run_dataset "${dataset}"
done

echo "All LLM detections completed."

if [[ "$shutdown_after" == true ]]; then
  echo "Shutting down now."
  sudo shutdown -h now
fi
