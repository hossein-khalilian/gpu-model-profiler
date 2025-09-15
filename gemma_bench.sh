#!/bin/bash

MODELS=(
  "gemma3:270m"
  "gemma3:270m-it-qat"
  "gemma3:270m-it-q8_0"
  "gemma3:270m-it-fp16"
  "gemma3:1b"
  "gemma3:1b-it-qat"
  "gemma3:1b-it-q4_K_M"
  "gemma3:1b-it-q8_0"
  "gemma3:1b-it-fp16"
  "gemma3:4b"
  "gemma3:4b-it-qat"
  "gemma3:4b-it-q4_K_M"
  "gemma3:4b-it-q8_0"
  "gemma3:4b-it-fp16"
  "gemma3:12b"
  "gemma3:12b-it-qat"
  "gemma3:12b-it-q4_K_M"
  "gemma3:12b-it-q8_0"
  "gemma3:12b-it-fp16"
  "gemma3:27b"
  "gemma3:27b-it-qat"
  "gemma3:27b-it-q4_K_M"
  "gemma3:27b-it-q8_0"
  "gemma3:27b-it-fp16"
)

OUTFILE="gemma3_mem_usage.csv"
echo "model,gpu_memory_mib" > "$OUTFILE"

check_gpu_empty_or_wait() {
  ATTEMPTS=0
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$USED" -lt 500 ]; then
      echo "[INFO] GPU is free ($USED MiB)."
      break
    else
      echo "[WARN] GPU still busy ($USED MiB used)."
      ATTEMPTS=$((ATTEMPTS+1))
      if [ $ATTEMPTS -ge 6 ]; then
        echo "[ERROR] GPU busy for too long, continuing anyway."
        break
      fi
      sleep 10
    fi
  done
}

for MODEL in "${MODELS[@]}"; do
  echo "===================================================="
  # echo "[START] Cleaning up leftover ollama processes..."
  # sudo pkill -9 ollama || true
  echo "[INFO] Starting test for model: $MODEL"

  # Ensure GPU is free before starting
  check_gpu_empty_or_wait

  # Run model once and show logs (download / load progress)
  echo "[INFO] Running: ollama run $MODEL (logs visible)"
  # Single-shot prompt, output visible so downloads/loading can be seen
  ollama run "$MODEL" <<< "Hello world"

  # Give it time to allocate memory
  sleep 5

  # Measure GPU memory
  MEM=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -nr | head -n1)
  if [ -z "$MEM" ]; then MEM=0; fi
  echo "[RESULT] $MODEL uses ${MEM} MiB of GPU memory"
  echo "$MODEL,$MEM" >> "$OUTFILE"

  # Stop the model explicitly
  echo "[INFO] Stopping model: $MODEL"
  ollama stop "$MODEL" > /dev/null 2>&1

  # Short pause before next loop
  sleep 5
done

echo "===================================================="
echo "[DONE] Results saved to $OUTFILE"

