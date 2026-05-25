#!/usr/bin/env bash


set -euo pipefail

# parse arguments
if [[ $# -lt 1 ]]; then
  echo "Usageee: $0 <python_script.py> [start_seed end_seed]" >&2
  exit 1
fi

PY_SCRIPT="$1"
shift

# default range 800-900 if nothing provided
START_SEED=${1:-800}
END_SEED=${2:-900}

# ensure start <= end
if (( START_SEED > END_SEED )); then
  echo "Error: start_seed ($START_SEED) must be <= end_seed ($END_SEED)" >&2
  exit 1
fi

# temp output files
OUTPUT_FILE="tmp_output.txt" # reused each run
RESULTS_FILE="tmp_results.txt" # keeps seed + average pairs

# run one seed & compute average
run_one_seed() {
  local seed="$1"
  # run training script with seed
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python "$PY_SCRIPT" --seed "$seed" --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48 --dct --att_out > "$OUTPUT_FILE"

  # grab last line and strip brackets
  local last_line
  last_line=$(tail -n 1 "$OUTPUT_FILE")
  local clean_line
  clean_line=$(echo "$last_line" | tr -d '[]')

  # split on commas, average
  local sum=0 count=0
  IFS=',' read -ra nums <<< "$clean_line"
  for n in "${nums[@]}"; do
    n_clean=$(echo "$n" | xargs) # trim spaces
    [[ -z $n_clean ]] && continue
    sum=$(echo "$sum + $n_clean" | bc)
    ((count++))
  done

  if [[ $count -eq 0 ]]; then
    echo "NaN" # failure
  else
    echo "$(echo "scale=6; $sum / $count" | bc)"
  fi
}

# run loop and collect results
> "$RESULTS_FILE" # truncate

# use seq to iterate over range
for seed in $(seq "$START_SEED" "$END_SEED"); do
  avg=$(run_one_seed "$seed")
  echo "$seed $avg" >> "$RESULTS_FILE"
done

# sort and pick best 5
echo "Top 5 seeds (smallest averages) from $START_SEED to $END_SEED:"
sort -k2,2n "$RESULTS_FILE" | head -n 5 | while read -r seed avg; do
  printf "  Seed %s -> avg = %s\n" "$seed" "$avg"
done

# clean
rm -f "$OUTPUT_FILE" "$RESULTS_FILE"