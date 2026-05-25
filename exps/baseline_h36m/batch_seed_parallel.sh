#!/usr/bin/env bash
# ------------------------------------
# batch_seed_parallel.sh
#
# usage:
#   ./batch_seed_parallel train.py [start_seed] [end_seed] [-j N]
#
#   - start_seed & end_seed (optional)  : numeric range, inclusive
#   - -j N (optional)                   : maximum number of concurrent jobs
# ----------------------------------------

set -euo pipefail

# parse arguments
SCRIPT_NAME="$0"
PY_SCRIPT=""
START_SEED=""
END_SEED=""
MAX_JOBS=5 # default

# shift through arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      shift
      MAX_JOBS=${1:-5}
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z $PY_SCRIPT ]]; then
        PY_SCRIPT="$1"
      elif [[ -z $START_SEED ]]; then
        START_SEED="$1"
      else
        END_SEED="$1"
      fi
      ;;
  esac
  shift
done

# sanity checks
[[ -z $PY_SCRIPT ]] && { echo "Error: You must specify the Python script to run." >&2; exit 1; }
[[ ! -f $PY_SCRIPT ]] && { echo "Error: $PY_SCRIPT not found." >&2; exit 1; }

#

# default range 800-900 if nothing provided
START_SEED=${START_SEED:-800}
END_SEED=${END_SEED:-900}

# ensure start <= end
(( START_SEED > END_SEED )) && { echo "Error: start_seed ($START_SEED) > end_seed ($END_SEED)" >&2; exit 1; }


# temp output files
TMP_OUT="tmp_output.txt" # reused each run
TMP_RES="tmp_results.txt" # keeps seed + average pairs
> "$TMP_RES"

# run one seed & compute average
run_one_seed() {
  local seed="$1"
  # run training script with seed
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python "$PY_SCRIPT" --seed "$seed" --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48 --dct --att_out > "$TMP_OUT"

  # grab last line and strip brackets
  local last_line
  last_line=$(tail -n 1 "$TMP_OUT")
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

  if (( count == 0 )); then
    echo "NaN" # failure
  else
    echo "$(echo "scale=6; $sum / $count" | bc)"
  fi
}

# use seq to iterate over range
for seed in $(seq "$START_SEED" "$END_SEED"); do
  # launch a subshell to capture output later
  (
    avg=$(run_one_seed "$seed")
    echo "$seed $avg"
  ) &
  # limit concurrency
  # if already have MAX_JOBS running, wait for *any* to finish
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 0.1
  done
done

# wait for all bkg jobs to finish before proceeding
wait

# sort and pick best 5
echo "Top 5 averages (seed avg): "
sort -k2,2n "$TMP_RES" | head -n 5 | \
while read -r seed avg; do
  printf "  Seed %s -> avg = %s\n" "$seed" "$avg"
done

# clean up
rm -f "$TMP_OUT" "$TMP_RES"

echo "Parallel sweep finished (max $MAX_JOBS jobs at once)."