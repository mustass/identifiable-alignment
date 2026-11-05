#!/bin/bash
# Data on the sphere, Fig. 3: the alignment runs (4 measures x 2 regimes x 10 seeds), their
# collection, the margin test (4 measures x 3 seeds) and the figure. CPU only; PAR runs at a
# time, one thread each. A run whose results.pkl exists is skipped, so the script resumes.
#
#   bash bash/sphere.sh                       # everything, into results/sphere
#   OUT=/tmp/sphere PAR=4 bash bash/sphere.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=${PY:-python}
OUT=${OUT:-$REPO/results/sphere}
PAR=${PAR:-8}
SEEDS=${SEEDS:-"10 11 12 13 14 15 16 17 18 19"}
DENSITIES=${DENSITIES:-"mixture_vmf mixture_vmf_eq uniform vmf"}   # the GW cells first: they take longest
MODES=${MODES:-"independent permuted"}
MARGIN_SEEDS=${MARGIN_SEEDS:-"0 1 2"}
mkdir -p "$OUT/logs" "$OUT/margin"

one_run() {
    local d=$1 m=$2 s=$3
    local res="$OUT/$d/round/$m/seed$s/results.pkl" log="$OUT/logs/${d}_${m}_s${s}.log"
    if [[ -f $res ]]; then echo "skip  $d $m s$s"; return 0; fi
    local t0=$(date +%s)
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 PYTHONPATH="$REPO/src" \
    "$PY" experiments/sphere/run.py --density "$d" --mode "$m" --seed "$s" --out "$OUT" > "$log" 2>&1
    local rc=$?
    echo "$( [[ $rc == 0 && -f $res ]] && echo done || echo FAIL)  $d $m s$s  $(( $(date +%s) - t0 ))s  rc=$rc"
}
one_margin() {
    local d=$1 s=$2
    local res="$OUT/margin/${d}_s${s}.json" log="$OUT/logs/margin_${d}_s${s}.log"
    if [[ -f $res ]]; then echo "skip  margin $d s$s"; return 0; fi
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$REPO/src" \
    "$PY" experiments/sphere/margin_test.py --density "$d" --seed "$s" --out "$OUT/margin" > "$log" 2>&1
    echo "$( [[ $? == 0 && -f $res ]] && echo done || echo FAIL)  margin $d s$s"
}
export -f one_run one_margin
export OUT REPO PY

# 1. the alignment runs
for d in $DENSITIES; do for m in $MODES; do for s in $SEEDS; do echo "$d $m $s"; done; done; done \
    | xargs -P "$PAR" -L 1 bash -c 'one_run "$0" "$1" "$2"'
PYTHONPATH="$REPO/src" "$PY" experiments/sphere/collect.py --root "$OUT" --out-dir "$OUT"

# 2. the margin test
for d in $DENSITIES; do for s in $MARGIN_SEEDS; do echo "$d $s"; done; done \
    | xargs -P "$PAR" -L 1 bash -c 'one_margin "$0" "$1"'

# 3. the figure
PYTHONPATH="$REPO/src" "$PY" experiments/sphere/figure.py --raw "$OUT/sphere_err_raw.json" \
    --full "$OUT/sphere_full_raw.json" --margin-sweep "$OUT/margin" --out "$OUT/sphere_main.pdf"
