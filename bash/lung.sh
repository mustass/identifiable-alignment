#!/usr/bin/env bash
# The lung experiment, end to end: scVI models of Tabula Sapiens lung cells, their decoder
# ensembles and pullback geometry, NGMA on carved samples, the ceilings, the baselines, the
# readouts and the paper's tables. Every stage skips work whose output exists, so the script can
# be rerun after an interruption.
#
#   ROOT=/path/to/lung SRC=/path/to/h5ad PY_SCVI=... PY=... bash bash/lung.sh [stage ...]
#
# Stages (default: all, in this order): prep fit heads geometry sources carve couple fits
# baselines readout tables. Two interpreters: PY_SCVI, an environment with scvi-tools, anndata
# and torch, for prep/fit/heads; PY, the repository's environment (JAX, see README.md), for the
# rest. PY_ROSETTA (default PY) runs Rosetta and mini-vec2vec, which need the unpaired-rosetta
# package (baselines/linear_baselines.py).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${ROOT:?set ROOT to the root directory of the experiment}"
: "${SRC:?set SRC to the directory holding lung_train.h5ad and ts_lung_full.h5ad}"
PY_SCVI="${PY_SCVI:?set PY_SCVI to the python of the scvi-tools environment}"
PY="${PY:-python}"
PY_ROSETTA="${PY_ROSETTA:-$PY}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
E=experiments/lung
SIZES="4900 9800 19600"
PAIRS="cross1 cross2 within1 within2"
MS="1 8 euc"
DRAWS="1 2 3"
N=3000
STAGES="${*:-prep fit heads geometry sources carve couple fits baselines readout tables}"
has () { [[ " $STAGES " == *" $1 "* ]]; }

if has prep; then
    [[ -f $ROOT/split.npz ]] || $PY_SCVI $E/prep.py --src "$SRC" --root "$ROOT"
fi
if has fit; then
    $PY_SCVI $E/fit_scvi.py --root "$ROOT"
fi
if has heads; then
    $PY_SCVI $E/decoder_heads.py --root "$ROOT"
fi
if has geometry; then
    for s in $SIZES; do for h in H1 H2; do for seed in 0 1; do
        tag="${h}_${s}_s${seed}"
        [[ -f $ROOT/geometry/$tag/geometry.json ]] && continue
        $PY $E/geometry.py --root "$ROOT" --tag "$tag" --M 1,8
    done; done; done
fi
if has sources; then
    $PY $E/make_sources.py --root "$ROOT" --M "${MS// /,}"
fi
if has carve; then
    # disjoint samples for the fits, 100 % overlap samples for the ceilings and the coupling accuracy
    for s in $SIZES; do for p in $PAIRS; do for m in $MS; do for d in $DRAWS; do
        src=$ROOT/sources/lung${s}_${p}_M${m}
        out=$ROOT/carves/lung${s}_${p}_M${m}/seed$d
        [[ -f $out/disjoint/gt_ab.npy ]] || $PY scripts/carve.py --src "$src" --out "$out/disjoint" --regime disjoint --n $N --seed $d
        [[ -f $out/frac100/gt_ab.npy ]] || $PY scripts/carve.py --src "$src" --out "$out/frac100" --regime frac --overlap 1.0 --n $N --seed $d
    done; done; done; done
fi
if has couple; then
    # the fused coupling on the pullback cost for M = 1 and 8; the GW coupling on the Euclidean
    # cost for the Euclidean comparator
    for s in $SIZES; do for p in $PAIRS; do for m in $MS; do
        if [[ $m == euc ]]; then mode=euclidean; coup=GW; else mode=ensemble_eol_samedec; coup=FGWG_ann; fi
        list=""
        for d in $DRAWS; do for reg in disjoint frac100; do
            c=$ROOT/carves/lung${s}_${p}_M${m}/seed$d/$reg
            [[ -f $c/alignments/${mode}_kn_$coup/pi.npy ]] || list="$list $c"
        done; done
        [[ -z $list ]] || $PY scripts/couple.py $list --mode $mode --coupling $coup
    done; done; done
fi
if has fits; then
    for s in $SIZES; do for p in $PAIRS; do for m in $MS; do for reg in disjoint frac100; do
        list=""
        for d in $DRAWS; do list="$list $ROOT/carves/lung${s}_${p}_M${m}/seed$d/$reg"; done
        $PY $E/fit_maps.py --root "$ROOT" $list
    done; done; done; done
fi
if has baselines; then
    # on the disjoint M = 8 carves: Rosetta and the official mini-vec2vec, GENOT-Q with the
    # Euclidean and the pullback cost, RelGeo on precomputed segment lengths
    $PY_ROSETTA baselines/linear_baselines.py --carve-glob "$ROOT/carves/lung*_M8/seed*/disjoint" \
        --methods rosetta,mv_official --out-root "$ROOT/baselines"
    mkdir -p "$ROOT/baselines/genot"
    for s in $SIZES; do for p in $PAIRS; do for d in $DRAWS; do
        c=$ROOT/carves/lung${s}_${p}_M8/seed$d/disjoint
        out=$ROOT/baselines/genot/lung${s}_${p}_s${d}_euclidean.csv
        [[ -f $out ]] || $PY baselines/genot_q.py --latent-dir "$c" --dataset lung --regime disjoint \
            --seed $d --batch-size $N --cost euclidean --out "$out"
        out=$ROOT/baselines/genot/lung${s}_${p}_s${d}_precomputed.csv
        [[ -f $out ]] || $PY baselines/genot_q.py --latent-dir "$c" --dataset lung --regime disjoint \
            --seed $d --batch-size $N --cost precomputed --cx "$c/C_ensemble_eol_samedec_a.npy" \
            --cy "$c/C_ensemble_eol_samedec_b.npy" --out "$out"
    done; done; done
    for s in $SIZES; do for p in $PAIRS; do for d in $DRAWS; do
        c=$ROOT/carves/lung${s}_${p}_M8/seed$d/disjoint
        [[ -f $c/relgeo/geo.npz ]] || $PY $E/relgeo_lengths.py --root "$ROOT" "$c"
        [[ -f $c/relgeo/out/results.csv ]] || $PY baselines/run_relgeo.py --data-dir "$c" \
            --out-dir "$c/relgeo/out" --geo-mode precomputed --geo-npz "$c/relgeo/geo.npz" --seed $d
    done; done; done
fi
if has readout; then
    $PY $E/read_fits.py --root "$ROOT"
    $PY $E/read_baselines.py --root "$ROOT"
fi
if has tables; then
    $PY $E/paper_tables.py --root "$ROOT" --out "$ROOT/tables"
fi
