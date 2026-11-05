#!/usr/bin/env bash
# The image experiments behind Table 1, for one dataset: models, sources, carves, couplings, maps,
# ceilings, baselines, and the table. Every step skips outputs that exist, so the script can be
# rerun after an interruption. Run from the repository root:
#
#   bash bash/images.sh mnist          # then fmnist, celeba
#   bash bash/images.sh table          # Table 1 from everything under results/
#
# Steps that need a GPU: training (STEP=train), the sources (prepare), the couplings, the maps and
# ceilings, GENOT-Q and the RelGeo lengths. The others run on CPU. The Rosetta and mini-vec2vec
# baselines run in a separate environment with the authors' package (see README.md); set ROSETTA_PY
# to its python.
#
# Data: MNIST and FashionMNIST download themselves into data/<dataset>. CelebA is read from
# data/celeba/celeba_{train,val,test}_{images,labels}.npy (64 x 64 x 3 float32 images in [0, 1],
# int32 attribute labels), prepared from the aligned-and-cropped CelebA release.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-python}
ROSETTA_PY=${ROSETTA_PY:-}
DS=${1:?mnist | fmnist | celeba | table}
STEP=${STEP:-all}       # all | train | prepare | carve | couple | fit | ceiling | baselines

if [[ $DS == table ]]; then
    $PY experiments/images/table1.py --results results
    exit 0
fi

# ---------------------------------------------------------------- the grid
# Models per prior: a b c d with seeds 42 123 7 8. Same-prior pairings ab ac ad bc bd cd; cross-prior
# pairings <vamp model><gauss model>. CelebA has fewer: Gaussian a b c d (pairings ab ac ad), VampPrior
# a b c (pairings ab ac), cross-prior aa ba.
declare -A SEED=([a]=42 [b]=123 [c]=7 [d]=8)
case $DS in
    mnist|fmnist)
        MODELS_gauss="a b c d"; MODELS_vamp="a b c d"
        PAIRS_gauss="ab ac ad bc bd cd"; PAIRS_vamp="ab ac ad bc bd cd"; PAIRS_cross="aa ab ba bb"
        N_POOL=6000; N=3000; Z=16 ;;
    celeba)
        MODELS_gauss="a b c d"; MODELS_vamp="a b c"
        PAIRS_gauss="ab ac ad"; PAIRS_vamp="ab ac"; PAIRS_cross="aa ba"
        N_POOL=15000; N=7500; Z=32 ;;
    *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac
DRAWS="1 2 3 4 5"                       # NGMA, its ceiling on CelebA, Rosetta and mini-vec2vec
ceiling_draws () { [[ $DS == celeba ]] && echo "1 2 3 4 5" || echo "1 2"; }
base_draws () {                         # the other baselines
    [[ $DS == celeba && $1 == celeba_vamp_ab ]] && echo "1 2 3 4 5" || echo "1 2"; }

roots () {                              # <root> <prior of a> <model a> <prior of b> <model b>
    for xy in $PAIRS_gauss; do echo "${DS}_gauss_${xy} gauss ${xy:0:1} gauss ${xy:1:1}"; done
    for xy in $PAIRS_vamp;  do echo "${DS}_vamp_${xy} vamp ${xy:0:1} vamp ${xy:1:1}"; done
    for xy in $PAIRS_cross; do echo "${DS}_cross_${xy} vamp ${xy:0:1} gauss ${xy:1:1}"; done
}
ckpt () { echo "checkpoints/${DS}_$1/vae_$2/best_model.pickle"; }
run () { [[ $STEP == all || $STEP == $1 ]]; }

# ---------------------------------------------------------------- 1. models
if run train; then
    for prior in gauss vamp; do
        models=MODELS_$prior
        for m in ${!models}; do
            [[ -f $(ckpt $prior $m) ]] && continue
            $PY scripts/train_vae.py --config-name vae_${DS}_${prior} training.seed=${SEED[$m]} general.run_name=vae_$m
        done
    done
fi

# ---------------------------------------------------------------- 2. sources: the pool encoded by both models, with its geometry
if run prepare; then
    while read -r root pa ma pb mb; do
        $PY scripts/prepare_sources.py --dataset $DS --data-root data/$DS --vae-a $(ckpt $pa $ma) --vae-b $(ckpt $pb $mb) \
            --n-points $N_POOL --seed 1 --out sources/$root
    done < <(roots)
fi

# ---------------------------------------------------------------- 3. carves: disjoint (NGMA and the baselines), paired (Procrustes, LFM anchors), frac100 (the ceilings)
if run carve; then
    while read -r root _; do
        for s in $DRAWS; do
            $PY scripts/carve.py --src sources/$root --out carves/$root/seed$s/disjoint --regime disjoint --n $N --seed $s
            $PY scripts/carve.py --src sources/$root --out carves/$root/seed$s/paired --regime paired --n $N --seed $s
            $PY scripts/carve.py --src sources/$root --out carves/$root/seed$s/frac100 --regime frac --overlap 1.0 --n $N --seed $s
        done
    done < <(roots)
fi

# ---------------------------------------------------------------- 4. couplings
if run couple; then
    while read -r root _; do
        for s in $DRAWS; do $PY scripts/couple.py carves/$root/seed$s/disjoint; done
    done < <(roots)
fi

# ---------------------------------------------------------------- 5. NGMA's map
if run fit; then
    mkdir -p results/fits
    while read -r root _; do
        for s in $DRAWS; do
            out=results/fits/${root}_s$s.json; [[ -f $out ]] && continue
            $PY scripts/fit_map.py carves/$root/seed$s/disjoint \
                --pi carves/$root/seed$s/disjoint/alignments/ensemble_eol_samedec_kn_FGWG_ann/pi.npy --out $out
        done
    done < <(roots)
fi

# ---------------------------------------------------------------- 6. the ceiling: NGMA's map class fitted on the true pairs
if run ceiling; then
    mkdir -p results/ceilings
    while read -r root _; do
        for s in $(ceiling_draws); do
            out=results/ceilings/${root}_s$s.json; [[ -f $out ]] && continue
            $PY scripts/fit_map.py carves/$root/seed$s/disjoint \
                --pi carves/$root/seed$s/disjoint/alignments/ensemble_eol_samedec_kn_FGWG_ann/pi.npy \
                --ceiling --oracle-carve carves/$root/seed$s/frac100 --out $out
        done
    done < <(roots)
fi

# ---------------------------------------------------------------- 7. baselines
if run baselines; then
    mkdir -p results/baselines/{cached,relreps,relgeo,genot}
    while read -r root pa ma pb mb; do
        for s in $(base_draws $root); do
            c=carves/$root/seed$s/disjoint
            [[ -f results/baselines/cached/${root}_s$s.csv ]] || \
                $PY baselines/cached_baselines.py --cache-dir carves/$root --seed $s --label $root --dataset $DS --z-dim $Z \
                    --out-csv results/baselines/cached/${root}_s$s.csv
            [[ -f results/baselines/relreps/${root}_s$s/results.csv ]] || \
                $PY baselines/run_relreps.py --data-dir $c --out-dir results/baselines/relreps/${root}_s$s --seed $s
            [[ -f results/baselines/relgeo/${root}_s$s/geo.npz ]] || \
                $PY baselines/relgeo_lengths.py --data-dir $c --checkpoint-a $(ckpt $pa $ma) --checkpoint-b $(ckpt $pb $mb) \
                    --z-dim $Z --model-class $([[ $DS == celeba ]] && echo ngma.models.celeba_vae_ilms.VAE || echo ngma.models.vae.VAE) \
                    --out results/baselines/relgeo/${root}_s$s/geo.npz
            [[ -f results/baselines/relgeo/${root}_s$s/results.csv ]] || \
                $PY baselines/run_relgeo.py --data-dir $c --out-dir results/baselines/relgeo/${root}_s$s \
                    --geo-mode precomputed --geo-npz results/baselines/relgeo/${root}_s$s/geo.npz --seed $s
            [[ -f results/baselines/genot/${root}_s${s}_euclidean.csv ]] || \
                $PY baselines/genot_q.py --latent-dir $c --dataset $DS --regime disjoint --seed $s --batch-size $N \
                    --cost euclidean --out results/baselines/genot/${root}_s${s}_euclidean.csv
            [[ -f results/baselines/genot/${root}_s${s}_pullback.csv ]] || \
                $PY baselines/genot_q.py --latent-dir $c --dataset $DS --regime disjoint --seed $s --batch-size $N \
                    --cost precomputed --cx $c/C_ensemble_eol_samedec_a.npy --cy $c/C_ensemble_eol_samedec_b.npy \
                    --out results/baselines/genot/${root}_s${s}_pullback.csv
        done
    done < <(roots)
    if [[ -n $ROSETTA_PY ]]; then
        $ROSETTA_PY baselines/linear_baselines.py --carve-glob "carves/${DS}_*/seed*/disjoint" \
            --methods rosetta,mv_official --out-root results/baselines
    else
        echo "ROSETTA_PY not set: the Rosetta and mini-vec2vec baselines were not run" >&2
    fi
fi
