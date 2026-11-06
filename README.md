# Neural Gromov-Monge Alignment (NGMA)

Code for *Identifiable alignment of unpaired representations*. Two latent spaces are trained
independently on unpaired data; each is treated as a metric measure space under its decoder's
pullback geometry, and a measure-preserving isometry between them is fitted from the geometry
alone, with no shared inputs, no anchors and no labels.

The repository reproduces three results of the paper. Each has one driver in `bash/` and its own
code in `experiments/`:

| result | driver | code |
|---|---|---|
| Table 1: instance retrieval between independently trained VAEs on MNIST, FashionMNIST, CelebA | `bash/images.sh` | `experiments/images/` |
| Figure 3: alignment and the margin test on the sphere | `bash/sphere.sh` | `experiments/sphere/` |
| Figure 4 and the lung table: scVI models of Tabula Sapiens lung cells | `bash/lung.sh` | `experiments/lung/` |

## Install

```bash
pip install -e .          # Python >= 3.10; JAX with GPU support as your platform requires
export PYTHONPATH=src     # every entry point needs this
```

Two parts run in their own environments: Rosetta and mini-vec2vec through the authors' package
(`pip install -e '.[rosetta]'`, see `baselines/rosetta.py`), and the lung experiment's scVI
models (`pip install -e '.[lung]'`).

## The method, as the code runs it

The method's entry points are in `scripts/`; every experiment calls them.

1. **Models.** VAEs with an ensemble of 8 decoders on one encoder, trained with β = 16 and a
   learned observation scale (`configs/vae_*.yaml`; `scripts/train_vae.py`). Latent dimension 16
   on MNIST and FashionMNIST, 32 on CelebA; standard Gaussian prior or a VampPrior.
2. **Sources** (`scripts/prepare_sources.py`). A pool of held-out images is encoded by both
   models. Each side's geometry is the decoder ensemble's expected pullback metric: the metric
   tensor at every pool code and the lengths of the straight segments between codes (32 steps,
   each step's length the root mean over decoders of the squared output change).
3. **Carves** (`scripts/carve.py`). A sample pair is two index sets into the pool: disjoint
   (no image on both sides; the regime of the tables), paired (true pairs, for the Procrustes
   baseline and the anchors of LFM) and 100 % overlap in independent orders (the carve the
   supervised ceilings are fitted on).
4. **Coupling** (`scripts/couple.py`). Entropic Gromov-Wasserstein between the two distance
   matrices, annealed in four stages from ε = 1 to 10⁻³, each stage started from the previous
   plan. At every Frank-Wolfe step the cost carries a linear term, Mémoli's third lower bound
   (the squared 2-Wasserstein distance between the two points' distance distributions), with
   weights α = 1, 2/3, 1/3, 0 over the stages and temperatures interpolating geometrically
   between GW's schedule and the lower bound's own, so the last stage is plain entropic GW.
   Float64, inner Sinkhorn to convergence with warm-started dual potentials.
5. **Map** (`scripts/fit_map.py`). Two maps φ(z) = Wz + t + h(z) and ψ(w) = W'w + t' + h'(w),
   with two-hidden-layer GELU MLPs of width 128, started at the plan's weighted Procrustes map
   and its inverse, trained on one objective: the isometry terms (the pullback metric of each
   side carried to the other), 0.3 times the fit to the plan's argmax targets, and 0.1 times the
   cycle term. Adam 10⁻³, batches of 256, 3000 steps; every 250 steps the pair is scored by the
   held-out mutual-nearest-neighbour fraction and the best checkpoint is kept, the rigid start
   being a candidate. Then up to three joint rounds of 1000 steps in which the mutual nearest
   neighbours of the combined ranking of the two maps replace the targets; the round with the
   largest held-out mutual-NN fraction is kept.
6. **Score.** The combined ranking of the two maps on the paired test split: R@1 is the
   fraction of test images whose true partner is ranked first.

The supervised ceiling (`fit_map.py --ceiling`) trains the same map class on the true pairs of
the 100 %-overlap carve, in both directions, and scores the same test split; the Procrustes
map on those pairs is scored beside it.

The baselines are in `baselines/`: Procrustes on the true pairs, RelReps-Centroid and the three
LFM variants from the carved codes (`cached_baselines.py`), RelReps (`run_relreps.py`), RelGeo
with path lengths under the same decoder geometry (`relgeo_lengths.py`, `run_relgeo.py`),
GENOT-Q on the Euclidean and on the pullback cost (`genot_q.py`), Rosetta and mini-vec2vec
through the authors' package (`linear_baselines.py`, `rosetta.py`). `baseline_metrics.py` is the
retrieval and CSLS scoring they share.

## Table 1: the image experiments

`bash bash/images.sh <mnist|fmnist|celeba>` runs, for one dataset, every step below and skips
outputs that exist; `bash bash/images.sh table` then writes
`results/tables/main_results.tex` and `results/tables/Headline_summary_disjoint.csv`
(`experiments/images/table1.py`). Data: MNIST and FashionMNIST download themselves; CelebA is
read from `data/celeba/celeba_{train,val,test}_{images,labels}.npy` (64 × 64 × 3 images in
[0, 1], attribute labels), built from the aligned-and-cropped release.

The grid of the table:

| | MNIST, FashionMNIST | CelebA |
|---|---|---|
| models per prior | a b c d (seeds 42 123 7 8) | Gaussian a b c d, VampPrior a b c |
| same-prior pairings | ab ac ad bc bd cd | Gaussian ab ac ad, VampPrior ab ac |
| cross-prior pairings (VampPrior model × Gaussian model) | aa ab ba bb | aa ba |
| pool, points per side | 6000, 3000 | 15000, 7500 |
| draws (carve seeds): NGMA, Rosetta, mini-vec2vec | 1–5 | 1–5 |
| draws: the ceiling | 1–2 | 1–5 |
| draws: the other baselines | 1–2 | 1–2 (VampPrior ab: 1–5) |

A cell of the table is a dataset and a prior; its mean is over pairings of the per-pairing mean
over draws, the spread the s.d. over all draws.

Every step is deterministic given its seed, but k-means, the QAP solver and GPU reductions differ
in their last digits across machines and library builds, so a rerun on other hardware agrees with
the paper to about 10⁻³ in R@1, not to the last digit.

Output layout:

```
checkpoints/<dataset>_<prior>/vae_<x>/best_model.pickle
sources/<dataset>_<prior>_<xy>/                 Za Zb labels, the test codes, C_<mode>_{a,b}, G_{a,b}
carves/<root>/seed<S>/{disjoint,paired,frac100}/
carves/<root>/seed<S>/disjoint/alignments/ensemble_eol_samedec_kn_FGWG_ann/pi.npy
results/fits/<root>_s<S>.json (+ _mapped.npz)   NGMA
results/ceilings/<root>_s<S>.json               the supervised ceiling
results/baselines/{cached,relreps,relgeo,genot,rosetta_wp,minivec2vec_official}/
results/tables/
```

## Figure 3: the sphere

`bash bash/sphere.sh` runs the alignments on the four measures of `configs/sphere_measures.yaml`
(independent and permuted samples, seeds 10–19, `experiments/sphere/run.py`, settings in
`configs/sphere.yaml`), collects them (`collect.py`), runs the margin test (`margin_test.py`,
seeds 0–2) and draws the figure (`figure.py`). The LDD-LP compares two landmarks by W₂² between
their distance distributions and is solved exactly by network simplex. Panel (b) adds, on the
mixtures with independent samples, NGMA's coupling and its annealed GW alone. The rotation is
fitted in closed form. CPU only; the full-cloud fused coupling takes one to three hours per run on
one thread, so the whole grid takes about a day at eight runs in parallel.

## The lung experiment: Figure 4 and the lung table

Two scVI models of the same RNA data, trained on different cells or on the same cells with
different seeds, aligned with NGMA's pullback geometry and scored on held-out cells that both
models encode. The scripts are in `experiments/lung/`.

- **Data.** Tabula Sapiens lung, the 62,954 10x 3' v3.1 cells of four donors, 3,000 genes, raw
  counts; Smart-seq2 cells dropped. 34 cell types, 4 compartments. No batch covariate. Inputs:
  `lung_train.h5ad` (the counts) and `ts_lung_full.h5ad` (the labels, joined by barcode).
  `prep.py`.
- **Split, by cell** (seed 0): 10,000 eval cells, scored with known pairs; a pool of 9,000
  held-out cells that both models encode; the rest cut into two disjoint training halves H1, H2,
  each with nested subsets of 4,900 < 9,800 < 19,600 cells.
- **Models.** scVI (latent 20, three hidden layers of 128, layer norm, gene dispersion, negative
  binomial, observed library size, 400 epochs), one per (half, size, seed in {0, 1}): 12 models.
  `fit_scvi.py`.
- **Decoder ensembles.** Head 0 is the scVI decoder, so M = 1 is plain scVI; heads 1-7 are
  trained with the encoder frozen, on codes sampled from its posterior q(z|x) of the model's own
  training cells, at its gene dispersions. `decoder_heads.py` refuses unless head 0 reproduces
  scVI's decoder.
- **Geometry.** φ = 2 √θ asinh(√(l_ref ρ / θ)) per head, the coordinate in which the Fisher
  metric of the negative-binomial mean is Euclidean (l_ref the median library size). The
  M-decoder ensemble metric and straight-segment lengths with T = 10 steps, for M = 1 and M = 8,
  on the pool and eval cells. `geometry.py`.
- **Pairs**, per training size: `cross1/2` (H1 against H2, different cells, seeds 0 and 1) and
  `within1/2` (two seeds on the same half).
- **Sources and carves.** One source per (pair, M), M in {1, 8, euc}; `euc` is the Euclidean
  comparator (Euclidean latent distances, identity metric). From each source, n = 3,000 cells
  per side, three draws, in two regimes: disjoint (no cell on both sides) and 100 % overlap
  (the same cells in independent orders, for the ceilings and the coupling accuracy). `make_sources.py`, then
  `scripts/carve.py`.
- **Coupling.** NGMA's coupling on the pullback cost for M = 1 and 8; the annealed GW coupling
  on the Euclidean cost for the comparator. `scripts/couple.py`.
- **Map.** `scripts/fit_map.py` with the settings above, both directions, scored by the combined
  ranking on the 10,000 eval cells. Every disjoint fit also reports the two ceilings on its
  100 %-overlap sibling: the same map class and the Procrustes map, both fitted on the true pairs
  and scored a → b. `fit_maps.py` runs the fits of several carves in one process.
- **Baselines**, on the disjoint M = 8 carves: Rosetta and the official mini-vec2vec
  (`baselines/linear_baselines.py`), GENOT-Q with the Euclidean and the pullback cost
  (`baselines/genot_q.py`), RelGeo on segment lengths to the class-centroid anchors
  (`relgeo_lengths.py`, then `baselines/run_relgeo.py --geo-mode precomputed`).
- **Readout.** `read_fits.py` (R@1, R@1 within cell type, cell-type and compartment accuracy,
  FOSCTTM, CSLS@1 and the ceilings, from each fit's saved mapped eval cells) and
  `read_baselines.py` (the baselines a → b). `paper_tables.py` writes the lung tables, Figure 4
  and the numbers the text quotes; the paper's lung table is `tab_lung_baselines.tex`.

Everything lives under one root directory outside the repository: `lung10x.h5ad`, `split.npz`,
`split_meta.json`, `labels.npz`, then `fits/`, `geometry/`, `sources/`, `carves/`, `results/`,
`baselines/`, `tables/`.

```bash
ROOT=/path/to/lung SRC=/path/to/h5ad_files PY_SCVI=/path/to/scvi/python PY=python \
    bash bash/lung.sh [stage ...]
```

runs the stages `prep fit heads geometry sources carve couple fits baselines readout tables` in
order, or the ones named; each skips outputs that exist. `PY_SCVI` is an environment with
`scvi-tools`, `anndata` and `torch`, for the models and heads; `PY` is the repository's JAX
environment, for the rest. The geometry, couplings and fits run on a GPU; the models, heads and
baselines on CPU.

## Layout

```
src/ngma/             models (VAE, priors), losses, training, geometry (segment lengths, metric
                      tensors, signatures), alignment (entropic GW with the linear term, closed-form
                      maps), evaluation (retrieval, CSLS, LFM), data loaders
scripts/              the method's entry points: train, prepare sources, carve, couple, fit the map
baselines/            the baselines and their shared scoring
experiments/images/   Table 1 from the result files
experiments/sphere/   the sphere experiment and Figure 3
experiments/lung/     the lung experiment, its tables and Figure 4
bash/                 one driver per experiment: images.sh, sphere.sh, lung.sh
configs/              the VAE configs and the sphere settings
```
