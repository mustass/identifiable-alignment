# Identifiable alignment of unpaired representations

Stas Syrota, Albert Lopez I Serrano, Johanne Franck, Søren Hauberg · Technical University of Denmark · NeurIPS 2026

**[Paper](https://syrota.me/files/ngma.pdf)** · **[Project page](https://syrota.me/identifiable-alignment/)**

Code for the paper. The project page walks through the theory and the results with interactive figures.

## What is where

```
src/ngma/             the package: VAEs and priors, losses, training, geometry (pullback metric,
                      segment lengths), alignment (GW, LDD-LP, maps), evaluation
scripts/              NGMA step by step: train a VAE, prepare sources, carve samples, couple, fit the map
baselines/            Rosetta, mini-vec2vec, GENOT-Q, RelReps, RelGeo, LFM and their scoring
experiments/images/   Table 1: MNIST, FashionMNIST, CelebA
experiments/sphere/   Figure 3: the sphere
experiments/lung/     Figure 4 and the lung table: scVI models of Tabula Sapiens lung cells
bash/                 one driver per experiment
configs/              VAE and sphere settings
```

## Reproducing the paper

```bash
pip install -e .        # Python >= 3.10, with JAX for your GPU
export PYTHONPATH=src
```

- **Table 1.** `bash bash/images.sh mnist` (likewise `fmnist` and `celeba`), then `bash bash/images.sh table`.
  MNIST and FashionMNIST download themselves. CelebA is read from
  `data/celeba/celeba_{train,val,test}_{images,labels}.npy` (aligned-and-cropped release, 64 × 64 × 3 in [0, 1]).
- **Figure 3.** `bash bash/sphere.sh`. CPU only, about a day at eight runs in parallel.
- **Figure 4 and the lung table.**
  `ROOT=/path/to/workdir SRC=/path/to/h5ad PY_SCVI=/path/to/scvi/python PY=python bash bash/lung.sh`.
  `SRC` holds `lung_train.h5ad` and `ts_lung_full.h5ad`, and `PY_SCVI` is a Python with scVI
  (`pip install -e '.[lung]'`).

The Rosetta and mini-vec2vec baselines run in their own environment (`pip install -e '.[rosetta]'`).
Every driver skips outputs that already exist. Reruns on other hardware agree with the paper to about 10⁻³ in R@1.

## Citation

```bibtex
@inproceedings{syrota2026identifiable,
  title     = {Identifiable alignment of unpaired representations},
  author    = {Syrota, Stas and Lopez I Serrano, Albert and Franck, Johanne and Hauberg, S{\o}ren},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by a research grant (42062) from VILLUM FONDEN. This project received
funding from the European Research Council (ERC) under the European Union's Horizon research and
innovation programme (grant agreement 101125993). The work was partly funded by the Novo Nordisk
Foundation through the Center for Basic Machine Learning Research in Life Science (NNF20OC0062606).
We acknowledge the DTU Computing Center for awarding this project access to the LUMI
supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI
consortium.
