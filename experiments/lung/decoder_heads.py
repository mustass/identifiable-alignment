#!/usr/bin/env python
"""Give every fitted scVI model an ensemble of K = 8 decoders on its own latent chart.

scVI has one decoder. NGMA's metric is the expected pullback metric of an ensemble that shares
one chart, so for each model we keep its encoder frozen and add decoders:

    head 0      the scVI decoder itself, so M = 1 is plain scVI
    heads 1-7   fresh decoders of the same architecture, trained from different initialisations
                on codes sampled from the model's frozen posterior q(z|x) of its own training
                cells, z = mu + sigma * eps afresh every batch, with the negative-binomial
                likelihood at the model's gene dispersions and the observed library sizes (the
                reconstruction term of the ELBO with the encoder held fixed)

The first M heads give the M-decoder ensemble, so M = 1 and M = 8 are nested.

Per model, <root>/fits/{tag}/heads.npz holds, framework-neutral for the JAX geometry:
    W{k}_{l}, b{k}_{l}   head k, layer l = 0, 1, 2 (128 units, LayerNorm without affine, ReLU)
                         and l = 3 (the softmax scale layer, 3000 genes)
    theta                gene dispersions of the model
    l_ref                reference library size (median over all 10x cells), one for all models
    nll_eval             per-head NB negative log-likelihood on the eval cells, at their posterior means

    python experiments/lung/decoder_heads.py --root <root> [--only TAG,...] [--threads 8]
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

K, EPOCHS, BATCH, LR = 8, 150, 128, 1e-3


class Dec(nn.Module):
    """The scVI decoder's mean path: three hidden layers, LayerNorm without affine, softmax."""

    def __init__(self, d_in=20, hidden=128, n_genes=3000):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(d_in, hidden), nn.Linear(hidden, hidden),
                                  nn.Linear(hidden, hidden), nn.Linear(hidden, n_genes)])
        self.ln = nn.LayerNorm(hidden, elementwise_affine=False)

    def forward(self, z):
        h = z
        for lin in self.lin[:3]:
            h = torch.relu(self.ln(lin(h)))
        return torch.softmax(self.lin[3](h), dim=-1)


def nb_nll(scale, y, lib, theta):
    m = lib[:, None] * scale
    return -(torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
             + theta * (torch.log(theta) - torch.log(theta + m))
             + y * (torch.log(m + 1e-30) - torch.log(theta + m))).sum(1).mean()


def parent_head(model, d_in, n_genes):
    """Copy the scVI decoder's weights into Dec, refusing any layer this port does not cover."""
    dec = model.module.decoder
    fcl = dec.px_decoder.fc_layers
    head = Dec(d_in, 128, n_genes)
    layers = list(fcl.children())
    if len(layers) != 3:
        raise SystemExit(f"REFUSING: scVI decoder has {len(layers)} hidden layers, the port assumes 3")
    for i, layer in enumerate(layers):
        lin = [m for m in layer if isinstance(m, nn.Linear)]
        ln = [m for m in layer if isinstance(m, nn.LayerNorm)]
        bn = [m for m in layer if isinstance(m, nn.BatchNorm1d)]
        if len(lin) != 1 or len(ln) != 1 or bn or ln[0].elementwise_affine:
            raise SystemExit(f"REFUSING: decoder layer {i} is not Linear + LayerNorm(no affine)")
        head.lin[i].load_state_dict(lin[0].state_dict())
    head.lin[3].load_state_dict(dec.px_scale_decoder[0].state_dict())
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. H1_4900_s0")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import anndata as ad
    import scvi

    root = Path(a.root)
    meta = json.load(open(root / "split_meta.json"))
    split = np.load(root / "split.npz")
    A = ad.read_h5ad(root / "lung10x.h5ad")
    l_ref = float(meta["l_ref_median_library"])
    tags = a.only.split(",") if a.only else sorted(p.name for p in (root / "fits").iterdir()
                                                   if (p / "fit.json").exists())
    for tag in tags:
        d = root / "fits" / tag
        if (d / "heads.npz").exists():
            print(f"{tag} cached", flush=True)
            continue
        t0 = time.time()
        half, size = tag.split("_")[0], int(tag.split("_")[1])
        tr_idx, ev_idx = split[f"{half}_{size}"], split["eval"]
        tr = A[tr_idx].copy()
        model = scvi.model.SCVI.load(str(d / "model"), adata=tr)
        model.to_device("cpu")
        mu = np.load(d / "latents.npz")["mu"]
        theta = torch.as_tensor(np.load(d / "theta.npy"), dtype=torch.float32)
        n_genes, d_in = A.n_vars, mu.shape[1]

        def counts(idx):
            X = A.X[idx]
            Y = torch.as_tensor(np.asarray(X.todense()), dtype=torch.float32)
            return Y, Y.sum(1)

        Ytr, Ltr = counts(tr_idx)
        Yev, Lev = counts(ev_idx)
        Ztr = torch.as_tensor(mu[tr_idx], dtype=torch.float32)
        Str = torch.as_tensor(np.sqrt(np.load(d / "latents.npz")["s2"][tr_idx]), dtype=torch.float32)
        Zev = torch.as_tensor(mu[ev_idx], dtype=torch.float32)

        heads = [parent_head(model, d_in, n_genes)]
        with torch.no_grad():                    # the port must reproduce scVI's own px_scale
            nc = len(model.module.decoder.px_decoder.n_cat_list)   # scVI passes a batch index even
            cats = [torch.zeros(256, 1)] * nc                       # with one batch
            ref = model.module.decoder("gene", Zev[:256], torch.log(Lev[:256, None]), *cats)[0]
            err = float((heads[0](Zev[:256]) - ref).abs().max())
        if err > 1e-5:
            raise SystemExit(f"REFUSING: {tag} head 0 differs from scVI's decoder by {err:.2e}")

        n = len(Ztr)
        for k in range(1, K):
            torch.manual_seed(10_000 * (zlib.crc32(tag.encode()) % 1000) + k)
            dec = Dec(d_in, 128, n_genes)
            opt = torch.optim.Adam(dec.parameters(), lr=LR)
            for ep in range(EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, BATCH):
                    b = perm[i:i + BATCH]
                    z = Ztr[b] + Str[b] * torch.randn_like(Ztr[b])
                    loss = nb_nll(dec(z), Ytr[b], Ltr[b], theta)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            heads.append(dec)
            print(f"  {tag} head {k} trained", flush=True)

        out = {"theta": theta.numpy().astype(np.float64), "l_ref": np.float64(l_ref)}
        nll = []
        with torch.no_grad():
            for k, h in enumerate(heads):
                for l, lin in enumerate(h.lin):
                    out[f"W{k}_{l}"] = lin.weight.numpy().astype(np.float64)
                    out[f"b{k}_{l}"] = lin.bias.numpy().astype(np.float64)
                nll.append(float(nb_nll(h(Zev), Yev, Lev, theta)))
        out["nll_eval"] = np.array(nll)
        np.savez(d / "heads.npz", **out)
        rec = {"tag": tag, "head0_port_err": err, "nll_eval": nll, "secs": round(time.time() - t0, 1)}
        json.dump(rec, open(d / "heads.json", "w"), indent=1)
        print(json.dumps(rec), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
