"""Artifact IO: atomic writes, a JSON sidecar per file, validated loads.

Every array or checkpoint written here appears complete or not at all, and gets a sidecar
`<file>.meta.json` with the caller's facts (mode, shapes, settings). `load_npy` can check
the sidecar against expected fields.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

SCHEMA = 1
log = logging.getLogger(__name__)


class ArtifactMismatch(RuntimeError):
    """Sidecar metadata conflicts with what the caller expected."""


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path: str) -> dict:
    """Path, content hash and size of an upstream input (for example a checkpoint)."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "sha256": sha256_file(path), "bytes": st.st_size}


def run_context() -> dict:
    return {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": os.path.basename(sys.argv[0]) if sys.argv else "unknown",
    }


def _atomic_write(path: str, writer) -> None:
    """writer(fh) -> None; the file appears at `path` complete or not at all."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            writer(fh)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def meta_path(path: str) -> str:
    return path + ".meta.json"


def write_meta(path: str, meta: dict | None) -> None:
    payload = {**run_context(), **(meta or {})}
    _atomic_write(meta_path(path),
                  lambda fh: fh.write(json.dumps(payload, indent=1, default=str).encode()))


def read_meta(path: str) -> dict | None:
    mp = meta_path(path)
    if not os.path.exists(mp):
        return None
    with open(mp) as fh:
        return json.load(fh)


def check_meta(path: str, expect: dict | None, strict: bool = False) -> dict | None:
    """Compare the keys in `expect` with the sidecar. A missing sidecar is a warning (an
    error when strict); a conflicting value raises ArtifactMismatch."""
    if not expect:
        return read_meta(path)
    meta = read_meta(path)
    if meta is None:
        msg = f"no sidecar for {path} (expected {expect})"
        if strict:
            raise ArtifactMismatch(msg)
        log.warning(msg)
        return None
    diffs = {k: (meta.get(k, "<absent>"), v) for k, v in expect.items()
             if json.loads(json.dumps(meta.get(k), default=str))
             != json.loads(json.dumps(v, default=str))}
    if diffs:
        lines = "\n".join(f"  {k}: sidecar={a!r}  expected={b!r}" for k, (a, b) in diffs.items())
        raise ArtifactMismatch(f"{path} is not what was asked for:\n{lines}")
    return meta


def save_npy(path: str, arr: np.ndarray, meta: dict | None = None, dtype=None) -> None:
    out = np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    _atomic_write(path, lambda fh: np.save(fh, out))
    write_meta(path, {"kind": "array", "shape": list(out.shape),
                      "dtype": str(out.dtype), **(meta or {})})


def load_npy(path: str, expect: dict | None = None, strict: bool = False) -> np.ndarray:
    check_meta(path, expect, strict=strict)
    return np.load(path)


def _class_path(obj) -> str:
    c = type(obj)
    return f"{c.__module__}.{c.__qualname__}"


def save_checkpoint(obj, path: str, payload: dict, meta: dict | None = None) -> None:
    """Atomic pickle with the object's class path embedded, plus a sidecar."""
    payload = {**payload, "class": _class_path(obj)}
    _atomic_write(path, lambda fh: pickle.dump(payload, fh))
    write_meta(path, {"kind": "checkpoint", "class": payload["class"], **(meta or {})})


def load_checkpoint(path: str, rngs=None, class_name: str | None = None,
                    expect: dict | None = None, strict: bool = False):
    """Construct the class recorded in the checkpoint and load the checkpoint into it.

    `class_name` is used when the checkpoint carries no class field.
    """
    check_meta(path, expect, strict=strict)
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    cls_path = payload.get("class") or class_name
    if cls_path is None:
        raise ArtifactMismatch(
            f"{path} has no embedded class and no class_name was given")
    from ngma.utils import load_obj
    cls = load_obj(cls_path)
    opts = payload.get("opts")
    if isinstance(opts, dict):
        kwargs = {"opts": opts}
    elif opts is not None:
        kwargs = {"opts": vars(opts) if hasattr(opts, "__dict__") else opts}
    else:
        kwargs = {}
    if rngs is None:
        from flax import nnx
        rngs = nnx.Rngs(0)   # construction randomness is overwritten by the restored state
    obj = cls(rngs=rngs, **kwargs)
    obj.load(path)
    return obj
