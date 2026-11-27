"""Per-policy cache of summed log-probabilities on evaluation pairs.

train.py --eval-data scores the freshly trained model on each file before it exits, so that
score_pairs.py does not have to load every model again. An entry is keyed by the SHA-256 of the
data file and the encoding (max_length, max_prompt, chat_prompt, truncation); anything else is a miss.
"""
import hashlib
from pathlib import Path

import torch


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _path(policy_dir, sha):
    return Path(policy_dir) / "eval_cache" / f"{sha[:16]}.pt"


def save(policy_dir, data_path, encoding, sums):
    """sums = (chosen_sum, rejected_sum, chosen_len, rejected_len) over every pair of data_path."""
    sha = file_sha256(data_path)
    dest = _path(policy_dir, sha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    torch.save(dict(sha256=sha, data=str(data_path), encoding=list(encoding),
                    sums=[t.float().cpu() for t in sums]), tmp)
    tmp.replace(dest)
    return dest


def load(policy_dir, data_path, encoding):
    """The cached sums, or None when there is no entry for this file content and encoding."""
    sha = file_sha256(data_path)
    path = _path(policy_dir, sha)
    if not path.exists():
        return None
    entry = torch.load(path, map_location="cpu")
    if entry["sha256"] != sha or tuple(entry["encoding"]) != tuple(encoding):
        return None
    return entry["sums"]
