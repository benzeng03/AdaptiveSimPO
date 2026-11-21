"""Combine several run directories into one evaluation directory without re-scoring.

Creates OUT with a directory junction (Windows) or symlink per adapter, merges the
dev/test score files (the base model is taken from the first run), re-selects the
fixed-margin baseline on dev over all fixed adapters, and copies cached generations
from the runs so that `run_experiment.ps1 -RunDir OUT -EvalOnly` only generates what is
missing. All runs must use the same dev and test data (checked by pair count and the
base model's gaps).

A run can be given as PATH=SUFFIX to append SUFFIX to the names of its adapters (and of
their cached generations), e.g. to merge the same variant trained at several learning rates:

    python research/merge_runs.py outputs/run-A outputs/run-B --out outputs/run-AB
    python research/merge_runs.py outputs/run-lr1e-4 outputs/run-lr2e-4=-lr2e-4 --out outputs/run-lr

`--adapters` keeps only the listed adapters (names after any suffix), so the baseline is
selected among the fixed adapters on that list only:

    python research/merge_runs.py outputs/run-AB outputs/run-C --out outputs/run-ABC \
        --adapters fixed-g0 fixed-g1 fixed-g2 calibrated
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_pairs import select_fixed  # noqa: E402


def link_dir(src: Path, dst: Path):
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=True)


def parse_run(spec):
    path, _, suffix = spec.partition("=") if "=" in spec[2:] else (spec, "", "")
    if "." in suffix or "/" in suffix or "\\" in suffix:
        raise SystemExit(f"bad suffix {suffix!r}: no '.', '/' or backslash")
    return Path(path), suffix


def merge_scores(runs, split, keep=None):
    merged, base = {}, None
    for run, suffix in runs:
        path = run / f"{split}_scores.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["adapter"] == "base":
                if base is None:
                    base = r
                elif max(abs(a - b) for a, b in zip(base["gaps"], r["gaps"])) > 1e-3 or base["pairs"] != r["pairs"]:
                    raise SystemExit(f"{split}: base model gaps differ between runs (different data?)")
                continue
            r = dict(r, adapter=r["adapter"] + suffix, source=str(run))
            if keep is not None and r["adapter"] not in keep:
                continue
            if r["adapter"] in merged:
                raise SystemExit(f"{split}: adapter {r['adapter']} appears in more than one run")
            merged[r["adapter"]] = r
    return [base] + [merged[k] for k in sorted(merged)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories, optionally PATH=SUFFIX")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--adapters", nargs="+",
                   help="keep only these adapters (names after any suffix); default: all")
    a = p.parse_args(argv)
    a.runs = [parse_run(r) for r in a.runs]
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    keep = set(a.adapters) if a.adapters else None
    if keep:
        for split in ("dev", "test"):
            missing = keep - {r["adapter"] for r in merge_scores(a.runs, split, keep)}
            if missing:
                raise SystemExit(f"{split}: adapters not found in any run: {sorted(missing)}")
    a.out.mkdir(parents=True)
    for split in ("dev", "test"):
        rows = merge_scores(a.runs, split, keep)
        (a.out / f"{split}_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[merge] {split}: {len(rows) - 1} adapters + base")
    dev = json.loads((a.out / "dev_scores.json").read_text(encoding="utf-8"))
    best = select_fixed(dev)
    (a.out / "selection.json").write_text(json.dumps(dict(
        best_fixed=best, rule="max dev accuracy, then min dev logloss, then min gamma",
        merged_from=[f"{r}={s}" if s else str(r) for r, s in a.runs]), indent=2), encoding="utf-8")
    print(f"[merge] dev-selected fixed baseline over all runs: {best}")
    names = {r["adapter"] for r in dev if r["adapter"] != "base"}
    for run, suffix in a.runs:
        own = {d.name for d in run.iterdir() if d.is_dir() and d.name + suffix in names}
        for d in sorted(run.iterdir()):
            if d.is_dir() and d.name in own and not (a.out / (d.name + suffix)).exists():
                link_dir(d.resolve(), a.out / (d.name + suffix))
            elif d.is_dir() and d.name.startswith("generations-"):
                dst = a.out / d.name
                dst.mkdir(exist_ok=True)
                if (d / "prompts.json").exists() and not (dst / "prompts.json").exists():
                    shutil.copy2(d / "prompts.json", dst / "prompts.json")
                for f in d.glob("[0-9][0-9][0-9]-*.json"):   # cached prompt groups only
                    group, name = f.stem.split("-", 1)
                    if name != "base" and name not in own:
                        continue
                    target = dst / f"{group}-{name if name == 'base' else name + suffix}.json"
                    if not target.exists():
                        shutil.copy2(f, target)
    print(f"[merge] wrote {a.out}")


if __name__ == "__main__":
    main()
