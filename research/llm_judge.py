"""Blind pairwise judging of two policies' generations by a GPT model through the Codex CLI.

Reads the cached generation groups GEN/NNN-<name>.json written by generate.py and judges
--variant against --baseline on every prompt, twice: once in each answer order, in separate
calls, so a call never sees both orders of the same prompt. Within a pass the variant sits in
slot A on even positions and in slot B on odd ones. As in judge.py, the variant wins a prompt
only if both orders prefer it, loses if both prefer the baseline, and otherwise ties.
Win rate = (W + 0.5 T) / N with a 95% prompt-bootstrap interval.

Each call sends --batch prompts and asks for JSON verdicts constrained by --output-schema.
Codex runs read-only, without the project instructions (its working directory is an empty
temporary folder), and each batch result is cached under GEN/llm-judge-<model>-<effort>-<variant>-vs-<baseline>/ so an
interrupted run resumes.

    python research/llm_judge.py outputs/run-C/generations-ae128-seed42-g16 \
        --variant calibrated --baseline fixed-g2 --model gpt-6.2-sol --effort medium
"""
import argparse
import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def find_codex():
    """codex.exe inside the installed Codex desktop app, else `codex` on PATH. The package folder
    changes on every app update, and WindowsApps cannot be listed, so ask Get-AppxPackage."""
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-AppxPackage -Name OpenAI.Codex).InstallLocation"],
                             capture_output=True, text=True, timeout=60)
        for loc in filter(None, map(str.strip, res.stdout.splitlines())):
            exe = Path(loc) / "app" / "resources" / "codex.exe"
            if exe.exists():
                return str(exe)
    except (OSError, subprocess.SubprocessError):
        pass
    return "codex"


INSTRUCTIONS = """You are an expert evaluator of AI assistant responses. For each item below you get a user instruction and two responses, A and B, from two different assistants. Decide which response better follows the instruction: helpful, correct, relevant, coherent and complete. Judge the content, not the length: a longer response is better only if the extra content is useful. Ignore which response comes first. Answer "tie" only if the two are genuinely equal in quality.

Do not run any commands or read any files; judge from the text below only. Return one verdict per item, using the item's id.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "winner"],
                "properties": {"id": {"type": "integer"}, "winner": {"type": "string", "enum": ["A", "B", "tie"]}},
            },
        }
    },
}


def load_generations(gen: Path, name: str):
    out = {}
    for f in sorted(gen.glob(f"[0-9][0-9][0-9]-{name}.json")):
        for g in json.loads(f.read_text(encoding="utf-8")):
            out[g["id"]] = g
    if not out:
        raise SystemExit(f"no cached generations for {name} in {gen}")
    return out


def build_prompt(items):
    parts = [INSTRUCTIONS]
    for it in items:
        parts.append(f"\n=== Item id {it['id']} ===\n[Instruction]\n{it['instruction']}\n\n"
                     f"[Response A]\n{it['A']}\n\n[Response B]\n{it['B']}\n")
    return "".join(parts)


def call_codex(a, prompt, out_file: Path, schema_file: Path):
    with tempfile.TemporaryDirectory() as cwd:
        cmd = [a.codex, "exec", "-m", a.model, "-c", f'model_reasoning_effort="{a.effort}"',
               "-s", "read-only", "--ephemeral", "--skip-git-repo-check",
               "--output-schema", str(schema_file), "-o", str(out_file), "-"]
        res = subprocess.run(cmd, input=prompt.encode("utf-8"), cwd=cwd, capture_output=True, timeout=a.timeout)
    if res.returncode != 0 or not out_file.exists():
        raise RuntimeError(f"codex failed ({res.returncode}): {res.stderr.decode('utf-8', 'replace')[-500:]}")
    return json.loads(out_file.read_text(encoding="utf-8"))["verdicts"]


def run_batch(a, key, items, cache: Path, schema_file: Path):
    path = cache / f"{key}.json"
    if path.exists():
        return key, json.loads(path.read_text(encoding="utf-8"))
    raw = cache / f"{key}.raw.json"
    want = {it["id"] for it in items}
    for attempt in range(a.retries + 1):
        try:
            verdicts = call_codex(a, build_prompt(items), raw, schema_file)
            got = {v["id"]: v["winner"] for v in verdicts}
            if set(got) != want:
                raise RuntimeError(f"ids mismatch: missing {sorted(want - set(got))}, extra {sorted(set(got) - want)}")
            rec = [dict(it, winner=got[it["id"]], A=None, B=None) for it in items]
            rec = [{k: v for k, v in r.items() if k not in ("A", "B", "instruction")} for r in rec]
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[llm-judge] {key}: done", flush=True)
            return key, rec
        except Exception as e:  # noqa: BLE001 - report and retry any failure of one call
            print(f"[llm-judge] {key}: attempt {attempt + 1} failed: {e}", flush=True)
    raise SystemExit(f"[llm-judge] {key}: giving up")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gen", type=Path, help="generation directory with NNN-<name>.json groups")
    p.add_argument("--variant", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--model", default="gpt-6.2-sol")
    p.add_argument("--effort", default="medium")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--limit", type=int, help="judge only the first N prompts (for testing)")
    p.add_argument("--sample", type=int, help="judge a random subset of N prompts (drawn with --sample-seed)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--codex", default=None, help="codex executable (default: the installed Codex app)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.codex = a.codex or find_codex()
    a.gen = a.gen.resolve()          # codex runs in a temporary directory, so paths must be absolute

    var, base = load_generations(a.gen, a.variant), load_generations(a.gen, a.baseline)
    ids = sorted(set(var) & set(base))
    if a.limit:
        ids = ids[:a.limit]
    if a.sample:
        ids = sorted(random.Random(a.sample_seed).sample(ids, a.sample))
    tag = (f"-limit{a.limit}" if a.limit else "") + (f"-sample{a.sample}s{a.sample_seed}" if a.sample else "")
    # One cache per comparison: batch keys (pass1-batch000, ...) are only unique within a comparison.
    cache = a.gen / f"llm-judge-{a.model}-{a.effort}-{a.variant}-vs-{a.baseline}{tag}"
    cache.mkdir(exist_ok=True)
    schema_file = cache / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA), encoding="utf-8")

    order = ids[:]
    random.Random(a.seed).shuffle(order)          # batch composition independent of prompt id
    jobs = []
    for pas in (1, 2):
        items = []
        for pos, i in enumerate(order):
            variant_in_a = (pos % 2 == 0) == (pas == 1)
            v, b = var[i]["output"], base[i]["output"]
            items.append(dict(id=i, instruction=var[i]["instruction"], pass_=pas, variant_slot="A" if variant_in_a else "B",
                              A=v if variant_in_a else b, B=b if variant_in_a else v))
        for k in range(0, len(items), a.batch):
            jobs.append((f"pass{pas}-batch{k // a.batch:03d}", items[k:k + a.batch]))
    with ThreadPoolExecutor(a.workers) as ex:
        results = dict(ex.map(lambda j: run_batch(a, j[0], j[1], cache, schema_file), jobs))

    verdict = {1: {}, 2: {}}
    pos_a = []
    for key, rec in results.items():
        for r in rec:
            pas = r["pass_"]
            w = r["winner"]
            pos_a.append(w == "A")
            verdict[pas][r["id"]] = "tie" if w == "tie" else ("variant" if w == r["variant_slot"] else "baseline")
    scores, rows = [], []
    for i in ids:
        v1, v2 = verdict[1][i], verdict[2][i]
        s = 1.0 if v1 == v2 == "variant" else 0.0 if v1 == v2 == "baseline" else 0.5
        scores.append(s)
        rows.append(dict(id=i, pass1=v1, pass2=v2, score=s, variant_tokens=var[i]["tokens"], baseline_tokens=base[i]["tokens"]))
    s = np.array(scores)
    rng = np.random.default_rng(0)
    boot = rng.choice(s, size=(10000, len(s)), replace=True).mean(1)
    consistent = np.mean([verdict[1][i] == verdict[2][i] for i in ids])
    out = dict(judge=f"{a.model} ({a.effort}) via Codex CLI, both orders", variant=a.variant, baseline=a.baseline,
               n=len(s), wins=int(np.sum(s == 1)), ties=int(np.sum(s == 0.5)), losses=int(np.sum(s == 0)),
               win_rate=float(s.mean()), ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
               order_consistency=float(consistent), share_slot_A=float(np.mean(pos_a)), per_prompt=rows)
    (cache / f"summary_{a.variant}_vs_{a.baseline}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{a.variant} vs {a.baseline} ({out['judge']}): {out['wins']}/{out['ties']}/{out['losses']}, "
          f"win rate {100 * out['win_rate']:.2f}% (95% CI {100 * out['ci95'][0]:.2f}-{100 * out['ci95'][1]:.2f}), "
          f"order-consistent {100 * consistent:.1f}%, slot A chosen {100 * out['share_slot_A']:.1f}%")


if __name__ == "__main__":
    main()
