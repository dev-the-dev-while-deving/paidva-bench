#!/usr/bin/env python3
"""
Paidva-Bench — Flagship Real-World AI Research & Reasoning Benchmark CLI.
Model-agnostic runner, deterministic + rubric grader, dataset validator, and statistical reporter.

Works with any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Ollama, LM Studio, vLLM)
as well as the Paidva CLI binary (--provider paidva).

Examples:
  python3 bench.py validate datasets/paidva_rw_public.jsonl
  python3 bench.py run --dataset datasets/paidva_rw_public.jsonl --tier smoke \
      --base-url http://localhost:1234/v1 --model spark-x2.6-4b-mlx
  python3 bench.py report results/*.jsonl --baselines baselines/published_baselines.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------- Statistical Rigor (Miller 2024, Wilson CI, Exact McNemar, Cluster SE, Calibration) ----------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + (z * z) / n
    c = (p + (z * z) / (2.0 * n)) / d
    h = z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_adjusted_se(rows: list[dict[str, Any]]) -> float:
    """Cluster-adjusted standard error for binary correctness (Miller 2024, Eq. 5)."""
    n = len(rows)
    if n <= 1:
        return 0.0
    mean = sum(1.0 if r["correct"] else 0.0 for r in rows) / n
    by_cluster: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        cid = str(r.get("cluster_id") or r["id"])
        by_cluster[cid].append(1.0 if r["correct"] else 0.0)
    c_count = len(by_cluster)
    if c_count <= 1:
        return math.sqrt(mean * (1.0 - mean) / n)
    # Sum of squared cluster residual totals
    ss = sum((sum(v - mean for v in vals)) ** 2 for vals in by_cluster.values())
    var = (c_count / (c_count - 1)) * ss / (n * n)
    return math.sqrt(max(0.0, var))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = 0.0
    term = 0.5 ** n
    for i in range(k + 1):
        cdf += term
        term *= (n - i) / (i + 1)
    return min(1.0, 2.0 * cdf)


def calibration_metrics(pairs: list[tuple[float, bool]], bins: int = 5) -> tuple[float, float]:
    """Expected Calibration Error (ECE) over equal-width bins, and Brier score."""
    if not pairs:
        return (0.0, 0.0)
    n = len(pairs)
    brier = sum((c - (1.0 if y else 0.0)) ** 2 for c, y in pairs) / n
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [p for p in pairs if p[0] >= lo and (p[0] < hi or (b == bins - 1 and p[0] <= 1.0))]
        if not bucket:
            continue
        k = len(bucket)
        acc = sum(1 for _, y in bucket if y) / k
        conf = sum(c for c, _ in bucket) / k
        ece += (k / n) * abs(acc - conf)
    return (ece, brier)


# ---------- Dataset Loading & Validation ----------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("//"):
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return items


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    items = load_dataset(path)
    base = path.parent
    ids: set[str] = set()
    missing_files: list[str] = []
    by_domain: collections.Counter[str] = collections.Counter()
    by_cat: collections.Counter[str] = collections.Counter()
    by_tier: collections.Counter[str] = collections.Counter()
    errors: list[str] = []

    for it in items:
        iid = it.get("id")
        if not iid:
            errors.append("item missing id")
            continue
        if iid in ids:
            errors.append(f"duplicate id: {iid}")
        ids.add(iid)
        for req in ("type", "question", "domain", "category"):
            if not it.get(req):
                errors.append(f"{iid}: missing {req}")
        by_domain[it.get("domain", "unknown")] += 1
        by_cat[it.get("category", "unknown")] += 1
        by_tier[it.get("tier", "full")] += 1
        for rel in it.get("files", []):
            fp = base / rel
            if not fp.exists():
                missing_files.append(f"{iid}: missing {fp}")

    print(f"Dataset: {path} ({len(items)} items)")
    print("  By Domain:  ", dict(sorted(by_domain.items())))
    print("  By Category:", dict(sorted(by_cat.items())))
    print("  By Tier:    ", dict(sorted(by_tier.items())))
    if missing_files:
        print(f"  ERROR: {len(missing_files)} missing fixture files (first 5): {missing_files[:5]}")
        return 1
    if errors:
        print(f"  ERROR: {len(errors)} schema errors (first 5): {errors[:5]}")
        return 1
    print("  STATUS: VALID (0 errors, 0 missing fixtures)")
    return 0


# ---------- Deterministic & Heuristic Grading ----------

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def extract_numbers(s: str) -> list[float]:
    out: list[float] = []
    # Fractions like 13/18
    for a, b in re.findall(r"(?<![\w.])(-?\d+)\s*/\s*(\d+)(?![\w.])", s):
        bf = float(b)
        if bf != 0.0:
            out.append(float(a) / bf)
    # Standard decimals/numbers (strip commas, handle %)
    for tok, pct in re.findall(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(%?)", s):
        try:
            v = float(tok.replace(",", ""))
            out.append(v)
            if pct == "%":
                out.append(v / 100.0)
        except ValueError:
            pass
    return out


ABSTAIN_PATTERNS = [
    r"\bcannot be (?:answered|determined|known|verified|proven)\b",
    r"\bno (?:evidence|statistically significant|significant difference|data)\b",
    r"\bfalse premise\b",
    r"\bnot (?:possible|available|provided|found|statistically significant)\b",
    r"\bdo(?:es)? not exist\b",
    r"\bnever (?:won|happened|existed|occurred)\b",
    r"\bdid not win\b",
    r"\bunanswerable\b",
    r"\babstain\b",
    r"\bmissing file\b",
    r"\bcannot predict\b",
    r"\bfuture\b.*\bunknown\b",
    r"\bp\s*(?:>|>=|=)\s*0\.[1-9]",
]


def grade_item(q: dict[str, Any], answer: str, judge_fn: Any = None) -> tuple[bool, float, str]:
    kind = q["type"]
    if kind == "factual":
        na = norm_text(answer)
        accept = q.get("accept", [])
        ok = any(norm_text(a) in na for a in accept)
        return (ok, 1.0 if ok else 0.0, f"accept={accept}")

    if kind == "maths":
        key = float(q["answer"])
        tol = float(q.get("tolerance", max(1e-3, 1e-3 * abs(key))))
        inputs = [v for v in extract_numbers(q["question"]) if abs(v - key) > tol]
        cands = [v for v in extract_numbers(answer) if not any(abs(v - iv) < 1e-12 for iv in inputs)]
        ok = any(abs(v - key) <= tol for v in cands)
        return (ok, 1.0 if ok else 0.0, f"key={key} +/- {tol}")

    if kind == "abstain":
        if judge_fn is not None:
            ok, reason = judge_fn("abstain", q, answer)
            return (ok, 1.0 if ok else 0.0, f"judge: {reason}")
        na = norm_text(answer)
        ok = any(re.search(pat, na) for pat in ABSTAIN_PATTERNS)
        return (ok, 1.0 if ok else 0.0, "heuristic_abstain")

    if kind in ("research", "analysis"):
        rubric = q.get("rubric", [])
        if judge_fn is not None and rubric:
            met_flags, reason = judge_fn("rubric", q, answer)
            met = sum(1 for m in met_flags[:len(rubric)] if m)
            score = met / max(1, len(rubric))
            return (score >= 0.75, score, f"rubric {met}/{len(rubric)}: {reason}")
        # Fallback lexical rubric overlap if no judge model is configured
        na = norm_text(answer)
        hits = 0
        for item in rubric:
            kws = [w for w in re.findall(r"[a-z0-9]{4,}", item.lower()) if w not in {"that", "with", "from", "this", "notes", "states", "mentions", "cites", "about", "their"}]
            if kws and sum(1 for w in kws if w in na) >= max(1, len(kws) // 3):
                hits += 1
        score = hits / max(1, len(rubric))
        return (score >= 0.75, score, f"lexical_rubric {hits}/{len(rubric)}")

    return (False, 0.0, f"unknown type {kind}")


# ---------- OpenAI-Compatible & Paidva Execution ----------

SYSTEM_PROMPT = (
    "You are being evaluated on a real-world research benchmark. "
    "Answer accurately, concisely, and honestly. If a question cannot be answered, rests on a false premise, "
    "references missing data, or shows no statistically significant effect, explicitly state that and abstain. "
    "Return ONLY valid JSON with keys: "
    '{"answer": string, "confidence": float between 0 and 1, "citations": [list of URLs or file names used], "abstained": boolean}.'
)


def call_openai_compat(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: int = 120) -> tuple[str, int]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"] or ""
    toks = int(body.get("usage", {}).get("total_tokens", 0))
    return (text, toks)


def parse_model_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer" in obj:
                return {
                    "answer": str(obj.get("answer", "")),
                    "confidence": float(obj.get("confidence", 0.5)),
                    "citations": [str(c) for c in obj.get("citations", []) if isinstance(c, str)],
                    "abstained": bool(obj.get("abstained", False)),
                }
        except Exception:
            pass
    return {"answer": s, "confidence": 0.5, "citations": [], "abstained": False}


def cmd_run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    items = load_dataset(dataset_path)
    base_dir = dataset_path.parent

    # Filter by tier, domain, category
    tier_rank = {"smoke": 0, "dev": 1, "full": 2}
    max_rank = tier_rank[args.tier]
    todo = [
        q for q in items
        if tier_rank.get(q.get("tier", "full"), 2) <= max_rank
        and (not args.domain or q.get("domain") == args.domain)
        and (not args.category or q.get("category") == args.category)
    ]
    if args.limit > 0:
        todo = todo[: args.limit]

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out_path = Path(args.out) if args.out else Path("results") / f"{dataset_path.stem}-{args.tier}-{args.provider}-{safe_model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and out_path.exists():
        out_path.unlink()

    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("//"):
                continue
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["id"])
            except Exception:
                pass
    else:
        manifest = {
            "_manifest": True,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "tier": args.tier,
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        }
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(manifest) + "\n")

    pending = [q for q in todo if q["id"] not in done]
    print(f"Running {len(pending)} items ({len(done)} already completed) -> {out_path}")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or ""

    for idx, q in enumerate(pending, 1):
        t0 = time.time()
        file_context = ""
        for rel in q.get("files", []):
            fp = base_dir / rel
            if fp.exists():
                txt = fp.read_text(encoding="utf-8", errors="replace")[:16000]
                file_context += f"\n\n--- FILE: {fp.name} ---\n{txt}\n"

        row: dict[str, Any] = {
            "id": q["id"], "type": q["type"], "category": q.get("category", ""),
            "domain": q.get("domain", ""), "cluster_id": q.get("cluster_id", q["id"]),
            "provider": args.provider, "model": args.model,
            "answer": "", "confidence": 0.5, "abstained": False,
            "correct": None, "score": 0.0, "citations": [], "tokens": 0, "secs": 0.0,
            "note": "", "error": "",
        }
        try:
            if args.provider == "paidva":
                cmd = [args.paidva_bin, "ask", q["question"]]
                for rel in q.get("files", []):
                    cmd.extend(["--file", str(base_dir / rel)])
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or f"paidva exit {proc.returncode}")
                row["answer"] = proc.stdout.strip()
            else:
                msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
                prior_turns = q.get("turns", [])
                for t_idx, turn in enumerate(prior_turns):
                    content = turn + (file_context if t_idx == 0 else "")
                    msgs.append({"role": "user", "content": content})
                    reply, toks = call_openai_compat(args.base_url, api_key, args.model, msgs, args.timeout)
                    row["tokens"] += toks
                    msgs.append({"role": "assistant", "content": reply})
                final_user = q["question"] + (file_context if not prior_turns else "")
                msgs.append({"role": "user", "content": final_user})
                raw_reply, toks = call_openai_compat(args.base_url, api_key, args.model, msgs, args.timeout)
                row["tokens"] += toks
                parsed = parse_model_json(raw_reply)
                row["answer"] = parsed["answer"]
                row["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
                row["citations"] = parsed["citations"]
                row["abstained"] = parsed["abstained"]

            ok, sc, note = grade_item(q, row["answer"])
            row["correct"] = ok
            row["score"] = sc
            row["note"] = note
        except Exception as e:
            row["error"] = str(e)

        row["secs"] = round(time.time() - t0, 2)
        status = "OK" if row["correct"] else ("ERR" if row["error"] else "WRONG")
        print(f"  [{idx}/{len(pending)}] {q['id']} ({q.get('domain')}/{q.get('category')}) -> {status} ({row['secs']}s)")
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return 0


# ---------- Statistical Reporting ----------

def load_results(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("_manifest"):
            continue
        latest[r["id"]] = r
    return list(latest.values())


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r.get("correct") is not None]
    n = len(graded)
    k = sum(1 for r in graded if r["correct"])
    acc = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    cse = cluster_adjusted_se(graded)
    pairs = [(float(r.get("confidence", 0.5)), bool(r["correct"])) for r in graded]
    ece, brier = calibration_metrics(pairs)
    ranked = sorted(pairs, key=lambda x: x[0], reverse=True)
    top_half = ranked[: max(1, (len(ranked) + 1) // 2)]
    sel_acc = sum(1 for _, y in top_half if y) / len(top_half) if top_half else 0.0
    return {
        "n": n, "correct": k, "acc": acc, "ci_lo": lo, "ci_hi": hi,
        "cluster_se": cse, "ece": ece, "brier": brier, "selective_acc": sel_acc,
        "errors": sum(1 for r in rows if r.get("error")),
        "mean_secs": sum(float(r.get("secs", 0.0)) for r in rows) / max(1, len(rows)),
        "total_tokens": sum(int(r.get("tokens", 0)) for r in rows),
    }


def cmd_report(args: argparse.Namespace) -> int:
    loaded: list[tuple[str, list[dict[str, Any]]]] = []
    for p_str in args.files:
        p = Path(p_str)
        rows = load_results(p)
        loaded.append((p.name, rows))
        s = summarize_group(rows)
        print(f"\n=== {p.name} ({len(rows)} rows, {s['n']} graded, {s['errors']} errors) ===")
        print(
            f"  Overall: {s['acc']*100:.1f}% (95% Wilson CI: {s['ci_lo']*100:.1f}-{s['ci_hi']*100:.1f}%, "
            f"cluster SE: +/-{s['cluster_se']*100:.1f} pts)"
        )
        print(
            f"  Calibration: ECE={s['ece']:.3f} | Brier={s['brier']:.3f} | "
            f"Top-50% Selective Acc={s['selective_acc']*100:.1f}% | Mean {s['mean_secs']:.1f}s/q"
        )

        # By Category
        by_cat: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        by_dom: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by_cat[r.get("category") or r.get("type") or "unknown"].append(r)
            if r.get("domain"):
                by_dom[r["domain"]].append(r)

        print("  By Skill Category:")
        for cat, g in sorted(by_cat.items()):
            gs = summarize_group(g)
            print(f"    {cat:<22s} n={gs['n']:3d}  acc={gs['acc']*100:5.1f}%  (CI {gs['ci_lo']*100:4.1f}-{gs['ci_hi']*100:4.1f}%)")

        if by_dom:
            print("  By Domain:")
            for dom, g in sorted(by_dom.items()):
                gs = summarize_group(g)
                print(f"    {dom:<22s} n={gs['n']:3d}  acc={gs['acc']*100:5.1f}%  (CI {gs['ci_lo']*100:4.1f}-{gs['ci_hi']*100:4.1f}%)")

    # Paired exact McNemar comparison against first file
    if len(loaded) >= 2:
        name_a, rows_a = loaded[0]
        map_a = {r["id"]: r for r in rows_a if r.get("correct") is not None}
        for name_b, rows_b in loaded[1:]:
            map_b = {r["id"]: r for r in rows_b if r.get("correct") is not None}
            shared = set(map_a) & set(map_b)
            only_a = sum(1 for i in shared if map_a[i]["correct"] and not map_b[i]["correct"])
            only_b = sum(1 for i in shared if map_b[i]["correct"] and not map_a[i]["correct"])
            both = sum(1 for i in shared if map_a[i]["correct"] and map_b[i]["correct"])
            p_val = mcnemar_exact(only_a, only_b)
            print(f"\nPaired McNemar ({name_a} vs {name_b}, n={len(shared)} shared):")
            print(f"  both={both}, only_first={only_a}, only_second={only_b}, exact p={p_val:.4f}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench.py", description="Paidva-Bench Flagship Evaluation CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate dataset schema, counts, and fixture files")
    p_val.add_argument("dataset", help="Path to JSONL dataset")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="Run a model or agent on a dataset (resumable)")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--tier", choices=["smoke", "dev", "full"], default="smoke")
    p_run.add_argument("--domain", default=None)
    p_run.add_argument("--category", default=None)
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--provider", choices=["openai", "paidva"], default="openai")
    p_run.add_argument("--base-url", default="http://localhost:1234/v1")
    p_run.add_argument("--api-key", default="")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--paidva-bin", default="paidva")
    p_run.add_argument("--timeout", type=int, default=120)
    p_run.add_argument("--out", default=None)
    p_run.add_argument("--fresh", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Generate statistical scorecard + paired McNemar report")
    p_rep.add_argument("files", nargs="+", help="Result JSONL files")
    p_rep.add_argument("--baselines", default=None)
    p_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
