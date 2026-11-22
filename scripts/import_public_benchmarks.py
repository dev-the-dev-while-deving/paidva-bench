#!/usr/bin/env python3
"""
Public Benchmark Bridge Importer for Paidva-Bench.
Downloads standardized public datasets (SimpleQA Verified, FRAMES, GSM-Symbolic, SciFact)
via the HuggingFace Datasets Server API (no extra dependencies required) and converts them
into Paidva-Bench's unified JSONL schema with deterministic seeded sampling.

Usage:
  python3 scripts/import_public_benchmarks.py --benchmark simpleqa_verified --limit 1000
  python3 scripts/import_public_benchmarks.py --benchmark frames --limit 824
  python3 scripts/import_public_benchmarks.py --benchmark gsm_symbolic --limit 500
  python3 scripts/import_public_benchmarks.py --benchmark all
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"

SOURCES: dict[str, dict[str, str]] = {
    "simpleqa_verified": {
        "dataset": "google/SimpleQA-Verified",
        "config": "default",
        "split": "eval",
        "license": "CC-BY-4.0 / verify upstream",
    },
    "frames": {
        "dataset": "google/frames-benchmark",
        "config": "default",
        "split": "test",
        "license": "Apache-2.0 / verify upstream",
    },
    "gsm_symbolic": {
        "dataset": "apple/GSM-Symbolic",
        "config": "main",
        "split": "test",
        "license": "CC-BY-NC-4.0 / verify upstream",
    },
    "scifact": {
        "dataset": "allenai/scifact",
        "config": "claims",
        "split": "validation",
        "license": "CC-BY-2.0 / verify upstream",
    },
}


def fetch_hf_rows(dataset: str, config: str, split: str, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 100
    while len(rows) < max_rows:
        length = min(page, max_rows - len(rows))
        q = urllib.parse.urlencode({
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        })
        url = f"{HF_ROWS_API}?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": "paidva-bench/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break
    return rows


def extract_final_number(text: str) -> float | None:
    m = re.search(r"####\s*([-+]?[0-9][0-9,\.]*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"[-+]?[0-9][0-9,\.]*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def convert_benchmark(name: str, limit: int, seed: int, out_dir: Path) -> Path:
    spec = SOURCES[name]
    print(f"Fetching {name} ({spec['dataset']}, split={spec['split']}, up to {limit} rows)...")
    raw = fetch_hf_rows(spec["dataset"], spec["config"], spec["split"], limit)
    rng = random.Random(seed)
    if len(raw) > limit:
        raw = rng.sample(raw, limit)

    out_path = out_dir / f"{name}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    converted: list[dict[str, Any]] = []

    for idx, r in enumerate(raw):
        qid = f"{name}-{idx+1:04d}"
        if name == "simpleqa_verified":
            q = r.get("problem") or r.get("question") or ""
            ans = str(r.get("answer") or "").strip()
            topic = str(r.get("topic") or "general").lower()
            converted.append({
                "id": qid, "type": "factual", "category": "public_factuality",
                "domain": topic, "cluster_id": qid, "tier": "full",
                "question": q, "accept": [ans] if ans else [],
            })
        elif name == "frames":
            q = r.get("Prompt") or r.get("prompt") or r.get("question") or ""
            ans = str(r.get("Answer") or r.get("answer") or "").strip()
            converted.append({
                "id": qid, "type": "factual", "category": "public_multihop",
                "domain": "multi_hop", "cluster_id": qid, "tier": "full",
                "question": q, "accept": [ans] if ans else [],
            })
        elif name == "gsm_symbolic":
            q = r.get("question") or ""
            ans_num = extract_final_number(str(r.get("answer") or ""))
            if ans_num is None:
                continue
            converted.append({
                "id": qid, "type": "maths", "category": "public_math_robustness",
                "domain": "math", "cluster_id": qid, "tier": "full",
                "question": q, "answer": ans_num, "tolerance": max(1e-3, 1e-3 * abs(ans_num)),
            })
        elif name == "scifact":
            claim = r.get("claim") or ""
            label = str(r.get("evidence_label") or "NOT_ENOUGH_INFO").upper()
            q = (
                f"Evaluate the following scientific claim and state clearly whether it is "
                f"SUPPORTED, REFUTED, or NOT_ENOUGH_INFO, with brief evidence: {claim}"
            )
            accept = [label.lower(), label.replace("_", " ").lower()]
            if label == "SUPPORT":
                accept.append("supported")
            elif label == "CONTRADICT":
                accept.extend(["refuted", "contradicted", "false"])
            converted.append({
                "id": qid, "type": "factual", "category": "public_claim_verification",
                "domain": "science_climate", "cluster_id": qid, "tier": "full",
                "question": q, "accept": accept,
            })

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"// Imported public benchmark: {name} ({spec['dataset']}) | n={len(converted)} | license: {spec['license']}\n")
        for it in converted:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(converted)} items -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Import standardized public benchmarks into Paidva-Bench JSONL format.")
    ap.add_argument("--benchmark", choices=list(SOURCES.keys()) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/public"))
    args = ap.parse_args()

    targets = list(SOURCES.keys()) if args.benchmark == "all" else [args.benchmark]
    for name in targets:
        try:
            convert_benchmark(name, args.limit, args.seed, args.out_dir)
        except Exception as e:
            print(f"WARN: failed to import {name}: {e}")


if __name__ == "__main__":
    main()
