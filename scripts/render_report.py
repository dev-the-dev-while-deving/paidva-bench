#!/usr/bin/env python3
"""
Render a publication-ready Markdown + standalone HTML benchmark report
combining Paidva-Bench measured runs with published external baselines.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sys
from pathlib import Path
from typing import Any

# Reuse exact stats from bench.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402


def render_markdown(results_files: list[Path], baselines_path: Path | None) -> str:
    lines: list[str] = [
        "# Paidva-Bench Evaluation Report",
        "",
        "## 1. Measured Runs (In-Harness, Paired)",
        "",
        "| Run / System | n (graded) | Accuracy | 95% Wilson CI | Cluster SE | ECE | Brier | Top-50% Selective | Mean Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    loaded: list[tuple[str, list[dict[str, Any]]]] = []
    for p in results_files:
        rows = bench.load_results(p)
        loaded.append((p.stem, rows))
        s = bench.summarize_group(rows)
        lines.append(
            f"| `{p.stem}` | {s['n']} | **{s['acc']*100:.1f}%** | "
            f"{s['ci_lo']*100:.1f}–{s['ci_hi']*100:.1f}% | ±{s['cluster_se']*100:.1f} pts | "
            f"{s['ece']:.3f} | {s['brier']:.3f} | {s['selective_acc']*100:.1f}% | {s['mean_secs']:.1f}s |"
        )

    if loaded:
        name0, rows0 = loaded[0]
        by_dom: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rows0:
            if r.get("domain"):
                by_dom[r["domain"]].append(r)
        if by_dom:
            lines.extend([
                "",
                f"## 2. Domain Breakdown (`{name0}`)",
                "",
                "| Domain | n | Accuracy | 95% Wilson CI |",
                "| :--- | :---: | :---: | :---: |",
            ])
            for dom, g in sorted(by_dom.items()):
                gs = bench.summarize_group(g)
                lines.append(f"| {dom} | {gs['n']} | {gs['acc']*100:.1f}% | {gs['ci_lo']*100:.1f}–{gs['ci_hi']*100:.1f}% |")

    if baselines_path and baselines_path.exists():
        bdata = json.loads(baselines_path.read_text(encoding="utf-8"))
        lines.extend([
            "",
            "## 3. Published External Baselines (Not Re-Run in This Harness)",
            "",
            "> **Methodological Note:** Values below are official published numbers from papers/model cards (cited in `baselines/published_baselines.json`). Tooling parity differs where noted.",
            "",
            "| Model | Tooling Mode | SimpleQA Verified | FRAMES | GSM-Symbolic |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ])
        for m in bdata.get("models", []):
            sc = m.get("scores", {})
            def fmt(key: str) -> str:
                entry = sc.get(key, {})
                v = entry.get("value")
                if v is None:
                    return "—"
                return f"{v}% ({entry.get('unit', '')})"
            lines.append(
                f"| {m['model']} | {m['tooling_mode']} | {fmt('simpleqa_verified')} | {fmt('frames')} | {fmt('gsm_symbolic')} |"
            )

    return "\n".join(lines) + "\n"


def render_html(md_text: str) -> str:
    escaped = html.escape(md_text)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Paidva-Bench Report</title>"
        "<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#111}"
        "pre{background:#f6f8fa;padding:1.25rem;border-radius:8px;overflow-x:auto;font-size:0.9rem}</style>"
        "</head><body><h1>Paidva-Bench Report</h1><pre>" + escaped + "</pre></body></html>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Markdown and HTML benchmark report.")
    ap.add_argument("files", nargs="*", help="Result JSONL files")
    ap.add_argument("--baselines", type=Path, default=Path("baselines/published_baselines.json"))
    ap.add_argument("--md-out", type=Path, default=Path("REPORT.md"))
    ap.add_argument("--html-out", type=Path, default=Path("report.html"))
    args = ap.parse_args()

    md = render_markdown([Path(f) for f in args.files], args.baselines)
    args.md_out.write_text(md, encoding="utf-8")
    args.html_out.write_text(render_html(md), encoding="utf-8")
    print(f"Wrote {args.md_out} and {args.html_out}")


if __name__ == "__main__":
    main()
