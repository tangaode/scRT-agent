"""CLI entry point for scRT-agent figure mode."""

from __future__ import annotations

import argparse

from scrt_agent.figure_mode import build_publication_figure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="scRT-agent v2 figure mode: generate a publication-style multi-panel figure."
    )
    parser.add_argument("--rna-h5ad-path", required=True, help="Path to the RNA .h5ad file.")
    parser.add_argument("--tcr-path", required=True, help="Path to the TCR annotation table.")
    parser.add_argument("--output-dir", required=True, help="Output directory for figure artifacts.")
    parser.add_argument("--figure-name", default="scrt_publication_figure", help="Base name for figure outputs.")
    args = parser.parse_args()

    result = build_publication_figure(
        rna_h5ad_path=args.rna_h5ad_path,
        tcr_path=args.tcr_path,
        output_dir=args.output_dir,
        figure_name=args.figure_name,
    )
    print(f"PNG: {result.png_path}")
    print(f"PDF: {result.pdf_path}")
    print(f"Summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
