"""Command-line interface for och_annotate.

    och-annotate embed -c config/octopus_chierchiae.yaml --dry-run
    och-annotate embed -c config/octopus_chierchiae.yaml
    och-annotate sae   -c config/octopus_chierchiae.yaml
    och-annotate umap  -c config/octopus_chierchiae.yaml --color chromosome --out umap.html
"""

from __future__ import annotations

import argparse
import json
import sys

from och_annotate.config import load_config


def _summary(title: str, data: dict) -> None:
    print(f"\n{title}")
    for key, value in data.items():
        if key == "errors":
            if value:
                print(f"  errors ({len(value)}): showing up to 5")
                for e in value[:5]:
                    print(f"    - {e}")
            continue
        print(f"  {key}: {value}")


def _apply_overrides(cfg, args) -> None:
    """Let CLI flags override SAE / run settings from the config YAML."""
    if getattr(args, "top_k", None) is not None:
        cfg.sae.top_k = args.top_k
    if getattr(args, "sae_model", None):
        cfg.sae.models = list(args.sae_model)
    if getattr(args, "no_sae", False):  # embed-only variant: skip SAE this run
        cfg.sae.models = []
    if getattr(args, "max_workers", None) is not None:
        cfg.run.max_workers = args.max_workers


def cmd_embed(args) -> int:
    from och_annotate.pipeline import EmbeddingPipeline

    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    pipeline = EmbeddingPipeline(cfg)
    summary = pipeline.run(dry_run=args.dry_run, limit=args.limit)
    _summary("Embedding summary" + (" (dry run)" if args.dry_run else ""), summary.as_dict())
    return 0


def cmd_sae(args) -> int:
    from och_annotate.sae import SAEPipeline

    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    summary = SAEPipeline(cfg).run(limit=args.limit)
    _summary("SAE summary", summary.as_dict())
    return 0


def cmd_umap(args) -> int:
    from och_annotate.analysis import load_embeddings, plot_umap, run_umap, save_html

    cfg = load_config(args.config)
    df = load_embeddings(cfg, prefer_cache=not args.from_baserow)
    if df.empty:
        print("No embeddings found (run `embed` first).", file=sys.stderr)
        return 1
    print(f"Loaded {len(df)} embeddings; running UMAP...")
    coords = run_umap(
        df,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
    )
    if args.out:
        fig = plot_umap(coords, color=args.color, title=f"{cfg.name} UMAP")
        save_html(fig, args.out)
        print(f"Wrote interactive plot to {args.out}")
    else:
        print(coords[["umap_0", "umap_1"]].describe().to_string())
        print("Pass --out plot.html to save an interactive figure.")
    return 0


def cmd_fields(args) -> int:
    from och_annotate.baserow import BaserowClient

    cfg = load_config(args.config)
    cfg.require_tokens(baserow=True, biohub=False)
    client = BaserowClient(cfg.baserow.base_url, cfg.baserow_token)
    names = sorted(client.field_names(cfg.baserow.table_id))
    print(json.dumps(names, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="och-annotate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p):
        p.add_argument("-c", "--config", required=True, help="Path to proteome config YAML")

    def add_sae_overrides(p):
        p.add_argument("--top-k", type=int, default=None,
                       help="Override sae.top_k: number of high-scoring SAE features kept per protein")
        p.add_argument("--sae-model", action="append", default=None, metavar="ID",
                       help="Override sae.models (repeatable), e.g. esmc-6b-2024-12-sae-layer60-k64-codebook16384")
        p.add_argument("--max-workers", type=int, default=None,
                       help="Override run.max_workers: concurrent Biohub requests (<=64)")

    p_embed = sub.add_parser("embed", help="Fetch sequences and write ESMC embeddings (+SAE if configured)")
    add_config(p_embed)
    p_embed.add_argument("--dry-run", action="store_true", help="Report counts without calling Biohub")
    p_embed.add_argument("--limit", type=int, default=None, help="Cap number of proteins embedded")
    p_embed.add_argument("--no-sae", action="store_true",
                         help="Embed-only: skip SAE even if sae.models is configured (cheaper run)")
    add_sae_overrides(p_embed)
    p_embed.set_defaults(func=cmd_embed)

    p_sae = sub.add_parser("sae", help="Back-fill top-K SAE features onto already-embedded proteins")
    add_config(p_sae)
    p_sae.add_argument("--limit", type=int, default=None)
    add_sae_overrides(p_sae)
    p_sae.set_defaults(func=cmd_sae)

    p_umap = sub.add_parser("umap", help="UMAP over stored embeddings")
    add_config(p_umap)
    p_umap.add_argument("--color", default=None, help="Metadata column to color points by")
    p_umap.add_argument("--out", default=None, help="Write interactive HTML to this path")
    p_umap.add_argument("--from-baserow", action="store_true", help="Ignore cache, load from Baserow")
    p_umap.add_argument("--n-neighbors", type=int, default=15)
    p_umap.add_argument("--min-dist", type=float, default=0.1)
    p_umap.add_argument("--metric", default="cosine")
    p_umap.set_defaults(func=cmd_umap)

    p_fields = sub.add_parser("fields", help="List columns of the Baserow table")
    add_config(p_fields)
    p_fields.set_defaults(func=cmd_fields)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
