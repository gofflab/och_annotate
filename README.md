# och_annotate

Annotate proteomes with **ESMC** (Biohub) protein-language-model embeddings and
**SAE** interpretability features, then explore the proteome with **UMAP**.

Built for *Octopus chierchiae* (Baserow database `206` / table `1026`,
`Transcripts`) but configuration-driven so additional proteomes are just another
YAML file.

## What it does

1. **Fetch** protein sequences + metadata (`gene_id`, `transcript_id`,
   `chromosome`, `start`, `end`, `strand`, `Ochierchiae_name`) from Baserow.
2. **Embed** each sequence with ESMC (default `esmc-6b-2024-12`) via the Biohub
   API, storing one **mean-pooled vector per protein**.
3. **Persist** the embedding back to Baserow (new columns) **and** a local
   parquet cache, so re-runs and analysis never re-spend Biohub tokens.
4. **SAE** (opt-in): when `sae.models` is set, the top-K high-scoring SAE
   features per protein are computed in the **same** Biohub call as the
   embedding (no extra token cost) and written to the `sae` column. The
   published SAEs for `esmc-6b-2024-12` are
   `esmc-6b-2024-12-sae-layer60-k64-codebook16384` (16k features, used in the
   Atlas) and `...-codebook65536` (65k, finer). The standalone `sae` command
   remains for back-filling SAE onto proteins embedded before it was enabled.
   `top_k` (the count written to Baserow) is the **tunable summary depth**
   (config `sae.top_k` / `--top-k`). Separately, set `sae.store_full: true` to
   persist the **full** pooled vector (every non-zero feature) to a local sparse
   `float32` `.npz` (`sae.feature_store_path`, default
   `data/sae_feature_matrix.npz`) kept **outside Baserow**. Needs a re-run to
   populate. Reload with `analysis.SaeFeatureStore(path).to_csr()`.
5. **Explore** the whole proteome with UMAP + interactive plots, colored by any
   metadata column.

## Token frugality

Biohub calls are the limiting resource, so the pipeline:

- only embeds proteins **missing** an embedding (checks both cache and Baserow);
- **checkpoints** to cache + Baserow every chunk (`run.batch_size`, default 32) —
  an interrupted or stalled run resumes for free and loses at most one chunk;
- runs requests concurrently but bounded (`run.max_workers`, default 16) with a
  tight per-request timeout (`esmc.request_timeout`), so a few slow/stuck
  requests can't freeze the run;
- offers `--dry-run` to report exactly how many proteins a real run would embed;
- offers `--limit N` to cap a first trial run, and `--no-sae` for a cheaper
  embed-only pass.

## Setup

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate och_annotate
```

Provide credentials (never committed) via the environment or a local `.env`
(copy from `.env.example`):

```bash
export BASEROW_TOKEN=...      # read + create-field/edit-row rights on table 1026
export BIOHUB_API_TOKEN=...   # ESMC API token (ESM_API_KEY also accepted)
```

> **Network note (Claude Code on the web):** the default web environment's
> network policy only allows pypi/npm/github. To run the live pipeline from a web
> session, recreate the environment with a custom egress allowlist that adds
> `baserow.gofflab.org` and `biohub.ai`. Locally / on HPC there is no such
> restriction.

## Usage

```bash
# Preview workload without spending any Biohub tokens
och-annotate embed -c config/octopus_chierchiae.yaml --dry-run

# Embed (resumable, writes to Baserow + local cache)
och-annotate embed -c config/octopus_chierchiae.yaml

# Try a handful first
och-annotate embed -c config/octopus_chierchiae.yaml --limit 10

# Embed-only variant (cheaper): skip SAE even if it's configured, then explore,
# then back-fill SAE later when the Biohub budget allows.
och-annotate embed -c config/octopus_chierchiae.yaml --no-sae
och-annotate umap  -c config/octopus_chierchiae.yaml --color chromosome --out umap.html
och-annotate sae   -c config/octopus_chierchiae.yaml      # back-fill SAE onto embedded proteins

# Override SAE / concurrency settings per run (no need to edit the YAML)
och-annotate embed -c config/octopus_chierchiae.yaml --top-k 128
och-annotate embed -c config/octopus_chierchiae.yaml --max-workers 8
och-annotate embed -c config/octopus_chierchiae.yaml \
    --sae-model esmc-6b-2024-12-sae-layer60-k64-codebook65536 --top-k 100

# Top-K SAE features (after setting sae.models in the config)
och-annotate sae -c config/octopus_chierchiae.yaml

# UMAP exploration -> interactive HTML
och-annotate umap -c config/octopus_chierchiae.yaml --color chromosome --out umap.html

# Color the UMAP by a single SAE feature's activation (0 where not in a protein's top-K)
och-annotate umap -c config/octopus_chierchiae.yaml --sae-feature 8523 --out feat8523.html

# Inspect Baserow columns
och-annotate fields -c config/octopus_chierchiae.yaml
```

Or use `notebooks/explore_proteome.ipynb` for interactive exploration, including
**Leiden clustering + SAE-feature enrichment** (proteins as "cells", SAE features
as "genes" — the scanpy marker-gene workflow):

```bash
pip install -e ".[cluster]"   # scanpy, leidenalg, igraph
```

Cluster on the ESMC embedding kNN graph (`build_anndata` → `sc.pp.neighbors(use_rep="X_esmc")`
→ `sc.tl.leiden`), then `sae_enrichment(adata)` ranks the SAE features enriched in
each cluster (Wilcoxon, FDR-corrected). Helpers live in `analysis.py`
(`build_anndata`, `sae_feature_matrix`, `sae_enrichment`).

## Project layout

```
config/                       # one YAML per proteome (Baserow ids, model, SAE, run opts)
src/och_annotate/
  config.py                   # typed config + env-token resolution
  baserow.py                  # Baserow REST client (read rows, ensure columns, write back)
  esmc.py                     # ESMC SDK wrapper: mean-pool embeddings + top-K SAE features
  cache.py                    # parquet embedding cache (idempotency)
  pipeline.py                 # fetch -> embed -> Baserow + cache (resumable, frugal)
  sae.py                      # separate SAE feature-extraction step
  analysis.py                 # load embeddings -> UMAP -> interactive plot
  cli.py                      # `och-annotate` command-line interface
notebooks/explore_proteome.ipynb
tests/                        # mocked unit tests (no live API / no torch needed)
```

## Running at scale (Biohub daily credit limit)

Biohub bills `logits` at `seq_len` tokens (1 credit = 10,000 tokens); `encode` is
free. An embedding and a **combined embedding+SAE** call cost the *same* (one
`logits` pass), so always run them together — splitting them doubles the cost.

For the *O. chierchiae* proteome (~35.8k proteins, mean 463 aa) the whole
embed+SAE job is **~1,657 credits, one-time**. On a 100-credit/day account that's
~16 days of daily resumes (~2,100 proteins/day); with a higher daily limit it's
hours.

The pipeline is built for this: each run resumes (skipping anything already done
in Baserow) and **stops early the moment the daily credit limit is hit**, so a
daily run is fast and safe. `.github/workflows/daily-embed.yml` runs it on a
daily cron — add `BASEROW_TOKEN` and `BIOHUB_API_TOKEN` as repository secrets,
and adjust the cron time to land just after your quota resets. Trigger it
manually (Actions → *Daily ESMC embed* → *Run workflow*) right after a limit bump
to drain the backlog at once.

### Multiple Biohub tokens

To finish the backlog faster, set **`BIOHUB_API_TOKENS`** to several tokens
(comma- or whitespace-separated — e.g. multiple accounts). The pipeline builds
one client per token and **round-robins requests across them**, so your effective
daily budget is the *sum* of the accounts. `run.max_workers` (default 16) is the
concurrency **per token**, and the executor pool **auto-scales to `max_workers ×
#tokens`** — so throughput rises with each token added, no retuning needed. When
a token hits its cap it is **retired from the pool** and the run continues on the
rest; only when *all* tokens are exhausted does it stop early. A single
`BIOHUB_API_TOKEN` still works exactly as before.

## Adding another proteome

Copy `config/octopus_chierchiae.yaml`, change `name`, the Baserow
`database_id`/`table_id`, the column names, and (optionally) the model. Then run
the same commands against the new config.

## Baserow columns written

| column             | type      | contents                                   |
| ------------------ | --------- | ------------------------------------------ |
| `esmc_embedding`   | long text | JSON `list[float]` mean-pooled vector      |
| `esmc_model`       | long text | model id that produced the embedding       |
| `esmc_embedded_at` | long text | ISO-8601 UTC timestamp                      |
| `sae_top_features` | long text | JSON `{sae_model: {indices, activations}}` |

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests run fully offline (esm/torch mocked)
```
