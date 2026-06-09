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
5. **Explore** the whole proteome with UMAP + interactive plots, colored by any
   metadata column.

## Token frugality

Biohub calls are the limiting resource, so the pipeline:

- only embeds proteins **missing** an embedding (checks both cache and Baserow);
- **checkpoints** to cache every batch — an interrupted run resumes for free;
- offers `--dry-run` to report exactly how many proteins a real run would embed;
- offers `--limit N` to cap a first trial run.

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

# Override SAE settings per run (no need to edit the YAML)
och-annotate embed -c config/octopus_chierchiae.yaml --top-k 128
och-annotate embed -c config/octopus_chierchiae.yaml \
    --sae-model esmc-6b-2024-12-sae-layer60-k64-codebook65536 --top-k 100

# Top-K SAE features (after setting sae.models in the config)
och-annotate sae -c config/octopus_chierchiae.yaml

# UMAP exploration -> interactive HTML
och-annotate umap -c config/octopus_chierchiae.yaml --color chromosome --out umap.html

# Inspect Baserow columns
och-annotate fields -c config/octopus_chierchiae.yaml
```

Or use `notebooks/explore_proteome.ipynb` for interactive exploration.

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
