"""Typed configuration loaded from a YAML file + environment tokens.

Tokens are never stored in the YAML; they come from the environment
(``BASEROW_TOKEN`` / ``BIOHUB_API_TOKEN``), optionally via a local ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class BaserowConfig:
    base_url: str
    database_id: int
    table_id: int
    sequence_column: str
    id_column: str
    metadata_columns: list[str]
    output_columns: dict[str, str]


@dataclass
class ESMCConfig:
    model: str
    url: str
    pooling: str = "mean"
    max_sequence_length: int = 2048
    request_timeout: int = 300


@dataclass
class SAEConfig:
    models: list[str] = field(default_factory=list)
    normalize_features: bool = True
    top_k: int = 64  # features kept per protein in the Baserow/cache summary (tunable)
    pooling: str = "max"
    # When true, also store per-feature [start, end, peak] residue spans for the
    # top-K features (computed from the same per-residue activations the API
    # already returns — no extra Biohub cost; needs a re-run to populate a cache).
    residue_regions: bool = False
    # A feature's span = residues with activation >= region_threshold * its peak.
    region_threshold: float = 0.5
    # Persist the FULL pooled SAE activation vector (every non-zero feature) to a
    # local sparse float32 store OUTSIDE Baserow. Independent of ``top_k`` above,
    # which is the *Baserow* summary depth (the tunable knob); this store is
    # always the complete vector. Off by default — needs a re-run to populate
    # (per-residue activations aren't recoverable from the cache; set
    # run.skip_existing false, or run the `sae` step, to reprocess existing rows).
    store_full: bool = False
    # Where the store lives; defaults to ``<run.cache_dir>/sae_feature_matrix.npz``.
    feature_store_path: str | None = None
    # Drop pooled activations <= this when building the store record (dust).
    store_min_activation: float = 0.0


@dataclass
class RunConfig:
    batch_size: int = 16
    max_attempts: int = 10
    max_workers: int = 16  # concurrent Biohub requests PER TOKEN (<=64); pool = this × #tokens
    cache_dir: str = ".cache"
    write_baserow: bool = True
    use_cache: bool = True
    skip_existing: bool = True


@dataclass
class Config:
    name: str
    baserow: BaserowConfig
    esmc: ESMCConfig
    sae: SAEConfig
    run: RunConfig

    # Tokens resolved from the environment (not persisted in YAML).
    baserow_token: str = ""
    biohub_token: str = ""  # primary Biohub token (first of the pool; back-compat)
    biohub_tokens: list[str] = field(default_factory=list)  # full token pool

    @property
    def biohub_token_pool(self) -> list[str]:
        """All Biohub tokens to round-robin across (falls back to the single one)."""
        if self.biohub_tokens:
            return self.biohub_tokens
        return [self.biohub_token] if self.biohub_token else []

    @property
    def cache_path(self) -> Path:
        return Path(self.run.cache_dir)

    def require_tokens(self, *, baserow: bool = True, biohub: bool = True) -> None:
        """Raise a clear error if a needed token is missing."""
        missing = []
        if baserow and not self.baserow_token:
            missing.append("BASEROW_TOKEN")
        if biohub and not self.biohub_token_pool:
            missing.append("BIOHUB_API_TOKEN")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them in your shell or a local .env (see .env.example)."
            )


def _parse_token_list(*values: str) -> list[str]:
    """Split comma/space/newline-separated token strings into a deduped list."""
    import re

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for tok in re.split(r"[\s,]+", (value or "").strip()):
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def load_config(path: str | Path) -> Config:
    """Load a proteome config YAML and merge in environment tokens."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    # Biohub tokens: a pool via BIOHUB_API_TOKENS / ESM_API_KEYS (comma- or
    # whitespace-separated), plus the single BIOHUB_API_TOKEN / ESM_API_KEY.
    # Multi-token entries come first; the primary token is the pool's head.
    tokens = _parse_token_list(
        os.environ.get("BIOHUB_API_TOKENS", ""),
        os.environ.get("ESM_API_KEYS", ""),
        os.environ.get("BIOHUB_API_TOKEN", ""),
        os.environ.get("ESM_API_KEY", ""),
    )

    cfg = Config(
        name=raw["name"],
        baserow=BaserowConfig(**raw["baserow"]),
        esmc=ESMCConfig(**raw["esmc"]),
        sae=SAEConfig(**raw.get("sae", {})),
        run=RunConfig(**raw.get("run", {})),
        baserow_token=os.environ.get("BASEROW_TOKEN", ""),
        biohub_token=tokens[0] if tokens else "",
        biohub_tokens=tokens,
    )
    return cfg
