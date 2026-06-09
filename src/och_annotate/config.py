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
    top_k: int = 64
    pooling: str = "max"
    # When true, also store per-feature [start, end, peak] residue spans for the
    # top-K features (computed from the same per-residue activations the API
    # already returns — no extra Biohub cost; needs a re-run to populate a cache).
    residue_regions: bool = False
    # A feature's span = residues with activation >= region_threshold * its peak.
    region_threshold: float = 0.5


@dataclass
class RunConfig:
    batch_size: int = 16
    max_attempts: int = 10
    max_workers: int = 16  # concurrent Biohub requests (<=64); lower is gentler/steadier
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
    biohub_token: str = ""

    @property
    def cache_path(self) -> Path:
        return Path(self.run.cache_dir)

    def require_tokens(self, *, baserow: bool = True, biohub: bool = True) -> None:
        """Raise a clear error if a needed token is missing."""
        missing = []
        if baserow and not self.baserow_token:
            missing.append("BASEROW_TOKEN")
        if biohub and not self.biohub_token:
            missing.append("BIOHUB_API_TOKEN")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them in your shell or a local .env (see .env.example)."
            )


def load_config(path: str | Path) -> Config:
    """Load a proteome config YAML and merge in environment tokens."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(
        name=raw["name"],
        baserow=BaserowConfig(**raw["baserow"]),
        esmc=ESMCConfig(**raw["esmc"]),
        sae=SAEConfig(**raw.get("sae", {})),
        run=RunConfig(**raw.get("run", {})),
        baserow_token=os.environ.get("BASEROW_TOKEN", ""),
        # The Biohub SDK also honors ESM_API_KEY; we accept either.
        biohub_token=os.environ.get("BIOHUB_API_TOKEN") or os.environ.get("ESM_API_KEY", ""),
    )
    return cfg
