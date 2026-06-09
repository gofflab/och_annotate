"""Thin wrapper around the Biohub ESMC SDK.

Responsibilities:
  * build the remote ``esmc_client``
  * mean-pool a sequence -> one embedding vector (``embed_one``)
  * pull + reduce SAE feature activations -> top-K per protein (``sae_one``)

The ``esm`` SDK (and its torch dependency) is imported lazily so that config,
caching, Baserow and analysis modules stay usable without the heavy ML stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from och_annotate.config import Config


@dataclass
class TopFeatures:
    """Top-K SAE features for one protein from one SAE model."""

    indices: list[int]
    activations: list[float]

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        return {"indices": self.indices, "activations": self.activations}


def _to_list(tensor) -> list[float]:
    """torch.Tensor (or array-like) -> plain python float list."""
    try:
        return tensor.detach().cpu().float().tolist()
    except AttributeError:
        return [float(x) for x in tensor]


class ESMCEmbedder:
    """Wraps a remote ESMC inference client for embeddings + SAE features."""

    def __init__(self, config: Config):
        self.config = config
        self._client = None  # lazily constructed
        self._api = None     # cached module handle (ESMProtein, LogitsConfig, SAEConfig)

    # ---- lazy client -------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return
        from esm.sdk import esmc_client  # noqa: WPS433 (lazy heavy import)
        from esm.sdk.api import ESMProtein, LogitsConfig, SAEConfig

        self._api = {
            "ESMProtein": ESMProtein,
            "LogitsConfig": LogitsConfig,
            "SAEConfig": SAEConfig,
        }
        self.config.require_tokens(baserow=False, biohub=True)
        self._client = esmc_client(
            model=self.config.esmc.model,
            url=self.config.esmc.url,
            token=self.config.biohub_token,
            request_timeout=self.config.esmc.request_timeout,
        )

    # ---- helpers -----------------------------------------------------------
    def _logits(self, sequence: str, logits_config):
        """encode + logits for one sequence, with bounded retries."""
        self._ensure_client()
        ESMProtein = self._api["ESMProtein"]
        last_err: Exception | None = None
        for attempt in range(self.config.run.max_attempts):
            try:
                protein = ESMProtein(sequence=sequence)
                tensor = self._client.encode(protein)
                return self._client.logits(tensor, logits_config)
            except Exception as err:  # noqa: BLE001 - retry transient API errors
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"ESMC logits failed after {self.config.run.max_attempts} attempts: {last_err}"
        )

    # ---- public API --------------------------------------------------------
    def embed_one(self, sequence: str) -> list[float]:
        """Return the mean-pooled embedding vector for a single sequence."""
        self._ensure_client()
        LogitsConfig = self._api["LogitsConfig"]
        if self.config.esmc.pooling == "mean":
            cfg = LogitsConfig(sequence=True, return_mean_embedding=True)
            out = self._logits(sequence, cfg)
            if out.mean_embedding is not None:
                return _to_list(out.mean_embedding)
            # fall through to local pooling if server didn't return it
        cfg = LogitsConfig(sequence=True, return_embeddings=True)
        out = self._logits(sequence, cfg)
        emb = out.embeddings
        # embeddings: [1, L, d] including BOS/EOS -> drop specials, mean over L.
        pooled = emb[0, 1:-1, :].mean(dim=0)
        return _to_list(pooled)

    def sae_one(self, sequence: str) -> dict[str, TopFeatures]:
        """Return top-K SAE features per configured SAE model for one sequence."""
        self._ensure_client()
        sae_cfg = self.config.sae
        if not sae_cfg.models:
            return {}
        LogitsConfig = self._api["LogitsConfig"]
        SAEConfig = self._api["SAEConfig"]
        cfg = LogitsConfig(
            sequence=True,
            sae_config=SAEConfig(
                models=list(sae_cfg.models),
                normalize_features=sae_cfg.normalize_features,
            ),
        )
        out = self._logits(sequence, cfg)
        results: dict[str, TopFeatures] = {}
        for model_name, acts in (out.sae_outputs or {}).items():
            results[model_name] = self._top_features(acts)
        return results

    def _top_features(self, activations) -> TopFeatures:
        """Pool per-residue SAE activations over the sequence, take top-K."""
        import torch  # local import; only needed in the SAE path

        acts = activations
        if acts.dim() == 3:  # [1, L, F] -> [L, F]
            acts = acts[0]
        if acts.dim() == 2:  # [L, F] -> drop BOS/EOS, pool over residues
            acts = acts[1:-1]
            pooled = acts.max(dim=0).values if self.config.sae.pooling == "max" else acts.mean(dim=0)
        else:
            pooled = acts
        k = min(self.config.sae.top_k, pooled.numel())
        top = torch.topk(pooled, k)
        return TopFeatures(
            indices=[int(i) for i in top.indices.tolist()],
            activations=[float(v) for v in top.values.tolist()],
        )
