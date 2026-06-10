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
    """Top-K SAE features for one protein from one SAE model.

    ``regions`` (optional) holds one ``[start, end, peak]`` residue span per
    top feature, aligned with ``indices`` — present only when residue-region
    extraction is enabled.
    """

    indices: list[int]
    activations: list[float]
    regions: list[list[int]] | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"indices": self.indices, "activations": self.activations}
        if self.regions is not None:
            out["regions"] = self.regions
        return out


def _residue_regions(acts2d, indices: list[int], threshold_frac: float) -> list[list[int]]:
    """Per-feature ``[start, end, peak]`` residue spans from a ``[L, F]`` matrix.

    ``acts2d`` is the per-residue activation matrix (BOS/EOS already dropped, so
    residue 0 is the first amino acid). For each feature in ``indices``: ``peak``
    is the argmax residue, and the span is the first/last residue whose activation
    reaches ``threshold_frac`` of that peak. Positions are 0-based.
    """
    import numpy as np

    acts = np.asarray(acts2d, dtype=float)
    regions: list[list[int]] = []
    for idx in indices:
        col = acts[:, idx]
        peak = int(col.argmax())
        peak_val = float(col[peak])
        if peak_val <= 0:
            regions.append([peak, peak, peak])
            continue
        active = np.where(col >= threshold_frac * peak_val)[0]
        regions.append([int(active.min()), int(active.max()), peak])
    return regions


# ESMC's accepted residue alphabet. Stop-codon markers ('*'), whitespace and any
# other stray characters are stripped before a sequence is sent, otherwise the
# server rejects the whole request with a 422 (and we'd burn a Biohub call).
_VALID_RESIDUES = frozenset("ABCDEFGHIKLMNOPQRSTUVWXYZ-.:_|")


def _sanitize_sequence(sequence: str) -> str:
    """Uppercase, drop whitespace, and remove characters ESMC rejects (e.g. ``*``)."""
    seq = "".join(sequence.split()).upper()
    return "".join(c for c in seq if c in _VALID_RESIDUES)


def _to_list(tensor) -> list[float]:
    """torch.Tensor (or array-like) -> flat plain-python float list.

    The server returns ``mean_embedding`` with leading singleton dims
    (e.g. ``[1, 1, d]``); flatten so we always store a 1-D ``list[float]``.
    """
    try:
        return tensor.detach().cpu().float().reshape(-1).tolist()
    except AttributeError:
        flat: list[float] = []
        for x in tensor:
            if isinstance(x, (list, tuple)):
                flat.extend(float(v) for v in x)
            else:
                flat.append(float(x))
        return flat


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
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig, SAEConfig

        self._api = {
            "ESMProtein": ESMProtein,
            "ESMProteinError": ESMProteinError,
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
    def _batch_executor(self):
        """Forge batch executor honoring ``run.max_workers`` (default SDK = 64).

        Fewer workers keeps a steadier request rate (the executor's AIMD limiter
        starts at this concurrency), which — together with a tight
        ``esmc.request_timeout`` — stops a few slow/stuck requests from freezing
        a whole chunk.
        """
        from esm.sdk import batch_executor  # ensures esm.sdk is initialised first

        max_workers = getattr(self.config.run, "max_workers", None)
        if max_workers:
            try:
                # Re-exported by esm.sdk; importing it here lets us set concurrency
                # (the public batch_executor() hardcodes 64 workers).
                from esm.sdk import ForgeBatchExecutor

                return ForgeBatchExecutor(
                    max_attempts=self.config.run.max_attempts,
                    max_workers=min(max_workers, 64),
                )
            except Exception:  # pragma: no cover - fall back to SDK default
                pass
        return batch_executor(max_attempts=self.config.run.max_attempts)

    def _logits(self, sequence: str, logits_config):
        """encode + logits for one sequence, with bounded retries."""
        self._ensure_client()
        ESMProtein = self._api["ESMProtein"]
        sequence = _sanitize_sequence(sequence)
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

    def embed_many(self, sequences: list[str]) -> list[list[float] | Exception]:
        """Mean-pool many sequences concurrently via the Forge batch executor.

        Returns one entry per input, in order: a ``list[float]`` embedding on
        success, or the ``ESMProteinError``/``Exception`` for sequences that
        failed after the executor's retries. The executor fans calls out across
        up to 64 workers with AIMD rate-limiting (auto back-off on 429/5xx), so
        the whole proteome embeds in minutes instead of hours.
        """
        self._ensure_client()
        ESMProtein = self._api["ESMProtein"]
        ESMProteinError = self._api["ESMProteinError"]
        LogitsConfig = self._api["LogitsConfig"]
        mean_cfg = LogitsConfig(sequence=True, return_mean_embedding=True)

        def _one(sequence: str):
            # One attempt; the executor handles retries on transient errors.
            tensor = self._client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
            if isinstance(tensor, ESMProteinError):
                return tensor
            out = self._client.logits(tensor, mean_cfg)
            if isinstance(out, ESMProteinError):
                return out
            if out.mean_embedding is not None:
                return _to_list(out.mean_embedding)
            # Server omitted the mean embedding: pool per-residue locally.
            out = self._client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
            if isinstance(out, ESMProteinError):
                return out
            return _to_list(out.embeddings[0, 1:-1, :].mean(dim=0))

        with self._batch_executor() as executor:
            return executor.execute_batch(_one, sequence=list(sequences))

    def embed_and_sae_many(
        self, sequences: list[str]
    ) -> list[dict[str, object] | Exception]:
        """Mean-pool embedding **and** top-K SAE features in a single pass.

        One ``logits`` call per protein returns both the mean embedding and the
        SAE activations, so getting both costs no more Biohub calls than the
        embedding alone. Returns one entry per input, in order: a dict
        ``{"embedding": list[float], "sae": {sae_model: {indices, activations}}}``
        on success, or the ``Exception`` for sequences that failed after retries.
        """
        self._ensure_client()
        ESMProtein = self._api["ESMProtein"]
        ESMProteinError = self._api["ESMProteinError"]
        LogitsConfig = self._api["LogitsConfig"]
        SAEConfig = self._api["SAEConfig"]
        sae_cfg = self.config.sae
        combined_cfg = LogitsConfig(
            sequence=True,
            return_mean_embedding=True,
            sae_config=SAEConfig(
                models=list(sae_cfg.models),
                normalize_features=sae_cfg.normalize_features,
            ),
        )

        def _one(sequence: str):
            tensor = self._client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
            if isinstance(tensor, ESMProteinError):
                return tensor
            out = self._client.logits(tensor, combined_cfg)
            if isinstance(out, ESMProteinError):
                return out
            if out.mean_embedding is not None:
                embedding = _to_list(out.mean_embedding)
            else:
                emb_out = self._client.logits(
                    tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                if isinstance(emb_out, ESMProteinError):
                    return emb_out
                embedding = _to_list(emb_out.embeddings[0, 1:-1, :].mean(dim=0))
            sae = {
                model_name: self._top_features(acts).as_dict()
                for model_name, acts in (out.sae_outputs or {}).items()
            }
            return {"embedding": embedding, "sae": sae}

        with self._batch_executor() as executor:
            return executor.execute_batch(_one, sequence=list(sequences))

    def sae_many(self, sequences: list[str]) -> list[dict[str, object] | Exception]:
        """Top-K SAE features for many sequences concurrently (no embedding).

        Used by the standalone ``sae`` back-fill step. Returns one entry per
        input: ``{sae_model: {indices, activations}}`` on success, or the
        ``Exception`` for sequences that failed after retries.
        """
        self._ensure_client()
        ESMProtein = self._api["ESMProtein"]
        ESMProteinError = self._api["ESMProteinError"]
        LogitsConfig = self._api["LogitsConfig"]
        SAEConfig = self._api["SAEConfig"]
        sae_cfg = self.config.sae
        cfg = LogitsConfig(
            sequence=True,
            sae_config=SAEConfig(
                models=list(sae_cfg.models),
                normalize_features=sae_cfg.normalize_features,
            ),
        )

        def _one(sequence: str):
            tensor = self._client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
            if isinstance(tensor, ESMProteinError):
                return tensor
            out = self._client.logits(tensor, cfg)
            if isinstance(out, ESMProteinError):
                return out
            return {
                model_name: self._top_features(acts).as_dict()
                for model_name, acts in (out.sae_outputs or {}).items()
            }

        with self._batch_executor() as executor:
            return executor.execute_batch(_one, sequence=list(sequences))

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
        if acts.is_sparse:  # SAE activations come back as a sparse COO tensor
            acts = acts.to_dense()
        if acts.dim() == 3:  # [1, L, F] -> [L, F]
            acts = acts[0]
        per_residue = None
        if acts.dim() == 2:  # [L, F] -> drop BOS/EOS, pool over residues
            acts = acts[1:-1]
            per_residue = acts  # keep for optional residue-region extraction
            pooled = acts.max(dim=0).values if self.config.sae.pooling == "max" else acts.mean(dim=0)
        else:
            pooled = acts
        k = min(self.config.sae.top_k, pooled.numel())
        top = torch.topk(pooled, k)
        indices = [int(i) for i in top.indices.tolist()]

        regions = None
        if getattr(self.config.sae, "residue_regions", False) and per_residue is not None:
            regions = _residue_regions(
                per_residue.detach().cpu().float().numpy(),
                indices,
                self.config.sae.region_threshold,
            )
        return TopFeatures(
            indices=indices,
            activations=[float(v) for v in top.values.tolist()],
            regions=regions,
        )
