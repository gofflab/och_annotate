"""Thin wrapper around the Biohub ESMC SDK.

Responsibilities:
  * build the remote ``esmc_client``
  * mean-pool a sequence -> one embedding vector (``embed_one``)
  * pull + reduce SAE feature activations -> top-K per protein (``sae_one``)

The ``esm`` SDK (and its torch dependency) is imported lazily so that config,
caching, Baserow and analysis modules stay usable without the heavy ML stack.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from och_annotate.config import Config


def _is_credit_limited(err: object) -> bool:
    """True if a failure is the Biohub daily credit-limit / rate cap (HTTP 429)."""
    msg = str(err).lower()
    return "credit limit" in msg or "usage cap" in msg


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
    # Store record: the FULL pooled vector (every non-zero feature), populated
    # only when ``sae.store_full`` is on. Kept OFF as_dict() so the Baserow/cache
    # summary stays the top_k subset; routed separately to the local store.
    store_indices: list[int] | None = None
    store_activations: list[float] | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"indices": self.indices, "activations": self.activations}
        if self.regions is not None:
            out["regions"] = self.regions
        return out

    def store_as_dict(self) -> dict[str, object] | None:
        if self.store_indices is None:
            return None
        return {"indices": self.store_indices, "activations": self.store_activations}


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
        self._client = None   # primary client (clients[0]); lazily constructed
        self._clients = None  # one client per Biohub token (round-robin pool)
        self._api = None      # cached module handle (ESMProtein, LogitsConfig, SAEConfig)
        self._exhausted: set[int] = set()  # token indices that hit their credit cap
        self._rr = 0          # round-robin cursor
        self._lock = threading.Lock()

    # ---- lazy client -------------------------------------------------------
    def _ensure_client(self):
        if self._clients is not None:
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
        tokens = self.config.biohub_token_pool
        # One client per token; requests are spread across them (so the daily
        # credit budget is the sum of all accounts) with failover on cap-out.
        self._clients = [
            esmc_client(
                model=self.config.esmc.model,
                url=self.config.esmc.url,
                token=token,
                request_timeout=self.config.esmc.request_timeout,
            )
            for token in tokens
        ]
        self._client = self._clients[0]
        if len(self._clients) > 1:
            per_token = min(getattr(self.config.run, "max_workers", 0) or 0, 64)
            print(f"Biohub: round-robining across {len(self._clients)} API tokens "
                  f"(with failover on credit-limit); up to {self._effective_max_workers()} "
                  f"concurrent requests ({per_token}/token).")

    # ---- multi-token routing ----------------------------------------------
    def _next_client(self):
        """Return ``(index, client)`` for the next non-exhausted token, or ``(-1, None)``."""
        with self._lock:
            n = len(self._clients)
            for _ in range(n):
                i = self._rr % n
                self._rr += 1
                if i not in self._exhausted:
                    return i, self._clients[i]
        return -1, None

    def _mark_exhausted(self, index: int) -> None:
        with self._lock:
            self._exhausted.add(index)

    def _call_with_failover(self, run):
        """Run ``run(client)`` on the next healthy token.

        On a credit-limit / usage-cap signal the token is retired from the pool
        and the error is raised so the batch executor retries the call on another
        token. Non-credit errors propagate unchanged (executor retries as before).
        """
        ESMProteinError = self._api["ESMProteinError"]
        index, client = self._next_client()
        if client is None:
            raise RuntimeError("All Biohub tokens reached their credit limit / usage cap.")
        try:
            result = run(client)
        except Exception as err:  # noqa: BLE001 - inspect, then re-raise for retry
            if _is_credit_limited(err):
                self._mark_exhausted(index)
            raise
        if isinstance(result, ESMProteinError) and _is_credit_limited(result):
            self._mark_exhausted(index)
            raise RuntimeError(f"Biohub credit limit on token #{index + 1}: {result}")
        return result

    # ---- helpers -----------------------------------------------------------
    def _effective_max_workers(self) -> int:
        """Concurrency for the executor: ``run.max_workers`` **per token**.

        ``max_workers`` is the per-token target (each token capped at 64, the SDK
        default); the executor pool auto-scales to that times the number of
        Biohub tokens, so adding tokens raises throughput without retuning.
        """
        per_token = min(getattr(self.config.run, "max_workers", 0) or 0, 64)
        n_tokens = max(len(self._clients or []), 1)
        return per_token * n_tokens

    def _batch_executor(self):
        """Forge batch executor sized by ``run.max_workers`` × token count.

        Fewer workers keeps a steadier request rate (the executor's AIMD limiter
        starts at this concurrency), which — together with a tight
        ``esmc.request_timeout`` — stops a few slow/stuck requests from freezing
        a whole chunk. With multiple tokens the pool scales up so each token
        carries its own ``max_workers`` share.
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
                    max_workers=self._effective_max_workers(),
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
            def run(client):
                tensor = client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
                if isinstance(tensor, ESMProteinError):
                    return tensor
                out = client.logits(tensor, mean_cfg)
                if isinstance(out, ESMProteinError):
                    return out
                if out.mean_embedding is not None:
                    return _to_list(out.mean_embedding)
                # Server omitted the mean embedding: pool per-residue locally.
                out = client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
                if isinstance(out, ESMProteinError):
                    return out
                return _to_list(out.embeddings[0, 1:-1, :].mean(dim=0))

            return self._call_with_failover(run)

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
            def run(client):
                tensor = client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
                if isinstance(tensor, ESMProteinError):
                    return tensor
                out = client.logits(tensor, combined_cfg)
                if isinstance(out, ESMProteinError):
                    return out
                if out.mean_embedding is not None:
                    embedding = _to_list(out.mean_embedding)
                else:
                    emb_out = client.logits(
                        tensor, LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    if isinstance(emb_out, ESMProteinError):
                        return emb_out
                    embedding = _to_list(emb_out.embeddings[0, 1:-1, :].mean(dim=0))
                sae, sae_store = self._split_sae(out.sae_outputs)
                return {"embedding": embedding, "sae": sae, "sae_store": sae_store}

            return self._call_with_failover(run)

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
            def run(client):
                tensor = client.encode(ESMProtein(sequence=_sanitize_sequence(sequence)))
                if isinstance(tensor, ESMProteinError):
                    return tensor
                out = client.logits(tensor, cfg)
                if isinstance(out, ESMProteinError):
                    return out
                sae, sae_store = self._split_sae(out.sae_outputs)
                return {"sae": sae, "sae_store": sae_store}

            return self._call_with_failover(run)

        with self._batch_executor() as executor:
            return executor.execute_batch(_one, sequence=list(sequences))

    def _split_sae(self, sae_outputs):
        """Reduce raw per-model activations to (top_k dict, store dict | None).

        ``top`` (always) is ``{model: {indices, activations}}`` for Baserow/cache,
        sized by ``sae.top_k``. ``store`` is the same shape over the FULL pooled
        vector (every non-zero feature), returned only when ``sae.store_full`` is
        set (else ``None``) — routed to the local feature store, never to Baserow.
        """
        enabled = getattr(self.config.sae, "store_full", False)
        top: dict[str, object] = {}
        store: dict[str, object] = {}
        for model_name, acts in (sae_outputs or {}).items():
            tf = self._top_features(acts)
            top[model_name] = tf.as_dict()
            sd = tf.store_as_dict()
            if sd is not None:
                store[model_name] = sd
        return top, (store if (enabled and store) else None)

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
        """Pool per-residue SAE activations over the sequence, take top-K.

        SAE activations arrive as a sparse COO ``[1, L, F]`` (or ``[L, F]``) with
        a huge codebook ``F`` (e.g. 16384). Densifying that full matrix costs up
        to ~134 MB per protein and, under concurrent workers, spikes memory hard
        enough to OOM the process. So in the default path we pool directly on the
        sparse tensor — only the length-``F`` pooled vector is ever dense. The
        full densify is kept solely for the optional residue-region path, and
        even then only the top-K feature columns are materialised.
        """
        import torch  # local import; only needed in the SAE path

        pool = self.config.sae.pooling
        residue_on = getattr(self.config.sae, "residue_regions", False)

        acts = activations
        if acts.is_sparse:
            acts = acts.coalesce()

        # Memory-light path: pool a sparse [.., L, F] over residues without
        # densifying the [L, F] matrix. SAE activations are non-negative
        # (post-ReLU / k-sparse), so an amax/sum scatter against zeros matches
        # the dense max/mean semantics exactly.
        if acts.is_sparse and acts.dim() >= 2 and not residue_on:
            idx, vals = acts.indices(), acts.values()
            f_dim = acts.shape[-1]
            ridx, fidx = idx[-2], idx[-1]
            length = acts.shape[-2]
            keep = (ridx >= 1) & (ridx <= length - 2)  # drop BOS/EOS rows
            fidx, vals = fidx[keep], vals[keep]
            pooled = torch.zeros(f_dim, dtype=vals.dtype if vals.numel() else torch.float32)
            if vals.numel():
                if pool == "max":
                    pooled.scatter_reduce_(0, fidx, vals, reduce="amax", include_self=True)
                else:  # mean over residues (inactive features contribute zeros)
                    pooled.scatter_reduce_(0, fidx, vals, reduce="sum", include_self=True)
                    pooled = pooled / max(length - 2, 1)
            return self._topk(pooled, per_residue=None)

        # Dense path: already-dense input, or residue-region extraction is on.
        if acts.is_sparse:
            acts = acts.to_dense()
        if acts.dim() == 3:  # [1, L, F] -> [L, F]
            acts = acts[0]
        per_residue = None
        if acts.dim() == 2:  # [L, F] -> drop BOS/EOS, pool over residues
            acts = acts[1:-1]
            per_residue = acts  # keep for optional residue-region extraction
            pooled = acts.max(dim=0).values if pool == "max" else acts.mean(dim=0)
        else:
            pooled = acts
        return self._topk(pooled, per_residue=per_residue if residue_on else None)

    def _topk(self, pooled, *, per_residue) -> TopFeatures:
        """Take top-K of a pooled [F] vector; optionally extract residue spans."""
        import torch

        k = min(self.config.sae.top_k, pooled.numel())
        top = torch.topk(pooled, k)
        indices = [int(i) for i in top.indices.tolist()]

        regions = None
        if per_residue is not None:
            regions = _residue_regions(
                per_residue.detach().cpu().float().numpy(),
                indices,
                self.config.sae.region_threshold,
            )

        # Store record: the FULL pooled vector (every non-zero feature above the
        # dust threshold) for the out-of-Baserow store. Independent of top_k.
        store_indices = store_activations = None
        if getattr(self.config.sae, "store_full", False):
            min_act = getattr(self.config.sae, "store_min_activation", 0.0)
            nz = torch.nonzero(pooled > min_act, as_tuple=False).flatten().tolist()
            store_indices = [int(i) for i in nz]
            store_activations = [float(pooled[i]) for i in nz]

        return TopFeatures(
            indices=indices,
            activations=[float(v) for v in top.values.tolist()],
            regions=regions,
            store_indices=store_indices,
            store_activations=store_activations,
        )
