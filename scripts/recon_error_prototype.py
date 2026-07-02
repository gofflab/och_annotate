"""Prototype: SAE reconstruction error (FVU) as a novelty signal for O. chierchiae.

One Biohub call per protein returns layer-60 hidden states; we run the official
ESMC SAE forward locally (transformers esmc_sae weights) to get per-residue
reconstruction error, aggregate per protein, and validate against an independent
novelty label vs the existing rarity composite.

Run with KMP_DUPLICATE_LIB_OK=TRUE.
"""
from __future__ import annotations
import os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from safetensors.torch import load_file

from och_annotate.config import load_config
from och_annotate.baserow import BaserowClient
from esm.sdk import esmc_client
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig, SAEConfig

REPO = "/Users/loyalgoff/repos/och_annotate"
SAE_PATH = "/tmp/saerecon/layer_60.safetensors"
N_SAMPLE = int(os.environ.get("N_SAMPLE", "400"))
WORKERS = int(os.environ.get("WORKERS", "16"))
OUT_CSV = os.path.join(REPO, "data/recon_error_sample.csv")
SEED = 0

cfg = load_config(os.path.join(REPO, "config/octopus_chierchiae.yaml"))
SAE_MODEL = cfg.sae.models[0]
TOKEN = cfg.biohub_token_pool[0]   # working 20k/day token only

# ---- SAE weights + official forward ----
W = load_file(SAE_PATH)
W_enc, W_dec, b_dec = W["W_enc"].float(), W["W_dec"].float(), W["b_dec"].float()
K = 64

def sae_metrics(h: torch.Tensor):
    """Official ESMC SAE forward on per-residue hidden states h:[L,d].
    Returns dict of per-protein aggregates of reconstruction error."""
    x = h - h.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-5)            # z-score per residue
    pre = F.relu((x - b_dec) @ W_enc)
    topk = torch.topk(pre, K, dim=-1)
    z = torch.zeros_like(pre).scatter(-1, topk.indices, topk.values)
    recon = z @ W_dec + b_dec
    res = recon - x
    sse = res.pow(2).sum(dim=-1)                            # per residue
    sst = x.pow(2).sum(dim=-1).clamp_min(1e-8)
    fvu = sse / sst                                         # per residue FVU
    mse = res.pow(2).mean(dim=-1)                           # official recon loss
    return {
        "fvu_mean": float(fvu.mean()),
        "fvu_max": float(fvu.max()),
        "fvu_p90": float(torch.quantile(fvu, 0.90)),
        "recon_loss_mean": float(mse.mean()),
        "n_res": int(h.shape[0]),
    }

# ---- pull sample from Baserow (read-only): id + sequence ----
print("Fetching rows from Baserow (read-only)...", flush=True)
br = BaserowClient(cfg.baserow.base_url, cfg.baserow_token, timeout=120)
rows = br.fetch_rows(
    cfg.baserow.table_id,
    fields=[cfg.baserow.id_column, cfg.baserow.sequence_column],
)
print(f"  got {len(rows)} rows", flush=True)

# join with novelty labels (orthogroup / missing_mouse / composite)
nov = pd.read_csv(os.path.join(REPO, "data/novelty_scores_saebasis.csv"))
nov = nov.drop_duplicates("transcript_id").set_index("transcript_id")

id_col, seq_col = cfg.baserow.id_column, cfg.baserow.sequence_column
pool = [(r[id_col], r[seq_col]) for r in rows
        if r.get(seq_col) and r.get(id_col) in nov.index]
print(f"  {len(pool)} rows have both a sequence and a novelty label", flush=True)

random.seed(SEED)
random.shuffle(pool)
sample = pool[:N_SAMPLE]
print(f"Sampling {len(sample)} proteins (seed={SEED}).", flush=True)

# ---- one Biohub call each: layer-60 hidden states ----
client = esmc_client(model=cfg.esmc.model, url=cfg.esmc.url, token=TOKEN,
                     request_timeout=cfg.esmc.request_timeout)
lcfg = LogitsConfig(sequence=True, return_hidden_states=True, ith_hidden_layer=60)

VALID = frozenset("ABCDEFGHIKLMNOPQRSTUVWXYZ-.:_|")
def sanitize(s): return "".join(c for c in "".join(s.split()).upper() if c in VALID)

def one(tid, seq):
    s = sanitize(seq)
    for attempt in range(4):
        try:
            t = client.encode(ESMProtein(sequence=s))
            if isinstance(t, ESMProteinError):
                raise RuntimeError(str(t))
            out = client.logits(t, lcfg)
            if isinstance(out, ESMProteinError):
                raise RuntimeError(str(out))
            h = out.hidden_states.reshape(-1, W_enc.shape[0]).float()
            h = h[1:-1]                                    # drop BOS/EOS
            m = sae_metrics(h)
            m["transcript_id"] = tid
            return m
        except Exception as e:
            if attempt == 3:
                return {"transcript_id": tid, "error": str(e)[:200]}
            time.sleep(min(2 ** attempt, 20))

t0 = time.time()
results, errs = [], 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(one, tid, seq) for tid, seq in sample]
    for i, f in enumerate(as_completed(futs)):
        r = f.result()
        if "error" in r:
            errs += 1
        else:
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} done, {errs} errs, "
                  f"{(time.time()-t0)/(i+1):.2f}s/protein", flush=True)
elapsed = time.time() - t0
print(f"Done: {len(results)} ok, {errs} errs in {elapsed:.0f}s "
      f"({elapsed/max(len(sample),1):.2f}s/protein wall).", flush=True)

df = pd.DataFrame(results)
df = df.merge(nov.reset_index()[[
    "transcript_id", "composite", "n_active", "idf_sum", "surprise_sum",
    "orthogroup", "singleton_og", "missing_mouse"]], on="transcript_id", how="left")
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}  ({len(df)} rows)", flush=True)
print(df[["fvu_mean", "fvu_max", "recon_loss_mean", "composite",
          "singleton_og", "missing_mouse"]].describe(include="all").to_string())
