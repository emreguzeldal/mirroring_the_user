"""
Data pull and feature extraction pipeline for the 'Mirroring the User' study.

Streams multi-turn English conversations from the WildChat-1M corpus,
extracts the lexicon-based features defined in `measurement.py`, and
constructs the framing and register composites. Saves the analyzable
parquet to disk.

Usage
-----
    python run_production.py --target 20000 --out ./out

Output
------
    {out}/wildchat_raw.parquet        # raw turn-level rows after filtering
    {out}/multiturn_features.parquet  # raw + features + composites + lags

Runtime estimates (8-core laptop, multiprocessing enabled):
    - Data pull:        15-25 minutes for 20,000 conversations
    - Feature extract:  20-40 minutes for ~75,000 turn-level rows
    - Total:            ~1 hour
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Import the measurement module from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measurement import (
    extract_features,
    build_composite,
    FRAMING_WEIGHTS,
    REGISTER_WEIGHTS,
)


# -------------------- Step 1: Pull conversations --------------------

def pull_conversations(target_n: int, max_scan: int = 800_000) -> pd.DataFrame:
    """Stream WildChat-1M and collect multi-turn English conversations.

    Filters retained conversations to:
        - Primary language: English (per corpus metadata)
        - Not flagged as redacted
        - At least 2 valid (user, assistant) turn pairs
        - Prompt length 20-3000 chars, response length 50-5000 chars

    Returns a turn-level DataFrame with one row per (user, assistant) pair.
    """
    print(f"=== Pulling {target_n:,} conversations from WildChat-1M ===")
    ds = load_dataset("allenai/WildChat-1M", streaming=True, split="train")

    rows = []
    n_kept = 0
    n_scanned = 0
    last_log = time.time()
    model_counts = {}

    for r in ds:
        n_scanned += 1
        if n_scanned > max_scan:
            print(f"Hit max_scan={max_scan:,}, stopping early")
            break

        if r.get("language") != "English":
            continue
        if r.get("redacted"):
            continue
        if r.get("turn", 0) < 2:
            continue

        conv = r.get("conversation") or []
        if not conv:
            continue

        # Pair user -> assistant turns
        pairs = []
        i = 0
        while i < len(conv) - 1:
            if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
                u, a = conv[i], conv[i + 1]
                uc = u.get("content") or ""
                ac = a.get("content") or ""
                if 20 <= len(uc) <= 3000 and 50 <= len(ac) <= 5000:
                    pairs.append((uc, ac))
                i += 2
            else:
                i += 1

        if len(pairs) < 2:
            continue

        for ti, (p, resp) in enumerate(pairs):
            rows.append({
                "conversation_hash": r["conversation_hash"],
                "model": r["model"],
                "turn_idx": ti,
                "n_turns_in_conv": len(pairs),
                "prompt": p,
                "response": resp,
                "country": r.get("country"),
            })
        n_kept += 1
        model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1

        if time.time() - last_log > 30:
            print(f"  scanned={n_scanned:,}  kept={n_kept:,}  "
                  f"rows={len(rows):,}  models={model_counts}")
            last_log = time.time()

        if n_kept >= target_n:
            break

    print(f"\n  Scanned {n_scanned:,}, kept {n_kept:,} conversations, "
          f"{len(rows):,} turn-level rows")
    print(f"  Models: {model_counts}")
    return pd.DataFrame(rows)


# -------------------- Step 2: Feature extraction --------------------

_NLP = None


def _init_worker():
    """Worker initializer: load spaCy once per worker process."""
    global _NLP
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _featurize_one(text: str) -> dict:
    """Worker function that returns a feature dict for one text."""
    return extract_features(text)


def featurize_parallel(texts: list, n_workers: int = None, desc: str = "") -> pd.DataFrame:
    """Run feature extraction across worker processes."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print(f"  Featurizing {len(texts):,} texts on {n_workers} workers...")

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    feats = []
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for f in tqdm(
            pool.imap(_featurize_one, texts, chunksize=64),
            total=len(texts), desc=desc,
        ):
            feats.append(f)
    return pd.DataFrame(feats)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000,
                    help="Target number of conversations to keep")
    ap.add_argument("--out", type=str, default="./out",
                    help="Output directory")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of feature-extraction workers (default: cpu_count-1)")
    ap.add_argument("--max-scan", type=int, default=800_000,
                    help="Max rows to scan from WildChat (safety cap)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1) Pull
    raw = pull_conversations(args.target, max_scan=args.max_scan)
    raw.to_parquet(out / "wildchat_raw.parquet")
    print(f"\nSaved {out/'wildchat_raw.parquet'} ({len(raw):,} rows)")

    # 2) Featurize prompts and responses
    print("\n=== Extracting prompt features ===")
    pf = featurize_parallel(raw["prompt"].tolist(), args.workers, "prompts")
    pf = pf.add_prefix("p_")

    print("\n=== Extracting response features ===")
    rf = featurize_parallel(raw["response"].tolist(), args.workers, "responses")
    rf = rf.add_prefix("r_")

    # 3) Build composites
    print("\n=== Building composites ===")
    df = pd.concat([raw.reset_index(drop=True), pf, rf], axis=1)

    p_for = pf.rename(columns=lambda c: c[2:])
    r_for = rf.rename(columns=lambda c: c[2:])
    df["framing_score"] = build_composite(p_for, FRAMING_WEIGHTS)
    df["register_score"] = build_composite(r_for, REGISTER_WEIGHTS)
    df["model_version"] = np.where(df["model"].str.contains("gpt-4"), "gpt4", "gpt35")
    df["log_p_tokens"] = np.log1p(df["p_n_tokens"])
    df["log_r_tokens"] = np.log1p(df["r_n_tokens"])

    # 4) Lagged register and framing for autoregressive specifications
    df = df.sort_values(["conversation_hash", "turn_idx"]).reset_index(drop=True)
    df["lag_register"] = df.groupby("conversation_hash")["register_score"].shift(1)
    df["lag_framing"] = df.groupby("conversation_hash")["framing_score"].shift(1)

    # 5) Save
    df.to_parquet(out / "multiturn_features.parquet")
    print(f"\nSaved {out/'multiturn_features.parquet'} "
          f"({len(df):,} rows, {df.shape[1]} columns)")

    elapsed = time.time() - t0
    print(f"\n=== Pipeline finished in {elapsed/60:.1f} minutes ===")
    print(f"\nNext step:")
    print(f"  python run_regressions.py --in {out/'multiturn_features.parquet'}")


if __name__ == "__main__":
    main()
