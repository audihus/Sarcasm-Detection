# -*- coding: utf-8 -*-
"""
pragmatic_features.py
Extracts 6 handcrafted pragmatic features from raw text.

Features (in order):
  0: log1p(exclamation_count)
  1: log1p(question_count)
  2: log1p(ellipsis_count)
  3: log1p(emoji_count)        -- requires `pip install emoji`
  4: caps_ratio                -- ALL-UPPERCASE tokens (len>=2) / total tokens
  5: log1p(char_repeat_count)  -- words with 3+ consecutive identical characters
"""
import os
import re
import math

import numpy as np

try:
    import emoji as _emoji_pkg
    _EMOJI_OK = True
except ImportError:
    _EMOJI_OK = False
    print("[pragmatic_features] WARNING: `emoji` package not found. "
          "Feature 3 (emoji_count) will always be 0. Install: pip install emoji")

_CHAR_REPEAT_RE = re.compile(r"(.)\1{2,}")


def extract_pragmatic_features(text: str) -> np.ndarray:
    """
    Extract 6 pragmatic features from raw text.

    Args:
        text: Raw input string (no preprocessing required).

    Returns:
        np.ndarray shape (6,), dtype float32.
    """
    tokens = text.split()
    n_tok  = max(len(tokens), 1)

    f0 = math.log1p(text.count("!"))
    f1 = math.log1p(text.count("?"))
    f2 = math.log1p(text.count("..."))

    if _EMOJI_OK:
        f3 = math.log1p(len(_emoji_pkg.emoji_list(text)))
    else:
        f3 = 0.0

    f4 = sum(1 for t in tokens if t.isupper() and len(t) >= 2) / n_tok

    f5 = math.log1p(
        sum(1 for w in tokens if _CHAR_REPEAT_RE.search(w))
    )

    return np.array([f0, f1, f2, f3, f4, f5], dtype=np.float32)


def compute_and_cache(texts: list, cache_path: str) -> np.ndarray:
    """
    Compute pragmatic features for a list of texts, with .npy disk cache.

    Args:
        texts:      List of raw strings.
        cache_path: Path to .npy cache file. Created on first call.

    Returns:
        np.ndarray shape (N, 6), dtype float32.
    """
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        return arr.astype(np.float32)

    print(f"[pragmatic] Computing features for {len(texts):,} texts...")
    features = np.stack([extract_pragmatic_features(t) for t in texts])
    np.save(cache_path, features)
    print(f"[pragmatic] Cached to: {cache_path}")
    return features
