"""
Lexicon-based measurement module for the 'Mirroring the User' study.

Constructs two parallel composite scores using the same instrument family,
applied symmetrically to user prompts and model responses:

- FRAMING SCORE on user prompts:
  Measures how socially the user frames the model (0-end = tool framing,
  high-end = social-actor framing). Combines pronouns, politeness markers,
  relational verbs, question/imperative structure, and Empath social-cluster
  categories.

- REGISTER SCORE on model responses:
  Measures how affiliatively the model responds (0-end = analytic register,
  high-end = affiliative register). Combines first-person and second-person
  pronouns, affect words, hedges, Empath social-cluster categories, and
  analytic markers (articles + prepositions, negatively loaded).

Each composite is the mean of signed z-scores over its component features,
following the construction logic of LIWC composite scores (e.g., Analytic,
Clout, Tone). The composites are computed across the entire corpus, so
within-corpus comparison is meaningful.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
import spacy
from empath import Empath


# -------------------- Lexicons --------------------

FIRST_PERSON_SG = {
    "i", "me", "my", "mine", "myself",
    "i'm", "i've", "i'd", "i'll",
}
FIRST_PERSON_PL = {
    "we", "us", "our", "ours", "ourselves",
    "we're", "we've", "we'd", "we'll",
}
SECOND_PERSON = {
    "you", "your", "yours", "yourself", "yourselves",
    "you're", "you've", "you'd", "you'll",
}

POLITENESS_MARKERS = {
    "please", "thanks", "thank", "kindly", "appreciate", "appreciated",
    "would", "could", "might", "may", "sorry", "pardon", "excuse",
}

# Cognitive / relational / volitional verbs - signal social-actor framing
RELATIONAL_VERBS = {
    "think", "thought", "feel", "felt", "want", "wanted", "wish", "hope",
    "believe", "wonder", "wondering", "guess", "suppose", "imagine",
    "help", "love", "like", "enjoy", "agree", "disagree", "remember",
    "understand", "know", "knows", "mean", "consider",
}

# Hedges / tentative markers - signal affiliative register
HEDGES = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "seem",
    "appears", "rather", "somewhat", "fairly", "kind", "sort", "sorta",
    "kinda", "probably", "likely", "guess", "suppose",
}

# Affect words: positive and negative emotion lexicon (basic)
AFFECT_WORDS = {
    "happy", "happily", "joy", "love", "loved", "wonderful", "great",
    "amazing", "excited", "delighted", "enjoy", "fun", "nice", "lovely",
    "sad", "sadly", "angry", "afraid", "fear", "worried", "anxious",
    "frustrated", "upset", "sorry", "unfortunate", "difficult", "hard",
    "hope", "care", "feel", "feels",
}

# Empath categories aggregated into the social-cluster signal
EMPATH_SOCIAL_CATS = [
    "help", "friends", "family", "love", "communication",
    "social_media", "affection", "positive_emotion", "negative_emotion",
    "trust",
]


# -------------------- Lazy loaders --------------------

@lru_cache(maxsize=1)
def _nlp():
    """Load spaCy model once per process. NER and lemmatizer disabled for speed."""
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


@lru_cache(maxsize=1)
def _empath():
    """Load Empath lexicon once per process."""
    return Empath()


# -------------------- Sentence-level structure --------------------

def _is_imperative(sent) -> bool:
    """Heuristic imperative detection: leading bare verb, no preceding subject.

    Returns True for sentences like 'Translate this' or 'Please summarize'.
    Does not flag interrogatives or declaratives.
    """
    toks = [t for t in sent if not t.is_punct and not t.is_space]
    if not toks:
        return False
    first = toks[0]
    # Imperative typically: leading bare verb (VB), no preceding subject
    if first.tag_ == "VB" and first.pos_ == "VERB":
        return True
    # 'Please <verb> ...' is also imperative
    if first.lower_ == "please" and len(toks) > 1 and toks[1].pos_ == "VERB":
        return True
    return False


# -------------------- Feature extraction --------------------

def _basic_counts(text: str) -> Dict[str, float]:
    """Token-level lexicon counts and structural features, normalized appropriately."""
    nlp = _nlp()
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    n = max(len(tokens), 1)

    lower_words = [t.lower_ for t in tokens]

    fp_sg = sum(1 for w in lower_words if w in FIRST_PERSON_SG)
    fp_pl = sum(1 for w in lower_words if w in FIRST_PERSON_PL)
    sp = sum(1 for w in lower_words if w in SECOND_PERSON)
    polite = sum(1 for w in lower_words if w in POLITENESS_MARKERS)
    rel = sum(1 for w in lower_words if w in RELATIONAL_VERBS)
    hedges = sum(1 for w in lower_words if w in HEDGES)
    affect = sum(1 for w in lower_words if w in AFFECT_WORDS)

    # Analytic markers: articles + prepositions, normalized.
    # Inspired by LIWC's Analytic composite, where these features index
    # formal/analytic register.
    articles = sum(1 for t in tokens if t.lower_ in {"a", "an", "the"})
    preps = sum(1 for t in tokens if t.pos_ == "ADP")
    analytic_raw = (articles + preps) / n

    # Sentence-level structure
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)
    n_questions = sum(1 for s in sents if "?" in s.text)
    n_imperatives = sum(1 for s in sents if _is_imperative(s))

    return {
        "n_tokens": float(n),
        "first_person_sg": fp_sg / n,
        "first_person_pl": fp_pl / n,
        "second_person": sp / n,
        "politeness": polite / n,
        "relational_verbs": rel / n,
        "hedges": hedges / n,
        "affect_words": affect / n,
        "analytic_raw": analytic_raw,
        "questions": n_questions / n_sents,
        "imperatives": n_imperatives / n_sents,
    }


def _empath_social(text: str) -> float:
    """Aggregated Empath score across social-cluster categories."""
    lex = _empath()
    res = lex.analyze(text, categories=EMPATH_SOCIAL_CATS, normalize=True)
    if res is None:
        return 0.0
    return float(sum(res.values()))


def extract_features(text: str) -> Dict[str, float]:
    """Extract all features for a single text. Returns a dict of feature values."""
    feats = _basic_counts(text)
    feats["empath_social"] = _empath_social(text)
    return feats


# -------------------- Composite construction --------------------

# Feature -> sign for FRAMING composite (applied to prompts).
# Positive sign = pulls composite toward 'social-actor framing'.
# Negative sign = pulls toward 'tool framing'.
FRAMING_WEIGHTS = {
    "first_person_sg":  +1.0,
    "second_person":    +1.0,
    "politeness":       +1.0,
    "relational_verbs": +1.0,
    "questions":        +1.0,
    "empath_social":    +1.0,
    "imperatives":      -1.0,
}

# Feature -> sign for REGISTER composite (applied to responses).
# Positive sign = pulls composite toward 'affiliative register'.
# Negative sign = pulls toward 'analytic/formal register'.
REGISTER_WEIGHTS = {
    "first_person_sg":  +1.0,
    "second_person":    +1.0,
    "affect_words":     +1.0,
    "hedges":           +1.0,
    "empath_social":    +1.0,
    "analytic_raw":     -1.0,
}


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize to zero mean, unit variance. Returns zeros if SD is zero."""
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def build_composite(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Construct a composite score as the mean of signed z-scores over weighted features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns referenced in `weights`.
    weights : Dict[str, float]
        Mapping from feature name to signed weight (+1 or -1).

    Returns
    -------
    pd.Series
        Composite score per row. Mean approximately zero, scale comparable across composites.
    """
    z_parts = []
    for feat, sign in weights.items():
        if feat not in df.columns:
            continue
        z_parts.append(sign * _zscore(df[feat]))
    if not z_parts:
        raise ValueError("No matching features found in df")
    stacked = pd.concat(z_parts, axis=1)
    return stacked.mean(axis=1)
