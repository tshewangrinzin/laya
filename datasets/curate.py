#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curate high-quality post-training datasets for Laya (typed decisions: choice / score / noul).

This is the data-processing engine behind `notebooks/laya_colab_post_training_rl.ipynb`.
It pulls *raw* rows from open sources (Hugging Face Hub, optionally Kaggle), runs a full
processing pipeline, and writes Laya-schema JSONL ready for SFT + RLCD post-training.

    python3 datasets/curate.py --sources all --out-dir datasets/curated
    python3 datasets/curate.py --demo                 # offline smoke test, no downloads
    python3 datasets/curate.py --sources typed_decisions,emotion --per-source-limit 4000

Pipeline stages (every stage is counted in the emitted `curated_report.json`):

    fetch        -> load raw rows from HF datasets / Kaggle / built-in demo generator
    catalog      -> per-choice-question label catalogs from dataset-wide frequencies; long
                    tails (e.g. CLINC's 150 intents) folded to "other" so every option stays
                    legible inside the model's head_max_len token budget
    clean        -> mojibake repair, HTML/entity stripping, PII scrubbing (URLs, e-mails,
                    long numbers), unicode/whitespace normalisation, signature removal
    filter       -> English heuristic, length bands, garbage/digit/repetition ratios
    label-norm   -> canonical label slugs, top-K + "other" folding, soft targets clipped away
                    from 0/1 and label-smoothed (strictly proper scoring: no raw one-hots)
    dedupe       -> exact hash + MinHash/LSH near-duplicate removal (word-shingle signatures)
    quality      -> per-case quality score; low-quality drop; low-margin -> "ambiguous"
    split        -> leakage-safe per-case splits: train / val / calib (temperature fitting) /
                    test; the source's own test split is forced into `test` for comparability
    balance      -> per-(question, catalog) label caps for TRAIN only; val/test untrimmed
    validate     -> optional laya build_sequence token-budget check with --validate-tokenizer-dir

Output schema (one JSON object per case; compatible with what the Laya training loop consumes
from `LocalLLaMA/typed-decisions`):

    {"id": "...", "workflow": "...", "source": "hf:...",
     "state": <json str>, "questions": <json str>, "gold": <json str {qid: {type,label,probabilities}}>,
     "factors": <json str>, "n_questions": 4, "quality": 0.91, "flags": [...], "split": "train"}

Only the Python standard library is required for `--demo`. `datasets` (pip install datasets)
is needed for HF sources; `kaggle` + ~/.kaggle/kaggle.json for the optional Kaggle sources.

Source licenses are listed in datasets/README.md; check each dataset card before redistributing
derivatives. The curated files are small text excerpts intended for research/fine-tuning use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------------------------
# Import hygiene. This file lives inside a folder literally named `datasets/`; run as a script
# (`python datasets/curate.py`) the script dir is prepended to sys.path and would shadow
# Hugging Face's `datasets` package with this very folder (namespace package). Drop our own
# directory from sys.path before any third-party import so `from datasets import ...`
# resolves to site-packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]


# ---------------------------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------------------------
HF_SOURCES = {
    "typed_decisions": dict(
        kind="hf", repo="LocalLLaMA/typed-decisions", config="all",
        train="train", test="test", license="see dataset card",
        builder="build_typed_decisions",
        note="official RLCD benchmark: 4 workflows, teacher-consensus soft labels",
    ),
    "typed_decisions_synth": dict(
        kind="hf", repo="n4ze3m/typed-decisions-synth", config="default",
        train="train", test="validation", license="mit",
        builder="build_typed_decisions_synth",
        note="synthetic typed decisions, 52 domains + 100 workflows, teacher soft labels",
    ),
    "ag_news": dict(
        kind="hf", repo="fancyzhx/ag_news", config="default",
        train="train", test="test", license="see dataset card",
        builder="build_ag_news", note="news topic classification (4 classes)",
    ),
    "emotion": dict(
        kind="hf", repo="dair-ai/emotion", config="default",
        train="train", test="test", license="see dataset card (SemEval-2018 Task 1)",
        builder="build_emotion", note="6-way emotion classification",
    ),
    "clinc_oos": dict(
        kind="hf", repo="clinc/clinc_oos", config="small",
        train="train", test="test", license="see dataset card",
        builder="build_clinc",
        note="CLINC150 assistant intents + out-of-scope; catalog folded to top-K + 'other'",
    ),
    "sms_spam": dict(
        kind="hf", repo="ucirvine/sms_spam", config="plain_text",
        train="train", test=None, license="unknown (academic collection)",
        builder="build_sms_spam", note="SMS spam/ham -> calibrated noul",
    ),
    "prompt_injections": dict(
        kind="hf", repo="deepset/prompt-injections", config="default",
        train="train", test="test", license="apache-2.0",
        builder="build_prompt_injections", note="prompt-injection yes/no -> noul (guardrails)",
    ),
    "sst5": dict(
        kind="hf", repo="SetFit/sst5", config="default",
        train="train", test="validation", license="see dataset card (derived from SST)",
        builder="build_sst5", note="5-way sentiment -> ordinal `score` question",
    ),
}

KAGGLE_SOURCES = {
    "support_tickets": dict(
        kind="kaggle", slug="waseemalastal/customer-support-ticket-dataset",
        license="see Kaggle page", builder="build_kaggle_tickets",
        note="~8k support tickets -> triage-style question pack (weak labels from metadata)",
    ),
    "customer_intent": dict(
        kind="kaggle", slug="scodepy/customer-support-intent-dataset",
        license="see Kaggle page", builder="build_kaggle_bitext",
        note="Bitext customer-service intents CSV -> catalog choice",
    ),
}


# ---------------------------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------------------------
_MOJIBAKE = re.compile(
    r"[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030"
    r"\u0160\u2039\u0152\u017d\u2018\u2019\u201c\u201d\u2022\u2013\u2014\u02dc\u2122\u0161\u203a"
    r"\u0153\u017e\u0178]")
_RE_HTML = re.compile(r"<[^>]{1,200}>")
_RE_CSS = re.compile(r"\s*(?:body|div|p|a|td|table)\s*\{[^}]*\}", re.I)
_RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_LONGNUM = re.compile(r"(?<!\d)\d{12,20}(?!\d)")
_RE_REPCHAR = re.compile(r"(.)\1{6,}")
_RE_REPW = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.I)
_RE_MULTINL = re.compile(r"\n{3,}")
_RE_MULTISP = re.compile(r"[ \t]{2,}")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_RE_SIG = re.compile(r"\n\s*(?:--|__)\s*\n.*$", re.I | re.S)
_RE_ENT = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|amp|lt|gt|quot|apos|nbsp);")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 payloads decoded as latin-1/cp1252 ('cafÃ©' -> 'café')."""
    if not _MOJIBAKE.search(text):
        return text
    for enc in ("latin-1", "cp1252"):
        try:
            cand = text.encode(enc, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not _MOJIBAKE.search(cand):
            return unicodedata.normalize("NFC", cand)
    return text


def clean_text(raw, *, scrub_pii: bool = True):
    """Return (cleaned_text, flags) for one free-text field."""
    flags = []
    if raw is None:
        return "", flags
    if isinstance(raw, (dict, list)):
        return raw, flags  # structural states are cleaned field-wise in clean_state()
    t = str(raw)
    if _MOJIBAKE.search(t):
        flags.append("fix:mojibake")
        t = fix_mojibake(t)
    if "<" in t and ">" in t:
        t2 = _RE_HTML.sub(" ", t)
        if t2 != t:
            flags.append("fix:html")
            t = t2
    if _RE_ENT.search(t):
        t = html.unescape(t).replace("\xa0", " ")
        flags.append("fix:entities")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = unicodedata.normalize("NFC", t)
    n0 = len(t)
    t = _RE_CTRL.sub("", t)
    if len(t) != n0:
        flags.append("fix:control-chars")
    t = _RE_CSS.sub(" ", t)
    if scrub_pii:
        t, n = _RE_URL.subn(" <url> ", t)
        flags += ["pii:url"] * bool(n)
        t, n = _RE_EMAIL.subn(" <email> ", t)
        flags += ["pii:email"] * bool(n)
        t, n = _RE_LONGNUM.subn(" <number> ", t)
        flags += ["pii:number"] * bool(n)
    t = _RE_SIG.sub("\n", t)
    t = _RE_REPCHAR.sub(lambda m: m.group(1) * 3, t)
    t = _RE_REPW.sub(lambda m: m.group(1), t)
    t = _RE_MULTINL.sub("\n\n", t)
    t = _RE_MULTISP.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n")).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t, flags


def clean_state(state, *, scrub_pii: bool = True):
    """Clean every text leaf of a state (str | dict | list), preserving structure."""
    flags = []
    if isinstance(state, str):
        return clean_text(state, scrub_pii=scrub_pii)
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            kk, fk = clean_text(k, scrub_pii=False)
            vv, fv = clean_state(v, scrub_pii=scrub_pii)
            out[kk] = vv
            flags += fk + fv
        return out, flags
    if isinstance(state, list):
        out_l = []
        for v in state:
            vv, fv = clean_state(v, scrub_pii=scrub_pii)
            out_l.append(vv)
            flags += fv
        return out_l, flags
    return state, flags


# ---------------------------------------------------------------------------------------------
# Language / quality heuristics (stdlib-only on purpose: runs anywhere, no model download)
# ---------------------------------------------------------------------------------------------
_EN_STOP = set(
    "the be to of and a in that have it for not on with he as you do at this but his by from "
    "they we say her she or an will my one all would there their what so up out if about who get "
    "which go me when make can like time no just him know take people into year your good some "
    "could them then now look only come over think also after use how our work first well way "
    "even new want because any these give day most us is are was were please thanks hi hello".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def is_englishish(text: str, min_stop: float = 0.06) -> bool:
    """Cheap English gate for the English checkpoint: Latin-script check + stopword coverage.

    Never perfect; it exists to keep obviously-foreign rows out of an English training mix.
    The multilingual checkpoint is fine to train on anything -- pass --min-stopwords 0 to disable.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    latinish = sum(1 for ch in letters if ch.isascii() or "\u00c0" <= ch <= "\u00ff")
    if latinish / len(letters) < 0.85:
        return False
    words = [w.lower() for w in _WORD.findall(text)]
    if min_stop <= 0 or len(words) < 4:
        return True
    return sum(1 for w in words if w in _EN_STOP) / len(words) >= min_stop


def quality_score(text: str, margin, n_flags: int, kind: str) -> float:
    """0..1 heuristic quality: length band + lexical diversity + teacher margin + cleanliness."""
    n = len(text)
    if kind == "text":
        if n < 20:
            return 0.0
        if n <= 320:
            s_len = math.log(max(20, n) / 20.0) / math.log(320.0 / 20.0)
        else:
            s_len = max(0.4, 1.0 - math.log(n / 320.0) / math.log(14.0))
        words = _WORD.findall(text)
        ttr = (len(set(words)) / max(1, len(words))) if len(words) >= 12 else 0.6
        base = 0.45 * min(1.0, s_len) + 0.2 * min(1.0, ttr / 0.55) + 0.15 * min(1.0, len(words) / 14.0)
    else:
        base = 0.7
    s_margin = 0.45 if margin is None else min(1.0, max(0.0, margin))
    s_clean = max(0.0, 1.0 - 0.06 * n_flags)
    return round(min(1.0, 0.55 * base + 0.25 * s_margin + 0.20 * s_clean), 4)


def text_of_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------------------------
_LABEL_OK = re.compile(r"[^a-z0-9]+")


def slugify(label) -> str:
    s = _LABEL_OK.sub("_", str(label).strip().lower()).strip("_")
    return (s or "label")[:48]


def soften(probs, keys, label: str, eps: float) -> dict:
    """Hard labels -> smoothed, soft labels -> clipped away from 0/1, renormalised to sum 1."""
    keys = list(keys)
    probs = probs if isinstance(probs, dict) else {}
    out = {k: max(0.0, float(probs.get(k, 0.0) or 0.0)) for k in keys}
    s = sum(out.values())
    if s <= 1e-9:
        out = {k: (1.0 - eps if k == label else eps / max(1, len(keys) - 1)) for k in keys}
        s = sum(out.values())
    out = {k: min(1.0 - eps, max(eps, v / s)) for k, v in out.items()}
    s = sum(out.values())
    return {k: round(v / s, 6) for k, v in out.items()}


def margin_of(probs: dict):
    """top1 - top2 probability: the ambiguity signal used for quality + calibration routing."""
    if not probs:
        return None
    v = sorted(probs.values(), reverse=True)
    return v[0] - (v[1] if len(v) > 1 else 0.0)


def fold_to_catalog(crit: dict, probs: dict, label: str, max_options: int,
                    counts: Counter, min_freq: int):
    """Fold the long tail of a choice catalog into 'other'.

    Laya splits a fixed head budget between options and document, so >12-option questions are
    trained and served at reduced quality at default settings (README: 'High-cardinality choice
    questions'). Curating to <= max_options keeps every label legible at head_max_len=192.
    """
    if len(crit) <= 1:
        return None, None, None, ["skip:catalog-pending"]
    keep, tail = [], []
    for k in crit:
        if k == "other":
            continue
        if counts.get(k, 0) < min_freq and len(crit) > max_options:
            tail.append(k)
        else:
            keep.append(k)
    if len(keep) + (1 if ("other" in crit or tail) else 0) > max_options:
        keep.sort(key=lambda k: -counts.get(k, 0))
        tail += keep[max(1, max_options - 1):]
        keep = keep[: max(1, max_options - 1)]
    flags = []
    crit_new = {k: crit[k] for k in keep}
    probs_new = {k: float(probs.get(k, 0.0) or 0.0) for k in keep}
    fold = sorted(set(tail)) + (["other"] if "other" in crit else [])
    if fold:
        p = sum(float(probs.get(k, 0.0) or 0.0) for k in fold)
        parts = [crit.get("other") or ""]
        if tail:
            parts.append("any of: " + ", ".join(sorted(tail))[:200])
        crit_new["other"] = " ; ".join(x for x in parts if x)[:250] or "none of the other options fits"
        probs_new["other"] = probs_new.get("other", 0.0) + p
        if label in fold:
            flags.append("fold:label->other")
            label = "other"
    if not crit_new:
        return None, None, None, ["drop:no-options"]
    return crit_new, probs_new, label, flags


def validate_questions(questions: dict):
    """Structural validity of the Laya question pack (types, criteria shape)."""
    errs = []
    if not questions:
        return ["drop:no-questions"]
    for qid, q in questions.items():
        t = q.get("type")
        if t == "choice":
            if not isinstance(q.get("criteria"), dict) or len(q["criteria"]) < 2:
                errs.append("bad-criteria:%s" % qid)
        elif t == "score":
            if not isinstance(q.get("criteria"), list) or len(q["criteria"]) < 3:
                errs.append("bad-criteria:%s" % qid)
        elif t != "noul":
            errs.append("bad-type:%s" % qid)
    return errs


# ---------------------------------------------------------------------------------------------
# Near-duplicate detection: pure-python MinHash over word shingles + LSH banding
# ---------------------------------------------------------------------------------------------
class MinHashIndex:
    def __init__(self, num_perm: int = 64, bands: int = 16, threshold: float = 0.85):
        self.rows = num_perm // bands
        self.num_perm, self.bands, self.threshold = num_perm, bands, threshold
        self.seeds_a = [zlib.crc32(b"laya-a%d" % i) & 0xFFFFFFFF for i in range(num_perm)]
        self.seeds_b = [zlib.crc32(b"laya-b%d" % i) & 0xFFFFFFFF for i in range(num_perm)]
        self.M = (1 << 61) - 1
        self.buckets = defaultdict(dict)   # band idx -> {band hash: [uid, ...]}
        self.sigs = {}                     # uid -> signature (kept entries only)
        self.kept = 0
        self.banned = 0

    @staticmethod
    def shingles(text: str, k: int = 4):
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        if not words:
            return ()
        if len(words) <= k:
            return (" ".join(words),)
        return tuple(" ".join(words[i: i + k]) for i in range(len(words) - k + 1))

    def signature(self, text: str):
        sh = self.shingles(text)
        if not sh:
            return None
        hs = [struct.unpack("<Q", hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest())[0]
              for s in sh]
        return [min((a * h + b) % self.M for h in hs) for a, b in zip(self.seeds_a, self.seeds_b)]

    def _jaccard_est(self, a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / float(len(a))

    def add(self, uid: str, text: str) -> bool:
        """Return True if kept (novel enough), False if a near-duplicate is already stored."""
        sig = self.signature(text)
        if sig is None:
            self.kept += 1
            return True
        cand = set()
        keys = []
        for bi in range(self.bands):
            band = tuple(sig[bi * self.rows:(bi + 1) * self.rows])
            key = zlib.crc32(repr(band).encode("utf-8")) & 0xFFFFFFFF
            keys.append((bi, key))
            cand.update(self.buckets[bi].get(key, ()))
        cand.discard(uid)
        for other in cand:
            if other in self.sigs and self._jaccard_est(sig, self.sigs[other]) >= self.threshold:
                self.banned += 1
                return False
        for bi, key in keys:
            self.buckets[bi].setdefault(key, []).append(uid)
        self.sigs[uid] = sig
        self.kept += 1
        return True


# ---------------------------------------------------------------------------------------------
# Canonical case + per-source builders (raw row -> canonical case)
# ---------------------------------------------------------------------------------------------
def make_case(uid, source, workflow, state, questions, gold, factors=None):
    return dict(id=uid, source=source, workflow=workflow, state=state,
                questions=questions, gold=gold, factors=factors or {},
                n_questions=len(questions))


def _pick(row, *names, default=None):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def _jload(v, default=None):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return default
    return v if v is not None else default


def build_typed_decisions(ds, split, src, lic, cfg):
    """LocalLLaMA/typed-decisions: state/questions/gold JSON with teacher-consensus probabilities."""
    for i, row in enumerate(ds):
        state = _jload(row.get("state"))
        questions = _jload(row.get("questions"))
        gold = _jload(row.get("gold"))
        if not (isinstance(state, (dict, list, str)) and isinstance(questions, dict)
                and isinstance(gold, dict) and questions and gold):
            continue
        yield make_case("%s:%s:%06d" % (src, split, i), src,
                        row.get("workflow", "typed_decisions"), state, questions, gold,
                        {"split_src": split})


def build_typed_decisions_synth(ds, split, src, lic, cfg):
    """n4ze3m/typed-decisions-synth: soft labels live in `teacher`, gold answers in `gold`."""
    for i, row in enumerate(ds):
        questions = _jload(row.get("questions"), {})
        gold_raw = _jload(row.get("gold"), {})
        teacher = _jload(row.get("teacher"), {})
        state = row.get("state")
        if row.get("state_is_json"):
            state = _jload(state, state)
        if not isinstance(questions, dict) or not questions:
            continue
        gold = {}
        for qid, q in questions.items():
            t = q.get("type")
            if t not in ("choice", "score", "noul"):
                continue
            tr = (teacher or {}).get(qid) or {}
            probs, g = None, (gold_raw or {}).get(qid)
            if isinstance(tr, dict) and isinstance(tr.get("probabilities"), dict):
                probs = tr["probabilities"]
            elif isinstance(tr, dict) and tr.get("noul") is not None:
                p = float(tr["noul"])
                probs = {"false": 1.0 - p, "true": p}
            if probs is None and g is not None:
                crit = q.get("criteria")
                if t == "choice":
                    keys = list((crit or {}).keys())
                elif t == "score":
                    keys = [str(j) for j in range(len(crit or []) or 4)]
                else:
                    keys = ["false", "true"]
                if str(g) in keys:
                    probs = {k: (1.0 if k == str(g) else 0.0) for k in keys}
            if not isinstance(probs, dict) or not probs:
                continue
            label = max(probs, key=probs.get)
            if t == "noul":
                label = "true" if float(probs.get("true", 0.0)) >= 0.5 else "false"
            gold[qid] = {"type": t, "label": str(label), "probabilities": probs}
        if not gold:
            continue
        yield make_case("%s:%s:%06d" % (src, split, i), src,
                        str(row.get("domain") or "synth"), state, questions, gold,
                        {"split_src": split, "teacher_model": row.get("teacher_model")})


_AG_DESC = {"world": "world news, politics, conflict, diplomacy",
            "sports": "games, teams, athletes, matches, scores",
            "business": "companies, markets, earnings, economy, mergers",
            "sci_tech": "science, technology, gadgets, software, space, medicine research"}


def build_ag_news(ds, split, src, lic, cfg):
    names = cfg.get("names") or ["World", "Sports", "Business", "Sci/Tech"]
    crit = {slugify(k): _AG_DESC.get(slugify(k)) for k in names}
    for i, row in enumerate(ds):
        text, lab = _pick(row, "text"), _pick(row, "label")
        if text is None or lab is None:
            continue
        if isinstance(lab, int):
            lab = names[lab] if 0 <= lab < len(names) else None
        key = slugify(lab) if lab is not None else None
        if key not in crit:
            continue
        q = {"intent": {"type": "choice", "instructions": "Which topic does `text` belong to?",
                        "criteria": dict(crit)}}
        g = {"intent": {"type": "choice", "label": key,
                        "probabilities": {k: (0.9 if k == key else 0.1 / max(1, len(crit) - 1))
                                          for k in crit}}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "topic", str(text), q, g,
                        {"orig_label": lab, "split_src": split})


_EMOTIONS = {"sadness": "feeling down, grief, disappointed, hopeless",
             "joy": "happy, grateful, proud, delighted",
             "love": "affection and care for a person, pet or thing",
             "anger": "furious, irritated, outraged, mad",
             "fear": "scared, anxious, worried, nervous, panicked",
             "surprise": "astonished, shocked, unexpected news"}


def build_emotion(ds, split, src, lic, cfg):
    order = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    crit = {k: _EMOTIONS[k] for k in order}
    for i, row in enumerate(ds):
        text, lab = _pick(row, "text"), _pick(row, "label")
        if text is None or lab is None:
            continue
        if isinstance(lab, int):
            names = cfg.get("names") or order
            lab = names[lab] if 0 <= lab < len(names) else None
        key = slugify(lab) if lab is not None else None
        if key not in crit:
            continue
        q = {"emotion": {"type": "choice", "instructions": "What emotion does `text` mainly express?",
                         "criteria": dict(crit)}}
        g = {"emotion": {"type": "choice", "label": key,
                         "probabilities": {k: (0.92 if k == key else 0.08 / (len(crit) - 1)) for k in crit}}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "emotion", str(text), q, g,
                        {"orig_label": lab, "split_src": split})


def build_clinc(ds, split, src, lic, cfg):
    """CLINC150 -> per-row: noul 'is_out_of_scope' + choice 'intent' (catalog completed later)."""
    names = cfg.get("names") or []
    for i, row in enumerate(ds):
        text, lab = _pick(row, "text"), _pick(row, "intent", "label")
        if text is None or lab is None:
            continue
        if isinstance(lab, int):
            name = names[lab] if 0 <= lab < len(names) else "oos"
        else:
            name = str(lab)
        oos = name == "oos"
        key = "oos" if oos else slugify(name)
        q = {"is_out_of_scope": {"type": "noul",
                                 "instructions": "Is `text` something this assistant cannot act on at all (no supported intent fits)?"}}
        g = {"is_out_of_scope": {"type": "noul", "label": "true" if oos else "false",
                                 "probabilities": {"false": 0.03 if oos else 0.97,
                                                   "true": 0.97 if oos else 0.03}}}
        if not oos:
            q["intent"] = {"type": "choice",
                           "instructions": "Which assistant intent does `text` express?",
                           "criteria": {key: None}}
            g["intent"] = {"type": "choice", "label": key, "probabilities": {key: 1.0}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "assistant", str(text), q, g,
                        {"orig_label": name, "split_src": split})


def build_sms_spam(ds, split, src, lic, cfg):
    for i, row in enumerate(ds):
        text, lab = _pick(row, "sms", "text"), _pick(row, "label")
        if text is None or lab is None:
            continue
        if isinstance(lab, int):
            names = cfg.get("names") or ["ham", "spam"]
            lab = names[lab] if 0 <= lab < len(names) else "ham"
        spam = str(lab).lower() == "spam"
        q = {"is_spam": {"type": "noul",
                         "instructions": "Is this SMS unsolicited marketing, a scam, or spam?",
                         "criteria": {"true": "spam, scam, unsolicited marketing",
                                      "false": "a genuine personal or business message"}}}
        g = {"is_spam": {"type": "noul", "label": "true" if spam else "false",
                         "probabilities": {"false": 0.97 if not spam else 0.03,
                                          "true": 0.97 if spam else 0.03}}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "spam_filter", str(text), q, g,
                        {"orig_label": lab, "split_src": split})


def build_prompt_injections(ds, split, src, lic, cfg):
    for i, row in enumerate(ds):
        text, lab = _pick(row, "text"), _pick(row, "label")
        if text is None or lab is None:
            continue
        inj = (int(lab) == 1) if isinstance(lab, (int, float)) else str(lab).lower() in ("injection", "1", "true", "yes")
        q = {"is_injection": {"type": "noul",
                              "instructions": "Does `prompt` try to hijack, inject into, or override the instructions of an AI system?",
                              "criteria": {"true": "prompt injection, jailbreak, or instruction override",
                                           "false": "an ordinary user request"}}}
        g = {"is_injection": {"type": "noul", "label": "true" if inj else "false",
                              "probabilities": {"false": 0.03 if inj else 0.97,
                                                "true": 0.97 if inj else 0.03}}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "guardrails", {"prompt": str(text)}, q, g,
                        {"orig_label": lab, "split_src": split})


_SST5_LEVELS = ["terribly negative", "negative", "neutral or mixed", "positive", "very positive"]


def build_sst5(ds, split, src, lic, cfg):
    for i, row in enumerate(ds):
        text, lab = _pick(row, "sentence", "text"), _pick(row, "label")
        if text is None or lab is None or not str(lab).isdigit() or not (0 <= int(lab) < 5):
            continue
        idx = int(lab)
        probs = {str(j): (0.8 if j == idx else 0.05) for j in range(5)}
        q = {"sentiment": {"type": "score",
                           "instructions": "Rate the sentiment expressed about the subject of `review`.",
                           "criteria": list(_SST5_LEVELS)}}
        g = {"sentiment": {"type": "score", "label": str(idx), "probabilities": probs}}
        yield make_case("%s:%s:%06d" % (src, split, i), src, "ordinal_sentiment",
                        {"review": str(text)}, q, g, {"orig_label": lab, "split_src": split})


# ---- Kaggle (optional) -------------------------------------------------------------------
def kaggle_download(slug, dest_dir):
    """`kaggle datasets download` + unzip into dest_dir. Raises if CLI/creds are missing."""
    os.makedirs(dest_dir, exist_ok=True)
    have_csv = any(fn.lower().endswith(".csv") for _, _, fns in os.walk(dest_dir) for fn in fns)
    if not have_csv:
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest_dir], check=True,
                       capture_output=True)
        import zipfile
        for fn in sorted(os.listdir(dest_dir)):
            if fn.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(dest_dir, fn)) as z:
                    z.extractall(dest_dir)
                break
    return dest_dir


def _kaggle_csv_rows(dest, max_rows_per_file=20000):
    for root, _, files in os.walk(dest):
        for fn in sorted(files):
            if not fn.lower().endswith(".csv"):
                continue
            with io.open(os.path.join(root, fn), "r", encoding="utf-8", errors="replace", newline="") as f:
                try:
                    reader = csv.DictReader(f)
                except csv.Error:
                    continue
                for n, row in enumerate(reader):
                    if n >= max_rows_per_file:
                        break
                    yield row


def build_kaggle_tickets(rows, src, lic, cfg):
    """Customer tickets: state dict + triage questions; weak labels from ticket metadata."""
    for i, row in enumerate(rows):
        subj = _pick(row, "TicketSubject", "subject", "Subject", "title")
        body = _pick(row, "TicketDetails", "TicketHistory", "body", "description", "Content")
        if not body or len(str(body)) < 40:
            continue
        state = {"body": re.sub(r"<[^>]+>", " ", str(body))}
        if subj:
            state["subject"] = str(subj)
        for k_in, k_out in (("TicketPriority", "priority"), ("TicketStatus", "status"),
                            ("Category", "category")):
            v = row.get(k_in)
            if v not in (None, ""):
                state[k_out] = str(v)[:60]
        pri = str(_pick(row, "TicketPriority", "priority", default="") or "").lower()
        urgent = pri in ("critical", "high", "urgent") or re.search(
            r"\basap\b|immediately|not working|blocked|deadline", str(body), re.I)
        q = {"is_urgent": {"type": "noul",
                           "instructions": "Does the ticket in `body` demand immediate action (outage, money at risk, hard deadline)?"},
             "frustration": {"type": "score", "instructions": "How frustrated is the customer in `body`?",
                             "criteria": ["calm and factual", "concerned but civil",
                                          "annoyed, stern wording", "angry or threatening to leave"]}}
        g = {"is_urgent": {"type": "noul", "label": "true" if urgent else "false",
                           "probabilities": {"false": 0.15 if urgent else 0.85,
                                             "true": 0.85 if urgent else 0.15}}}
        cs = row.get("CustomerSatisfactionScore")
        try:
            if cs is not None and str(cs).strip():
                lv = max(0, min(3, int(float(cs)) - 1))  # 1..5 -> level 0..3 (weak, low trust)
                g["frustration"] = {"type": "score", "label": str(lv),
                                    "probabilities": {str(j): (0.7 if j == lv else 0.075) for j in range(4)}}
        except (ValueError, TypeError):
            pass
        yield make_case("%s:%06d" % (src, i), src, "tickets", state, q, g,
                        {"split_src": "train", "weak_label": "metadata-derived"})


def build_kaggle_bitext(rows, src, lic, cfg):
    """Bitext customer-service CSVs: FullText + IntentName -> catalog choice (completed later)."""
    for i, row in enumerate(rows):
        text = _pick(row, "FullText", "full_text", "text", "Message")
        intent = _pick(row, "IntentName", "intent", "Intent")
        if not text or not intent:
            continue
        key = slugify(intent)
        q = {"intent": {"type": "choice", "instructions": "What is the customer asking for in `message`?",
                        "criteria": {key: None}}}
        g = {"intent": {"type": "choice", "label": key, "probabilities": {key: 1.0}}}
        yield make_case("%s:%06d" % (src, i), src, "banking_intent", {"message": str(text)}, q, g,
                        {"split_src": "train"})


# ---------------------------------------------------------------------------------------------
# Demo source: synthetic *raw* rows, deliberately dirty, exercising every filter stage.
# Works fully offline -- this is how datasets/curated/* is smoke-tested and regenerated.
# ---------------------------------------------------------------------------------------------
def build_demo_cases(seed: int):
    rng = random.Random(seed)
    cases = []
    dept_crit = {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                 "sales": "pricing, demos, new contracts", "other": "everything else"}
    raw_rows = [
        ("billing", "Hi, I was charged twice for my subscription this month. Please refund the duplicate charge."),
        ("billing", "Invoice&nbsp;#4411 has the WRONG amount!! Fix it now or we cancel everything!!"),
        ("technical", "the app crashes on login with err 500 after the update, logs at https://example.io/x attached, sent to support@acme-corp.example"),
        ("technical", "API returns 429 for all calls since 6AM UTC, our integration is completely down and customers are churning"),
        ("sales", "What does the enterprise plan cost for 250 seats, and is there a discount for annual payment?"),
        ("sales", "can we get a demo next week for the platform team? we are evaluating against a competitor"),
        ("other", "anyone know a good cafe near the office for a weekend team offsite?"),
        ("other", "bonjour, je souhaite resilier mon abonnement immediatement merci beaucoup pour votre aide"),
        ("other", "Hello, hello, hello, hello, hello, hello, hello, hello, hello, hello, hello, hello"),
        ("technical", "1234567890 1234567890 1234567890 1234567890"),
        ("sales", "pricing"),
        ("billing", "Refund please. My account 1234567890123456 was charged twice, contact me at a.b@corp.example or +1 415 555 0137 today please"),
        ("technical", "My password does not work anymore, lockout email went to jane.doe@corp.example, hard deadline today"),
        ("other", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ]
    for i, (gold_label, text) in enumerate(raw_rows):
        probs = {k: 0.03 for k in dept_crit}
        probs[gold_label] = 0.91
        urgent = any(w in text.lower() for w in ("now", "today", "cancel", "deadline", "down", "!!"))
        q = {"department": {"type": "choice", "instructions": "Which team should handle `message`?",
                            "criteria": dict(dept_crit)},
             "is_urgent": {"type": "noul", "instructions": "Does `message` demand immediate action?"}}
        g = {"department": {"type": "choice", "label": gold_label, "probabilities": probs},
             "is_urgent": {"type": "noul", "label": "true" if urgent else "false",
                           "probabilities": {"false": 0.08 if urgent else 0.92,
                                            "true": 0.92 if urgent else 0.08}}}
        cases.append(make_case("demo:triage:%06d" % i, "demo:triage", "email_triage",
                               {"message": text}, q, g, {"split_src": "train"}))
    # near-duplicates (one extra clause) -> MinHash should remove them
    for j in range(6):
        lab, base = raw_rows[j]
        probs = {k: 0.03 for k in dept_crit}
        probs[lab] = 0.91
        q = {"department": {"type": "choice", "instructions": "Which team should handle `message`?",
                            "criteria": dict(dept_crit)}}
        g = {"department": {"type": "choice", "label": lab, "probabilities": probs}}
        cases.append(make_case("demo:dup:%06d" % j, "demo:triage", "email_triage",
                               {"message": base + " (fwd 2)"}, q, g, {"split_src": "train"}))
    # ordinal reviews; every 9th has a flat teacher distribution (ambiguous -> calib split)
    for k in range(40):
        star = rng.randint(1, 5)
        sentiment = ["a complete disaster, nothing worked", "pretty disappointing overall",
                     "it is fine, nothing to report", "quite good, would return",
                     "absolutely loved it, exceeded every expectation"][star - 1]
        if k % 9 == 0:
            probs = {str(j): 0.2 for j in range(5)}
        else:
            probs = {str(j): (0.8 if j == star - 1 else 0.05) for j in range(5)}
        q = {"sentiment": {"type": "score", "instructions": "Rate the sentiment of `review`.",
                           "criteria": list(_SST5_LEVELS)}}
        g = {"sentiment": {"type": "score", "label": str(star - 1), "probabilities": probs}}
        cases.append(make_case("demo:review:%06d" % k, "demo:reviews", "ordinal_sentiment",
                               {"review": "The hotel was " + sentiment + " (ref %d)" % k}, q, g,
                               {"split_src": "train"}))
    # spam noul
    for k in range(40):
        spam = k % 3 == 0
        text = (("URGENT winner: you qualify for a free cruise, claim code %d now" % k) if spam else
                ("hey, are we still on for lunch tomorrow at 1? bring the design deck"))
        q = {"is_spam": {"type": "noul", "instructions": "Is this SMS unsolicited marketing or a scam?"}}
        g = {"is_spam": {"type": "noul", "label": "true" if spam else "false",
                         "probabilities": {"false": 0.03 if spam else 0.97, "true": 0.97 if spam else 0.03}}}
        cases.append(make_case("demo:sms:%06d" % k, "demo:sms", "spam_filter", text, q, g,
                               {"split_src": "train"}))
    # single-label rows -> demo catalog completion + long-tail folding
    for k in range(30):
        lab = "intent_%02d" % (k % 24)
        q = {"intent": {"type": "choice", "instructions": "Which intent does `message` express?",
                        "criteria": {lab: None}}}
        g = {"intent": {"type": "choice", "label": lab, "probabilities": {lab: 1.0}}}
        cases.append(make_case("demo:fold:%06d" % k, "demo:foldsrc", "assistant_intent",
                               {"message": "Please help with %s right away, order 44%s" % (lab.replace("_", " "), k)},
                               q, g, {"split_src": "train"}))
    rng.shuffle(cases)
    return cases


# ---------------------------------------------------------------------------------------------
# Catalog completion for single-label choice questions (clinc / bitext / demo:fold)
# ---------------------------------------------------------------------------------------------
def complete_catalogs(cases, per_src_counts, opts):
    """Rows that carry their own label only get a shared catalog: top-K labels + 'other'."""
    catalogs = {}
    for key, cnt in per_src_counts.items():
        if not cnt:
            continue
        keep = [k for k, _ in cnt.most_common(max(1, opts.max_options - 1))]
        crit = {k: None for k in keep}
        if len(cnt) > len(keep):
            crit["other"] = "anything not matching the listed options"
        if len(crit) >= 2:
            catalogs[key] = crit
    for c in cases:
        for qid, q in c["questions"].items():
            if q.get("type") != "choice":
                continue
            if len(q.get("criteria") or {}) > 1:
                continue
            cat = catalogs.get((c["source"], qid))
            g = c["gold"].get(qid)
            if not cat or not isinstance(g, dict):
                continue
            lab = slugify(g.get("label", ""))
            if lab not in cat:
                if "other" in cat:
                    g["label"] = lab = "other"
                    c["flags"] = sorted(set(c.get("flags", []) + ["fold:label->other"]))
                else:
                    continue
            q["criteria"] = dict(cat)
            g["probabilities"] = {k: (1.0 if k == lab else 0.0) for k in cat}
    return cases


# ---------------------------------------------------------------------------------------------
# Per-case transform: clean -> filter -> label-normalise (+ question-level repair)
# ---------------------------------------------------------------------------------------------
def transform(case, opts, stats, src_counts):
    flags = []
    state, f = clean_state(case["state"], scrub_pii=not opts.no_pii_scrub)
    flags += f
    body = text_of_state(state)
    if len(body) < opts.min_chars or len(body) > opts.max_chars:
        stats["drop:length"] += 1
        return None
    if not is_englishish(body, opts.min_stopwords):
        stats["drop:not-english"] += 1
        return None
    alnum = sum(ch.isalnum() for ch in body)
    digits = sum(ch.isdigit() for ch in body)
    letters = sum(ch.isalpha() for ch in body)
    if alnum == 0 or digits / max(1, alnum) > 0.55 or letters / max(1, alnum) < 0.55:
        stats["drop:garbage"] += 1
        return None
    if len(set(body)) < 6:
        stats["drop:repetitive"] += 1
        return None

    questions, gold = {}, {}
    for qid, q in (case["questions"] or {}).items():
        qid = slugify(qid)
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            stats["drop:q-bad-type"] += 1
            continue
        g = (case.get("gold") or {}).get(next((k for k in (case.get("gold") or {}) if slugify(k) == qid), qid))
        if not isinstance(g, dict):
            stats["drop:q-no-gold"] += 1
            continue
        ins, fi = clean_text(q.get("instructions") or "Answer the question about the state.", scrub_pii=False)
        flags += fi
        t = q["type"]
        cnt = src_counts.get((case["source"], next((k for k in (case["questions"] or {}) if slugify(k) == qid), qid)), Counter())
        if t == "choice":
            crit = q.get("criteria")
            if isinstance(crit, list):
                crit = {cc: None for cc in crit}
            if not isinstance(crit, dict) or not crit:
                stats["drop:q-bad-criteria"] += 1
                continue
            crit_n, seen_keys = {}, {}
            for k, v in crit.items():
                vk, fk = clean_text(v, scrub_pii=False) if v not in (None, "") else (None, [])
                flags += fk
                sk = slugify(k)
                if sk in crit_n:
                    continue
                crit_n[sk] = vk or None
            probs = {}
            for k, p in (g.get("probabilities") or {}).items():
                if isinstance(p, (int, float)):
                    probs[slugify(k)] = probs.get(slugify(k), 0.0) + float(p)
            label = slugify(g.get("label", max(probs, key=probs.get) if probs else ""))
            if not probs and label:
                probs = {label: 1.0}
            crit_f, probs_f, label_f, fold_flags = fold_to_catalog(
                crit_n, probs, label, opts.max_options, cnt, opts.min_label_freq)
            if crit_f is None:
                stats["drop:q-no-options"] += 1
                continue
            flags += fold_flags
            probs_f = soften(probs_f, list(crit_f), label_f, opts.label_smooth)
            label_f = max(probs_f, key=probs_f.get)   # keep label consistent with final probs
            questions[qid] = {"type": "choice", "instructions": ins or "Answer the question about the state.",
                              "criteria": crit_f}
            gold[qid] = {"type": "choice", "label": label_f, "probabilities": probs_f}
        elif t == "noul":
            probs = g.get("probabilities") or {}
            p = g.get("noul")
            if probs and "true" in probs:
                p_true = float(probs.get("true", 0.5))
            elif p is not None:
                p_true = float(p)
            else:
                p_true = 1.0 if str(g.get("label", "")).lower() in ("true", "1", "yes") else 0.0
            p_true = min(1 - opts.label_smooth, max(opts.label_smooth, p_true))
            crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
            qq = {"type": "noul", "instructions": ins or "Answer the question about the state."}
            if crit.get("true") or crit.get("false"):
                qq["criteria"] = {"false": crit.get("false") or "no, the statement does not hold",
                                  "true": crit.get("true") or "yes, the statement holds"}
            questions[qid] = qq
            gold[qid] = {"type": "noul", "label": "true" if p_true >= 0.5 else "false",
                         "probabilities": {"false": round(1 - p_true, 6), "true": round(p_true, 6)}}
        else:  # score
            crit = q.get("criteria")
            if not isinstance(crit, list) or len(crit) < 3:
                stats["drop:q-bad-criteria"] += 1
                continue
            crit_n, extra = [], []
            for cc in crit:
                v, fc = clean_text(cc, scrub_pii=False)
                crit_n.append(v or "level")
                extra += fc
            flags += extra
            keys = [str(i) for i in range(len(crit_n))]
            probs = {str(kk): float(v) for kk, v in (g.get("probabilities") or {}).items()
                     if isinstance(v, (int, float))}
            label = str(g.get("label", max(probs, key=probs.get) if probs else "0"))
            probs_f = soften(probs, keys, label, opts.label_smooth)
            questions[qid] = {"type": "score", "instructions": ins or "Answer the question about the state.",
                              "criteria": crit_n}
            gold[qid] = {"type": "score", "label": max(probs_f, key=probs_f.get), "probabilities": probs_f}
    if not questions or not gold:
        stats["drop:no-questions"] += 1
        return None
    if validate_questions(questions):
        stats["drop:invalid-questions"] += 1
        return None
    margins = {}
    for qid, g in gold.items():
        if g["type"] in ("choice", "score"):
            margins[qid] = margin_of(g["probabilities"])
        else:
            margins[qid] = abs(g["probabilities"]["true"] - 0.5) * 2
    min_margin = min([m for m in margins.values() if m is not None], default=None)
    ambiguous = min_margin is not None and min_margin < opts.min_margin
    q_score = quality_score(body, min_margin, len(set(flags)),
                            "text" if isinstance(state, str) else "json")
    if q_score < opts.min_quality and not ambiguous:
        stats["drop:quality"] += 1
        return None
    out = dict(case)
    out.update(state=state, questions=questions, gold=gold, quality=q_score,
               flags=sorted(set(flags) | ({"ambiguous"} if ambiguous else set())),
               factors=dict(case.get("factors") or {},
                            margins={k: (round(v, 4) if v is not None else None) for k, v in margins.items()},
                            n_state_chars=len(body)))
    return out


# ---------------------------------------------------------------------------------------------
# Splitting (leakage-safe, case-keyed) then balancing (train only)
# ---------------------------------------------------------------------------------------------
def balance_and_split(cases, opts, rng, stats):
    out = {"train": [], "val": [], "calib": [], "test": []}
    for c in cases:
        if c["factors"].get("split_src") == "test":
            split = "test"                     # official holdouts stay official holdouts
        elif "ambiguous" in c["flags"]:
            split = "calib"                    # low teacher margin -> calibration slice
        else:
            h = int(hashlib.md5(c["id"].encode()).hexdigest(), 16) % 1000
            a, b, z = opts.split
            split = "train" if h < a else "val" if h < a + b else "test" if h < a + b + z else "train"
        out[split].append(c)

    if opts.balance:
        def key_for(c, qid):
            crit = c["questions"][qid].get("criteria") or {}
            keys = sorted(crit) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]
            return (qid, tuple(keys))

        per = defaultdict(Counter)
        for c in out["train"]:
            for qid, g in c["gold"].items():
                if g["type"] == "choice":
                    per[key_for(c, qid)][g["label"]] += 1
        caps = {}
        for key, cnt in per.items():
            if len(cnt) > 2:
                med = sorted(cnt.values())[len(cnt) // 2]
                caps[key] = max(opts.balance_min, int(med * opts.balance_factor))
        used, kept = Counter(), []
        for c in out["train"]:
            ok = True
            for qid, g in c["gold"].items():
                if g["type"] != "choice":
                    continue
                key = key_for(c, qid)
                if key in caps and used[key + (g["label"],)] >= caps[key]:
                    ok = False
                    break
            if ok:
                kept.append(c)
                for qid, g in c["gold"].items():
                    if g["type"] != "choice":
                        continue
                    key = key_for(c, qid)
                    used[key + (g["label"],)] += 1
            else:
                stats["drop:imbalance"] += 1
        out["train"] = kept
    rng.shuffle(out["train"])
    return out


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------
def encode_row(c, split):
    return {
        "id": c["id"], "workflow": c["workflow"], "source": c["source"],
        "state": json.dumps(c["state"], ensure_ascii=False),
        "questions": json.dumps(c["questions"], ensure_ascii=False),
        "gold": json.dumps(c["gold"], ensure_ascii=False),
        "factors": json.dumps(c.get("factors") or {}, ensure_ascii=False),
        "n_questions": len(c["questions"]), "quality": c["quality"],
        "flags": c["flags"], "split": split,
    }


def write_outputs(splits, out_dir, opts, stats, report_extra):
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for name, cases in splits.items():
        if not cases:
            continue
        path = os.path.join(out_dir, "laya_posttrain_%s.jsonl" % name)
        with io.open(path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(encode_row(c, name), ensure_ascii=False) + "\n")
        counts[name] = len(cases)
    label_dist, primitive_dist = {}, defaultdict(int)
    for name, cases in splits.items():
        d = defaultdict(Counter)
        for c in cases:
            for qid, g in c["gold"].items():
                d[qid][g["label"]] += 1
                primitive_dist[g["type"]] += 1
        label_dist[name] = {qid: dict(cnt.most_common(24)) for qid, cnt in d.items()}
    report = {
        "pipeline": "laya-dataset-curation-v1",
        "sources": list(opts.active_sources),
        "stats": dict(sorted(stats.items())),
        "counts": counts,
        "primitive_counts": dict(primitive_dist),
        "label_distribution": label_dist,
        "options": {k: getattr(opts, k) for k in
                    ("max_options", "min_quality", "label_smooth", "dedup", "dedup_threshold",
                     "balance", "min_margin", "min_label_freq", "per_source_limit", "min_chars",
                     "max_chars", "seed", "split", "demo")},
        "licenses": {k: [v.get("license"), v.get("note", "")]
                     for k, v in list(HF_SOURCES.items()) + list(KAGGLE_SOURCES.items())
                     if k in opts.active_sources},
        **report_extra,
    }
    with io.open(os.path.join(out_dir, "curated_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return counts, report


def try_token_budget_validation(cases, tokenizer_dir, max_len, head_max_len):
    """Optional hard guarantee: every case fits the model's head+state token budget."""
    try:
        from laya.common import build_sequence, render_options
        from transformers import AutoTokenizer
    except Exception as e:
        print("  [validate] skipped (laya/transformers unavailable): %s" % e)
        return None
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    bad = 0
    for c in cases:
        for qid, q in c["questions"].items():
            qq = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            try:
                _, markers = build_sequence(tok, c["state"], qq, max_len, head_max_len)
                if len(markers) != len(render_options(qq)):
                    bad += 1
            except Exception:
                bad += 1
    return bad


# ---------------------------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------------------------
def fetch_hf(name, limit, cache_dir, strict):
    spec = HF_SOURCES[name]
    try:
        from datasets import load_dataset
    except ImportError as e:
        msg = "pip install datasets (needed for HF source %s)" % spec["repo"]
        if strict:
            raise RuntimeError(msg) from e
        print("  [skip] %s: %s" % (name, msg))
        return []
    builder = globals()[spec["builder"]]
    out = []
    for split_key in ("train", "test"):
        split = spec.get(split_key)
        if not split:
            continue
        try:
            kw = {"split": split}
            if spec.get("config") and spec["config"] != "default":
                kw["name"] = spec["config"]
            ds = load_dataset(spec["repo"], cache_dir=cache_dir, **kw)
        except Exception as e:
            if strict:
                raise
            print("  [skip] %s/%s: %s" % (name, split, str(e)[:200]))
            continue
        cfg = {"names": None}
        try:
            for col in ("label", "intent"):
                feat = ds.features.get(col)
                if feat is not None and hasattr(feat, "names"):
                    cfg["names"] = feat.names
                    break
        except Exception:
            pass
        start = len(out)
        for j, case in enumerate(builder(ds, split_key, "hf:" + spec["repo"], spec["license"], cfg)):
            if limit and j >= limit:
                break
            out.append(case)
        print("    %s/%s -> %d cases" % (name, split, len(out) - start))
    return out


def fetch_kaggle(name, limit, data_dir, strict):
    spec = KAGGLE_SOURCES[name]
    try:
        dest = kaggle_download(spec["slug"], os.path.join(data_dir, name))
        rows = list(_kaggle_csv_rows(dest))
    except Exception as e:
        msg = str(e)[:200]
        if strict:
            raise
        print("  [skip] %s: %s (needs `pip install kaggle` + ~/.kaggle/kaggle.json)" % (name, msg))
        return []
    builder = globals()[spec["builder"]]
    out = []
    for j, case in enumerate(builder(rows, "kaggle:" + spec["slug"], spec["license"], {})):
        if limit and j >= limit:
            break
        out.append(case)
    print("    %s -> %d cases" % (name, len(out)))
    return out


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Curate Laya post-training datasets from open sources.")
    ap.add_argument("--sources", default="all",
                    help="comma list from: %s | kaggle: %s | all | demo"
                         % (",".join(HF_SOURCES), ",".join(KAGGLE_SOURCES)))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "curated"))
    ap.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    ap.add_argument("--data-dir", default=None, help="raw download dir (default: <repo>/datasets/raw)")
    ap.add_argument("--per-source-limit", type=int, default=12000, help="max raw rows per source split (0=all)")
    ap.add_argument("--max-options", type=int, default=12, help="cap on choice options; tail folds into 'other'")
    ap.add_argument("--min-label-freq", type=int, default=4, help="min examples per label before it may stay in a big catalog")
    ap.add_argument("--min-chars", dest="min_chars", type=int, default=24)
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=6000)
    ap.add_argument("--min-quality", type=float, default=0.30)
    ap.add_argument("--min-margin", type=float, default=0.25,
                    help="below this teacher top1-top2 margin a case is 'ambiguous' -> calibration split")
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--min-stopwords", type=float, default=0.06, help="english gate: min stopword coverage (0 disables)")
    ap.add_argument("--dedup-threshold", type=float, default=0.85, help="near-dup Jaccard threshold (0 disables)")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false", default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True)
    ap.add_argument("--balance-min", type=int, default=40)
    ap.add_argument("--balance-factor", type=float, default=3.0)
    ap.add_argument("--no-pii-scrub", dest="no_pii_scrub", action="store_true")
    ap.add_argument("--split", type=int, nargs=3, default=(760, 80, 160), metavar=("TRAIN", "VAL", "TEST"),
                    help="parts per 1000 of the non-ambiguous pool (remainder -> train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="offline: curate the built-in dirty demo data")
    ap.add_argument("--strict", action="store_true", help="fail on any source error instead of skipping")
    ap.add_argument("--with-kaggle", action="store_true", help="also pull the optional Kaggle sources")
    ap.add_argument("--validate-tokenizer-dir", default=None,
                    help="tokenizer dir used to verify every case fits the head budget (needs laya installed)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    opts = ap.parse_args(argv)

    rng = random.Random(opts.seed)
    stats = Counter()
    if opts.demo or opts.sources == "demo":
        active = ["demo"]
    elif opts.sources in ("all", "core"):
        active = list(HF_SOURCES) + (list(KAGGLE_SOURCES) if opts.with_kaggle else [])
    else:
        active = [s.strip() for s in opts.sources.split(",") if s.strip()]
    opts.active_sources = active
    data_dir = opts.data_dir or os.path.join(_HERE, "raw")
    print("Curating laya post-training data | sources=%s | out=%s" % (",".join(active), opts.out_dir))

    raw = []
    for name in active:
        if name == "demo":
            demo = build_demo_cases(opts.seed)
            raw.extend(demo)
            stats["fetched:demo"] += len(demo)
        elif name in HF_SOURCES:
            print("  fetch hf: %s" % HF_SOURCES[name]["repo"])
            rows = fetch_hf(name, opts.per_source_limit, opts.cache_dir, opts.strict)
            raw.extend(rows)
            stats["fetched:%s" % name] += len(rows)
        elif name in KAGGLE_SOURCES:
            print("  fetch kaggle: %s" % KAGGLE_SOURCES[name]["slug"])
            rows = fetch_kaggle(name, opts.per_source_limit, data_dir, opts.strict)
            raw.extend(rows)
            stats["fetched:%s" % name] += len(rows)
        else:
            raise SystemExit("unknown source %r; known: %s"
                             % (name, ",".join(list(HF_SOURCES) + list(KAGGLE_SOURCES) + ["demo"])))
    print("  raw cases: %d" % len(raw))
    if not raw:
        raise SystemExit("no data fetched (network down or all sources skipped); try --demo")

    # dataset-wide label counts per (source, question) -> catalogs + folding decisions
    src_counts = defaultdict(Counter)
    for c in raw:
        for qid, q in (c["questions"] or {}).items():
            if not isinstance(q, dict) or q.get("type") != "choice":
                continue
            crit = q.get("criteria") or {}
            if isinstance(crit, dict) and len(crit) >= 2:
                continue  # fixed catalogs are already complete; don't fold against them
            if isinstance(crit, dict) and len(crit) == 1:
                for k in crit:
                    src_counts[(c["source"], qid)][slugify(k)] += 1
            else:
                g = (c["gold"] or {}).get(qid) or {}
                if g.get("label"):
                    src_counts[(c["source"], qid)][slugify(g["label"])] += 1

    raw = complete_catalogs(raw, src_counts, opts)

    dedup = MinHashIndex(threshold=opts.dedup_threshold) if (opts.dedup and opts.dedup_threshold > 0) else None
    seen_exact = set()
    curated = []
    for case in raw:
        out_case = transform(case, opts, stats, src_counts)
        if out_case is None:
            continue
        body = text_of_state(out_case["state"])
        ex = hashlib.blake2b(re.sub(r"\W+", "", body.lower()).encode("utf-8")).hexdigest()
        if ex in seen_exact:
            stats["drop:dup-exact"] += 1
            continue
        seen_exact.add(ex)
        if dedup is not None and not dedup.add(out_case["id"], body):
            stats["drop:dup-near"] += 1
            continue
        curated.append(out_case)
    stats["curated"] = len(curated)
    print("  after cleaning / label-norm / dedupe: %d" % len(curated))

    splits = balance_and_split(curated, opts, rng, stats)
    counts, report = write_outputs(splits, opts.out_dir, opts, stats, {
        "sources_registry": {k: {kk: vv for kk, vv in v.items() if kk != "builder"}
                             for k, v in list(HF_SOURCES.items()) + list(KAGGLE_SOURCES.items())},
    })
    print("  wrote: %s" % counts)
    print("  stats: %s" % json.dumps(report["stats"], sort_keys=True))
    if opts.validate_tokenizer_dir:
        bad = try_token_budget_validation(curated, opts.validate_tokenizer_dir, opts.max_len, opts.head_max_len)
        if bad is not None:
            print("  token-budget validation: %d invalid cases (must be 0)" % bad)
            report.setdefault("validation", {})["head_budget_invalid"] = bad
            with io.open(os.path.join(opts.out_dir, "curated_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
    print("done. report: %s" % os.path.join(opts.out_dir, "curated_report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
