# `datasets/` — Laya post-training data curation

`curate.py` is the data engine behind
[`notebooks/laya_colab_post_training_rl.ipynb`](../notebooks/laya_colab_post_training_rl.ipynb).
It pulls **raw** rows from open sources, runs a full cleaning/normalisation pipeline in pure
Python, and emits Laya-schema JSONL ready for SFT + RLCD post-training. Every stage counts its
drops into `curated_report.json`, so the provenance of the final mix is fully auditable.

```bash
python3 datasets/curate.py --sources all --out-dir datasets/curated     # full pipeline
python3 datasets/curate.py --demo                                        # offline, stdlib-only
python3 datasets/curate.py --sources emotion,sms_spam --per-source-limit 4000
```

Requires `pip install datasets` for the HF sources (nothing else). `--with-kaggle` additionally
needs `pip install kaggle` and `~/.kaggle/kaggle.json`. `--demo` needs no network and no deps.

## Sources

| key | repo / slug | what it contributes | primitive | license |
|---|---|---|---|---|
| `typed_decisions` | `LocalLLaMA/typed-decisions` (config `all`) | official RLCD benchmark: 4 support workflows with teacher-consensus soft labels — passed through with resplit | choice / score / noul | see [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) |
| `typed_decisions_synth` | `n4ze3m/typed-decisions-synth` | synthetic typed decisions (52 domains, 100 workflows), teacher JSON or noul floats | choice / noul | MIT |
| `ag_news` | `fancyzhx/ag_news` | news topic classification, 4 classes | choice | see [card](https://huggingface.co/datasets/fancyzhx/ag_news) |
| `emotion` | `dair-ai/emotion` | SemEval-2018 Task 1, 6-way emotion tweets | choice | see [card](https://huggingface.co/datasets/dair-ai/emotion) |
| `clinc_oos` | `clinc/clinc_oos` (config `small`) | assistant intents incl. out-of-scope; 150-label catalog folded to top-K + `other` | choice | see [card](https://huggingface.co/datasets/clinc/clinc_oos) |
| `sms_spam` | `ucirvine/sms_spam` | spam/ham SMS | noul | unknown (academic collection) |
| `prompt_injections` | `deepset/prompt-injections` | prompt-injection detection (546/116) | noul | Apache-2.0 |
| `sst5` | `SetFit/sst5` | 5-way sentiment re-mapped onto a written 0–4 ordinal rubric | score | see [card](https://huggingface.co/datasets/SetFit/sst5) |
| `support_tickets` *(optional)* | Kaggle `waseemalastal/customer-support-ticket-dataset` | ~8k tickets → triage question pack, weak labels from metadata | choice / score / noul | see Kaggle page |
| `customer_intent` *(optional)* | Kaggle `scodepy/customer-support-intent-dataset` | Bitext customer-service intents CSV | choice | see Kaggle page |

`--demo` builds ~130 deliberately *dirty* rows (mojibake, HTML entities, PII, near-dupes,
non-English, garbage) covering every primitive and exercises all ten pipeline stages offline.

Check each dataset card before redistributing derivatives; the curated files are small text
excerpts intended for research / fine-tuning use.

## Pipeline (every stage counted in `curated_report.json`)

1. **fetch** — raw rows from HF `datasets` / Kaggle CSVs / the demo generator
2. **catalog** — per-question label catalogs from dataset-wide frequencies; long tails folded
   to `other` so every option stays legible inside `head_max_len` (192) token budget
3. **clean** — mojibake repair, HTML/entity stripping, PII scrubbing (URLs → `⟨url⟩`, e-mails →
   `⟨email⟩`, 5+ digit runs → `⟨num⟩`), unicode NFKC + whitespace normalisation, signature stripping
4. **filter** — English heuristic (stopword coverage), length bands 24–6000 chars,
   garbage / digit / repetition ratios
5. **label-norm** — canonical label slugs, top-K + `other` folding, soft targets clipped away
   from 0/1 and smoothed (`--label-smooth 0.05`): strictly-proper scoring never sees raw one-hots
6. **dedupe** — exact hash + MinHash/LSH near-dup removal (word-shingle Jaccard ≥ 0.85)
7. **quality** — per-case score; below `--min-quality 0.30` dropped; weak top-1/top-2 margin →
   flagged `ambiguous`
8. **split** — leakage-safe per-case splits `train / val / calib / test` (760/80/160 weights);
   a source's own test split is *forced* into `test` so numbers stay comparable to the repo
   benchmark; `calib` is held out for temperature fitting
9. **balance** — per-(question, catalog) label caps for **train only** (median × 3, floor 40);
   val/calib/test stay untrimmed
10. **validate** — optional token-budget check through laya's own `build_sequence`
    (`--validate-tokenizer-dir <dir>` / automatic in the Colab notebook)

## Output

`curated/laya_posttrain_{train,val,calib,test}.jsonl` — one case per line, same schema as
`LocalLLaMA/typed-decisions` (so the trainer consumes either interchangeably):

```json
{"id": "…", "workflow": "…", "source": "hf:emotion", "state": "<json str>",
 "questions": "<json str {qid: {type, instructions, criteria}}>",
 "gold": "<json str {qid: {type, label, probabilities}}>",
 "factors": "<json str>", "n_questions": 4, "quality": 0.91, "flags": [], "split": "train"}
```

plus `curated_report.json` with per-stage stats, per-label distributions and primitive counts.

## Useful flags

`--sources all|demo|<comma list>` · `--per-source-limit 12000` · `--max-options 12` ·
`--split 760 80 160` · `--seed 42` · `--min-quality 0.30` · `--label-smooth 0.05` ·
`--dedup-threshold 0.85` / `--no-dedup` · `--no-balance` · `--no-pii-scrub` ·
`--with-kaggle` · `--strict` (fail on any source error) · `--demo`

## Limitations

* Weak labels on `support_tickets` (metadata-derived) — the report flags the source; drop it with
  `--sources` if you want teacher-labeled data only.
* The English gate is a stopword heuristic tuned for short social/ticket text; formal long-form
  English can score low (widen with `--min-stopwords 0`).
* `sms_spam` has no test split upstream → its cases are resampled by the global split weights.
* PII scrubbing is regex-grade, not NER-grade: fine for public benchmarks, not a substitute for
  a privacy review of private data.
