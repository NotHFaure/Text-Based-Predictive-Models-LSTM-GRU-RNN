# scripts/

## `prepare_corpus.py`

A standard-library-only script that regenerates this project's corpus —
`pg766.txt` and `cleaned_chapters/` (64 files) — from a pinned source,
reproducing the notebook's cleaning pipeline byte-for-byte. As of this
fix, this script **is** how those two artefacts get regenerated: they are
no longer committed to the repository (see "Why they're no longer
committed" below). It does **not** regenerate `cleaned_data/cleaned_text.txt`
— see that same section for why.

### What it does

1. Resolves a verified copy of the corpus, in priority order:
   1. an already-cached, hash-verified copy at `--cache` (default
      `.corpus_cache/pg766.txt`, gitignored);
   2. the pinned 2024 snapshot tracked at `scripts/pg766_source.txt` —
      **this is the default path and needs no network access**;
   3. a live download from Project Gutenberg, only if both of the above
      are unavailable, still hash-verified before being accepted.
2. Strips the Project Gutenberg header/footer and splits the remaining
   text into per-chapter files.
3. Cleans each chapter exactly as the notebook's `clean_and_preprocess`
   does: lowercasing, contraction expansion, digit-to-word conversion,
   punctuation removal, residual chapter-heading removal, whitespace
   normalisation. Output goes to `--output` (default
   `.corpus_cache/chapters/`, gitignored) — pointing it at `cleaned_chapters/`
   is exactly how you regenerate what the training notebook expects at the
   repository root (see below); an earlier version of this script refused
   that path as a safety check against overwriting a *committed* copy, but
   neither `cleaned_chapters/` nor `cleaned_data/` is committed any more,
   so that check no longer applies and was removed.

### How to run

```bash
python scripts/prepare_corpus.py
# or, with explicit paths:
python scripts/prepare_corpus.py --cache /tmp/pg766.txt --output /tmp/chapters
```

No arguments are required, and no network access is needed by default —
the corpus is regenerated from the tracked `scripts/pg766_source.txt`.
Run from anywhere — the script locates the repository root relative to
its own file location, not via a hardcoded path.

To regenerate the files `MLASS4v2.ipynb` (the notebook that actually
trains and evaluates the RNN/GRU/LSTM models) expects at the repository
root:

```bash
python scripts/prepare_corpus.py --cache pg766.txt --output cleaned_chapters
```

**`cleaned_data/cleaned_text.txt` is deliberately not reproduced by this
script**, and its removal needs no regeneration path. It is *not* an
artefact of `MLASS4v2.ipynb`'s `clean_and_preprocess` → `cleaned_chapters/`
pipeline this script targets — confirmed directly: `MLASS4v2.ipynb` never
reads `cleaned_data/` at all, it reads `cleaned_chapters/` straight into
memory via `load_cleaned_chapters`. `cleaned_data/cleaned_text.txt` was
instead written by `MLASS4.ipynb` (cell 17) — the early-draft notebook,
already documented in the repository README as "data exploration and
n-gram experiments only; no RNN/GRU/LSTM training" — from its own separate
`clean_text` pipeline (stopword removal via NLTK + WordNet lemmatization,
not part of `clean_and_preprocess`). Confirmed by direct comparison: the
64 cleaned chapters joined together do **not** match the committed
`cleaned_data/cleaned_text.txt` byte-for-byte, or even in length (1,853,212
vs. 1,805,712 characters) — they are genuinely different pipelines over
the same source text, not one derived from the other. Reproducing it would
mean hand-rolling NLTK's stopword list and `WordNetLemmatizer`, for an
artefact nothing in this repository's documented, working results
actually depends on -- out of scope for what this script exists to do.

### Source, licence and provenance

- **Source:** [Project Gutenberg eBook #766](https://www.gutenberg.org/ebooks/766), *David Copperfield* by Charles Dickens.
- **Licence:** Public domain in the United States. Project Gutenberg's own
  trademark terms apply to the standard header/footer text bundled with the
  file, not to the novel itself.
- **Corpus identity:** confirmed directly from the file's own Project
  Gutenberg header (`Title: David Copperfield` / `Author: Charles Dickens`)
  and cross-checked against both committed notebooks, which load and print
  the same header when run.

### How the stdlib-only cleaning step was verified

The committed `cleaned_chapters/` this script used to be checked against
(and which `scripts/pg766_source.txt` + this script can still exactly
reproduce) was produced by `MLASS4v2.ipynb`'s `clean_and_preprocess` cell,
which depends on three third-party packages: `contractions` (contraction
expansion), `num2words` (digit-to-word conversion), and `BeautifulSoup`
(HTML handling). `A-33`'s approval requires this script to stay
standard-library only, so it does not import any of them — it reproduces
their effect on this specific, pinned corpus with hand-rolled equivalents
instead:

- **Contractions:** the script embeds the complete case-folded mapping the
  `contractions` package's `fix()` actually applies (base dict plus its
  "leftover" and "slang" extensions, both apostrophe variants of every
  entry), matched with an explicit non-alphanumeric boundary check rather
  than a plain `\b` (some entries are themselves bounded by a
  non-alphanumeric character, e.g. the slang entry `"r "` → `"are "`).
  Verified byte-for-byte against `contractions.fix()` across the full
  lowercased raw corpus, not just the 64 chapters.
- **Numbers:** the script's `_number_to_words` reproduces `num2words`'s
  English cardinal style for 0-999, verified against `num2words` across
  that entire range. It deliberately does not implement thousand/million
  grouping — the pinned corpus only ever needs single digits (chapter 28
  contains the only two: "9" and "1").
- **HTML:** the notebook's `BeautifulSoup(text, "html.parser").get_text()`
  step is confirmed a byte-for-byte no-op on this corpus (plain Gutenberg
  text, no HTML tags or entities), so it is correctly omitted rather than
  approximated.

**Result: `prepare_corpus.py` now reproduces all 64 committed chapters
byte-for-byte**, verified via direct comparison against the archived
committed copies, using only the pinned `scripts/pg766_source.txt` and no
network access. (An earlier version of this script only reproduced chapter
*count* and boundaries, not cleaned content — see the execution report
referenced below for that finding and how it was closed.)

### Why the corpus is no longer committed, and why `pg766_source.txt` is

`pg766.txt`, `cleaned_chapters/` and `cleaned_data/cleaned_text.txt` were
removed from this repository once this script was verified to reproduce
them exactly (`A-33`). Removing them without a reliable regeneration path
would have made the repository non-reproducible, so `scripts/pg766_source.txt`
— a byte-for-byte copy of the `pg766.txt` originally committed here in
2024 — stays tracked specifically to make regeneration possible without
depending on a live download. That matters because:

> Project Gutenberg periodically revises the text and boilerplate it
> serves for a given eBook ID. A live download of eBook #766, verified
> during this repository's own history, did **not** match the SHA-256 of
> the `pg766.txt` committed here in 2024 — traced to a single character
> deep in the novel text (Gutenberg's current copy has since corrected a
> stray typographic quote mark). Treating a live download as equivalent to
> the corpus this repository was actually built and trained on would be
> silently substituting one text for another. `scripts/pg766_source.txt`
> sidesteps the question entirely: it's what this repository was actually
> built against, tracked and hash-verified, not re-fetched.

A live download via `--url` remains available for anyone who wants to
compare against upstream Gutenberg, and is still hash-verified against the
same pinned constant before being accepted — it is just never the default.

### Known limitations

- Does not regenerate `tokenizer.json` or `vocab.json` — this script has no
  bearing on the tokenizer-artefact gap described in the repository
  README's Known Limitations section.
- Does not concatenate `cleaned_chapters/` into `cleaned_data/cleaned_text.txt`
  — that's a one-line join, documented above rather than duplicated in code.
