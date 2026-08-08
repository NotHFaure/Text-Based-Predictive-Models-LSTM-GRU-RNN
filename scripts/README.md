# scripts/

## `prepare_corpus.py`

A standard-library-only reproducibility script. It downloads the corpus this
project trains on, verifies it, and reproduces the chapter-splitting step of
the preprocessing pipeline documented in `MLASS4v2.ipynb`.

**It is a reproducibility aid, not a replacement.** `pg766.txt`,
`cleaned_chapters/` and `cleaned_data/cleaned_text.txt` all remain committed
to this repository exactly as they were. Nothing here deletes, moves, or
overwrites any of them.

### What it does

1. Downloads Project Gutenberg eBook #766 to a local, gitignored cache path
   (default `.corpus_cache/pg766.txt`), skipping the download if a cached
   copy already exists and matches the expected hash.
2. Verifies the corpus by SHA-256 against a constant recorded in the script,
   captured from the committed `pg766.txt` during this repository's own
   `EP-2026-07-31-003` execution. A mismatch is a hard error — the script
   refuses to proceed on unverified content rather than silently accepting
   whatever the source currently serves.
3. Strips the Project Gutenberg header/footer and splits the remaining text
   into per-chapter files, written to an output directory you supply
   (default `.corpus_cache/chapters/`, also gitignored). The default, and
   the script's own internal check, make it impossible to point the output
   at the tracked `cleaned_chapters/` or `cleaned_data/` directories.

### How to run

```bash
python scripts/prepare_corpus.py
# or, with explicit paths:
python scripts/prepare_corpus.py --cache /tmp/pg766.txt --output /tmp/chapters
```

No arguments are required. Run from anywhere — the script locates the
repository root relative to its own file location, not via a hardcoded path.

### Source, licence and provenance

- **Source:** [Project Gutenberg eBook #766](https://www.gutenberg.org/ebooks/766), *David Copperfield* by Charles Dickens.
- **Licence:** Public domain in the United States. Project Gutenberg's own
  trademark terms apply to the standard header/footer text bundled with the
  file, not to the novel itself.
- **Corpus identity:** confirmed directly from the file's own Project
  Gutenberg header (`Title: David Copperfield` / `Author: Charles Dickens`)
  and cross-checked against both committed notebooks, which load and print
  the same header when run.

### Known differences from the committed `cleaned_chapters/`

The committed `cleaned_chapters/` was produced by `MLASS4v2.ipynb`'s
`clean_and_preprocess` cell, which depends on three third-party packages:
`contractions` (contraction expansion, e.g. "can't" → "cannot"),
`num2words` (digit-to-word conversion), and `BeautifulSoup` (HTML-entity
handling). `A-33`'s approval requires this script to be standard-library
only, so it cannot import any of them.

`prepare_corpus.py` therefore reproduces exactly: the Project Gutenberg
boilerplate stripping, the chapter-boundary splitting (both are pure
`re` in the notebook already), lowercasing, punctuation removal, residual
chapter-heading removal, and whitespace normalisation. It does **not**
expand contractions or convert digits to words, and it approximates the
BeautifulSoup HTML-entity step with the standard library's `html.unescape`
rather than a real HTML parser.

The practical effect: chapter boundaries and chapter counts match the
committed output exactly, but the cleaned text within each chapter is not
byte-identical — contractions and any digit sequences in the source text
remain unexpanded. The exact comparison from this script's validation run
is recorded in `Execution/Reports/ER-2026-07-31-003-corpus-script.md`
(Code HQ), not duplicated here.

### A note on the corpus source itself

Project Gutenberg periodically revises the text and boilerplate it serves
for a given eBook ID. A live download of eBook #766, verified this same
session, did **not** match the SHA-256 of the `pg766.txt` committed to this
repository in 2024 — investigation traced the difference to a single
character deep in the novel text (Gutenberg's current copy has since
corrected a stray typographic quote mark). This script's SHA-256 check is
what catches exactly this kind of drift: it refuses to treat a currently-
served file as equivalent to the corpus this repository was actually built
and trained on, and stops rather than substituting one for the other
silently. Details are in the execution report referenced above.

### Known limitations

- Does not regenerate `tokenizer.json` or `vocab.json` — this script has no
  bearing on the tokenizer-artefact gap described in the repository
  README's Known Limitations section.
- Requires network access to `gutenberg.org` on first run (or on any run
  where the local cache is absent or stale).
