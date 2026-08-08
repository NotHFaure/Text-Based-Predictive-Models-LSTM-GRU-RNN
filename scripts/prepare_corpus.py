#!/usr/bin/env python3
"""Deterministic corpus download and chapter-preparation script.

Downloads Project Gutenberg eBook #766 (David Copperfield, public domain in
the US), verifies it by SHA-256 against the hash of the corpus already
committed at ``pg766.txt``, and reproduces the chapter-splitting step of the
preprocessing pipeline used in ``MLASS4v2.ipynb`` into a directory the
caller supplies.

This script is a reproducibility aid. It does not replace ``pg766.txt``,
``cleaned_chapters/`` or ``cleaned_data/cleaned_text.txt``, all of which
remain committed. See ``scripts/README.md`` for the full picture, including
where this script's output is known to differ from the committed
``cleaned_chapters/`` (the notebook's cleaning step depends on third-party
packages this script does not import).

Standard library only. No network access is attempted unless the corpus is
not already cached with a verified hash.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/766/pg766.txt"

# Captured from the committed pg766.txt during EP-2026-07-31-003's execution
# (2026-08-08). This is the corpus this script is meant to reproduce, not
# necessarily whatever Project Gutenberg is serving today -- see
# scripts/README.md's "Known differences" section for why those can diverge.
EXPECTED_SHA256 = "5b3e58058f9bb67fe66e1de92bd1d28b24fa02eade777042726a8a2f3195144e"

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_CORPUS_FILE = REPO_ROOT / "pg766.txt"
TRACKED_CHAPTERS_DIR = REPO_ROOT / "cleaned_chapters"
TRACKED_CLEANED_DATA_DIR = REPO_ROOT / "cleaned_data"

DEFAULT_CACHE_PATH = REPO_ROOT / ".corpus_cache" / "pg766.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".corpus_cache" / "chapters"

CHAPTER_PATTERN = re.compile(r"chapter\s+(?:[ivxlcdm]+|\d+)\.\s*", re.IGNORECASE)
START_MARKER = re.compile(
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
END_MARKER = re.compile(
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_corpus(cache_path: Path, url: str, expected_sha256: str) -> Path:
    """Return a local path to a corpus file verified against expected_sha256.

    Skips the download if a cached file already exists and its hash matches.
    Downloads once otherwise. Never returns a file whose hash does not match
    -- a mismatch is a fatal error, not a warning, since it means the source
    has drifted from the snapshot this script was written against.
    """
    if cache_path.exists():
        cached_hash = sha256_of(cache_path)
        if cached_hash == expected_sha256:
            print(f"Using cached corpus at {cache_path} (SHA-256 verified, no download)")
            return cache_path
        print(
            f"Cached file at {cache_path} does not match the expected hash "
            f"(got {cached_hash}); re-downloading.",
            file=sys.stderr,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc

    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_sha256:
        raise RuntimeError(
            "Downloaded corpus does not match the expected SHA-256.\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual_hash}\n"
            "Refusing to proceed with unverified content. This can mean the "
            "upstream source has changed since this script's hash constant "
            "was captured -- see scripts/README.md before updating it."
        )

    cache_path.write_bytes(data)
    print(f"Downloaded and verified {cache_path} (SHA-256 matched)")
    return cache_path


def remove_boilerplate(text: str) -> str:
    """Strip the Project Gutenberg header/footer, exactly as the notebook does."""
    start_match = START_MARKER.search(text)
    end_match = END_MARKER.search(text)
    if start_match and end_match:
        text = text[start_match.end() : end_match.start()]
    return text.strip()


def extract_chapters(text: str) -> list[str]:
    """Split on chapter headings, exactly as the notebook's extractChapters does."""
    matches = list(CHAPTER_PATTERN.finditer(text))
    chapters = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append(text[match.end() : end].strip())
    return chapters


def clean_chapter(text: str) -> str:
    """Standard-library-only approximation of the notebook's clean_and_preprocess.

    Reproduces exactly: lowercasing, punctuation removal, residual chapter-
    heading removal, and whitespace normalisation.

    Does NOT reproduce (both require third-party packages this script must
    not import, per A-33's approval): contraction expansion (`contractions`)
    and digit-to-word conversion (`num2words`). The notebook's HTML-entity
    stripping via BeautifulSoup is approximated with the standard-library
    `html.unescape`, which is not a full HTML parser. See scripts/README.md.
    """
    import html as html_module

    text = text.lower()
    text = html_module.unescape(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"chapter\s+\w+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_safe_output_dir(output_dir: Path) -> Path:
    """Refuse an output directory that could collide with tracked corpora."""
    resolved = output_dir.resolve()
    forbidden = {TRACKED_CHAPTERS_DIR.resolve(), TRACKED_CLEANED_DATA_DIR.resolve()}
    for candidate in (resolved, *resolved.parents):
        if candidate in forbidden:
            raise SystemExit(
                f"Refusing to write into a tracked corpus directory ({candidate}). "
                "Choose a different --output."
            )
    return resolved


def prepare_chapters(corpus_path: Path, output_dir: Path) -> int:
    output_dir = resolve_safe_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = corpus_path.read_text(encoding="utf-8")
    body = remove_boilerplate(raw)
    chapters = extract_chapters(body)

    for idx, chapter in enumerate(chapters, start=1):
        cleaned = clean_chapter(chapter)
        (output_dir / f"chapter_{idx}.txt").write_text(cleaned, encoding="utf-8")

    return len(chapters)


def corpus_stats(corpus_path: Path) -> tuple[int, int]:
    """Character and word counts on the raw corpus, matching the notebook's
    len(data) / re.findall(r'\\b\\w+\\b', data) figures."""
    raw = corpus_path.read_text(encoding="utf-8")
    char_count = len(raw)
    word_count = len(re.findall(r"\b\w+\b", raw))
    return char_count, word_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Local cache path for the downloaded corpus (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for the regenerated chapters. Must not be the "
            f"tracked cleaned_chapters/ or cleaned_data/ (default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument("--url", default=GUTENBERG_URL, help="Source URL")
    args = parser.parse_args(argv)

    corpus_path = ensure_corpus(args.cache, args.url, EXPECTED_SHA256)
    char_count, word_count = corpus_stats(corpus_path)
    print(f"Corpus: {char_count} characters, {word_count} words")

    count = prepare_chapters(corpus_path, args.output)
    print(f"Wrote {count} chapter files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
