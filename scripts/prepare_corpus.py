#!/usr/bin/env python3
"""Deterministic corpus download and chapter-preparation script.

Verifies and prepares Project Gutenberg eBook #766 (David Copperfield,
public domain in the US), then reproduces the chapter-splitting and
cleaning steps of the preprocessing pipeline used in ``MLASS4v2.ipynb``
into a directory the caller supplies.

This script is what regenerates ``pg766.txt`` and ``cleaned_chapters/`` --
neither is committed to this repository (``A-33``); this script is the
reproducibility path for both, verified byte-for-byte. It does not
regenerate ``cleaned_data/cleaned_text.txt``, which has no regeneration
path -- see ``scripts/README.md`` for why. See ``scripts/README.md`` for
the full picture.

Standard library only -- including the cleaning step. The notebook's
``clean_and_preprocess`` depends on three third-party packages
(``contractions``, ``num2words``, ``BeautifulSoup``); this script
reproduces their effect on this specific, pinned corpus with hand-rolled,
stdlib-only equivalents instead of importing them, per A-33's standing
constraint that this script must not import a third-party package:

- Contraction expansion: ``_CONTRACTIONS`` below is the complete
  case-folded mapping the ``contractions`` package's ``fix()`` actually
  applies (base dict + its "leftover" and "slang" extensions, with both
  the straight-apostrophe and curly-apostrophe (U+2019) form of every
  entry -- ``contractions.fix()`` matches both without any separate
  normalisation step, so both are included as literal keys here rather
  than folded via a normalisation pass). Matching uses a longest-key-first
  alternation with explicit non-alphanumeric boundary checks on both sides
  (not a plain ``\b``, because some entries are themselves bounded by a
  non-alphanumeric character, e.g. the slang entry ``"r "`` -> ``"are "``;
  a naive ``\b`` would treat the embedded space as satisfying the right
  boundary and over-match). Verified byte-for-byte against
  ``contractions.fix()`` across the full lowercased raw corpus (not just
  the 64 committed chapters) during this fix's development.
- Number-to-words: ``_number_to_words`` reproduces ``num2words``'s English
  cardinal style (e.g. ``"one hundred and twenty-three"``) for 0-999,
  verified against ``num2words`` for that full range. The pinned corpus
  only ever needs single digits (chapter 28 contains the only two: "9"
  and "1"), so this intentionally does not implement thousand/million
  grouping -- extending it is straightforward if a future corpus needs it,
  but nothing here does.
- HTML handling: the notebook runs ``BeautifulSoup(text, "html.parser")
  .get_text()`` before contraction expansion. Confirmed a byte-for-byte
  no-op on this corpus (plain Gutenberg text, no HTML tags or entities) --
  so this step is correctly omitted rather than approximated.

Source pinning: this script's default corpus source is the tracked
snapshot at ``scripts/pg766_source.txt`` -- a byte-for-byte copy of the
``pg766.txt`` originally committed to this repository in 2024 -- not a
live download. Project Gutenberg has since revised the text it serves for
this eBook ID (confirmed during this fix: a live download's SHA-256 no
longer matches, traced to a single corrected typographic quote mark deep
in the novel text), so treating the current upstream copy as equivalent
would silently change what this script reproduces. A live download
remains available via ``--url``/``--cache`` for anyone who wants to
compare against upstream, but is never used by default, and is still
hash-verified against the same pinned constant before being accepted.
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

# Captured from the pg766.txt originally committed to this repository in
# 2024 (during EP-2026-07-31-003's execution, 2026-08-08). This is the
# corpus this script reproduces -- not necessarily whatever Project
# Gutenberg is serving today -- see scripts/README.md for why those can
# diverge, and where the tracked snapshot below comes from.
EXPECTED_SHA256 = "5b3e58058f9bb67fe66e1de92bd1d28b24fa02eade777042726a8a2f3195144e"

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

# The pinned 2024 source, tracked in git alongside this script -- see the
# module docstring's "Source pinning" section.
BUNDLED_SOURCE = SCRIPT_DIR / "pg766_source.txt"

DEFAULT_CACHE_PATH = REPO_ROOT / ".corpus_cache" / "pg766.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".corpus_cache" / "chapters"

CHAPTER_PATTERN = re.compile(r"chapter\s+(?:[ivxlcdm]+|\d+)\.\s*", re.IGNORECASE)
START_MARKER = re.compile(
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
END_MARKER = re.compile(
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)

# Complete case-folded mapping used by the `contractions` package's fix()
# (base dict + leftover + slang extensions), both apostrophe variants
# included as literal keys. See the module docstring for how this was
# derived and verified. One genuine key collision existed in the source
# package ("ima" resolves ambiguously under case-insensitive matching
# depending on which original-case variant is considered); the value
# `contractions.fix()` itself actually returns for lowercase input is the
# one kept here. It does not occur anywhere in this corpus.
_CONTRACTIONS: dict[str, str] = {
    "'aight": 'alright',
    "'all": '',
    "'am": '',
    "'cause": 'because',
    "'coz": 'because',
    "'d": ' would',
    "'em": 'them',
    "'ll": ' will',
    "'re": ' are',
    "'tis": 'it is',
    "'twas": 'it was',
    'abt': 'about',
    'acct': 'account',
    "ain't": 'are not',
    'aint': 'are not',
    'ain’t': 'are not',
    'altho': 'although',
    "amn't": 'am not',
    'amnt': 'am not',
    'amn’t': 'am not',
    'apr.': 'april',
    "aren't": 'are not',
    'arent': 'are not',
    'aren’t': 'are not',
    'asap': 'as soon as possible',
    'aug.': 'august',
    'avg': 'average',
    'b4': 'before',
    'bc': 'because',
    'bday': 'birthday',
    'btw': 'by the way',
    'can cause': 'can cause',
    "can't": 'cannot',
    "can't've": 'cannot have',
    "can'tve": 'cannot have',
    'cant': 'cannot',
    "cant've": 'cannot have',
    'cantve': 'cannot have',
    'can’t': 'cannot',
    'can’t’ve': 'cannot have',
    'cause': 'because',
    'convo': 'conversation',
    'could cause': 'could cause',
    "could've": 'could have',
    "couldn't": 'could not',
    "couldn't've": 'could not have',
    "couldn'tve": 'could not have',
    'couldnt': 'could not',
    "couldnt've": 'could not have',
    'couldntve': 'could not have',
    'couldn’t': 'could not',
    'couldn’t’ve': 'could not have',
    'couldve': 'could have',
    'could’ve': 'could have',
    'cya': 'see ya',
    "daren't": 'dare not',
    'darent': 'dare not',
    'daren’t': 'dare not',
    "daresn't": 'dare not',
    'daresnt': 'dare not',
    'daresn’t': 'dare not',
    "dasn't": 'dare not',
    'dasnt': 'dare not',
    'dasn’t': 'dare not',
    'dec.': 'december',
    "didn't": 'did not',
    'didnt': 'did not',
    'didn’t': 'did not',
    'diff': 'different',
    "doesn't": 'does not',
    'doesnt': 'does not',
    'doesn’t': 'does not',
    "doin'": 'doing',
    'doin’': 'doing',
    "don't": 'do not',
    'dont': 'do not',
    'don’t': 'do not',
    'dunno': 'do not know',
    "e'er": 'ever',
    'eer': 'ever',
    'em': 'them',
    "everyone's": 'everyone is',
    'everyones': 'everyone is',
    'everyone’s': 'everyone is',
    'e’er': 'ever',
    'feb.': 'february',
    'finna': 'fixing to',
    "g'day": 'good day',
    'gimme': 'give me',
    "goin'": 'going',
    'goin’': 'going',
    "gon't": 'go not',
    'gonna': 'going to',
    'gont': 'go not',
    'gon’t': 'go not',
    'gotta': 'got to',
    "hadn't": 'had not',
    "hadn't've": 'had not have',
    "hadn'tve": 'had not have',
    'hadnt': 'had not',
    "hadnt've": 'had not have',
    'hadntve': 'had not have',
    'hadn’t': 'had not',
    'hadn’t’ve': 'had not have',
    "hasn't": 'has not',
    'hasnt': 'has not',
    'hasn’t': 'has not',
    "haven't": 'have not',
    'havent': 'have not',
    'haven’t': 'have not',
    "havin'": 'having',
    'havin’': 'having',
    "he'd": 'he would',
    "he'd've": 'he would have',
    "he'dve": 'he would have',
    "he'll": 'he will',
    "he'll've": 'he will have',
    "he'llve": 'he will have',
    "he's": 'he is',
    "he've": 'he have',
    'hed': 'he would',
    "hed've": 'he would have',
    'hedve': 'he would have',
    "hell've": 'he will have',
    'hellve': 'he will have',
    "here's": 'here is',
    'heres': 'here is',
    'here’s': 'here is',
    'heve': 'he have',
    'he’d': 'he would',
    'he’d’ve': 'he would have',
    'he’ll': 'he will',
    'he’ll’ve': 'he will have',
    'he’s': 'he is',
    'he’ve': 'he have',
    "how'd": 'how did',
    "how'd'y": 'how do you',
    "how'dy": 'how do you',
    "how'll": 'how will',
    "how're": 'how are',
    "how's": 'how is',
    'howd': 'how did',
    "howd'y": 'how do you',
    'howdy': 'how do you',
    'howll': 'how will',
    'howre': 'how are',
    'hows': 'how is',
    'how’d': 'how did',
    'how’d’y': 'how do you',
    'how’ll': 'how will',
    'how’re': 'how are',
    'how’s': 'how is',
    "i'd": 'i would',
    "i'd've": 'i would have',
    "i'dve": 'i would have',
    "i'll": 'i will',
    "i'll've": 'i will have',
    "i'llve": 'i will have',
    "i'm": 'i am',
    "i'm'a": 'i am about to',
    "i'm'o": 'i am going to',
    "i'ma": 'i am about to',
    "i'mo": 'i am going to',
    "i've": 'i have',
    "id've": 'i would have',
    'idk': 'i do not know',
    'idve': 'i would have',
    "ill've": 'i will have',
    'illve': 'i will have',
    'im': 'i am',
    "im'a": 'i am about to',
    "im'o": 'i am going to',
    'ima': 'i am about to',
    'imma': 'i am going to',
    'imo': 'i am going to',
    'innit': 'is it not',
    "isn't": 'is not',
    'isnt': 'is not',
    'isn’t': 'is not',
    "it'd": 'it would',
    "it'd've": 'it would have',
    "it'dve": 'it would have',
    "it'll": 'it will',
    "it'll've": 'it will have',
    "it'llve": 'it will have',
    "it's": 'it is',
    'itd': 'it would',
    "itd've": 'it would have',
    'itdve': 'it would have',
    'itll': 'it will',
    "itll've": 'it will have',
    'itllve': 'it will have',
    'it’d': 'it would',
    'it’d’ve': 'it would have',
    'it’ll': 'it will',
    'it’ll’ve': 'it will have',
    'it’s': 'it is',
    'iunno': 'i do not know',
    'ive': 'i have',
    'i’d': 'i would',
    'i’d’ve': 'i would have',
    'i’ll': 'i will',
    'i’ll’ve': 'i will have',
    'i’m': 'i am',
    'i’m’a': 'i am about to',
    'i’m’o': 'i am going to',
    'i’ve': 'i have',
    'jan.': 'january',
    'jul.': 'july',
    'jun.': 'june',
    'kinda': 'kind of',
    'kk': 'okay',
    'lemme': 'let me',
    "let's": 'let us',
    'lets': 'let us',
    'let’s': 'let us',
    "lovin'": 'loving',
    'lovin’': 'loving',
    'luv': 'love',
    "ma'am": 'madam',
    'maam': 'madam',
    'mar.': 'march',
    'may cause': 'may cause',
    "may've": 'may have',
    "mayn't": 'may not',
    'maynt': 'may not',
    'mayn’t': 'may not',
    'mayve': 'may have',
    'may’ve': 'may have',
    'ma’am': 'madam',
    'might cause': 'might cause',
    "might've": 'might have',
    "mightn't": 'might not',
    "mightn't've": 'might not have',
    "mightn'tve": 'might not have',
    'mightnt': 'might not',
    "mightnt've": 'might not have',
    'mightntve': 'might not have',
    'mightn’t': 'might not',
    'mightn’t’ve': 'might not have',
    'mightve': 'might have',
    'might’ve': 'might have',
    'msg': 'message',
    'must cause': 'must cause',
    "must've": 'must have',
    "mustn't": 'must not',
    "mustn't've": 'must not have',
    "mustn'tve": 'must not have',
    'mustnt': 'must not',
    "mustnt've": 'must not have',
    'mustntve': 'must not have',
    'mustn’t': 'must not',
    'mustn’t’ve': 'must not have',
    'mustve': 'must have',
    'must’ve': 'must have',
    "ne'er": 'never',
    "needn't": 'need not',
    "needn't've": 'need not have',
    "needn'tve": 'need not have',
    'neednt': 'need not',
    "neednt've": 'need not have',
    'needntve': 'need not have',
    'needn’t': 'need not',
    'needn’t’ve': 'need not have',
    'neer': 'never',
    'ne’er': 'never',
    "nothin'": 'nothing',
    'nothin’': 'nothing',
    'nov.': 'november',
    'nvm': 'nevermind',
    "o'": 'of',
    "o'clock": 'of the clock',
    "o'er": 'over',
    'oclock': 'of the clock',
    'oct.': 'october',
    'oer': 'over',
    'ofc': 'of course',
    'ol': 'old',
    "ol'": 'old',
    'ol’': 'old',
    "oughtn't": 'ought not',
    "oughtn't've": 'ought not have',
    "oughtn'tve": 'ought not have',
    'oughtnt': 'ought not',
    "oughtnt've": 'ought not have',
    'oughtntve': 'ought not have',
    'oughtn’t': 'ought not',
    'oughtn’t’ve': 'ought not have',
    'o’': 'of',
    'o’clock': 'of the clock',
    'o’er': 'over',
    'ppl': 'people',
    'prolly': 'probably',
    'pymnt': 'payment',
    'r ': 'are ',
    'rlly': 'really',
    'rly': 'really',
    'rn': 'right now',
    'sep.': 'september',
    "sha'n't": 'shall not',
    "sha'nt": 'shall not',
    'shall cause': 'shall cause',
    "shalln't": 'shall not',
    'shallnt': 'shall not',
    'shalln’t': 'shall not',
    "shan't": 'shall not',
    "shan't've": 'shall not have',
    "shan'tve": 'shall not have',
    'shant': 'shall not',
    "shant've": 'shall not have',
    'shantve': 'shall not have',
    'shan’t': 'shall not',
    'shan’t’ve': 'shall not have',
    'sha’n’t': 'shall not',
    "she'd": 'she would',
    "she'd've": 'she would have',
    "she'dve": 'she would have',
    "she'll": 'she will',
    "she's": 'she is',
    'shed': 'she would',
    "shed've": 'she would have',
    'shedve': 'she would have',
    'shell': 'she will',
    'shes': 'she is',
    'she’d': 'she would',
    'she’d’ve': 'she would have',
    'she’ll': 'she will',
    'she’s': 'she is',
    'should cause': 'should cause',
    "should've": 'should have',
    "shouldn't": 'should not',
    "shouldn't've": 'should not have',
    "shouldn'tve": 'should not have',
    'shouldnt': 'should not',
    "shouldnt've": 'should not have',
    'shouldntve': 'should not have',
    'shouldn’t': 'should not',
    'shouldn’t’ve': 'should not have',
    'shouldve': 'should have',
    'should’ve': 'should have',
    "so's": 'so is',
    "so've": 'so have',
    "somebody's": 'somebody is',
    'somebodys': 'somebody is',
    'somebody’s': 'somebody is',
    "someone's": 'someone is',
    'someones': 'someone is',
    'someone’s': 'someone is',
    "somethin'": 'something',
    "something's": 'something is',
    'somethings': 'something is',
    'something’s': 'something is',
    'somethin’': 'something',
    'sos': 'so is',
    'sove': 'so have',
    'so’s': 'so is',
    'so’ve': 'so have',
    'spk': 'spoke',
    'sux': 'sucks',
    'tbh': 'to be honest',
    "that'd": 'that would',
    "that'd've": 'that would have',
    "that'dve": 'that would have',
    "that'll": 'that will',
    "that're": 'that are',
    "that's": 'that is',
    'thatd': 'that would',
    "thatd've": 'that would have',
    'thatdve': 'that would have',
    'thatll': 'that will',
    'thatre': 'that are',
    'thats': 'that is',
    'that’d': 'that would',
    'that’d’ve': 'that would have',
    'that’ll': 'that will',
    'that’re': 'that are',
    'that’s': 'that is',
    "there'd": 'there would',
    "there'd've": 'there would have',
    "there'dve": 'there would have',
    "there'll": 'there will',
    "there're": 'there are',
    "there's": 'there is',
    'thered': 'there would',
    "thered've": 'there would have',
    'theredve': 'there would have',
    'therell': 'there will',
    'therere': 'there are',
    'theres': 'there is',
    'there’d': 'there would',
    'there’d’ve': 'there would have',
    'there’ll': 'there will',
    'there’re': 'there are',
    'there’s': 'there is',
    "these're": 'these are',
    'thesere': 'these are',
    'these’re': 'these are',
    "they'd": 'they would',
    "they'd've": 'they would have',
    "they'dve": 'they would have',
    "they'll": 'they will',
    "they'll've": 'they will have',
    "they'llve": 'they will have',
    "they're": 'they are',
    "they've": 'they have',
    'theyd': 'they would',
    "theyd've": 'they would have',
    'theydve': 'they would have',
    'theyll': 'they will',
    "theyll've": 'they will have',
    'theyllve': 'they will have',
    'theyre': 'they are',
    'theyve': 'they have',
    'they’d': 'they would',
    'they’d’ve': 'they would have',
    'they’ll': 'they will',
    'they’ll’ve': 'they will have',
    'they’re': 'they are',
    'they’ve': 'they have',
    "this'd": 'this would',
    "this'll": 'this will',
    "this's": 'this is',
    'thisd': 'this would',
    'thisll': 'this will',
    'thiss': 'this is',
    'this’d': 'this would',
    'this’ll': 'this will',
    'this’s': 'this is',
    'tho': 'though',
    "those're": 'those are',
    'thosere': 'those are',
    'those’re': 'those are',
    'thx': 'thanks',
    'tis': 'it is',
    'tlked': 'talked',
    'tmmw': 'tomorrow',
    'tmr': 'tomorrow',
    'tmrw': 'tomorrow',
    'to cause': 'to cause',
    "to've": 'to have',
    'tove': 'to have',
    'to’ve': 'to have',
    'twas': 'it was',
    'u': 'you',
    'ur': 'you are',
    'wanna': 'want to',
    "wasn't": 'was not',
    'wasnt': 'was not',
    'wasn’t': 'was not',
    "we'd": 'we would',
    "we'd've": 'we would have',
    "we'dve": 'we would have',
    "we'll": 'we will',
    "we'll've": 'we will have',
    "we'llve": 'we will have',
    "we're": 'we are',
    "we've": 'we have',
    "wed've": 'we would have',
    'wedve': 'we would have',
    "well've": 'we will have',
    'wellve': 'we will have',
    "weren't": 'were not',
    'werent': 'were not',
    'weren’t': 'were not',
    'weve': 'we have',
    'we’d': 'we would',
    'we’d’ve': 'we would have',
    'we’ll': 'we will',
    'we’ll’ve': 'we will have',
    'we’re': 'we are',
    'we’ve': 'we have',
    "what'd": 'what did',
    "what'll": 'what will',
    "what'll've": 'what will have',
    "what'llve": 'what will have',
    "what're": 'what are',
    "what's": 'what is',
    "what've": 'what have',
    'whatcha': 'what are you',
    'whatd': 'what did',
    'whatll': 'what will',
    "whatll've": 'what will have',
    'whatllve': 'what will have',
    'whatre': 'what are',
    'whats': 'what is',
    'whatve': 'what have',
    'what’d': 'what did',
    'what’ll': 'what will',
    'what’ll’ve': 'what will have',
    'what’re': 'what are',
    'what’s': 'what is',
    'what’ve': 'what have',
    "when's": 'when is',
    "when've": 'when have',
    'whens': 'when is',
    'whenve': 'when have',
    'when’s': 'when is',
    'when’ve': 'when have',
    "where'd": 'where did',
    "where're": 'where are',
    "where's": 'where is',
    "where've": 'where have',
    'whered': 'where did',
    'wherere': 'where are',
    'wheres': 'where is',
    'whereve': 'where have',
    'where’d': 'where did',
    'where’re': 'where are',
    'where’s': 'where is',
    'where’ve': 'where have',
    "which's": 'which is',
    'whichs': 'which is',
    'which’s': 'which is',
    "who'd": 'who would',
    "who'd've": 'who would have',
    "who'dve": 'who would have',
    "who'll": 'who will',
    "who'll've": 'who will have',
    "who'llve": 'who will have',
    "who're": 'who are',
    "who's": 'who is',
    "who've": 'who have',
    'whod': 'who would',
    "whod've": 'who would have',
    'whodve': 'who would have',
    'wholl': 'who will',
    "wholl've": 'who will have',
    'whollve': 'who will have',
    'whos': 'who is',
    'whove': 'who have',
    'who’d': 'who would',
    'who’d’ve': 'who would have',
    'who’ll': 'who will',
    'who’ll’ve': 'who will have',
    'who’re': 'who are',
    'who’s': 'who is',
    'who’ve': 'who have',
    "why'd": 'why did',
    "why're": 'why are',
    "why's": 'why is',
    "why've": 'why have',
    'whyd': 'why did',
    'whyre': 'why are',
    'whys': 'why is',
    'whyve': 'why have',
    'why’d': 'why did',
    'why’re': 'why are',
    'why’s': 'why is',
    'why’ve': 'why have',
    'will cause': 'will cause',
    "will've": 'will have',
    'willve': 'will have',
    'will’ve': 'will have',
    "won't": 'will not',
    "won't've": 'will not have',
    "won'tve": 'will not have',
    'wont': 'will not',
    "wont've": 'will not have',
    'wontve': 'will not have',
    'won’t': 'will not',
    'won’t’ve': 'will not have',
    'would cause': 'would cause',
    "would've": 'would have',
    'woulda': 'would have',
    "wouldn't": 'would not',
    "wouldn't've": 'would not have',
    "wouldn'tve": 'would not have',
    'wouldnt': 'would not',
    "wouldnt've": 'would not have',
    'wouldntve': 'would not have',
    'wouldn’t': 'would not',
    'wouldn’t’ve': 'would not have',
    'wouldve': 'would have',
    'would’ve': 'would have',
    "y'all": 'you all',
    "y'all'd": 'you all would',
    "y'all'd've": 'you all would have',
    "y'all'dve": 'you all would have',
    "y'all're": 'you all are',
    "y'all've": 'you all have',
    "y'alld": 'you all would',
    "y'alld've": 'you all would have',
    "y'alldve": 'you all would have',
    "y'allre": 'you all are',
    "y'allve": 'you all have',
    'yall': 'you all',
    "yall'd": 'you all would',
    "yall'd've": 'you all would have',
    "yall'dve": 'you all would have',
    "yall're": 'you all are',
    "yall've": 'you all have',
    'yalld': 'you all would',
    "yalld've": 'you all would have',
    'yalldve': 'you all would have',
    'yallre': 'you all are',
    'yallve': 'you all have',
    "you'd": 'you would',
    "you'd've": 'you would have',
    "you'dve": 'you would have',
    "you'll": 'you will',
    "you'll've": 'you shall have',
    "you'llve": 'you shall have',
    "you're": 'you are',
    "you've": 'you have',
    'youd': 'you would',
    "youd've": 'you would have',
    'youdve': 'you would have',
    'youll': 'you will',
    "youll've": 'you shall have',
    'youllve': 'you shall have',
    'youre': 'you are',
    'youve': 'you have',
    'you’d': 'you would',
    'you’d’ve': 'you would have',
    'you’ll': 'you will',
    'you’ll’ve': 'you shall have',
    'you’re': 'you are',
    'you’ve': 'you have',
    'y’all': 'you all',
    'y’all’d': 'you all would',
    'y’all’d’ve': 'you all would have',
    'y’all’re': 'you all are',
    'y’all’ve': 'you all have',
    '’all': '',
    '’am': '',
    '’cause': 'because',
    '’coz': 'because',
    '’d': ' would',
    '’em': ' them',
    '’ll': ' will',
    '’re': ' are',
    '’tis': 'it is',
    '’twas': 'it was',
}

_contraction_keys_by_length = sorted(_CONTRACTIONS, key=len, reverse=True)
_CONTRACTION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(k) for k in _contraction_keys_by_length)
    + r")(?![A-Za-z0-9_])"
)

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_corpus(cache_path: Path, url: str, expected_sha256: str) -> Path:
    """Return a local path to a corpus file verified against expected_sha256.

    Priority order: (1) an already-cached, hash-verified copy at
    ``cache_path``; (2) the tracked 2024 snapshot bundled alongside this
    script (``BUNDLED_SOURCE``), copied into the cache once verified; (3) a
    live download from ``url``, only as a last resort, still hash-verified
    before being accepted. This means a normal run needs no network access
    at all -- see the module docstring's "Source pinning" section for why
    a live download is not the default.
    """
    if cache_path.exists():
        cached_hash = sha256_of(cache_path)
        if cached_hash == expected_sha256:
            print(f"Using cached corpus at {cache_path} (SHA-256 verified, no download)")
            return cache_path
        print(
            f"Cached file at {cache_path} does not match the expected hash "
            f"(got {cached_hash}); trying the bundled source next.",
            file=sys.stderr,
        )

    if BUNDLED_SOURCE.exists():
        bundled_hash = sha256_of(BUNDLED_SOURCE)
        if bundled_hash == expected_sha256:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(BUNDLED_SOURCE.read_bytes())
            print(f"Using bundled source at {BUNDLED_SOURCE} (SHA-256 verified, no download)")
            return cache_path
        print(
            f"Bundled source at {BUNDLED_SOURCE} does not match the expected hash "
            f"(got {bundled_hash}); falling back to a live download.",
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


def expand_contractions(text: str) -> str:
    """Stdlib-only equivalent of ``contractions.fix(text)``. See the module
    docstring for how this table and its matching rule were derived and
    verified."""
    return _CONTRACTION_PATTERN.sub(lambda m: _CONTRACTIONS[m.group(0)], text)


def _two_digits(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _number_to_words(n: int) -> str:
    """Stdlib-only equivalent of ``num2words(n)``'s English cardinal style,
    verified against ``num2words`` for the full 0-999 range. Deliberately
    does not implement thousand/million grouping -- see the module
    docstring for why the pinned corpus never needs it."""
    if n == 0:
        return "zero"
    if n >= 1000:
        raise ValueError(f"_number_to_words is only verified for 0-999, got {n}")
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _two_digits(rest)
    if rest == 0:
        return f"{_ONES[hundreds]} hundred"
    return f"{_ONES[hundreds]} hundred and {_two_digits(rest)}"


def convert_numbers(text: str) -> str:
    """Stdlib-only equivalent of the notebook's convert_numbers (num2words)."""
    def replace_num(match: re.Match[str]) -> str:
        try:
            return _number_to_words(int(match.group()))
        except ValueError:
            return match.group()

    return re.sub(r"\b\d+\b", replace_num, text)


def clean_chapter(text: str) -> str:
    """Stdlib-only reproduction of the notebook's clean_and_preprocess.

    Reproduces, in the notebook's own order: lowercasing, contraction
    expansion, digit-to-word conversion, punctuation removal, residual
    chapter-heading removal, and whitespace normalisation. The notebook's
    BeautifulSoup HTML-entity step is omitted -- confirmed a byte-for-byte
    no-op on this corpus, not approximated. See the module docstring for
    how each stdlib-only replacement was derived and verified."""
    text = text.lower()
    text = expand_contractions(text)
    text = convert_numbers(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"chapter\s+\w+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_chapters(corpus_path: Path, output_dir: Path) -> int:
    """Write each cleaned chapter to output_dir.

    No longer refuses `cleaned_chapters/` as a destination: that guard
    existed to protect a tracked, committed copy from being overwritten,
    and neither `cleaned_chapters/` nor `cleaned_data/` is tracked as of
    `A-33` -- see the module docstring and scripts/README.md.
    """
    output_dir = output_dir.resolve()
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
        help=f"Local cache path for the corpus (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the regenerated chapters (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--url",
        default=GUTENBERG_URL,
        help="Source URL, only used if both the cache and the bundled source are unavailable",
    )
    args = parser.parse_args(argv)

    corpus_path = ensure_corpus(args.cache, args.url, EXPECTED_SHA256)
    char_count, word_count = corpus_stats(corpus_path)
    print(f"Corpus: {char_count} characters, {word_count} words")

    count = prepare_chapters(corpus_path, args.output)
    print(f"Wrote {count} chapter files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
