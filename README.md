# Text-Based Predictive Models — LSTM / GRU / RNN

**ZEIT4151 Machine Learning, Assignment 4B — Module 3A: Word-Level Predictive Text Module**
**Author:** Harrison Faure
**Date:** November 4, 2024

---

## Table of Contents

- [Introduction](#introduction)
- [Goals](#goals)
- [Dataset](#dataset)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements and Dependencies](#requirements-and-dependencies)
- [Folder Structure](#folder-structure)
- [Example Outputs](#example-outputs)
- [Technologies](#technologies)
- [Licence](#licence)
- [Known Limitations](#known-limitations)
- [Maintenance Status](#maintenance-status)
- [Complexity Rating](#complexity-rating)
- [Acknowledgements](#acknowledgements)

---

## Introduction

This project develops a predictive text module for an HMI system aimed at supporting communication for ALS patients. The module provides word-level predictive text, suggesting the next word based on previous words typed. Three recurrent architectures (RNN, GRU, LSTM) were implemented and compared for this task.

## Goals

Build a word-level next-word predictor that offers three suggestions per keystroke, aimed at reducing typing effort for users with ALS, and compare RNN, GRU and LSTM architectures on accuracy, training behaviour and inference speed to justify the final model choice.

## Dataset

The dataset is *David Copperfield* by Charles Dickens — Project Gutenberg eBook [#766](https://www.gutenberg.org/ebooks/766), United States public domain. The committed `pg766.txt` was verified directly against its own Project Gutenberg header (`Title: David Copperfield` / `Author: Charles Dickens`) and against the notebooks' own "Dataset Statistics" cells, which load and print the same header when run.

- **Raw file size:** 1,954,026 characters (`pg766.txt`, matches the notebook's `len(data)` output)
- **Total words (raw regex tokenisation):** 367,022 (`MLASS4.ipynb`, cell 8 / `MLASS4v2.ipynb`, cell 3)
- **Vocabulary size:** reported differently at different preprocessing stages in the notebooks — 14,223 unique tokens from the raw-corpus statistics cell, versus 16,748–17,066 from an intermediate cleaned-token count. The model-training pipeline (`cleaned_chapters/` → oversampled sequences) builds its own tokenizer vocabulary separately again. No single "unique word count" in the notebooks is authoritative for all stages, so no one figure is asserted as *the* vocabulary size.
- **Chapters:** 64, extracted from the raw text and stored individually under `cleaned_chapters/`
- **Preprocessing:** boilerplate removal, chapter splitting, lowercasing, contraction expansion, number-to-word conversion, punctuation removal, tokenisation, rare-word oversampling

## Features

- Word-level next-word prediction with top-3 suggestions
- Three trained architectures (RNN, GRU, LSTM) for direct comparison
- Preprocessing pipeline from raw Gutenberg text to per-chapter cleaned corpus
- Token-distribution and word-cloud visualisations of the corpus
- Per-architecture training-curve and inference-speed figures

## Machine Learning Models

Three models were developed and evaluated, each on the same underlying corpus:

1. **Recurrent Neural Network (RNN):** implemented in PyTorch (`MLASS4v2.ipynb`, "RNN" section) — two stacked `nn.RNN` layers (128, 64 units) with a linear output head.
2. **Gated Recurrent Unit (GRU):** implemented in Keras (`MLASS4v2.ipynb`, "GRU" section) — two stacked GRU layers.
3. **Long Short-Term Memory (LSTM):** implemented in Keras across five iterations (`MLASS4v2.ipynb`, "Model 1"–"Model 5"); Model 5 is the final, best-performing version and is the one referred to as "LSTM" in the results below.

## Results

All figures below are read directly from the training-output cells of `MLASS4v2.ipynb` (the notebook explicitly labelled "Updated notebook with final results and parameter tuning"), not re-run or estimated.

| Model | Final-epoch (20/20) validation accuracy | Best epoch observed | Validation loss (final epoch) |
| :--- | :--- | :--- | :--- |
| LSTM (Model 5) | **98.17%** | 98.46% (epoch 17) | 0.123 |
| GRU | 7.84% | 9.78% (epoch 7) | 5.580 |
| RNN | 10.26% | 13.27% (epoch 2) | 6.079 |

The LSTM model is a clear, substantial winner on this corpus and this training run — the GRU and RNN models did not converge to a useful validation accuracy in the 20 epochs trained. This matches the original assignment's conclusion, though the original README's headline number (98.17%) was previously attributed to the wrong book; it is retained here because it is the correct figure for the LSTM model, independently confirmed against the notebook.

**Inference speed is not measured for direct comparison.** The notebook records "Average FPS" for all three models, but:
- The RNN figure (~2,775 samples/sec) was measured in PyTorch with single-sample forward passes on CPU/GPU as available.
- The LSTM and GRU figures (~27 and ~27 samples/sec respectively) were measured via Keras `model.predict()` calls, which carry per-call framework overhead not present in the raw PyTorch path — so the ~100x gap is very likely a framework-overhead artefact, not a genuine architecture-speed difference.
- The GRU and LSTM FPS-measurement cells both load the file `best_model.keras` before timing it, so the GRU figure may not reflect the GRU model at all. This appears to be a bug in the original notebook, reproduced here as an observation rather than corrected, since fixing model code is out of scope for this documentation pass.

For these reasons, FPS is reported per-model above as raw notebook output, but is **not** presented as a fair three-way comparison.

## Installation

```bash
git clone https://github.com/NotHFaure/Text-Based-Predictive-Models-LSTM-GRU-RNN.git
cd Text-Based-Predictive-Models-LSTM-GRU-RNN
```

No `requirements.txt` or environment file is committed. Install the libraries listed under [Requirements and Dependencies](#requirements-and-dependencies) manually, or reconstruct requirements from the notebooks' own `import` cells before running them.

## Usage

1. Ensure `pg766.txt`, `cleaned_chapters/` and `cleaned_data/` are present (all committed).
2. Open `MLASS4v2.ipynb` — this is the notebook with the final RNN/GRU/LSTM comparison and results in this README. `MLASS4.ipynb` is an earlier draft (data exploration and n-gram experiments only; no RNN/GRU/LSTM training).
3. Run cells top to bottom within a given model's section (Model 1–5 for LSTM, then GRU, then RNN). Each section is largely self-contained but depends on `cleaned_chapters/` having already been generated by an earlier cell in the same notebook run.
4. The RNN section uses `torch`, `torch.nn`, `TensorDataset` and `DataLoader` without its own import statement in the committed notebook — add these imports manually before running that section standalone.
5. Tokenizer/vocabulary artefacts (previously `tokenizer.pickle`, `tok.pickle`, `tok_model_4.pickle`, `word_to_index.pickle`, `index_to_word.pickle`) were removed for security reasons — see [Known Limitations](#known-limitations) for regeneration instructions.

## Requirements and Dependencies

Inferred from the notebooks' `import` statements (no version pins are committed):

`tensorflow` / `keras` · `torch` · `numpy` · `pandas` · `scikit-learn` · `nltk` · `spacy` · `beautifulsoup4` · `contractions` · `num2words` · `wordcloud` · `matplotlib`

`nltk` requires the `stopwords` and `wordnet` corpora (downloaded via `nltk.download(...)` in-notebook).

## Folder Structure

```
.
├── Images/                  # 13 figures: token distributions, word cloud, training curves (RNN/GRU/LSTM)
├── cleaned_chapters/        # 64 per-chapter cleaned text files, generated from pg766.txt
├── cleaned_data/            # Concatenated cleaned corpus
├── pg766.txt                # Raw corpus: Project Gutenberg eBook #766, "David Copperfield"
├── best_model.keras         # Saved Keras model (LSTM, Model 5 — see Known Limitations on provenance)
├── MLASS4.ipynb             # Early draft: data exploration, n-gram experiments
├── MLASS4v2.ipynb           # Final notebook: RNN, GRU and LSTM (Models 1-5) training and comparison
├── Training_Output_LSTM.txt # Captured training log for one LSTM run
├── LICENSE                  # MIT (code only — see licence note below)
├── .gitattributes
├── .gitignore
└── README.md
```

## Example Outputs

Selected figures from `Images/` (all pre-existing, not regenerated for this documentation pass):

- `Images/Training_LSTM_Model_5.png` — training/validation accuracy and loss curves for the final LSTM model
- `Images/Training_GRU.png`, `Images/Training_RNN.png` — training curves for the comparison architectures
- `Images/Word_Cloud.png`, `Images/Top_20_Words.png` — corpus vocabulary visualisations
- `Images/PreSampling_TokenizerDistribution_Top100.png`, `Images/PostSampling_TokenizerDistribution_Top100.png` — effect of rare-word oversampling on token frequency distribution

## Technologies

Python, TensorFlow/Keras, PyTorch, scikit-learn, NLTK, spaCy, Jupyter Notebook.

## Licence

MIT — see [`LICENSE`](LICENSE). The licence covers the author's own code (notebooks, preprocessing and model-definition logic) only. It does not extend to the text corpus, which is Project Gutenberg eBook #766 ("David Copperfield" by Charles Dickens, US public domain), and it does not assert any right over the original UNSW Canberra assignment brief, which has been removed from this repository (see below).

## Known Limitations

- **Single novel, single language:** trained and validated on one English-language novel; no evidence in the notebooks of testing against other corpora.
- **Word-level, not sub-word:** the tokenizer operates on whole words, so out-of-vocabulary words are mapped to a single `<UNK>` token rather than handled via sub-word units.
- **Coursework scope:** originally an assessed university assignment (ZEIT4151, UNSW Canberra), not a production system.
- **GRU and RNN did not converge** to a useful validation accuracy in the 20 epochs trained (see [Results](#results)); this is an accurate report of what the notebook shows, not a bug in this documentation.
- **FPS figures are not a fair cross-architecture comparison** — see [Results](#results) for why.
- **`best_model.keras` provenance:** the file is committed, and the FPS-measurement cells load it by that filename, but no cell in either committed notebook contains the `model.save(...)` call that produced it. It is presumed to be the Model 5 LSTM checkpoint based on how it is loaded and used, but this cannot be independently confirmed from the notebooks alone.
- **Previously shipped pickle artefacts (remediated 2026-07-29):** this repository formerly committed five `.pickle` files (`tokenizer.pickle`, `tok.pickle`, `tok_model_4.pickle`, `word_to_index.pickle`, `index_to_word.pickle`). Pickle deserialisation executes arbitrary code by design, and this is a public repository, so all five were removed rather than kept or blindly converted. To regenerate safe equivalents: re-run the Model 5 tokenizer-fitting cell in `MLASS4v2.ipynb` against the already-committed `cleaned_chapters/` text (this does not require retraining the neural network itself), then export with Keras's `tokenizer.to_json()` for `tokenizer.json`, and dump `tokenizer.word_index` as `vocab.json`. This was not done as part of this documentation pass, to avoid deserialising the removed pickles or fabricating their contents.
- **No committed `requirements.txt` or environment file** — dependencies must be reconstructed from the notebooks' `import` cells.
- **Assignment brief removed:** the original UNSW Canberra assignment brief PDF has been removed from this repository (2026-07-29) — it is the institution's material, not the author's to publish, independent of any privacy concern.

## Maintenance Status

**Archived coursework.** Completed for assessment in November 2024; not under active development. This 2026-07-29 update is a documentation and security correction pass only — no model code was changed or re-run.

## Complexity Rating

**Intermediate.** Custom preprocessing pipeline (chapter extraction, cleaning, oversampling) plus three distinct model implementations across two frameworks (Keras and PyTorch), but no distributed training, custom loss functions, or production deployment concerns.

## Acknowledgements

- ZEIT4151 Machine Learning, UNSW Canberra — unit under which this assignment was completed.
- Project Gutenberg, for the public-domain corpus (*David Copperfield*, Charles Dickens, eBook #766).
