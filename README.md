# ZEIT4151 Machine Learning Assignment 4B

**Predictive Text Module Development**  
**Module 3A: Word-Level Predictive Text Module**  
**Author:** Harrison Faure  
**Student ID:** z5364422  
**Date:** November 4, 2024

---

## Table of Contents

- [Introduction](#introduction)
- [Objective](#objective)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Introduction

This project develops a predictive text module for an HMI system aimed at supporting communication for ALS patients. The module provides word-level predictive text, suggesting the next word based on previous words typed. Three machine learning models (RNN, GRU, LSTM) were implemented to compare performance and suitability for text prediction.

## Objective

The goal was to create a word-level predictive text system that suggests the next word, improving typing efficiency for ALS patients by reducing input time. The module offers three suggestions for the next word and was trained on textual data from Project Gutenberg.

## Dataset

The dataset used in this project is *The Adventures of Tom Sawyer* by Mark Twain, sourced from Project Gutenberg. Key characteristics include:
- **Size:** 1,954,026 characters; 367,022 words
- **Unique Words:** 18,450
- **Preprocessing:** Text cleaning steps included lowercasing, punctuation removal, and word embedding for dimensionality reduction.

The dataset structure and characteristics are analyzed for effective model training, including handling class imbalance and overfitting through oversampling and data shuffling.

## Machine Learning Models

Three models were developed and evaluated:

1. **Recurrent Neural Network (RNN):** Baseline model, capturing short-term dependencies.
2. **Gated Recurrent Unit (GRU):** Improved RNN with mechanisms for long-term dependency.
3. **Long Short-Term Memory (LSTM):** Best-performing model, efficiently handling long sequences.

Each model’s performance was measured in terms of accuracy, speed, and resource efficiency. The LSTM model, configured with optimized parameters, outperformed the other models.

## Results

- **Parameter Tuning:** Extensive tuning was performed on the LSTM model, adjusting embedding dimensions, sequence lengths, and learning rates.
- **Performance Comparison:** The LSTM model achieved 98.17% validation accuracy, outperforming RNN and GRU.
- **FPS Comparison:** All models were benchmarked for FPS, with the LSTM performing slightly slower due to its complexity.

The results highlight the importance of model selection and parameter tuning in creating an effective predictive text solution.

## Code Structure

The project directory includes the following files and folders:
ASSIGNMENT 4/ ├── cleaned_chapters/ # Preprocessed text data for model training ├── cleaned_data/ # Raw and cleaned datasets ├── Images/ # Visualizations (e.g., token distributions, training progress) │ ├── PreSampling_TokenizerDistribution_Top100.png │ ├── PostSampling_TokenizerDistribution_Bottom100.png │ └── ... (additional images) ├── Assignment 4B-Text.pdf # Report detailing model development and results ├── best_model.keras # Saved model for deployment ├── MLASS4.ipynb # Main Jupyter notebook with code and explanations ├── MLASS4v2.ipynb # Updated notebook with final results and parameter tuning ├── tok_model_4.pickle # Tokenizer for data processing └── Training_Output_LSTM.txt # Training results and metrics


## Usage

1. **Dataset Preparation:** Ensure the text data is available in `cleaned_data` folder.
2. **Run Jupyter Notebook:** Open `MLASS4.ipynb` or `MLASS4v2.ipynb` in Jupyter Notebook to reproduce the results. All code and explanations are included in the notebook.
3. **Model Training:** Execute the cells to train the models (RNN, GRU, LSTM) and view visualizations of training progress.
4. **Evaluate Model Performance:** Compare validation accuracy, loss, and FPS between models.
5. **Visualization:** All images are stored in the `Images` folder for reference.

## Acknowledgments

This project is part of the ZEIT4151 Machine Learning assignment at UNSW, designed to help students apply machine learning techniques to real-life challenges in predictive text for accessibility. Special thanks to Project Gutenberg for the dataset used in this project.

--- 
