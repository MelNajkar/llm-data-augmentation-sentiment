# LLM-Based Synthetic Data Augmentation for Low-Resource Sentiment Classification

## Overview

Training NLP models with limited labeled data often leads to poor generalization and unstable performance. In many real-world scenarios, collecting annotated text data is expensive and time-consuming.

This project investigates whether Large Language Models (Google Gemini) can generate synthetic training examples to improve sentiment classification performance under low-resource conditions.

---

## Objective

Evaluate the effectiveness of LLM-generated synthetic Yelp reviews in improving classification performance when only a small number of real training samples are available.

---

## Methodology

A balanced subset of the Yelp Review dataset was used to simulate low-resource scenarios.

### Experimental Pipeline

1. Subsample the dataset to create small balanced training sets (20–200 samples per class)
2. Design structured prompts for Google Gemini to generate synthetic positive and negative reviews
3. Augment the real dataset with generated synthetic samples
4. Convert text into numerical features using **TF-IDF vectorization**
5. Train a **Linear Support Vector Machine (SVM)** classifier
6. Evaluate performance using:
   - Accuracy
   - Macro F1-score
   - Per-class precision and recall

Baseline models trained on real data only were compared against augmented models trained on real + synthetic data.

---

## Results

- Synthetic augmentation improved accuracy by approximately **6–7%** when only 20 samples per class were available.
- Performance gains diminished as the real training data increased beyond 200 samples.
- The approach is most effective in extremely low-resource settings.

These findings indicate that LLM-based data augmentation is a practical strategy when labeled data is scarce.

---

## Tech Stack

- Python
- scikit-learn
- Google Gemini API
- TF-IDF vectorization
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/llm-data-augmentation-sentiment.git
cd llm-data-augmentation-sentiment

# Install dependencies:
pip install -r requirements.txt

# Set your Gemini API key:
export GOOGLE_API_KEY="your_api_key_here"

## Usage
Run the main notebook:

llm-data-augmentation-sentiment.ipynb

## Project Structure

.
├── llm-data-augmentation-sentiment.ipynb
├── nlp-Najkar.pdf
├── requirements.txt
└── README.md

## Future Improvements:

Compare multiple LLM providers (Gemini vs GPT vs open-source models)
Replace TF-IDF with transformer-based embeddings
Automate prompt optimization
Evaluate robustness under domain shift
Convert notebook into a modular Python package
