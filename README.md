# LLM-Based Data Augmentation for Sentiment Classification

Using Large Language Models (Gemini) to generate synthetic Yelp reviews for improving text classification under low-resource conditions.

## Overview
This project investigates whether augmenting small, balanced sentiment datasets with LLM-generated examples improves classifier performance. Experiments compare baseline models trained on real data only vs. models trained on real + synthetic data.

## Key Features
- Synthetic data generation using Google Gemini API
- Sentiment classification on Yelp reviews (positive/negative)
- Multiple prompting strategies for data augmentation
- TF-IDF feature extraction
- Linear SVM classifier
- Performance evaluation across different training set sizes

## Technologies
- Python
- Google Gemini API
- scikit-learn
- TF-IDF vectorization
- Pandas, NumPy
- Jupyter Notebook

## Methodology
1. Subsample balanced Yelp review dataset to simulate low-resource scenario
2. Design LLM prompts to generate synthetic reviews
3. Train Linear SVM on:
   - Real data only (baseline)
   - Real + synthetic data (augmented)
4. Evaluate using accuracy, macro F1-score, and per-class metrics

## Key Findings
- Synthetic augmentation provides 6-7% improvement with only 20 samples per class
- Benefits diminish as real training data increases beyond 200 samples
- Most effective in extremely low-resource settings

## Files
- `llm-data-augmentation-sentiment.ipynb` - Main implementation
- `nlp-Najkar.pdf` - Detailed technical report

## Requirements
```
datasets
scikit-learn
google-generativeai
pandas
numpy
matplotlib
```

## Note
Gemini API key required for synthetic data generation.
```

---

**4. For Upwork portfolio:**

**Project title:** "LLM Data Augmentation for NLP"

**Description:**
```
Research project using Large Language Models (Google Gemini) to generate synthetic training data for sentiment classification. Designed prompting strategies, implemented data augmentation pipeline, and evaluated performance improvements in low-resource settings. Achieved 6-7% accuracy improvement with small datasets using Linear SVM and TF-IDF features.
