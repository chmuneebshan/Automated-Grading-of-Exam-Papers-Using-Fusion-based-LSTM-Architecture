# 📚 Automated Grading of Exam Papers Using Fusion-Based LSTM Architecture

This repository contains the full implementation of the MSc Data Science final project titled **"Automated Grading of Exam Papers Using Fusion-Based LSTM Architecture"**.  
It includes feature engineering, sentence vectorization, model development (fusion and baseline models), hyperparameter tuning, and evaluation.

---

## 🚀 Project Overview

Manual essay grading is time-consuming, subjective, and inconsistent.  
This project proposes a deep learning-based **fusion architecture** combining CNN and BiLSTM to automate grading by learning:
- Vocabulary-level features
- Readability-level features
- Sentence-level semantic vectors
- Chapter-level topic distributions

✅ Achieved **QWK = 0.981**, **R² = 0.962**, outperforming baseline models and prior studies.

---

## 📁 Repository Structure

```
├── App/
│   └── main.py                # Main script to run the application
│
├── Code/
│   ├── Feature Engineering.ipynb        # Feature extraction from essays
│   ├── Modelling.ipynb                   # Fusion model training
│   └── Baseline LSTM Modeling.ipynb       # Baseline non-fusion LSTM model training
│
├── Data/
│   ├── Raw/                     # Raw ASAP dataset (input essays)
│   ├── Processed/                # Preprocessed cleaned data
│   ├── Features/                 # Vocabulary, readability, and chapter-level features
│   └── nltk_data/                # Downloaded NLTK resources
│
├── Outputs/
│   ├── Model/
│   │   ├── word2vec_model.model      # Word2Vec trained model
│   │   ├── doc2vec_model.model       # Doc2Vec model
│   │   ├── lda_model.model           # LDA topic model
│   │   ├── lsi_model.model           # LSI semantic model
│   │   └── corpus_dictionary.dict    # Gensim dictionary for topic models
│   └── Baseline/             # Predictions, evaluation plots, etc.
│
├── LICENSE
├── README.md
└── requirements.txt          # Required Python packages
```

---

## 🔥 Key Features

- **Feature Engineering:** Length-based, POS-based, punctuation-based, readability indices, semantic embeddings.
- **Fusion Model Architecture:**
  - Vocabulary → CNN
  - Readability → CNN
  - Sentence Vectors → BiLSTM
  - Chapter-Level Topics → BiLSTM
  - Final Dense fusion layers
- **Baseline Non-Fusion Model:** Basic LSTM with vocab inputs for comparison.
- **Hyperparameter Tuning:** RandomSearch using KerasTuner.

---

## 📊 Results Summary

| Metric | Fusion Model | Baseline LSTM |
|:------:|:------------:|:-------------:|
| MSE    | 2.92          | 10.30         |
| MAE    | 0.93          | 2.47          |
| R²     | 0.962         | 0.868         |
| QWK    | 0.981         | 0.933         |

✅ Fusion model clearly outperforms the baseline and literature benchmarks.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/automated-essay-grading.git
   cd automated-essay-grading
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the **ASAP Automated Essay Scoring** dataset from Kaggle and place it inside `Data/Raw/`.

4. Run the notebooks in `Code/` step-by-step:
   - Feature extraction
   - Sentence vectorization
   - Model training
   - Evaluation

---

## 🧪 Major Libraries Used

- **TensorFlow / Keras**
- **Gensim** (Word2Vec, Doc2Vec, LDA, LSI)
- **Scikit-learn** (Evaluation metrics)
- **TextStat** (Readability scoring)
- **NLTK** (Text preprocessing)

---

## 💻 How to Run

- Run `Feature Engineering.ipynb` to generate features.
- Train models using `Modelling.ipynb` (Fusion model) or `Baseline LSTM Modeling.ipynb`.
- Run `main.py` to start app.

---

> 📢 **Note:**  
> This repository showcases that integrating handcrafted features with deep learning representations can build practical, scalable, and reliable automated grading solutions.
