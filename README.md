# 🐦 Twitter Sentiment Analysis using DistilBERT + Logistic Regression

> **Natural Language Processing (NLP) — Course Project**
> American International University - Bangladesh (AIUB), Dhaka, Bangladesh

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![sklearn](https://img.shields.io/badge/scikit--learn-Classifier-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Author](#author)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

This project performs **Twitter Sentiment Analysis** using a transfer learning approach. Tweets are encoded into rich numerical representations using **DistilBERT** — a lightweight, fast version of Google's BERT — and then classified using **Logistic Regression** into one of three sentiment categories:

| Label | Sentiment |
|---|---|
| `1` | Positive |
| `0` | Neutral |
| `-1` | Negative |

Instead of training a deep learning model from scratch, this project uses DistilBERT purely as a **feature extractor**, converting raw tweet text into 768-dimensional embeddings. These embeddings are then fed into a Logistic Regression classifier — making the pipeline fast, lightweight, and effective.

---

## Author

| Name | ID | Group |
|---|---|---|
| Salahuddin Elias Khan | 23-93143-3 | Group 9 |

**Course:** Natural Language Processing 
**American International University - Bangladesh (AIUB), Dhaka, Bangladesh

---

## Dataset

- **File:** `Twitter_Data.csv`
- **Total Records:** ~192,000 tweets
- **Columns:**
  - `clean_text` — preprocessed tweet text
  - `category` — sentiment label (`-1`, `0`, or `1`)
- **Subset Used:** First 2,000 rows (`batch_1`) for this experiment

### Sample Data

| clean_text | category |
|---|---|
| when modi promised minimum government maximum... | -1 |
| talk all the nonsense and continue all the dra... | 0 |
| what did just say vote for modi welcome bjp t... | 1 |

---

## Architecture
Raw Tweet Text
↓
DistilBERT Tokenizer  →  Token IDs + Padding + Attention Mask
↓
DistilBERT Model      →  768-dim CLS Token Embedding per Tweet
↓
Logistic Regression   →  Sentiment Prediction (-1 / 0 / 1)

**Why DistilBERT?**
DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its language understanding capability. It is ideal for resource-constrained experiments without sacrificing meaningful accuracy.

**Why Logistic Regression on top?**
Using a simple classifier on top of powerful pre-trained embeddings is a well-established NLP pattern. It avoids overfitting on small datasets and trains extremely fast compared to fine-tuning the full transformer.

---

## Pipeline

### Step 1 — Load Data
```python
df = pd.read_csv("Twitter_Data.csv", header=None)
batch_1 = df[:2000]
batch_1 = batch_1.dropna()
```

### Step 2 — Load DistilBERT
```python
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel,
    ppb.DistilBertTokenizer,
    'distilbert-base-uncased'
)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```

### Step 3 — Tokenize Tweets
```python
tokenized = batch_1[0].apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True)
)
```

### Step 4 — Pad Sequences
```python
max_len = max(len(i) for i in tokenized.values)
padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
```

### Step 5 — Create Attention Masks
```python
attention_mask = np.where(padded != 0, 1, 0)
```
Tells DistilBERT which tokens are real (`1`) and which are padding (`0`).

### Step 6 — Generate DistilBERT Embeddings
```python
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()
```
Extracts the **[CLS] token embedding** (first token) from DistilBERT's output — a 768-dimensional vector summarizing the entire tweet's meaning.

### Step 7 — Train Logistic Regression
```python
labels = batch_1[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
```

### Step 8 — Evaluate
```python
lr_clf.score(test_features, test_labels)
```

---

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/twitter-sentiment-distilbert.git
cd twitter-sentiment-distilbert

# Install dependencies
pip install transformers torch scikit-learn pandas numpy
```

Or install all at once using the requirements file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
torch
transformers
scikit-learn
pandas
numpy
---

## Usage

### Running in Google Colab (Recommended)
1. Upload `Twitter_Data.csv` to your Colab session
2. Open the `.ipynb` notebook in Colab
3. Run all cells top to bottom

### Running Locally

```bash
# Make sure Twitter_Data.csv is in the same directory as the notebook
jupyter notebook Salah_ud_Din_Elias_Khan_NLP_Group_9_Version_1_2_.ipynb
```

> **Note:** The first run will automatically download the `distilbert-base-uncased` pretrained model (~250MB) from HuggingFace. An internet connection is required for this step.

---

## Results

The model classifies tweets into three sentiment categories — Negative (`-1`), Neutral (`0`), and Positive (`1`) — using DistilBERT embeddings fed into Logistic Regression.

| Component | Detail |
|---|---|
| Pretrained Model | `distilbert-base-uncased` |
| Classifier | Logistic Regression |
| Training/Test Split | 75% / 25% (default sklearn split) |
| Dataset Subset Used | 2,000 tweets |
| Embedding Dimension | 768 (CLS token) |

---

## Key Concepts

**Transfer Learning** — Rather than training a language model from scratch, DistilBERT's pre-trained weights (trained on Wikipedia + BookCorpus) are used directly to extract meaningful tweet features.

**CLS Token** — In BERT-style models, the `[CLS]` token at position `0` of the output captures a summary representation of the entire input sequence, making it ideal for classification tasks.

**Attention Mask** — A binary mask that tells the transformer which token positions contain real text (`1`) and which are padding (`0`), ensuring padding does not affect the model's attention computation.

---

## References

1. Sanh V, et al. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108. 2019.
2. Devlin J, et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805. 2018.
3. HuggingFace Transformers Documentation. https://huggingface.co/docs/transformers
4. Scikit-learn Documentation. https://scikit-learn.org

---

*Natural Language Processing [MScCS] | Military Institute of Science and Technology (MIST) | Dhaka, Bangladesh*
