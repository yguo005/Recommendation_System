# Recommendation System

A comprehensive recommendation system implementation using the Amazon Beauty dataset. This project demonstrates a complete recommendation pipeline from data preprocessing to model evaluation, including both retrieval and ranking stages.

## Overview

This project implements a two-stage recommendation system:
1. **Retrieval Stage**: Candidate generation using embedding-based models
2. **Ranking Stage**: Precise ranking of candidates using machine learning models

The system is built using popular frameworks including RecBole, PyTorch, XGBoost, and FAISS for efficient similarity search.

## Dataset

**Amazon Beauty Dataset**

The project uses the Amazon Beauty product review dataset, which includes:
- User-item interactions (ratings, timestamps)
- Item metadata (product features)
- Train/validation/test splits for model evaluation

## Project Structure

```
.
├── download_amazon_beauty.py      # Script to download and preprocess dataset
├── amazon_beauty_EDA.ipynb        # Exploratory Data Analysis
├── baseline.ipynb                 # Baseline models (Most Popular, Item-KNN)
├── Retrieval_self.ipynb          # Two-Tower retrieval model implementation
├── Retrieval_v2.ipynb            # Enhanced retrieval model (Colab-ready)
├── ranking_xgboost_v2.ipynb      # XGBoost ranking model
├── ranking_deepfm.ipynb          # DeepFM ranking model (RecBole)
├── eval_utils.py                 # Evaluation utilities for consistent metrics
├── requirements.txt              # Python dependencies
├── deepfm_config.yaml           # Configuration for DeepFM model
└── dataset/                      # Dataset directory (created after download)
    └── amazon-beauty/
        ├── amazon-beauty.inter   # All interactions
        ├── amazon-beauty.item    # Item features
        ├── amazon-beauty-train.inter
        ├── amazon-beauty-valid.inter
        └── amazon-beauty-test.inter
```

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv recbole_env
source recbole_env/bin/activate  # On Windows: recbole_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
python3 download_amazon_beauty.py
```

This will automatically download the Amazon Beauty dataset and create train/validation/test splits in the `dataset/amazon-beauty/` directory.

## Usage

### Step 1: Exploratory Data Analysis

Run the EDA notebook to understand the dataset characteristics:

```bash
jupyter notebook amazon_beauty_EDA.ipynb
```

**What it does:**
- Analyzes user interaction patterns
- Examines item popularity distributions
- Identifies rating patterns and temporal trends
- Provides data preprocessing insights

### Step 2: Baseline Models

Establish baseline performance with simple models:

```bash
jupyter notebook baseline.ipynb
```

**Models included:**
- **Most Popular**: Recommends the most popular items
- **Item-KNN**: Collaborative filtering using item-item similarity

**Baseline Results:**
- Most Popular HR@10: ~0.77%
- Item-KNN HR@10: ~1.23%

### Step 3: Retrieval Models

Generate candidate items using embedding-based approaches:

```bash
jupyter notebook Retrieval_self.ipynb
# or
jupyter notebook Retrieval_v2.ipynb
```

**What it does:**
- Implements Two-Tower architecture
- Learns separate embeddings for users and items
- Uses FAISS for efficient similarity search
- Generates top-K candidate items for each user

**Two-Tower Model:**
- User Tower: Maps user_id to user embedding
- Item Tower: Maps item_id to item embedding
- Training: Learns embeddings that bring positive pairs closer

### Step 4: Ranking Models

Rank candidate items using advanced machine learning models:

#### Option A: XGBoost Ranker

```bash
jupyter notebook ranking_xgboost_v2.ipynb
```

**Features:**
- Gradient boosting for ranking
- Automatic GPU/CPU selection
- Feature engineering with categorical encoding
- Timestamp normalization

**Performance:**
- Train AUC: ~0.92
- Validation/Test AUC: ~0.70

#### Option B: DeepFM Ranker

```bash
jupyter notebook ranking_deepfm.ipynb
```

**Features:**
- Deep Factorization Machine architecture
- Combines factorization machines with deep neural networks
- RecBole framework integration
- Configurable via `deepfm_config.yaml`

**Performance:**
- Recall@10: ~99.8% (train), ~99.9% (test)
- High recall indicates excellent candidate generation

## Models and Approaches

### 1. Baseline Models
- **Most Popular**: Simple popularity-based recommendation
- **Item-KNN**: Collaborative filtering using cosine similarity

### 2. Retrieval Models
- **Two-Tower Model**: Embedding-based retrieval with separate user and item towers
- **FAISS Integration**: Efficient approximate nearest neighbor search

### 3. Ranking Models
- **XGBoost**: Gradient boosting with 1:1 negative sampling
- **DeepFM**: Neural network combining FM and deep learning

## Evaluation Metrics

The project uses consistent evaluation metrics across all models:

- **Hit Rate@K (HR@K)**: Percentage of test cases where the true item appears in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain, considers ranking position
- **AUC**: Area Under the ROC Curve for binary classification
- **Recall@K**: Proportion of relevant items found in top-K recommendations

All models use the same evaluation protocol via `eval_utils.py` for fair comparison.

## Key Dependencies

- **RecBole** (1.2.0): Recommendation system library
- **PyTorch** (2.9.1): Deep learning framework
- **scikit-learn** (1.7.2): Machine learning utilities
- **pandas** (2.3.3): Data manipulation
- **numpy** (2.3.5): Numerical computing
- **FAISS**: Efficient similarity search (via RecBole)
- **Jupyter**: Interactive notebooks

**Note:** If you plan to use the XGBoost ranking notebook (`ranking_xgboost_v2.ipynb`), you'll need to install XGBoost separately:
```bash
pip install xgboost
```

See `requirements.txt` for complete list of dependencies.

## Running on Google Colab

The `Retrieval_v2.ipynb` notebook includes a Colab badge for easy execution in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yguo005/Recommendation_System/blob/main/Retrieval_v2.ipynb)

## Results Summary

| Model | Metric | Score | Notes |
|-------|--------|-------|-------|
| Most Popular | HR@10 | 0.77% | Simple baseline |
| Item-KNN | HR@10 | 1.23% | Collaborative filtering |
| Two-Tower | - | - | Candidate generation |
| XGBoost | AUC | 0.70 | Test set performance |
| DeepFM | Recall@10 | 99.9% | Test set performance |

## Development Workflow

1. **Data Download**: Use `download_amazon_beauty.py` to get the dataset
2. **EDA**: Explore data characteristics with `amazon_beauty_EDA.ipynb`
3. **Baseline**: Establish baseline metrics with `baseline.ipynb`
4. **Retrieval**: Generate candidates with `Retrieval_self.ipynb` or `Retrieval_v2.ipynb`
5. **Ranking**: Rank candidates with `ranking_xgboost_v2.ipynb` or `ranking_deepfm.ipynb`

## Notes

- The project uses RecBole's data preparation utilities for consistent data handling
- All models use the same train/validation/test splits for fair comparison
- Evaluation utilities in `eval_utils.py` ensure consistent random negative sampling
- DeepFM configuration can be customized via `deepfm_config.yaml`
- XGBoost automatically detects and uses GPU if available

## License

This project is for educational and research purposes.

## Acknowledgments

- Amazon Beauty dataset from Amazon Product Data
- RecBole framework for recommendation systems
- FAISS library for efficient similarity search
