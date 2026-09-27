# Categorical Feature Encoding Challenge II

Notes and code exploring how to handle categorical variables in machine learning, based on the `cat-in-the-dat-ii` Kaggle competition dataset.

## Topics covered

- Nominal vs. ordinal, binary, and cyclic categorical variables
- Label Encoding vs. One-Hot Encoding (and when to use each)
- Sparse vs. dense matrices — memory comparison for high-cardinality OHE
- Handling missing values as a separate category (`"NONE"`)
- Handling unseen/rare categories in production (`"RARE"` bucket)
- Combining train + test to learn the full category vocabulary, without overfitting (via proper cross-validation design)
- Creating new features by combining categorical columns
- Baseline models: Logistic Regression (OHE) vs. Random Forest (Label Encoding)


## Setup

```bash
pip install pandas numpy scikit-learn xgboost kagglehub
```

Data is downloaded via `kagglehub` and is not committed to this repo — run the notebook's download cell to fetch it into `input/`.

## Running the pipeline

```bash
cd src
python create_folds.py       # creates cat_train_folds.csv with stratified k-folds
python ohe_logres.py         # One-Hot Encoding + Logistic Regression baseline
python lbl_rf.py             # Label Encoding + Random Forest baseline
```

## Key takeaway

Logistic Regression with One-Hot Encoding outperformed Random Forest with Label Encoding on this dataset without any hyperparameter tuning — a reminder that encoding strategy should match the model type (linear models favor OHE; tree-based models tolerate label encoding).
