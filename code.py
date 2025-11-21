# Data load

import os, json, numpy as np, pandas as pd

# Paths
TRAIN_PATH = "train_data.json"
TEST_PATH = "test_data.json"
METRIC_NAMES_PATH = "metric_names.json"
METRIC_EMB_PATH = "metric_name_embeddings.npy"

# Load
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)
with open(METRIC_NAMES_PATH, "r", encoding="utf-8") as f:
    metric_names = json.load(f)
metric_embeddings = np.load(METRIC_EMB_PATH)

# DataFrames
train_df = pd.DataFrame(train_data)
test_df  = pd.DataFrame(test_data)

# Types & nulls
train_df["score"] = train_df["score"].astype(float)
for df in (train_df, test_df):
    df["system_prompt"] = df["system_prompt"].fillna("")
    df["user_prompt"] = df["user_prompt"].fillna("")
    df["response"] = df["response"].fillna("")
    df["combined_text"] = (
        df["system_prompt"].astype(str)
        + " [USER] " + df["user_prompt"].astype(str)
        + " [BOT] " + df["response"].astype(str)
    )

# Metric index map
metric_to_idx = {name: i for i, name in enumerate(metric_names)}

# Attach metric embeddings
def map_metric_emb(name):
    return metric_embeddings[metric_to_idx[name]] if name in metric_to_idx else np.zeros(metric_embeddings.shape[1])

train_df["metric_embedding"] = train_df["metric_name"].apply(map_metric_emb)
test_df["metric_embedding"]  = test_df["metric_name"].apply(map_metric_emb)

# Quick report
print("Train samples:", len(train_df), "| Test samples:", len(test_df))
print("Metric embeddings shape:", metric_embeddings.shape, "| Distinct metrics in train:", train_df["metric_name"].nunique())
print("Example train row:")
print(train_df[["metric_name","score"]].head(3))
print("\nCombined text preview:\n", train_df["combined_text"].iloc[0][:300], "...")


# TF-IDF + SVD text representations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack
import numpy as np
import os, gc

# Combine corpus for consistent vocab
corpus = train_df["combined_text"].tolist() + test_df["combined_text"].tolist()

print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    strip_accents="unicode"
)
X_tfidf = tfidf.fit_transform(corpus)
print("TF-IDF shape:", X_tfidf.shape)

# Dimensionality reduction
print("Running Truncated SVD (256-D)...")
svd = TruncatedSVD(n_components=256, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Split back into train/test
X_train_text = X_svd[: len(train_df)]
X_test_text  = X_svd[len(train_df):]

# Save caches (optional)
np.save("train_text_svd.npy", X_train_text)
np.save("test_text_svd.npy",  X_test_text)

print("Text embeddings ready:", X_train_text.shape, X_test_text.shape)
gc.collect();


# Build features + balanced training loop (regression model)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# ----- Targets -----
y = train_df["score"].values.astype(float)

# Feature matrix: text SVD + metric embeddings
# metric_embedding column currently holds 1D numpy arrays
metric_mat = np.stack(train_df["metric_embedding"].values)  

# Concatenate text and metric features
X_train_feats = np.hstack([X_train_text, metric_mat])       
X_train_feats = pd.DataFrame(X_train_feats)                

# ----- Stratification bins for scores -----
num_bins = 5
y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

# ----- CV training loop -----
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
models = []

for fold, (tr, va) in enumerate(kf.split(X_train_feats, y_bins), 1):
    model = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=fold,
    )
    model.fit(X_train_feats.iloc[tr], y[tr])
    preds = model.predict(X_train_feats.iloc[va])
    oof[va] = preds
    rmse = mean_squared_error(y[va], preds) ** 0.5
    print(f"Fold {fold} RMSE: {rmse:.4f}")
    models.append(model)

cv_rmse = mean_squared_error(y, oof) ** 0.5
print(f"OOF RMSE: {cv_rmse:.4f}")


# Build test features + inference & regression submission

import numpy as np, pandas as pd

# Build X_test_feats: [text SVD + metric embeddings]
metric_mat_test = np.stack(test_df["metric_embedding"].values)   
X_test_feats = np.hstack([X_test_text, metric_mat_test])      

# Ensemble predictions (regression)
preds = np.zeros(len(X_test_feats), dtype=np.float32)
for model in models:
    preds += model.predict(X_test_feats)
preds /= len(models)

# Clip and round to whole numbers in [0, 10]
preds = np.clip(preds, 0, 10)
preds = np.rint(preds).astype(int)
preds_reg_int = preds.copy()

# Save regression submission 
submission_reg = pd.DataFrame({
    "ID": np.arange(1, len(preds) + 1),
    "score": preds
})
submission_reg.to_csv("submission_reg.csv", index=False)

print("✅ Saved submission_reg.csv")
print(submission_reg["score"].value_counts().sort_index())
print(submission_reg.head(10))


# Ordinal classification (10 thresholds) with class-balanced logistic models

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Ensure labels are integers in [0..10]
y_int = y.astype(int)

# We'll train 10 binary models for "y >= k" (k = 1..10)
Ks = list(range(1, 11))
models_ord = []
probs_test = []

print("Training ordinal classifiers for thresholds k=1..10")

for k in Ks:
    y_bin = (y_int >= k).astype(int)
    clf = make_pipeline(
        StandardScaler(with_mean=False), 
        LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            C=1.0,
        )
    )
    clf.fit(X_train_feats, y_bin)
    models_ord.append(clf)
    p = clf.predict_proba(X_test_feats)[:, 1]
    probs_test.append(np.clip(p, 0.0, 1.0))


probs_test = np.vstack(probs_test) 
exp_score = probs_test.sum(axis=0)

preds_ord = np.rint(np.clip(exp_score, 0, 10)).astype(int)
preds_ord_int = preds_ord.copy()

submission_ord = pd.DataFrame({
    "ID": np.arange(1, len(preds_ord) + 1),
    "score": preds_ord
})
submission_ord.to_csv("submission_ord.csv", index=False)

print("✅ Saved submission_ord.csv (ordinal classification). Distribution:")
print(submission_ord["score"].value_counts().sort_index())
print(submission_ord.head(12))


# Weighted blend of Ordinal Classification + Regression

import numpy as np
import pandas as pd

# Blend integer predictions from ordinal and regression models
assert "preds_ord_int" in globals(), "Ordinal predictions not available"
assert "preds_reg_int" in globals(), "Regression predictions not available"
assert len(preds_ord_int) == len(preds_reg_int)

w_ord, w_reg = 0.6, 0.4
score_blend = (w_ord * preds_ord_int + w_reg * preds_reg_int).clip(0, 10)
score_blend_int = np.rint(score_blend).astype(int)

submission_blend = pd.DataFrame({
    "ID": np.arange(1, len(score_blend_int) + 1),
    "score": score_blend_int
})
submission_blend.to_csv("submission_blended.csv", index=False)

print("✅ Saved submission_blended.csv (ensemble of ordinal + regression)")
print(submission_blend["score"].value_counts().sort_index())
print(submission_blend.head(10))
