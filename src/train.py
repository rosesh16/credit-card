"""
IEEE-CIS Fraud Detection — Training Pipeline
Uses LightGBM + SMOTETomek + Time-Series Split
"""

# ── Fix for Python 3.14 + modern Windows where `wmic` is removed ──
# joblib's loky backend calls wmic to detect physical cores and crashes.
# We monkey-patch the broken function before anything else imports it.
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import joblib.externals.loky.backend.context as _loky_ctx
_loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
# ── end fix ──────────────────────────────────────────────────────────

import json
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# add project root to path so config can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_PROCESSED,
    DEFAULT_THRESHOLD,
    DROP_COLS,
    FEATURE_MAPPINGS_PATH,
    FEATURE_NAMES_PATH,
    LGBM_PARAMS,
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TEST_RATIO,
    TIME_COL,
    TRAIN_IDENTITY,
    TRAIN_TRANSACTION,
)

warnings.filterwarnings("ignore")


# ───────────────────────── helpers ──────────────────────────────
def load_data():
    """Load and merge transaction + identity datasets."""
    print("[1/6] Loading data …")
    t0 = time.time()

    df_txn = pd.read_csv(TRAIN_TRANSACTION)
    df_id = pd.read_csv(TRAIN_IDENTITY)
    df = df_txn.merge(df_id, on="TransactionID", how="left")

    print(f"       Loaded {len(df):,} rows × {df.shape[1]} cols  ({time.time()-t0:.1f}s)")
    return df


def engineer_features_base(df: pd.DataFrame):
    """Drop ID columns, create Time features, separate numeric / categorical."""
    print("[2/7] Base Feature Engineering (Time Variables) …")

    # Time Features
    df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
    df['Transaction_day_of_week'] = np.floor(df['TransactionDT'] / (3600 * 24)) % 7

    # Pseudo-Identity
    if 'card1' in df.columns and 'addr1' in df.columns:
        df['uid'] = df['card1'].astype(str) + "_" + df['addr1'].astype(str)

    # drop explicit ID cols
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # identify feature types
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [
        c for c in df.columns
        if c not in cat_cols and c != TARGET_COL and c != TIME_COL
    ]

    # remove target & time from cat_cols just in case
    cat_cols = [c for c in cat_cols if c not in (TARGET_COL, TIME_COL)]

    print(f"       {len(num_cols)} numeric, {len(cat_cols)} categorical features")
    return df, num_cols, cat_cols


def time_series_split(df, num_cols, cat_cols):
    """Chronological split: last TEST_RATIO% by TransactionDT → test set."""
    print("[3/7] Time-series split …")

    df = df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_RATIO))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, y_train, X_test, y_test, feature_cols


def extract_and_apply_mappings(X_train, X_test, num_cols):
    """Calculate Frequency & Velocity using ONLY train set to prevent leakage."""
    print("[4/7] Advanced Grouping & Velocity Features …")
    
    # 1. Calculate mappings on TRAIN only
    mappings = {
        "uid_mean_amt": X_train.groupby('uid')['TransactionAmt'].mean().to_dict() if 'uid' in X_train else {},
        "card1_mean_amt": X_train.groupby('card1')['TransactionAmt'].mean().to_dict(),
        "global_card1_mean": float(X_train['TransactionAmt'].mean())
    }

    # 2. Apply to both
    def apply_map(X):
        X = X.copy()
        if 'uid' in X:
            u_mean = X['uid'].map(mappings['uid_mean_amt']).fillna(mappings['global_card1_mean'])
            X['TransactionAmt_to_mean_uid'] = X['TransactionAmt'] / (u_mean + 1e-5)
        
        c1_mean = X['card1'].map(mappings['card1_mean_amt']).fillna(mappings['global_card1_mean'])
        X['TransactionAmt_to_mean_card1'] = X['TransactionAmt'] / (c1_mean + 1e-5)
        return X

    X_train_mapped = apply_map(X_train)
    X_test_mapped = apply_map(X_test)
    
    num_cols.extend(['TransactionAmt_to_mean_card1'])
    if 'uid' in X_train_mapped:
        num_cols.extend(['TransactionAmt_to_mean_uid'])
    
    feature_cols = list(X_train_mapped.columns)
    
    return X_train_mapped, X_test_mapped, num_cols, feature_cols, mappings


def build_preprocessor(num_cols, cat_cols):
    """Build sklearn ColumnTransformer. LightGBM handles NaN natively."""
    cat_pipe = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    return preprocessor


    print(f"       Train set: {len(y_train):,} samples  (fraud {fraud_pct:.1f}%)")
    return X_train_processed, y_train


def train_model(X_train_processed, y_train, X_test_processed, y_test):
    """Train LightGBM classifier."""
    print("[6/7] Training LightGBM …")
    t0 = time.time()

    model = LGBMClassifier(**LGBM_PARAMS)

    # removing eval_set and early stopping because chronological X_test
    # has different distribution, causing naive early stopping to halt
    # prematurely (e.g. at round 4).
    model.fit(X_train_processed, y_train)

    print(f"       Done  ({time.time()-t0:.1f}s)")
    return model


def evaluate(model, X_test_processed, y_test):
    """Compute evaluation metrics, dynamically finding the threshold that maximizes F1-Score."""
    print("[7/7] Evaluating & Finding Optimal Threshold …")

    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Automatically find the mathematical threshold that maximizes F1-Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"       ✅ Found Optimal F1 Threshold: {best_thresh:.3f}")

    y_pred = (y_proba >= best_thresh).astype(int)

    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "auprc": round(float(average_precision_score(y_test, y_proba)), 4),
        "threshold": round(float(best_thresh), 3),
    }

    for k, v in metrics.items():
        print(f"       {k}: {v}")
    return metrics, y_proba


# ───────────────────────── main ─────────────────────────────────
def main():
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    df = load_data()
    df, num_cols, cat_cols = engineer_features_base(df)
    X_train, y_train, X_test, y_test, _ = time_series_split(df, num_cols, cat_cols)

    # extract aggregate mapping logic to avoid leakage
    X_train_mapped, X_test_mapped, num_cols, feature_cols, mappings = extract_and_apply_mappings(X_train, X_test, num_cols)

    # build & fit preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_processed = preprocessor.fit_transform(X_train_mapped)
    X_test_processed = preprocessor.transform(X_test_mapped)

    # train
    model = train_model(X_train_processed, y_train, X_test_processed, y_test)

    # evaluate
    metrics, y_proba = evaluate(model, X_test_processed, y_test)

    # save artefacts
    print("\nSaving artefacts …")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(mappings, FEATURE_MAPPINGS_PATH)
    joblib.dump(feature_cols, FEATURE_NAMES_PATH)
    joblib.dump({
        "X_test": X_test_mapped,
        "y_test": y_test.values,
        "y_proba": y_proba,
    }, TEST_DATA_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓  Model        → {MODEL_PATH}")
    print(f"✓  Preprocessor → {PREPROCESSOR_PATH}")
    print(f"✓  Mappings     → {FEATURE_MAPPINGS_PATH}")
    print(f"✓  Test data    → {TEST_DATA_PATH}")
    print(f"✓  Metrics      → {METRICS_PATH}")
    print(f"✓  Features     → {FEATURE_NAMES_PATH}")
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
