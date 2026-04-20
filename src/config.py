"""
Configuration for the IEEE-CIS Fraud Detection system.
"""
import os

# ──────────────────────────── paths ────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "Data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "Data", "processed")

TRAIN_TRANSACTION = os.path.join(DATA_RAW, "train_transaction.csv")
TRAIN_IDENTITY = os.path.join(DATA_RAW, "train_identity.csv")

MODEL_PATH = os.path.join(DATA_PROCESSED, "model_pipeline.joblib")
METRICS_PATH = os.path.join(DATA_PROCESSED, "metrics.json")
TEST_DATA_PATH = os.path.join(DATA_PROCESSED, "test_data.joblib")
FEATURE_NAMES_PATH = os.path.join(DATA_PROCESSED, "feature_names.joblib")
PREPROCESSOR_PATH = os.path.join(DATA_PROCESSED, "preprocessor.joblib")
FEATURE_MAPPINGS_PATH = os.path.join(DATA_PROCESSED, "feature_mappings.joblib")

# ──────────────────────────── target ───────────────────────────
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

# ──────────── columns to drop (IDs / leaky) ────────────────────
DROP_COLS = ["TransactionID"]

# ──────────── model hyper-parameters ───────────────────────────
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 9,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

# ───────────── train / test split ratio ────────────────────────
TEST_RATIO = 0.20  # last 20% by time for held-out test

# ───────────── default classification threshold ────────────────
DEFAULT_THRESHOLD = 0.5

# ───────────── near-miss band (±) ──────────────────────────────
NEAR_MISS_BAND = 0.02
