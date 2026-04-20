"""
IEEE-CIS Fraud Detection — FastAPI Backend
Endpoints: /metrics, /analyze, /simulate
"""

# ── Fix for Python 3.14 + modern Windows where `wmic` is removed ──
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import joblib.externals.loky.backend.context as _loky_ctx
_loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
# ── end fix ──────────────────────────────────────────────────────────
import json
import os
import sys

import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DEFAULT_THRESHOLD,
    FEATURE_MAPPINGS_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NEAR_MISS_BAND,
    PREPROCESSOR_PATH,
    TEST_DATA_PATH,
)

# ──────────────────────── app setup ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model artefacts on startup
    _load_artefacts()
    yield

app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="LightGBM fraud detection with LIME explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────── global state ─────────────────────────
_model = None
_preprocessor = None
_feature_names = None
_test_data = None
_metrics = None
_lime_explainer = None
_mappings = None


def _load_artefacts():
    global _model, _preprocessor, _feature_names, _test_data, _metrics, _lime_explainer, _mappings

    if _model is not None:
        return  # already loaded

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model not found. Run  python src/train.py  first."
        )

    _model = joblib.load(MODEL_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    _feature_names = joblib.load(FEATURE_NAMES_PATH)
    _test_data = joblib.load(TEST_DATA_PATH)
    _mappings = joblib.load(FEATURE_MAPPINGS_PATH)

    with open(METRICS_PATH) as f:
        _metrics = json.load(f)

    # build LIME explainer from test data
    X_test_processed = _preprocessor.transform(_test_data["X_test"])
    transformed_names = _preprocessor.get_feature_names_out().tolist()

    _lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_test_processed[:2000]),  # sample for speed
        feature_names=transformed_names,
        class_names=["Legitimate", "Fraud"],
        mode="classification",
        random_state=42,
    )


# Startup pre-loading is now handled via the lifespan context manager above.


# ──────────────────── /metrics endpoint ────────────────────────
@app.get("/metrics")
def get_metrics():
    """Return model performance on the held-out test set."""
    _load_artefacts()
    return _metrics


# ──────────────────── /curves endpoint ────────────────────────
@app.get("/curves")
def get_curves():
    """Return ROC and PR curve data for visualization."""
    _load_artefacts()
    y_test = _test_data["y_test"]
    y_proba = _test_data["y_proba"]

    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_proba)

    # Downsample and sanitize points (roc_curve returns np.inf for the first threshold)
    def sanitize(v):
        v = float(v)
        return 1.0 if (v == np.inf or v > 1.0) else v

    def downsample(x, y, t):
        if len(x) > 100:
            idx = np.linspace(0, len(x) - 1, 100, dtype=int)
            t_idx = [i for i in idx if i < len(t)]
            return [float(v) for v in x[idx]], [float(v) for v in y[idx]], [sanitize(t[i]) for i in t_idx]
        return [float(v) for v in x], [float(v) for v in y], [sanitize(v) for v in t]

    fpr_ds, tpr_ds, roc_t_ds = downsample(fpr, tpr, roc_thresh)
    prec_ds, rec_ds, pr_t_ds = downsample(prec, rec, pr_thresh)

    return {
        "roc": {"fpr": fpr_ds, "tpr": tpr_ds, "thresholds": roc_t_ds},
        "pr": {"precision": prec_ds, "recall": rec_ds, "thresholds": pr_t_ds}
    }


# ──────────────────── /simulate endpoint ───────────────────────
class SimulateRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="Classification threshold")


@app.post("/simulate")
def simulate_threshold(req: SimulateRequest):
    """Recalculate metrics at a given threshold."""
    _load_artefacts()

    threshold = req.threshold
    y_test = _test_data["y_test"]
    y_proba = _test_data["y_proba"]

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc = float(roc_auc_score(y_test, y_proba))
    auprc = float(average_precision_score(y_test, y_proba))

    # customer friction / financial loss labels
    if fp / max(1, (fp + tn)) > 0.10:
        friction = "High"
    elif fp / max(1, (fp + tn)) > 0.05:
        friction = "Medium"
    else:
        friction = "Low"

    if fn / max(1, (fn + tp)) > 0.30:
        loss = "High"
    elif fn / max(1, (fn + tp)) > 0.15:
        loss = "Medium"
    else:
        loss = "Low"

    return {
        "threshold": round(threshold, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "auprc": round(auprc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "customer_friction": friction,
        "financial_loss": loss,
    }


# ──────────────────── /analyze endpoint ────────────────────────
class AnalyzeRequest(BaseModel):
    """Transaction features as key-value dict."""
    features: dict = Field(..., description="Transaction feature dict")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class LimeFeature(BaseModel):
    feature: str
    weight: float
    direction: str


class AnalyzeResponse(BaseModel):
    prediction: str
    probability_score: float
    threshold: float
    is_near_miss: bool
    lime_explanation: list


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_transaction(
    req: AnalyzeRequest,
    explain: bool = Query(True, description="Include LIME explanation (slower)"),
):
    """Predict fraud probability for a single transaction."""
    _load_artefacts()

    # build DataFrame from input
    try:
        row = pd.DataFrame([req.features])
        
        # 1. Apply Time Features (if not already provided by pre-processed `/sample` calls)
        if 'Transaction_hour' not in row.columns:
            if 'TransactionDT' in row.columns and pd.notna(row['TransactionDT'].iloc[0]):
                dt = float(row['TransactionDT'].iloc[0])
                row['Transaction_hour'] = np.floor(dt / 3600) % 24
                row['Transaction_day_of_week'] = np.floor(dt / (3600 * 24)) % 7
            else:
                row['Transaction_hour'] = 0.0
                row['Transaction_day_of_week'] = 0.0

        # 2. Apply Grouping & Velocity Features from Feature Store Mappings
        if 'TransactionAmt_to_mean_card1' not in row.columns: 
            c1 = row['card1'].iloc[0] if 'card1' in row.columns else None
            a1 = row['addr1'].iloc[0] if 'addr1' in row.columns else None
            uid = str(c1) + "_" + str(a1)
            
            u_mean = _mappings.get('uid_mean_amt', {}).get(uid, _mappings.get('global_card1_mean', 0))
            
            c1_mean = _mappings.get('card1_mean_amt', {}).get(c1, _mappings.get('global_card1_mean', 0))
            amt = float(row['TransactionAmt'].iloc[0]) if 'TransactionAmt' in row.columns else 0.0
            
            row['TransactionAmt_to_mean_card1'] = amt / (c1_mean + 1e-5)
            row['TransactionAmt_to_mean_uid'] = amt / (u_mean + 1e-5)

        # 3. Ensure columns align exactly with training features
        for col in _feature_names:
            if col not in row.columns:
                row[col] = np.nan
        row = row[_feature_names]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature error: {e}")

    # preprocess & predict
    X_processed = _preprocessor.transform(row)
    proba = float(_model.predict_proba(X_processed)[0, 1])
    pred_label = "Fraud" if proba >= req.threshold else "Legitimate"
    is_near_miss = abs(proba - req.threshold) <= NEAR_MISS_BAND

    # LIME explanation
    lime_features = []
    if explain:
        try:
            exp = _lime_explainer.explain_instance(
                np.array(X_processed[0]),
                _model.predict_proba,
                num_features=10,
                top_labels=1,
            )
            for feat_name, weight in exp.as_list(label=1):
                lime_features.append({
                    "feature": feat_name,
                    "weight": round(float(weight), 6),
                    "direction": "positive" if weight > 0 else "negative",
                })
        except Exception:
            pass  # lime can occasionally fail on edge-case rows

    return AnalyzeResponse(
        prediction=pred_label,
        probability_score=round(proba, 6),
        threshold=req.threshold,
        is_near_miss=is_near_miss,
        lime_explanation=lime_features,
    )


# ──────────── /sample endpoint (for UI convenience) ────────────
@app.get("/sample")
def get_sample_transaction():
    """Return a random test-set transaction as a feature dict."""
    _load_artefacts()
    idx = np.random.randint(0, len(_test_data["X_test"]))
    row = _test_data["X_test"].iloc[idx]
    actual = int(_test_data["y_test"][idx])
    features = {}
    for col in _feature_names:
        val = row.get(col, None)
        if val is not None and pd.notna(val):
            features[col] = val if not isinstance(val, (np.integer, np.floating)) else val.item()
        else:
            features[col] = None
    return {"features": features, "actual_label": actual}

if __name__ == "__main__":
    import uvicorn
    # When run via `python src/app.py`, start the uvicorn server
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
