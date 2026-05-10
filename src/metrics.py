from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
)


LABELS = [-1, 0]


def binary_metrics(y_true, y_pred) -> dict[str, float]:
  return {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_negative": precision_score(
      y_true,
      y_pred,
      pos_label=-1,
      zero_division=0,
    ),
    "recall_negative": recall_score(
      y_true,
      y_pred,
      pos_label=-1,
      zero_division=0,
    ),
    "f1_negative": f1_score(
      y_true,
      y_pred,
      pos_label=-1,
      zero_division=0,
    ),
    "precision_macro": precision_score(
      y_true,
      y_pred,
      labels=LABELS,
      average="macro",
      zero_division=0,
    ),
    "recall_macro": recall_score(
      y_true,
      y_pred,
      labels=LABELS,
      average="macro",
      zero_division=0,
    ),
    "f1_macro": f1_score(
      y_true,
      y_pred,
      labels=LABELS,
      average="macro",
      zero_division=0,
    ),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
  }


def evaluate_predictions(
  df: pd.DataFrame,
  true_col: str,
  pred_col: str,
) -> tuple[dict[str, float], pd.DataFrame]:
  clean = df[[true_col, pred_col]].dropna().copy()

  clean[true_col] = clean[true_col].astype(int)
  clean[pred_col] = clean[pred_col].astype(int)

  metrics = binary_metrics(clean[true_col], clean[pred_col])

  cm = confusion_matrix(
    clean[true_col],
    clean[pred_col],
    labels=LABELS,
  )

  cm_df = pd.DataFrame(
    cm,
    index=["true_negative", "true_non_negative"],
    columns=["pred_negative", "pred_non_negative"],
  )

  return metrics, cm_df