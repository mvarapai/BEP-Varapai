from __future__ import annotations

import pandas as pd


NEGATIVE = -1
NON_NEGATIVE = 0


def to_binary_label(label: object) -> int:
  value = str(label).strip().lower()

  if value in {"negative", "neg", "-1"}:
    return NEGATIVE

  if value in {
    "positive",
    "neutral",
    "mixed",
    "non-negative",
    "non negative",
    "nonnegative",
    "0",
  }:
    return NON_NEGATIVE

  raise ValueError(f"Unknown sentiment label: {label!r}")


def parse_llm_label(label: object) -> float:
  if pd.isna(label):
    return float("nan")

  value = str(label).strip().lower()

  value = (
    value
    .replace(".", "")
    .replace(",", "")
    .replace(":", "")
    .replace('"', "")
    .replace("'", "")
  )

  if value in {"negative", "neg", "-1"}:
    return NEGATIVE

  if value in {
    "non-negative",
    "non negative",
    "nonnegative",
    "neutral",
    "positive",
    "0",
  }:
    return NON_NEGATIVE

  if "non-negative" in value or "non negative" in value or "nonnegative" in value:
    return NON_NEGATIVE

  if "negative" in value:
    return NEGATIVE

  return float("nan")