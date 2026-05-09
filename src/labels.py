# labels.py
# methods to convert raw labels to unified labels in [NEGATIVE, NON_NEGATIVE].

import numpy as np

NEGATIVE = -1
NON_NEGATIVE = 0

# label to int - fail if not recognized
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
  
  raise ValueError(f"Unknown label: {label!r}")

# label to float - NaN if not recognized
def parse_llm_label(label: object) -> float:
  value = str(label).strip().lower()

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

  return np.nan