# artifacts.py
# file containing helpers used when saving or loading files to/from the disk.

import pandas as pd
from pathlib import Path

# save .csv and avoid Cyrillic text encoding faults by enforcing utf-8-sig.
def save_csv(df: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(path, index=False, encoding="utf-8-sig")
  print(f"Saved {path} ({len(df)} rows)", flush=True)

# save selected candidate messages in Excel for further manual labeling.
def save_labeling_excel(candidates: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)

  if "text" not in candidates.columns:
    raise ValueError("Cannot create labeling Excel: missing text column")

  labeling = candidates[["text"]].copy()
  labeling.insert(1, "gold_label", "")

  labeling.to_excel(path, index=False, engine="openpyxl")

  print(f"Saved labeling Excel: {path} ({len(labeling)} rows)")

# safer version of reading .csv
def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
  if not path.exists():
    return None

  print(f"Using existing file: {path}", flush=True)
  return pd.read_csv(path, encoding="utf-8-sig")
