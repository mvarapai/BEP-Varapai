from __future__ import annotations

import json
import re
from pathlib import Path
import ast

import pandas as pd

from src.labels import to_binary_label


def load_novielli(path: str | Path) -> pd.DataFrame:
  df = pd.read_csv(
    path,
    sep=";",
    quotechar='"',
    encoding="utf-8",
  )

  required_columns = {"ID", "Polarity", "Text"}
  missing = required_columns - set(df.columns)

  if missing:
    raise ValueError(f"Novielli dataset is missing columns: {sorted(missing)}")

  df = df.rename(columns={
    "ID": "id",
    "Polarity": "original_label",
    "Text": "text",
  })

  df = df[["id", "text", "original_label"]].copy()

  df["text"] = df["text"].astype(str).str.strip()
  df["original_label"] = df["original_label"].astype(str).str.strip().str.lower()

  df = df[df["text"].ne("")]
  df = df[df["original_label"].ne("")]

  df["gold_label"] = df["original_label"].map(to_binary_label)

  return df[["id", "text", "original_label", "gold_label"]]


def _safe_dict(value: object) -> dict:
  if isinstance(value, dict):
    return value

  if pd.isna(value):
    return {}

  if isinstance(value, str):
    value = value.strip()

    if not value:
      return {}

    try:
      parsed = json.loads(value)
      if isinstance(parsed, dict):
        return parsed
    except json.JSONDecodeError:
      pass

    try:
      parsed = ast.literal_eval(value)
      if isinstance(parsed, dict):
        return parsed
    except (ValueError, SyntaxError):
      pass

  return {}

def load_coutinho(path: str | Path) -> pd.DataFrame:
  with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

  df = pd.DataFrame(data)

  required_columns = {
    "clean_message",
    "part2_aggregate",
    "tools",
  }

  missing = required_columns - set(df.columns)

  if missing:
    raise ValueError(f"Coutinho dataset is missing columns: {sorted(missing)}")

  df["part2_aggregate"] = df["part2_aggregate"].apply(_safe_dict)
  df["tools"] = df["tools"].apply(_safe_dict)

  df["polarity"] = df["part2_aggregate"].apply(lambda x: x.get("polarity"))

  if "discussion_polarity" in df.columns:
    mask = df["polarity"].eq("undefined")
    df.loc[mask, "polarity"] = df.loc[mask, "discussion_polarity"]

  df["text"] = df["clean_message"].astype(str).str.strip()

  df = df[df["text"].ne("")]
  df = df[df["polarity"].notna()]
  df = df[df["polarity"].ne("undefined")]

  df["gold_label"] = df["polarity"].apply(to_binary_label).astype(int)

  df["senticr_label"] = df["tools"].apply(lambda x: x.get("SentiCR"))
  df = df[df["senticr_label"].notna()]
  df["senticr_score"] = df["senticr_label"].apply(to_binary_label).astype(int)

  return df[
    [
      "text",
      "gold_label",
      "senticr_score",
    ]
  ].reset_index(drop=True)

def normalize_russian_text(text: object) -> str:
  if pd.isna(text):
    return ""

  value = str(text).strip()
  value = re.sub(r"http\S+|www\.\S+", " ", value)
  value = re.sub(r"@\w+", " ", value)
  value = re.sub(r"`{1,3}.*?`{1,3}", " ", value, flags=re.DOTALL)
  value = re.sub(r"[>#*_~]", " ", value)
  value = re.sub(r"\s+", " ", value)

  return value.strip()


def russian_score(text: str) -> float:
  letters = re.findall(r"[a-zA-Zа-яА-ЯёЁ]", text)

  if not letters:
    return 0.0

  cyrillic = re.findall(r"[а-яА-ЯёЁ]", text)
  return len(cyrillic) / len(letters)


def looks_russian(
  text: str,
  min_cyr_ratio: float = 0.5,
  min_cyr_chars: int = 3,
) -> bool:
  cyrillic = re.findall(r"[а-яА-ЯёЁ]", text)
  return len(cyrillic) >= min_cyr_chars and russian_score(text) >= min_cyr_ratio