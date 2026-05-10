from __future__ import annotations

import re

import pandas as pd
from langdetect import DetectorFactory
from langdetect import LangDetectException
from langdetect import detect


DetectorFactory.seed = 0

NON_RUSSIAN_CYRILLIC_RE = re.compile(r"[ЎўІіЇїЄєҐґ]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


def fix_mojibake(text: object) -> str:
  if pd.isna(text):
    return ""

  value = str(text)

  # UTF-8 Russian text sometimes appears as Windows-1252/Latin-1 mojibake:
  # "привет" -> "Ð¿Ñ€Ð¸Ð²ÐµÑ‚"
  if "Ð" not in value and "Ñ" not in value:
    return value

  try:
    return value.encode("latin1").decode("utf-8")
  except UnicodeError:
    return value


def has_cyrillic(text: object) -> bool:
  return bool(CYRILLIC_RE.search(str(text)))

def has_non_russian_cyrillic(text: object) -> bool:
  return bool(NON_RUSSIAN_CYRILLIC_RE.search(str(text)))


def count_cyrillic_chars(text: object) -> int:
  return len(CYRILLIC_RE.findall(str(text)))


def cyrillic_ratio(text: object) -> float:
  value = str(text)
  letters = LETTER_RE.findall(value)

  if not letters:
    return 0.0

  return count_cyrillic_chars(value) / len(letters)


def remove_quoted_previous_messages(text: object) -> str:
  value = "" if pd.isna(text) else str(text)

  kept_lines = []

  for line in value.splitlines():
    stripped = line.strip()

    if stripped.startswith(">"):
      continue

    kept_lines.append(line)

  return "\n".join(kept_lines)


def clean_text_basic(text: object) -> str:
  value = "" if pd.isna(text) else str(text)

  # Remove quoted previous comments first.
  value = remove_quoted_previous_messages(value)

  # Remove fenced and inline code.
  value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
  value = re.sub(r"`[^`]+`", " ", value)

  # Remove HTML comments and common generated sections.
  value = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
  value = re.sub(r"<details>.*?</details>", " ", value, flags=re.DOTALL | re.IGNORECASE)
  value = re.sub(r"<summary>.*?</summary>", " ", value, flags=re.DOTALL | re.IGNORECASE)

  # Remove markdown images and links but keep link text.
  value = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", value)
  value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)

  # Remove URLs, mentions, PR references.
  value = re.sub(r"https?://\S+|www\.\S+", " ", value)
  value = re.sub(r"@\w+", " ", value)
  value = re.sub(r"#\d+", " ", value)

  # Remove markdown/table noise.
  value = re.sub(r"^\s*\|.*\|\s*$", " ", value, flags=re.MULTILINE)
  value = re.sub(r"^\s*[-:| ]{3,}\s*$", " ", value, flags=re.MULTILINE)
  value = re.sub(r"[>*_~]", " ", value)

  # Remove common GitHub/system UI fragments.
  value = re.sub(r"(?i)\bresolved\b|\boutdated\b|\bcommittable suggestion\b", " ", value)

  value = re.sub(r"\s+", " ", value)

  return value.strip()


def detect_language_cleaned(text: object) -> str | None:
  value = "" if pd.isna(text) else str(text).strip()

  if not value:
    return None

  try:
    return detect(value)
  except LangDetectException:
    return None


def build_russian_candidates(
  df: pd.DataFrame,
  text_col: str = "body",
  min_chars: int = 8,
  max_chars: int = 500,
  min_cyrillic_chars: int = 4,
  min_cyrillic_ratio: float = 0.3,
) -> pd.DataFrame:
  if text_col not in df.columns:
    raise ValueError(f"Missing text column: {text_col}")

  result = df.copy()

  result["text"] = result[text_col].apply(clean_text_basic)
  result["char_len"] = result["text"].str.len()
  result["word_len"] = result["text"].str.split().str.len()
  result["has_cyrillic"] = result["text"].apply(has_cyrillic)
  result["has_non_russian_cyrillic"] = result["text"].apply(has_non_russian_cyrillic)
  result["cyrillic_chars"] = result["text"].apply(count_cyrillic_chars)
  result["cyrillic_ratio"] = result["text"].apply(cyrillic_ratio)

  result = result[
    result["has_cyrillic"]
    & ~result["has_non_russian_cyrillic"]
    & result["char_len"].between(min_chars, max_chars)
    & result["cyrillic_chars"].ge(min_cyrillic_chars)
    & result["cyrillic_ratio"].ge(min_cyrillic_ratio)
  ].copy()

  result["detected_language"] = result["text"].apply(detect_language_cleaned)
  result["is_russian"] = result["detected_language"].eq("ru")

  result = result[result["is_russian"]].copy()
  result = result.drop_duplicates(subset=["text"]).reset_index(drop=True)

  return result