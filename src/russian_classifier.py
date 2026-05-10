# russian_classifier.py
# this file provides methods for cleaning messages in Russian.
# main exports of this file - fix_mojibrake(..) and build_russian_candidates(..).

from __future__ import annotations
import re
import pandas as pd

# use langdetect as a more fine-grained Russian language classifier
from langdetect import DetectorFactory, LangDetectException, detect
DetectorFactory.seed = 0

# attempt to exclude Belarusian and Ukrainian language comments,
# which also use cyrillic, and which are most often being
# misclassified as Russian, unlike Bulgarian or Serbian.
NON_RUSSIAN_CYRILLIC_RE = re.compile(r"[ЎўІіЇїЄєҐґ]")

# high-level check - filter out texts using Cyrillic script
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# filter for texts using either Cyrillic or Latin script
LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


# functions for basic script check and filtering

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



# some entries come as answers to previous messages, e.g.:
# > > This needs to be fixed.
# >   Done, please inspect it.
#     Looks good!               <----- only this message should be left
def remove_quoted_previous_messages(text: object) -> str:
  value = "" if pd.isna(text) else str(text)
  kept_lines = []

  # goal: simply remove lines starting with '>'.
  for line in value.splitlines():
    stripped = line.strip()
    if stripped.startswith(">"):
      continue
    kept_lines.append(line)

  return "\n".join(kept_lines)


# GitHub PR discussion comments may come with all sorts of things:
# code snippets, references, figures, links, and others that
# do not influence emotional sentiment of the message conveyed.
def clean_text_basic(text: object) -> str:
  value = "" if pd.isna(text) else str(text)

  # remove quoted previous comments first.
  value = remove_quoted_previous_messages(value)

  # remove fenced and inline code.
  value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
  value = re.sub(r"`[^`]+`", " ", value)

  # remove HTML comments and common generated sections.
  value = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
  value = re.sub(r"<details>.*?</details>", " ", value, flags=re.DOTALL | re.IGNORECASE)
  value = re.sub(r"<summary>.*?</summary>", " ", value, flags=re.DOTALL | re.IGNORECASE)

  # remove markdown images and links but keep link text.
  value = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", value)
  value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)

  # remove URLs, mentions, PR references.
  value = re.sub(r"https?://\S+|www\.\S+", " ", value)
  value = re.sub(r"@\w+", " ", value)
  value = re.sub(r"#\d+", " ", value)

  # remove markdown/table noise.
  value = re.sub(r"^\s*\|.*\|\s*$", " ", value, flags=re.MULTILINE)
  value = re.sub(r"^\s*[-:| ]{3,}\s*$", " ", value, flags=re.MULTILINE)
  value = re.sub(r"[>*_~]", " ", value)

  # remove common GitHub/system UI fragments.
  value = re.sub(r"(?i)\bresolved\b|\boutdated\b|\bcommittable suggestion\b", " ", value)

  value = re.sub(r"\s+", " ", value)

  return value.strip()

# use third party software to classify language
def detect_language_cleaned(text: object) -> str | None:
  value = "" if pd.isna(text) else str(text).strip()

  if not value:
    return None

  try:
    return detect(value)
  except LangDetectException:
    return None

# main dataset cleaning function.
# applies multiple criteria to a candidate message to allow it to become part of the grading set.
def build_russian_candidates(
  df: pd.DataFrame,         # read-only df
  text_col: str = "body",   # column containing source text
  min_chars: int = 8,
  max_chars: int = 500,
  min_cyrillic_chars: int = 4,
  min_cyrillic_ratio: float = 0.3,
) -> pd.DataFrame:
  
  # make sure source text column exists
  if text_col not in df.columns:
    raise ValueError(f"Missing text column: {text_col}")

  # do not modify original df
  result = df.copy()

  # place cleaned message in ["text"]
  result["text"] = result[text_col].apply(clean_text_basic)

  # derive diverse message characteristics
  result["char_len"] = result["text"].str.len()
  result["word_len"] = result["text"].str.split().str.len()
  result["has_cyrillic"] = result["text"].apply(has_cyrillic)
  result["has_non_russian_cyrillic"] = result["text"].apply(has_non_russian_cyrillic)
  result["cyrillic_chars"] = result["text"].apply(count_cyrillic_chars)
  result["cyrillic_ratio"] = result["text"].apply(cyrillic_ratio)

  # step 1 of cleaning - on basis of text appearance
  # apply message filters - drop all invalid entries
  result = result[
    # result has to have at least one cyrillic character
    result["has_cyrillic"]

    # result MUST NOT have any non-Russian cyrillic, as that automatically makes text non-Russian
    & ~result["has_non_russian_cyrillic"]        

    # cleaned message length must be within specified bounds
    & result["char_len"].between(min_chars, max_chars)

    # cleaned message must have at least a specified amount of Cyrillic chars
    & result["cyrillic_chars"].ge(min_cyrillic_chars)

    # cleaned message must consist of at least a certain percentage of Cyrillic 
    & result["cyrillic_ratio"].ge(min_cyrillic_ratio)
  ].copy()

  # step 2 of cleaning - on basis of a third party language classifier
  result["detected_language"] = result["text"].apply(detect_language_cleaned)
  result["is_russian"] = result["detected_language"].eq("ru")

  result = result[result["is_russian"]].copy()

  # lastly, drop repeated messages, if any
  result = result.drop_duplicates(subset=["text"]).reset_index(drop=True)

  return result