from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.datasets import looks_russian
from src.datasets import normalize_russian_text
from src.datasets import russian_score


def prepare_russian(input_path: Path, output_path: Path) -> None:
  if not input_path.exists():
    raise FileNotFoundError(f"Missing raw Russian data: {input_path}")

  df = pd.read_csv(input_path)

  if "text" not in df.columns and "body" in df.columns:
    df = df.rename(columns={"body": "text"})

  if "text" not in df.columns:
    raise ValueError("Russian raw dataset must contain either 'text' or 'body' column.")

  df["text"] = df["text"].apply(normalize_russian_text)
  df["ru_score"] = df["text"].apply(russian_score)
  df["is_russian"] = df["text"].apply(looks_russian)
  df["char_len"] = df["text"].str.len()
  df["word_len"] = df["text"].str.split().str.len()

  df = df[
    df["is_russian"]
    & df["char_len"].between(8, 500)
  ].copy()

  df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

  if output_path.exists():
    raise FileExistsError(
      f"{output_path} already exists. "
      "Refusing to overwrite the frozen Russian dataset."
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_path, index=False, encoding="utf-8")

  print(f"Saved {len(df)} rows to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input",
    default="data/russian_generated/russian_raw.csv",
  )
  parser.add_argument(
    "--output",
    default="data/processed/russian.csv",
  )

  args = parser.parse_args()

  prepare_russian(
    input_path=Path(args.input),
    output_path=Path(args.output),
  )


if __name__ == "__main__":
  main()