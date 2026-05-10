from __future__ import annotations

from pathlib import Path

from src.datasets import load_novielli


def main() -> None:

  print("\nSTAGE 01: Prepare the dataset of Novielli et al.")

  input_path = Path("data/raw/novielli/github_gold.csv")
  output_path = Path("data/processed/novielli.csv")

  print("Goal: translate string labels into 0 for non-negative and -1 for negative")
  print(f"Writing: {input_path} -> {output_path}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Created a directory at {output_path.parent}")

  print("Loading Novielli et al. dataset...")
  df = load_novielli(input_path)
  print("Novielli et al. dataset loaded successfully into a DataFrame!")

  df.to_csv(output_path, index=False, encoding="utf-8")

  print(f"Saved {len(df)} rows to {output_path}")
  print("END STAGE 01\n")


if __name__ == "__main__":
  main()