from __future__ import annotations

from pathlib import Path

from src.datasets import load_coutinho


def main() -> None:
  print("\nSTAGE 02: prepare the dataset of Coutinho et al.")

  input_path = Path("data/raw/coutinho/dataset.json")
  output_path = Path("data/processed/coutinho.csv")

  print("Goal: only leave out (clean_message,gold_label,senticr_score)")
  print(f"Writing: {input_path} -> {output_path}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Created a directory at {output_path.parent}")

  print("Loading Coutinho et al. dataset...")
  df = load_coutinho(input_path)
  df.to_csv(output_path, index=False, encoding="utf-8")

  print(f"Saved {len(df)} rows to {output_path}")
  print("END STAGE 02\n")


if __name__ == "__main__":
  main()