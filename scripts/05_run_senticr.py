from __future__ import annotations

from pathlib import Path

import pandas as pd

from SentiCR import SentiCR


def run_senticr(input_path: Path, output_path: Path) -> None:
  df = pd.read_csv(input_path)

  if "text" not in df.columns:
    raise ValueError(f"{input_path} is missing required column: text")

  model = SentiCR(algo="GBT")

  df["senticr_score"] = model.get_sentiment_polarity_collection(
    df["text"].astype(str)
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_path, index=False, encoding="utf-8")

  print(f"Saved SentiCR predictions to {output_path}")


def main() -> None:
  run_senticr(
    input_path=Path("data/processed/novielli.csv"),
    output_path=Path("data/predictions/novielli_senticr.csv"),
  )

  coutinho_path = Path("data/processed/coutinho.csv")
  coutinho_output = Path("data/predictions/coutinho_senticr.csv")

  df = pd.read_csv(coutinho_path)

  if "senticr_score" in df.columns:
    coutinho_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(coutinho_output, index=False, encoding="utf-8")
    print(f"Copied existing Coutinho SentiCR predictions to {coutinho_output}")
  else:
    print("Coutinho dataset has no existing senticr_score column; skipping.")


if __name__ == "__main__":
  main()