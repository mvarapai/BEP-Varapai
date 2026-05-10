from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import evaluate_predictions


def evaluate_file(
  input_path: Path,
  true_col: str,
  pred_col: str,
  name: str,
) -> None:
  if not input_path.exists():
    print(f"Skipping missing file: {input_path}")
    return

  df = pd.read_csv(input_path)

  if true_col not in df.columns:
    print(f"Skipping {name}: missing true column {true_col}")
    return

  if pred_col not in df.columns:
    print(f"Skipping {name}: missing prediction column {pred_col}")
    return

  metrics, cm = evaluate_predictions(
    df,
    true_col=true_col,
    pred_col=pred_col,
  )

  out_dir = Path("results/generated")
  out_dir.mkdir(parents=True, exist_ok=True)

  pd.DataFrame([metrics]).to_csv(
    out_dir / f"{name}_metrics.csv",
    index=False,
  )

  cm.to_csv(out_dir / f"{name}_confusion_matrix.csv")

  print(f"\n{name}")
  print(metrics)
  print(cm)


def main() -> None:
  evaluate_file(
    input_path=Path("data/predictions/novielli_senticr.csv"),
    true_col="gold_label",
    pred_col="senticr_score",
    name="novielli_senticr",
  )

  evaluate_file(
    input_path=Path("data/predictions/novielli_llm.csv"),
    true_col="gold_label",
    pred_col="llm_score",
    name="novielli_llm",
  )

  evaluate_file(
    input_path=Path("data/predictions/coutinho_senticr.csv"),
    true_col="gold_label",
    pred_col="senticr_score",
    name="coutinho_senticr",
  )

  evaluate_file(
    input_path=Path("data/predictions/coutinho_llm.csv"),
    true_col="gold_label",
    pred_col="llm_score",
    name="coutinho_llm",
  )

  evaluate_file(
    input_path=Path("data/predictions/russian_llm.csv"),
    true_col="gold_label",
    pred_col="llm_score",
    name="russian_llm",
  )


if __name__ == "__main__":
  main()