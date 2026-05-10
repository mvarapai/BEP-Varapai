from __future__ import annotations

from pathlib import Path

import pandas as pd

from ollama_wrapper import OllamaBatchPrompt
from src.labels import parse_llm_label
from src.prompts import BINARY_SENTIMENT_TEMPLATE


CHECKPOINT_EVERY = 25


DATASETS = [
  {
    "name": "novielli",
    "input": Path("data/processed/novielli.csv"),
    "output": Path("data/predictions/novielli_llm.csv"),
    "text_col": "text",
  },
  {
    "name": "coutinho",
    "input": Path("data/processed/coutinho.csv"),
    "output": Path("data/predictions/coutinho_llm.csv"),
    "text_col": "text",
  },
  {
    "name": "russian",
    "input": Path("data/processed/russian.csv"),
    "output": Path("data/predictions/russian_llm.csv"),
    "text_col": "text",
  },
]


def make_llm() -> OllamaBatchPrompt:
  return OllamaBatchPrompt(
    model="qwen3:4b",
    system="Return only the label. No explanation.",
    template=BINARY_SENTIMENT_TEMPLATE,
    temperature=0.0,
    num_predict=3,
    keep_alive="30m",
    options={},
  )


def load_or_initialize(input_path: Path, output_path: Path) -> pd.DataFrame:
  if output_path.exists():
    print(f"Resuming from {output_path}")
    return pd.read_csv(output_path)

  print(f"Starting from {input_path}")
  df = pd.read_csv(input_path)

  if "llm_raw" not in df.columns:
    df["llm_raw"] = pd.NA

  if "llm_score" not in df.columns:
    df["llm_score"] = pd.NA

  return df


def save_checkpoint(df: pd.DataFrame, output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_path, index=False, encoding="utf-8")
  print(f"Saved checkpoint: {output_path}")


def run_llm_for_dataset(
  llm: OllamaBatchPrompt,
  name: str,
  input_path: Path,
  output_path: Path,
  text_col: str,
) -> None:
  print(f"\nRunning LLM for dataset: {name}")

  df = load_or_initialize(input_path, output_path)

  if text_col not in df.columns:
    raise ValueError(f"{input_path} is missing text column: {text_col}")

  pending_mask = df["llm_raw"].isna() | df["llm_raw"].astype(str).str.strip().eq("")
  pending_indices = df.index[pending_mask].tolist()

  total = len(df)
  pending = len(pending_indices)
  done = total - pending

  print(f"Total rows: {total}")
  print(f"Already done: {done}")
  print(f"Pending: {pending}")

  if pending == 0:
    print(f"Skipping {name}: all rows already predicted.")
    return

  for counter, row_index in enumerate(pending_indices, start=1):
    text = str(df.at[row_index, text_col])

    try:
      raw = llm.ask_one({"text": text}).strip()
    except Exception as exc:
      save_checkpoint(df, output_path)
      raise RuntimeError(
        f"LLM failed on dataset={name}, row_index={row_index}"
      ) from exc

    df.at[row_index, "llm_raw"] = raw
    df.at[row_index, "llm_score"] = parse_llm_label(raw)

    print(f"[{name}] {done + counter}/{total}: {raw!r}")

    if counter % CHECKPOINT_EVERY == 0:
      save_checkpoint(df, output_path)

  save_checkpoint(df, output_path)
  print(f"Finished LLM predictions for {name}")


def main() -> None:
  with make_llm() as llm:
    for dataset in DATASETS:
      run_llm_for_dataset(
        llm=llm,
        name=dataset["name"],
        input_path=dataset["input"],
        output_path=dataset["output"],
        text_col=dataset["text_col"],
      )


if __name__ == "__main__":
  main()