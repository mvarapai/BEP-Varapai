from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
  model = os.environ.get("OLLAMA_MODEL", "qwen3:4b-q4_K_M")
  base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

  print(f"Using Ollama model: {model}")
  print(f"Using Ollama base URL: {base_url}")

  return OllamaBatchPrompt(
    model=model,
    base_url=base_url,
    system=(
      "You are a sentiment classifier. "
      "Return exactly one label: negative or non-negative. "
      "Do not explain."
    ),
    template=BINARY_SENTIMENT_TEMPLATE,
    temperature=0.0,
    num_predict=32,
    keep_alive="30m",
    think=False,
    options={},
  )


def load_or_initialize(input_path: Path, output_path: Path) -> pd.DataFrame:
  if output_path.exists():
    df = pd.read_csv(output_path)
  else:
    df = pd.read_csv(input_path)

  if "llm_raw" not in df.columns:
    df["llm_raw"] = pd.NA

  if "llm_score" not in df.columns:
    df["llm_score"] = pd.NA

  df["llm_raw"] = df["llm_raw"].astype("string")
  df["llm_score"] = df["llm_score"].astype("Int64")

  return df


def save_checkpoint(df: pd.DataFrame, output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_path, index=False, encoding="utf-8")


def pending_indices(df: pd.DataFrame) -> list[int]:
  mask = df["llm_raw"].isna() | df["llm_raw"].astype(str).str.strip().eq("")
  return df.index[mask].tolist()


def load_all_datasets() -> list[dict]:
  loaded = []

  for dataset in DATASETS:
    name = dataset["name"]
    input_path = dataset["input"]
    output_path = dataset["output"]
    text_col = dataset["text_col"]

    df = load_or_initialize(input_path, output_path)

    if text_col not in df.columns:
      raise ValueError(f"{input_path} is missing text column: {text_col}")

    indices = pending_indices(df)

    loaded.append({
      "name": name,
      "input": input_path,
      "output": output_path,
      "text_col": text_col,
      "df": df,
      "pending_indices": indices,
      "done": len(df) - len(indices),
      "total": len(df),
    })

  return loaded


def print_dataset_summary(datasets: list[dict]) -> None:
  print("\nLLM prediction summary:")

  total_rows = 0
  total_done = 0
  total_pending = 0

  for dataset in datasets:
    name = dataset["name"]
    total = dataset["total"]
    done = dataset["done"]
    pending = len(dataset["pending_indices"])

    total_rows += total
    total_done += done
    total_pending += pending

    print(f"  {name}: {done}/{total} done, {pending} pending")

  print(f"\nTotal: {total_done}/{total_rows} done, {total_pending} pending\n")


def run_llm_for_all_datasets(llm: OllamaBatchPrompt, datasets: list[dict]) -> None:
  total_pending = sum(len(dataset["pending_indices"]) for dataset in datasets)

  if total_pending == 0:
    print("All LLM predictions already exist. Nothing to do.")
    return

  start_time = time.monotonic()

  with tqdm(
    total=total_pending,
    desc="LLM predictions",
    unit="prompt",
    dynamic_ncols=True,
  ) as progress:
    for dataset in datasets:
      name = dataset["name"]
      df = dataset["df"]
      output_path = dataset["output"]
      text_col = dataset["text_col"]
      indices = dataset["pending_indices"]

      if not indices:
        continue

      progress.set_postfix_str(f"dataset={name}")

      for local_count, row_index in enumerate(indices, start=1):
        text = str(df.at[row_index, text_col])

        try:
          raw = llm.ask_one({"text": text}).strip()

          if not raw:
            save_checkpoint(df, output_path)
            raise RuntimeError(
              f"Empty LLM response on dataset={name}, row_index={row_index}."
            )

          parsed = parse_llm_label(raw)

          df.at[row_index, "llm_raw"] = raw

          if pd.isna(parsed):
            df.at[row_index, "llm_score"] = pd.NA
          else:
            df.at[row_index, "llm_score"] = int(parsed)

        except Exception:
          save_checkpoint(df, output_path)
          raise

        if local_count % CHECKPOINT_EVERY == 0:
          save_checkpoint(df, output_path)

        elapsed = time.monotonic() - start_time
        completed = progress.n + 1
        avg_seconds = elapsed / completed
        remaining = total_pending - completed
        eta_seconds = remaining * avg_seconds

        progress.set_postfix({
          "dataset": name,
          "last": raw,
          "eta_min": f"{eta_seconds / 60:.1f}",
        })

        progress.update(1)

      save_checkpoint(df, output_path)

  print("\nLLM predictions finished.")


def main() -> None:
  datasets = load_all_datasets()
  print_dataset_summary(datasets)

  with make_llm() as llm:
    run_llm_for_all_datasets(llm, datasets)


if __name__ == "__main__":
  main()