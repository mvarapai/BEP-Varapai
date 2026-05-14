from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import httpx
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetConfig:
  dataset_id: str
  input_path: Path
  language: str


@dataclass(frozen=True)
class ModelConfig:
  model_id: str


@dataclass(frozen=True)
class PromptConfig:
  prompt_id: str
  template_en: str
  template_ru: str


DATASETS = [
  DatasetConfig(
    dataset_id="novielli",
    input_path=Path("data/processed/novielli.csv"),
    language="en",
  ),
  DatasetConfig(
    dataset_id="coutinho",
    input_path=Path("data/processed/coutinho.csv"),
    language="en",
  ),
  DatasetConfig(
    dataset_id="russian",
    input_path=Path("data/processed/russian.csv"),
    language="ru",
  ),
]


MODELS = [
  ModelConfig(model_id="phi4:14b"),
  ModelConfig(model_id="llama3.1:8b"),
  ModelConfig(model_id="gemma3:12b"),
]


PROMPTS = [
  PromptConfig(
    prompt_id="prompt0_direct_label",
    template_en=(
      "Please perform a sentiment classification task.\n"
      "Given the following software engineering text, classify its sentiment as "
      "either negative or non-negative.\n"
      "Return only one label: negative or non-negative.\n\n"
      "Text: $text"
    ),
    template_ru=(
      "Выполни задачу классификации тональности.\n"
      "Дан следующий текст из обсуждения разработки программного обеспечения. "
      "Определи его тональность как negative или non-negative.\n"
      "Верни только одну метку: negative или non-negative.\n\n"
      "Текст: $text"
    ),
  ),
  PromptConfig(
    prompt_id="prompt1_categorize",
    template_en=(
      "Please categorize the sentiment expressed in the following software "
      "engineering text as either (1) negative or (2) non-negative.\n"
      "Return only one label: negative or non-negative.\n\n"
      "Text: $text\n\n"
      "Output:"
    ),
    template_ru=(
      "Определи тональность следующего текста из обсуждения разработки ПО "
      "как (1) negative или (2) non-negative.\n"
      "Верни только одну метку: negative или non-negative.\n\n"
      "Текст: $text\n\n"
      "Ответ:"
    ),
  ),
  PromptConfig(
    prompt_id="prompt2_question",
    template_en=(
      "I will give you a software engineering text.\n"
      "Is the sentiment expressed in the text negative or non-negative?\n"
      "Return only one label: negative or non-negative.\n\n"
      "Text: $text\n\n"
      "Answer:"
    ),
    template_ru=(
      "Я дам тебе текст из обсуждения разработки программного обеспечения.\n"
      "Является ли тональность этого текста negative или non-negative?\n"
      "Верни только одну метку: negative или non-negative.\n\n"
      "Текст: $text\n\n"
      "Ответ:"
    ),
  ),
]


def safe_name(value: str) -> str:
  value = value.lower()
  value = re.sub(r"[^a-z0-9_.-]+", "_", value)
  value = value.strip("_")
  return value


def output_path(
  output_dir: Path,
  dataset_id: str,
  model_id: str,
  prompt_id: str,
) -> Path:
  return output_dir / f"{dataset_id}__{safe_name(model_id)}__{prompt_id}.csv"


def load_dataset(path: Path) -> pd.DataFrame:
  if not path.exists():
    raise FileNotFoundError(f"Missing dataset: {path}")

  df = pd.read_csv(path, encoding="utf-8-sig")

  if "text" not in df.columns:
    raise ValueError(f"{path} must contain a 'text' column")

  df = df.copy()
  df["text"] = df["text"].fillna("").astype(str).str.strip()
  df = df[df["text"].ne("")].reset_index(drop=True)

  return df


def prepare_output_frame(input_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
  if existing_path.exists():
    df = pd.read_csv(existing_path, encoding="utf-8-sig")

    required = {"text", "llm_raw", "llm_label"}

    if required <= set(df.columns):
      df["text"] = df["text"].fillna("").astype(str)
      df["llm_raw"] = df["llm_raw"].astype("object")
      df["llm_label"] = df["llm_label"].astype("object")
      return df

    print(f"Ignoring malformed checkpoint: {existing_path}")

  df = input_df.copy()
  df["llm_raw"] = pd.Series([""] * len(df), dtype="object")
  df["llm_label"] = pd.Series([""] * len(df), dtype="object")

  return df


def is_done(value: object) -> bool:
  if pd.isna(value):
    return False

  return str(value).strip() in {"negative", "non-negative"}


def parse_label(raw: str) -> str:
  value = raw.strip().lower()

  value = value.replace("nonnegative", "non-negative")
  value = value.replace("non negative", "non-negative")
  value = value.replace("not negative", "non-negative")

  value = re.sub(r"[^a-zа-яё\- ]+", " ", value)
  value = re.sub(r"\s+", " ", value).strip()

  negative_ru = {
    "негативный",
    "негативная",
    "негативное",
    "отрицательный",
    "отрицательная",
    "отрицательное",
  }

  non_negative_ru = {
    "не негативный",
    "не негативная",
    "не негативное",
    "не отрицательный",
    "не отрицательная",
    "не отрицательное",
    "нейтральный",
    "нейтральная",
    "нейтральное",
    "позитивный",
    "позитивная",
    "позитивное",
    "положительный",
    "положительная",
    "положительное",
  }

  if value == "negative":
    return "negative"

  if value == "non-negative":
    return "non-negative"

  if value in negative_ru:
    return "negative"

  if value in non_negative_ru:
    return "non-negative"

  if re.search(r"\bnon-negative\b", value):
    return "non-negative"

  if re.search(r"\bnegative\b", value):
    return "negative"

  for token in non_negative_ru:
    if token in value:
      return "non-negative"

  for token in negative_ru:
    if token in value:
      return "negative"

  return ""


def render_prompt(prompt: PromptConfig, language: str, text: str) -> str:
  template = prompt.template_ru if language == "ru" else prompt.template_en
  return Template(template).safe_substitute(text=text)


def ollama_chat(
  client: httpx.Client,
  base_url: str,
  model: str,
  prompt: str,
  temperature: float,
) -> str:
  payload: dict[str, Any] = {
    "model": model,
    "messages": [
      {
        "role": "user",
        "content": prompt,
      }
    ],
    "stream": False,
    "options": {
      "temperature": temperature,
      "num_predict": 16,
    },
  }

  response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
  response.raise_for_status()

  data = response.json()
  content = data.get("message", {}).get("content", "")

  return str(content).strip()


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(path, index=False, encoding="utf-8-sig")


def count_total_pending(output_dir: Path) -> int:
  total = 0

  for dataset in DATASETS:
    input_df = load_dataset(dataset.input_path)

    for model in MODELS:
      for prompt in PROMPTS:
        path = output_path(
          output_dir=output_dir,
          dataset_id=dataset.dataset_id,
          model_id=model.model_id,
          prompt_id=prompt.prompt_id,
        )

        df = prepare_output_frame(input_df, path)
        total += int(~df["llm_label"].apply(is_done).sum())

  return total


def run_single_job(
  client: httpx.Client,
  base_url: str,
  output_dir: Path,
  dataset: DatasetConfig,
  model: ModelConfig,
  prompt: PromptConfig,
  temperature: float,
  checkpoint_every: int,
  global_bar: tqdm,
) -> None:
  input_df = load_dataset(dataset.input_path)

  path = output_path(
    output_dir=output_dir,
    dataset_id=dataset.dataset_id,
    model_id=model.model_id,
    prompt_id=prompt.prompt_id,
  )

  df = prepare_output_frame(input_df, path)

  if len(df) != len(input_df):
    raise ValueError(
      f"Checkpoint row count mismatch for {path}: "
      f"{len(df)} rows in checkpoint, {len(input_df)} rows in input"
    )

  pending_indices = [
    index for index, value in df["llm_label"].items()
    if not is_done(value)
  ]

  job_label = f"{dataset.dataset_id} | {model.model_id} | {prompt.prompt_id}"

  with tqdm(
    total=len(pending_indices),
    desc=job_label,
    unit="row",
    leave=False,
  ) as job_bar:
    for local_count, row_index in enumerate(pending_indices, start=1):
      text = str(df.at[row_index, "text"])
      rendered_prompt = render_prompt(prompt, dataset.language, text)

      raw = ollama_chat(
        client=client,
        base_url=base_url,
        model=model.model_id,
        prompt=rendered_prompt,
        temperature=temperature,
      )

      label = parse_label(raw)

      df.at[row_index, "llm_raw"] = raw
      df.at[row_index, "llm_label"] = label

      job_bar.update(1)
      global_bar.update(1)

      if local_count % checkpoint_every == 0:
        save_checkpoint(df, path)

    save_checkpoint(df, path)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run 3x3x3 Ollama LLM sentiment labeling jobs with caching."
  )

  parser.add_argument(
    "--ollama-base-url",
    "--base-url",
    dest="ollama_base_url",
    default=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    help="Ollama API base URL.",
  )

  parser.add_argument(
    "--output-dir",
    default="data/predictions",
  )

  parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
  )

  parser.add_argument(
    "--checkpoint-every",
    type=int,
    default=25,
  )

  parser.add_argument(
    "--models",
    nargs="*",
    default=[model.model_id for model in MODELS],
    help="Optional subset of Ollama models to run.",
  )

  parser.add_argument(
    "--datasets",
    nargs="*",
    default=[dataset.dataset_id for dataset in DATASETS],
    help="Optional subset of datasets to run.",
  )

  parser.add_argument(
    "--prompts",
    nargs="*",
    default=[prompt.prompt_id for prompt in PROMPTS],
    help="Optional subset of prompts to run.",
  )

  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  selected_datasets = [
    dataset for dataset in DATASETS
    if dataset.dataset_id in set(args.datasets)
  ]

  selected_models = [
    model for model in MODELS
    if model.model_id in set(args.models)
  ]

  selected_prompts = [
    prompt for prompt in PROMPTS
    if prompt.prompt_id in set(args.prompts)
  ]

  if not selected_datasets:
    raise ValueError("No datasets selected")

  if not selected_models:
    raise ValueError("No models selected")

  if not selected_prompts:
    raise ValueError("No prompts selected")

  jobs = [
    (dataset, model, prompt)
    for dataset in selected_datasets
    for model in selected_models
    for prompt in selected_prompts
  ]

  print(f"Ollama base URL: {args.ollama_base_url}")
  print(f"Output directory: {output_dir}")
  print(f"Jobs: {len(jobs)}")

  total_pending = 0

  for dataset, model, prompt in jobs:
    input_df = load_dataset(dataset.input_path)
    path = output_path(output_dir, dataset.dataset_id, model.model_id, prompt.prompt_id)
    df = prepare_output_frame(input_df, path)
    pending = int((~df["llm_label"].apply(is_done)).sum())
    total_pending += pending

    print(
      f"{dataset.dataset_id} | {model.model_id} | {prompt.prompt_id}: "
      f"{pending}/{len(df)} pending -> {path}"
    )

  if total_pending == 0:
    print("All selected LLM runs are already complete.")
    return

  started_at = time.time()

  # accept higher timeout for read
  timeout = httpx.Timeout(
    connect=30.0,
    read=600.0,
    write=30.0,
    pool=30.0,
  )

  with httpx.Client(timeout=timeout) as client:
    with tqdm(
      total=total_pending,
      desc="All LLM runs",
      unit="row",
      position=0,
    ) as global_bar:
      for job_index, (dataset, model, prompt) in enumerate(jobs, start=1):
        elapsed = time.time() - started_at

        print(
          f"\nStage {job_index}/{len(jobs)}: "
          f"dataset={dataset.dataset_id}, "
          f"model={model.model_id}, "
          f"prompt={prompt.prompt_id}, "
          f"elapsed={elapsed / 60:.1f} min"
        )

        run_single_job(
          client=client,
          base_url=args.ollama_base_url,
          output_dir=output_dir,
          dataset=dataset,
          model=model,
          prompt=prompt,
          temperature=args.temperature,
          checkpoint_every=args.checkpoint_every,
          global_bar=global_bar,
        )

  print("\nDone.")


if __name__ == "__main__":
  main()