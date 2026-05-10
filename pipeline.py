from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
  print("\n" + "=" * 80)
  print("Running:", " ".join(command))
  print("=" * 80)
  subprocess.run(command, cwd=ROOT, check=True)


def require_file(path: str) -> None:
  file_path = ROOT / path

  if not file_path.exists():
    raise FileNotFoundError(
      f"Required file is missing: {path}\n"
      "This file is expected to exist before running the main pipeline."
    )


def run_default_pipeline() -> None:
  print("Starting replication pipeline")

  run([sys.executable, "scripts/01_prepare_novielli.py"])
  run([sys.executable, "scripts/02_prepare_coutinho.py"])

  print("\nChecking frozen Russian dataset...")
  require_file("data/processed/russian.csv")
  print("Found data/processed/russian.csv")

  run([sys.executable, "scripts/05_run_senticr.py"])
  run([sys.executable, "scripts/06_run_llm.py"])
  run([sys.executable, "scripts/07_evaluate.py"])

  print("\nPipeline finished successfully.")


def generate_russian_dataset() -> None:
  print("Generating Russian dataset")
  print(
    "This step is not part of the default replication pipeline. "
    "It is included to document how the frozen Russian dataset was prepared."
  )

  run([sys.executable, "scripts/04_prepare_russian.py"])

  print("\nRussian dataset generation finished.")


def collect_russian_dataset() -> None:
  print("Collecting Russian GitHub PR comments")
  print(
    "This step depends on GitHub state and should not be used for normal reproduction."
  )

  run([sys.executable, "scripts/03_collect_russian.py"])


def main() -> None:
  parser = argparse.ArgumentParser(description="Replication pipeline")

  parser.add_argument(
    "--generate-russian",
    action="store_true",
    help="Prepare the Russian dataset from already collected raw data.",
  )

  parser.add_argument(
    "--collect-russian",
    action="store_true",
    help="Collect Russian GitHub PR comments from GitHub. Not used by default.",
  )

  args = parser.parse_args()

  if args.collect_russian:
    collect_russian_dataset()
  elif args.generate_russian:
    generate_russian_dataset()
  else:
    run_default_pipeline()


if __name__ == "__main__":
  main()