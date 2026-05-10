# 03_collect_russian.py
# goal:   1. fetch PRs from GitHub API by Russian keywords
#         2. fetch PR review comments, comments and issues
#         3. perform several pruning steps and produce an Excel spreadsheet for manual labeling

from __future__ import annotations

from pathlib import Path

import argparse
import httpx

from src.artifacts import read_csv_if_exists, save_csv, save_labeling_excel
from src.github_api_helper import collect_comments, collect_prs, github_headers, normalize_comments, prune_comments, read_terms
from src.russian_classifier import build_russian_candidates


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Collect Russian PR discussion candidates from GitHub."
  )

  parser.add_argument(
    "--terms-file",
    default=None,
    help="Optional file with one Cyrillic retrieval term per line.",
  )

  parser.add_argument(
    "--output-dir",
    default="data/russian_generated",
  )

  parser.add_argument(
    "--search-pages",
    type=int,
    default=5,
  )

  parser.add_argument(
    "--labeling-excel",
    default=None,
    help="Optional path for manual-labeling Excel file. Defaults to russian_labeling.xlsx in output-dir.",
  )

  parser.add_argument(
    "--comment-pages",
    type=int,
    default=5,
  )

  parser.add_argument(
    "--max-comments-per-pr",
    type=int,
    default=10,
    help="Maximum accepted candidate comments from a single PR.",
  )

  parser.add_argument(
    "--target-candidates",
    type=int,
    default=1000,
    help="Stop after this many Russian candidate comments are collected.",
  )

  parser.add_argument(
    "--checkpoint-every",
    type=int,
    default=25,
  )

  parser.add_argument(
    "--min-chars",
    type=int,
    default=8,
  )

  parser.add_argument(
    "--max-chars",
    type=int,
    default=500,
  )

  parser.add_argument(
    "--force-pr-discovery",
    action="store_true",
    help="Ignore cached russian_pull_requests.csv and rerun PR discovery.",
  )

  parser.add_argument(
    "--force-comments",
    action="store_true",
    help="Ignore cached raw comments/checkpoint files and refetch comments.",
  )

  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  terms = read_terms(Path(args.terms_file) if args.terms_file else None)

  prs_path = output_dir / "russian_pull_requests.csv"
  prs_subset_path = output_dir / "russian_pull_requests_subset.csv"
  raw_path = output_dir / "russian_raw_comments.csv"
  checkpoint_path = output_dir / "russian_raw_comments_checkpoint.csv"
  candidates_path = output_dir / "russian_candidates.csv"

  if args.labeling_excel:
    labeling_excel_path = Path(args.labeling_excel)
  else:
    labeling_excel_path = output_dir / "russian_labeling.xlsx"

  print(f"PR cache path: {prs_path.resolve()}", flush=True)

  if args.force_comments:
    if checkpoint_path.exists():
      checkpoint_path.unlink()

    if raw_path.exists():
      raw_path.unlink()

  with httpx.Client(headers=github_headers(), timeout=60.0) as client:
    cached_prs = None

    if not args.force_pr_discovery:
      cached_prs = read_csv_if_exists(prs_path)

    if cached_prs is not None:
      prs = cached_prs
    else:
      prs = collect_prs(
        client=client,
        search_terms=terms,
        max_pages=args.search_pages,
      )

      save_csv(prs, prs_path)

    prs = prs.sort_values("updated_at", ascending=False).reset_index(drop=True)
    save_csv(prs, prs_subset_path)

    cached_comments = None

    if not args.force_comments:
      cached_comments = read_csv_if_exists(raw_path)

    if cached_comments is not None:
      comments = normalize_comments(cached_comments)
    else:
      comments = collect_comments(
        client=client,
        prs=prs,
        max_comment_pages=args.comment_pages,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        target_candidates=args.target_candidates,
        max_comments_per_pr=args.max_comments_per_pr,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
      )

      save_csv(comments, raw_path)

  comments = prune_comments(comments)

  candidates = build_russian_candidates(
    comments,
    text_col="body",
    min_chars=args.min_chars,
    max_chars=args.max_chars,
    min_cyrillic_chars=4,
    min_cyrillic_ratio=0.3,
  )

  if args.target_candidates is not None:
    candidates = candidates.head(args.target_candidates).reset_index(drop=True)

  save_csv(candidates, candidates_path)
  save_labeling_excel(candidates, labeling_excel_path)

  if not candidates.empty:
    print("\nCandidate source distribution:")
    print(candidates["comment_source"].value_counts())

    print("\nLanguage distribution:")
    print(candidates["detected_language"].value_counts(dropna=False))

    print("\nLength summary:")
    print(candidates["char_len"].describe())


if __name__ == "__main__":
  main()