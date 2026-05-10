from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tqdm import tqdm

from src.gh_helpers import build_russian_candidates


GITHUB_API = "https://api.github.com"


DEFAULT_SEARCH_TERMS = [
  "и",
  "в",
  "не",
  "на",
  "что",
  "это",
  "как",
  "так",
  "если",
  "можно",
  "нужно",
  "спасибо",
  "ошибка",
  "проблема",
  "работает",
  "исправить",
]


def github_headers() -> dict[str, str]:
  token = os.environ.get("GITHUB_TOKEN")

  headers = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "bep-varapai-russian-dataset-fetcher",
  }

  if token:
    headers["Authorization"] = f"Bearer {token}"

  return headers


def save_csv(df: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(path, index=False, encoding="utf-8-sig")
  print(f"Saved {path} ({len(df)} rows)", flush=True)

def save_labeling_excel(candidates: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)

  if "text" not in candidates.columns:
    raise ValueError("Cannot create labeling Excel: missing text column")

  labeling = candidates[["text"]].copy()
  labeling.insert(1, "gold_label", "")

  labeling.to_excel(path, index=False, engine="openpyxl")

  print(f"Saved labeling Excel: {path} ({len(labeling)} rows)")

def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
  if not path.exists():
    return None

  print(f"Using existing file: {path}", flush=True)
  return pd.read_csv(path, encoding="utf-8-sig")


def gh_get(
  client: httpx.Client,
  url: str,
  params: dict[str, Any] | None = None,
) -> Any:
  while True:
    response = client.get(url, params=params)

    if response.status_code in {403, 429}:
      reset = response.headers.get("x-ratelimit-reset")

      if reset is not None:
        wait_seconds = max(0, int(reset) - int(time.time())) + 5
      else:
        wait_seconds = 60

      print(f"GitHub rate/API limit. Sleeping {wait_seconds}s.", flush=True)
      time.sleep(wait_seconds)
      continue

    response.raise_for_status()
    return response.json()


def gh_get_paginated(
  client: httpx.Client,
  url: str,
  max_pages: int,
  params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  for page in range(1, max_pages + 1):
    page_params = dict(params or {})
    page_params["per_page"] = 100
    page_params["page"] = page

    data = gh_get(client, url, params=page_params)

    if not isinstance(data, list) or not data:
      break

    rows.extend(data)

    if len(data) < 100:
      break

  return rows


def search_prs_for_term(
  client: httpx.Client,
  term: str,
  max_pages: int,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  query = f'is:pr is:public comments:>0 "{term}"'

  for page in range(1, max_pages + 1):
    data = gh_get(
      client,
      f"{GITHUB_API}/search/issues",
      params={
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": 100,
        "page": page,
      },
    )

    items = data.get("items", [])

    for item in items:
      if "pull_request" not in item:
        continue

      rows.append({
        "pr_url": item.get("html_url"),
        "pr_api_url": item.get("url"),
        "repository_url": item.get("repository_url"),
        "pr_number": item.get("number"),
        "pr_title": item.get("title"),
        "retrieval_term": term,
        "retrieval_query": query,
        "updated_at": item.get("updated_at"),
        "created_at": item.get("created_at"),
      })

    if len(items) < 100:
      break

  return rows


def repo_from_repository_url(repository_url: str) -> str:
  prefix = f"{GITHUB_API}/repos/"

  if not repository_url.startswith(prefix):
    raise ValueError(f"Unexpected repository_url: {repository_url}")

  return repository_url.removeprefix(prefix)


def collect_prs(
  client: httpx.Client,
  search_terms: list[str],
  max_pages: int,
) -> pd.DataFrame:
  rows: list[dict[str, Any]] = []

  for term in tqdm(search_terms, desc="Searching PRs"):
    rows.extend(search_prs_for_term(client, term, max_pages=max_pages))
    time.sleep(0.2)

  df = pd.DataFrame(rows)

  if df.empty:
    return df

  df["repo_full_name"] = df["repository_url"].apply(repo_from_repository_url)

  df = df.drop_duplicates(
    subset=["repo_full_name", "pr_number"],
    keep="first",
  ).reset_index(drop=True)

  return df


def limit_prs(prs: pd.DataFrame, max_prs: int) -> pd.DataFrame:
  if prs.empty:
    return prs

  if len(prs) <= max_prs:
    return prs.reset_index(drop=True)

  print(f"Limiting PRs from {len(prs)} to {max_prs}", flush=True)

  if "updated_at" in prs.columns:
    prs = prs.sort_values("updated_at", ascending=False)

  return prs.head(max_prs).reset_index(drop=True)


def fetch_pr_discussion(
  client: httpx.Client,
  repo_full_name: str,
  pr_number: int,
  pr_url: str,
  max_comment_pages: int,
) -> list[dict[str, Any]]:
  owner, repo = repo_full_name.split("/", maxsplit=1)

  issue_comments = gh_get_paginated(
    client,
    f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments",
    max_pages=max_comment_pages,
  )

  review_comments = gh_get_paginated(
    client,
    f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/comments",
    max_pages=max_comment_pages,
  )

  reviews = gh_get_paginated(
    client,
    f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
    max_pages=max_comment_pages,
  )

  rows: list[dict[str, Any]] = []

  for comment in issue_comments:
    rows.append({
      "repo_full_name": repo_full_name,
      "pr_number": pr_number,
      "pr_url": pr_url,
      "comment_source": "issue_comment",
      "comment_id": comment.get("id"),
      "comment_url": comment.get("html_url"),
      "user_login": (comment.get("user") or {}).get("login"),
      "created_at": comment.get("created_at"),
      "updated_at": comment.get("updated_at"),
      "body": comment.get("body"),
      "path": None,
      "commit_id": None,
    })

  for comment in review_comments:
    rows.append({
      "repo_full_name": repo_full_name,
      "pr_number": pr_number,
      "pr_url": pr_url,
      "comment_source": "review_comment",
      "comment_id": comment.get("id"),
      "comment_url": comment.get("html_url"),
      "user_login": (comment.get("user") or {}).get("login"),
      "created_at": comment.get("created_at"),
      "updated_at": comment.get("updated_at"),
      "body": comment.get("body"),
      "path": comment.get("path"),
      "commit_id": comment.get("commit_id"),
    })

  for review in reviews:
    body = review.get("body")

    if body is None or not str(body).strip():
      continue

    rows.append({
      "repo_full_name": repo_full_name,
      "pr_number": pr_number,
      "pr_url": pr_url,
      "comment_source": "review_body",
      "comment_id": review.get("id"),
      "comment_url": review.get("html_url"),
      "user_login": (review.get("user") or {}).get("login"),
      "created_at": review.get("submitted_at"),
      "updated_at": review.get("submitted_at"),
      "body": body,
      "path": None,
      "commit_id": review.get("commit_id"),
    })

  return rows


def normalize_comments(df: pd.DataFrame) -> pd.DataFrame:
  if df.empty:
    return df

  if "comment_id" not in df.columns:
    return df

  df = df.dropna(subset=["comment_id"]).copy()

  df = df.drop_duplicates(
    subset=["comment_source", "comment_id"],
    keep="first",
  ).reset_index(drop=True)

  return df


def fetched_pr_keys(comments: pd.DataFrame) -> set[tuple[str, int]]:
  if comments.empty:
    return set()

  required = {"repo_full_name", "pr_number"}

  if not required <= set(comments.columns):
    return set()

  keys: set[tuple[str, int]] = set()

  for row in comments[["repo_full_name", "pr_number"]].dropna().itertuples(index=False):
    keys.add((str(row.repo_full_name), int(row.pr_number)))

  return keys


def estimate_candidate_count(comments: pd.DataFrame) -> int:
  if comments.empty:
    return 0

  candidates = build_russian_candidates(
    comments,
    text_col="body",
    min_chars=8,
    max_chars=500,
    min_cyrillic_chars=4,
    min_cyrillic_ratio=0.3,
  )

  return len(candidates)



# --- Bot/autogenerated pruning helpers ---
AUTOGENERATED_PATTERNS = [
  "auto-generated comment",
  "automated comment",
  "this is an auto-generated",
  "generated by",
  "generated with",
  "generated using",
  "ai-generated",
  "created by ai",
  "written by ai",

  "github-actions",
  "github actions",
  "coderabbit",
  "code rabbit",
  "copilot",
  "github copilot",
  "copilot review",
  "copilot summary",
  "copilot generated",
  "dependabot",
  "renovate bot",
  "stale bot",
  "prettier bot",

  "walkthrough_start",
  "walkthrough_end",
  "pre_merge_checks",
  "internal state start",
  "internal state end",
  "finishing_touch_checkbox",
  "tips_start",
  "tips_end",

  "rsi diff bot",
  "diff updated after",
  "pre-merge checks failed",
  "suggested reviewers",
  "estimated code review effort",
  "prompt for ai agents",
  "verify each finding against the current code",
  "committable suggestion",
]

def is_bot_user(user_login: object) -> bool:
  value = "" if pd.isna(user_login) else str(user_login).lower()

  if value.endswith("[bot]"):
    return True

  bot_names = {
    "github-actions",
    "github-actions[bot]",
    "coderabbitai[bot]",
    "dependabot[bot]",
    "renovate[bot]",
    "copilot-pull-request-reviewer[bot]",
    "copilot[bot]",
  }

  return value in bot_names

def is_autogenerated_body(body: object) -> bool:
  value = "" if pd.isna(body) else str(body).lower()

  return any(pattern in value for pattern in AUTOGENERATED_PATTERNS)

def prune_comments(df: pd.DataFrame) -> pd.DataFrame:
  if df.empty:
    return df

  result = df.copy()

  if "user_login" in result.columns:
    result = result[~result["user_login"].apply(is_bot_user)].copy()

  if "body" in result.columns:
    result = result[~result["body"].apply(is_autogenerated_body)].copy()

  return result.reset_index(drop=True)


# --- Updated collect_comments ---
def collect_comments(
  client: httpx.Client,
  prs: pd.DataFrame,
  max_comment_pages: int,
  checkpoint_path: Path,
  checkpoint_every: int,
  target_candidates: int,
  max_comments_per_pr: int,
  min_chars: int,
  max_chars: int,
) -> pd.DataFrame:
  cached = read_csv_if_exists(checkpoint_path)

  if cached is not None:
    rows_df = normalize_comments(cached)
    done_keys = fetched_pr_keys(rows_df)
    print(
      f"Resuming comments from checkpoint: "
      f"{len(rows_df)} comments, {len(done_keys)} PRs already used"
    )
  else:
    rows_df = pd.DataFrame()
    done_keys = set()

  if prs.empty:
    return rows_df

  current_rows = rows_df.to_dict("records")
  current_count = len(rows_df)

  remaining_prs = []

  for pr in prs.itertuples(index=False):
    key = (str(pr.repo_full_name), int(pr.pr_number))

    if key not in done_keys:
      remaining_prs.append(pr)

  print(f"Remaining PRs to inspect: {len(remaining_prs)}")
  print(f"Current accepted candidate comments: {current_count}/{target_candidates}")

  for count, pr in enumerate(
    tqdm(remaining_prs, total=len(remaining_prs), desc="Fetching PR discussions"),
    start=1,
  ):
    if current_count >= target_candidates:
      print("Target candidate count reached. Stopping.")
      break

    try:
      pr_rows = fetch_pr_discussion(
        client=client,
        repo_full_name=str(pr.repo_full_name),
        pr_number=int(pr.pr_number),
        pr_url=str(pr.pr_url),
        max_comment_pages=max_comment_pages,
      )
    except httpx.HTTPStatusError as exc:
      print(f"Skipping {pr.repo_full_name}#{pr.pr_number}: {exc}")
      continue
    except ValueError as exc:
      print(f"Skipping malformed PR row: {exc}")
      continue

    pr_df = normalize_comments(pd.DataFrame(pr_rows))
    pr_df = prune_comments(pr_df)

    if pr_df.empty:
      continue

    pr_candidates = build_russian_candidates(
      pr_df,
      text_col="body",
      min_chars=min_chars,
      max_chars=max_chars,
      min_cyrillic_chars=4,
      min_cyrillic_ratio=0.3,
    )

    if pr_candidates.empty:
      continue

    pr_candidates = pr_candidates.head(max_comments_per_pr)

    remaining_slots = target_candidates - current_count
    pr_candidates = pr_candidates.head(remaining_slots)

    current_rows.extend(pr_candidates.to_dict("records"))
    current_count += len(pr_candidates)

    if count % checkpoint_every == 0:
      checkpoint_df = normalize_comments(pd.DataFrame(current_rows))
      save_csv(checkpoint_df, checkpoint_path)
      print(f"Accepted candidate comments: {current_count}/{target_candidates}")

    time.sleep(0.15)

  final_df = normalize_comments(pd.DataFrame(current_rows))
  save_csv(final_df, checkpoint_path)

  return final_df


def read_terms(path: Path | None) -> list[str]:
  if path is None:
    return DEFAULT_SEARCH_TERMS

  terms: list[str] = []

  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()

    if not line or line.startswith("#"):
      continue

    terms.append(line)

  return terms


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