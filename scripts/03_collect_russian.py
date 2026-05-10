from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
import pandas as pd


GITHUB_API = "https://api.github.com"


def github_headers() -> dict[str, str]:
  token = os.environ.get("GITHUB_TOKEN")

  headers = {
    "Accept": "application/vnd.github+json",
  }

  if token:
    headers["Authorization"] = f"Bearer {token}"

  return headers


def search_issues(
  client: httpx.Client,
  query: str,
  max_pages: int,
) -> list[dict]:
  items = []

  for page in range(1, max_pages + 1):
    response = client.get(
      f"{GITHUB_API}/search/issues",
      params={
        "q": query,
        "per_page": 100,
        "page": page,
      },
    )
    response.raise_for_status()

    payload = response.json()
    items.extend(payload.get("items", []))

    if len(payload.get("items", [])) < 100:
      break

  return items


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--query",
    default='is:pr is:public comments:>0 "не"',
    help="GitHub issue-search query.",
  )
  parser.add_argument(
    "--max-pages",
    type=int,
    default=3,
  )
  parser.add_argument(
    "--output",
    default="data/russian_generated/russian_raw.csv",
  )

  args = parser.parse_args()

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with httpx.Client(headers=github_headers(), timeout=60.0) as client:
    items = search_issues(client, args.query, args.max_pages)

  rows = []

  for item in items:
    rows.append({
      "url": item.get("html_url"),
      "api_url": item.get("url"),
      "repository_url": item.get("repository_url"),
      "title": item.get("title"),
      "text": item.get("body") or "",
      "created_at": item.get("created_at"),
      "updated_at": item.get("updated_at"),
    })

  df = pd.DataFrame(rows)
  df.to_csv(output_path, index=False, encoding="utf-8")

  print(f"Saved {len(df)} candidate rows to {output_path}")


if __name__ == "__main__":
  main()