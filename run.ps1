# run-llm.ps1

$root = (Get-Location).Path -replace '\\', '/'

docker build -t paper-replication .

docker run --rm `
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" `
  --mount type=bind,source="${PWD}\data\processed",target=/app/data/processed `
  --mount type=bind,source="${PWD}\data\predictions",target=/app/data/predictions `
  --mount type=bind,source="${PWD}\data\russian_generated",target=/app/data/russian_generated `
  --mount type=bind,source="${PWD}\results\generated",target=/app/results/generated `
  paper-replication `
  python scripts/06_run_llm.py `
    --base-url "http://host.docker.internal:11434"