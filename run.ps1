$root = (Get-Location).Path -replace '\\', '/'

docker build -t paper-replication .
docker run --rm `
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" `
  -e OLLAMA_MODEL="qwen3:4b-q4_K_M" `
  --mount type=bind,source="${PWD}\data\processed",target=/app/data/processed `
  --mount type=bind,source="${PWD}\data\predictions",target=/app/data/predictions `
  --mount type=bind,source="${PWD}\results\generated",target=/app/results/generated `
  paper-replication