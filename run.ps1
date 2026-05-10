$root = (Get-Location).Path -replace '\\', '/'

docker build -t paper-replication .
docker run --rm `
  -v "${root}/data/processed:/app/data/processed" `
  -v "${root}/data/predictions:/app/data/predictions" `
  -v "${root}/results/generated:/app/results/generated" `
  paper-replication