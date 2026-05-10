param(
  [switch]$Build,
  [switch]$ForcePrDiscovery,
  [switch]$ForceComments,
  [switch]$ForceAll,

  [string]$ImageName = "paper-replication",

  # Must stay under data/russian_generated because that folder is mounted.
  [string]$OutputDir = "data/russian_generated",
  [string]$TermsFile = "data/russian_generated/cyrillic_terms.txt",

  [int]$SearchPages = 5,
  [int]$CommentPages = 5,
  [int]$MaxPrs = 500,
  [int]$TargetCandidates = 500,
  [int]$CheckpointEvery = 25,

  [int]$MinChars = 8,
  [int]$MaxChars = 500
)

$ErrorActionPreference = "Stop"

if ($ForceAll) {
  $ForcePrDiscovery = $true
  $ForceComments = $true
}

$generatedHostDir = Join-Path (Get-Location) "data\russian_generated"
$generatedHostDir = (Resolve-Path $generatedHostDir).Path

if (-not (Test-Path $generatedHostDir)) {
  New-Item -ItemType Directory -Force $generatedHostDir | Out-Null
}

if (-not (Test-Path $TermsFile)) {
  throw "Terms file not found: $TermsFile"
}

if ($Build) {
  docker build -t $ImageName .
}

$scriptArgs = @(
  "python", "scripts/03_collect_russian.py",
  "--terms-file", $TermsFile,
  "--output-dir", $OutputDir,
  "--search-pages", "$SearchPages",
  "--comment-pages", "$CommentPages",
  "--target-candidates", "$TargetCandidates",
  "--checkpoint-every", "$CheckpointEvery",
  "--min-chars", "$MinChars",
  "--max-chars", "$MaxChars"
)

if ($ForcePrDiscovery) {
  $scriptArgs += "--force-pr-discovery"
}

if ($ForceComments) {
  $scriptArgs += "--force-comments"
}

Write-Host "Image: $ImageName"
Write-Host "Mounted host dir: $generatedHostDir"
Write-Host "Container output dir: $OutputDir"
Write-Host "Force PR discovery: $ForcePrDiscovery"
Write-Host "Force comments: $ForceComments"

docker run --rm `
  -e "GITHUB_TOKEN=$env:GITHUB_TOKEN" `
  --mount "type=bind,source=$generatedHostDir,target=/app/data/russian_generated" `
  $ImageName `
  @scriptArgs