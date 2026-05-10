# Bachelor End Project (BEP) by Mikalai Varapai - Replication Package

This repository features all code that was used to obtain results presented in the work.

## Overview

### Initial State

Replication artifacts will be stored in `data/processed` for harmonized column names and labels of raw datasets, and in `data/predictions` for SentiCR and LLM labeling.

Execution results are stored in `results/generated`, and could be compared with `results/example` for correspondence.

## Requirements
- Docker version (TODO)
- RAM/CPU/GPU requirements (TODO)
- Approximate runtime (TODO)
- Disk space (TODO)
- Internet required or fully offline (TODO)

## Quick Start

```bash
docker build -t paper-replication .
docker run --rm `
  -v "${PWD}\data\processed:/app/data/processed" `
  -v "${PWD}\data\predictions:/app/data/predictions" `
  -v "${PWD}\results\generated:/app/results/generated" `
  paper-replication
```