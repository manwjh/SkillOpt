#!/usr/bin/env bash
# Download SpreadsheetBench data for SkillOpt official adapter.
# Usage: ./scripts/download_spreadsheetbench.sh [sample|full]
set -euo pipefail

MODE="${1:-sample}"
ROOT="${2:-data/spreadsheetbench}"
mkdir -p "$ROOT"
cd "$ROOT"

if [ "$MODE" = "full" ]; then
  if [ ! -f all_data_912.json ] && [ ! -f ../all_data_912.json ]; then
    echo "Downloading SpreadsheetBench all_data_912.tar.gz (~large) ..."
    curl -L -o all_data_912.tar.gz \
      "https://github.com/RUCKBReasoning/SpreadsheetBench/raw/main/data/all_data_912.tar.gz"
    tar -xzf all_data_912.tar.gz
  fi
  echo "Full dataset ready under $(pwd)"
  echo "Set config: dataset.data_root=../../data/spreadsheetbench/all_data_912 dataset.manifest=dataset.json dataset.limit=null"
else
  if [ ! -d sample_data_200 ] && [ ! -f sample_data_200.jsonl ]; then
    echo "Downloading SpreadsheetBench sample_data_200.tar.gz ..."
    curl -L -o sample_data_200.tar.gz \
      "https://github.com/RUCKBReasoning/SpreadsheetBench/raw/main/data/sample_data_200.tar.gz"
    tar -xzf sample_data_200.tar.gz
  fi
  echo "Sample dataset ready: $(pwd)/sample_data_200"
  echo "Use: skillopt import-spreadsheetbench data/spreadsheetbench/sample_data_200 --output benchmarks/spreadsheet/tasks_official.yaml"
fi
