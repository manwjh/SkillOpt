#!/usr/bin/env bash
# Full SpreadsheetBench test: unit tests → optimize → test-split eval → baselines
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

CONFIG="benchmarks/spreadsheet/config.official.full.kimi_code.yaml"
ARTIFACTS="benchmarks/spreadsheet/artifacts_spreadsheetbench_full"
LOG="$ARTIFACTS/full_test.log"

mkdir -p "$ARTIFACTS"
exec > >(tee -a "$LOG") 2>&1

echo "=== SpreadsheetBench full test $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "--- 1/4 pytest (spreadsheet) ---"
pytest tests/test_spreadsheetbench_adapter.py tests/test_spreadsheet_harness.py tests/test_kimi_code_harness.py -v --tb=short

echo "--- 2/4 SkillOpt optimize (train split, Kimi Code agent) ---"
cd benchmarks/spreadsheet
skillopt optimize config.official.full.kimi_code.yaml

echo "--- 3/4 Evaluate test split (initial + best skill) ---"
skillopt spreadsheetbench config.official.full.kimi_code.yaml --split test --skill-path initial_skill_official.md --no-resume --report-suffix _initial
BEST_SKILL="artifacts_spreadsheetbench_full/best_skill.md"
if [[ -f "$BEST_SKILL" ]]; then
  skillopt spreadsheetbench config.official.full.kimi_code.yaml --split test --skill-path "$BEST_SKILL" --no-resume --report-suffix _best
fi

echo "--- 4/4 Baselines on test split ---"
skillopt baselines config.official.full.kimi_code.yaml

echo "=== Done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
