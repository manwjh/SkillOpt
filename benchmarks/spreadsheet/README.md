# SpreadsheetBench Preset

Two modes:

1. **Mock subset** (`tasks.yaml`, 6 tasks) — CI-friendly, openpyxl runtime
2. **Official sample** (`config.official.yaml`) — SpreadsheetBench `sample_data_200`

## Mock (default)

```bash
cd benchmarks/spreadsheet
skillopt optimize config.yaml
skillopt baselines config.yaml
skillopt baselines config.baselines.yaml --external
```

## Official SpreadsheetBench

```bash
# From repo root
./scripts/download_spreadsheetbench.sh
cd benchmarks/spreadsheet
skillopt optimize config.official.yaml
skillopt import-spreadsheetbench ../../data/spreadsheetbench/sample_data_200 \
  --output tasks_official_sample.yaml --limit 20
```

## Kimi K2.6

```bash
# Kimi Code CLI agent（需先 ./scripts/install_kimi_cli.sh）
skillopt optimize config.kimi_code.yaml
skillopt optimize config.official.kimi_code.yaml

# Kimi API only（单轮 JSON writes，非 Agent）
skillopt optimize config.kimi.yaml
```
