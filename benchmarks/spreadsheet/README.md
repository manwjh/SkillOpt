# SpreadsheetBench Preset

## Layout

| Path | Purpose |
|------|---------|
| `profiles/mock-spreadsheet.yaml` | Mock 6 tasks, CI-friendly (default) |
| `profiles/official-*.yaml` | Official `sample_data_200` variants |
| `_base/` | Shared fragments (dataset split, Kimi models, optimization) |

All runnable configs live under `profiles/`. They compose `_base/` fragments via `extends`.

## Quick commands

### Mock (default)

```bash
cd benchmarks/spreadsheet
skillopt optimize profiles/mock-spreadsheet.yaml
skillopt baselines profiles/mock-spreadsheet.yaml
skillopt baselines profiles/mock-baselines.yaml --external
```

### Official SpreadsheetBench

```bash
# From repo root
./scripts/download_spreadsheetbench.sh
cd benchmarks/spreadsheet
skillopt optimize profiles/official-mock.yaml
skillopt import-spreadsheetbench ../../data/spreadsheetbench/sample_data_200 \
  --output tasks_official_sample.yaml --limit 20
```

### Kimi K2.6

```bash
# Kimi Code CLI agent（需先 ./scripts/install_kimi_cli.sh）
skillopt optimize profiles/mock-kimi-code.yaml              # mock smoke
skillopt optimize profiles/official-kimi-code-smoke.yaml    # official limit 4
skillopt optimize profiles/official-kimi-code-full.yaml     # full 200

# Kimi API only（单轮 JSON writes，非 Agent）
skillopt optimize profiles/mock-kimi-api.yaml
```

### Custom preset

Create `profiles/my-run.yaml`:

```yaml
extends:
  - ../_base/official-dataset.yaml
  - ../_base/kimi-code-harness.yaml
  - ../_base/kimi-models.yaml
name: my-run
dataset:
  limit: 20
output_dir: artifacts_my_run
```

Then: `skillopt optimize profiles/my-run.yaml`

## Profile index

| Profile | Harness | Dataset | Notes |
|---------|---------|---------|-------|
| `mock-spreadsheet` | spreadsheet | mock 6 | default CI |
| `mock-baselines` | spreadsheet | mock 6 | + external baselines |
| `mock-kimi-api` | spreadsheet | mock 6 | Kimi API target+optimizer |
| `mock-kimi-code` | kimi_code | mock 6 | Kimi CLI agent |
| `mock-claude-code` | claude_code | mock 6 | Claude CLI agent |
| `official-mock` | spreadsheet | official limit 10 | |
| `official-kimi-api` | spreadsheet | official limit 8 | Kimi API |
| `official-kimi-api-r2` | spreadsheet | official limit 12 | paper-aligned r2 |
| `official-kimi-code-smoke` | kimi_code | official limit 4 | smoke |
| `official-kimi-code-full` | kimi_code | official 200 | full benchmark |
