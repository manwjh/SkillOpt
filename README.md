# SkillOpt v1.0

Text-space optimizer for agent skills — train compact skill documents without modifying model weights.

Based on [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904).

## Features

- **Optimization Engine** — rollout → reflection → bounded edit → validation gate
- **Multi-Harness** — Direct Chat, Codex, Claude Code (workspace-based)
- **Multi-Provider** — Mock, OpenAI, Anthropic, Azure OpenAI
- **Web Console** — REST API + browser dashboard
- **Skill Library** — catalog, review workflow, export
- **Transfer & A/B** — cross-harness/benchmark testing, skill comparison
- **CI/CD** — regression testing via `skillopt regression`
- **Cost Tracking** — tokens per point of gain

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run demo (no API key needed)
cd examples/demo_qa && skillopt optimize config.yaml

# Start Web Console
skillopt serve
# → http://localhost:8080
```

## CLI Commands

```bash
skillopt optimize config.yaml          # Run optimization
skillopt evaluate skill.md tasks.yaml    # Evaluate skill
skillopt transfer skill.md tasks.yaml    # Transfer test
skillopt compare skill_a.md skill_b.md tasks.yaml  # A/B compare
skillopt regression skill.md tasks.yaml --min-score 0.8  # CI regression
skillopt serve --port 8080               # Web Console
skillopt benchmarks                      # List benchmark presets
skillopt library list                    # Skill library
skillopt library add skill.md "Name" domain
skillopt library review skill-id --status published
```

## Benchmark Presets

```bash
skillopt benchmarks
skillopt optimize benchmarks/spreadsheet/profiles/mock-spreadsheet.yaml
skillopt optimize benchmarks/office_qa/config.yaml
```

## Configuration

```yaml
name: my-run
skill_path: initial_skill.md
dataset_path: tasks.yaml
harness: direct_chat  # direct_chat | codex | claude_code
output_dir: artifacts

models:
  target: mock        # mock | openai | anthropic | azure
  optimizer: openai
  openai_model: gpt-4o

optimization:
  epochs: 4
  rollout_batch_size: 40
  rollout_accumulation_steps: 1
  reflection_minibatch_size: 8
  reflection_refinement_rounds: 3
  merge_batch_size: 8
  reflection_workers: 4
  learning_rate: 4
  learning_rate_min: 2
  schedule: cosine  # constant | linear | cosine | autonomous
  edit_mode: patch  # patch | rewrite
  slow_update_samples: 20
```

## Project Structure

**Core (~2k LOC)** — the optimizer engine; everything else is adapters and presets.

```
src/skillopt/
├── optimizer/      Loop, Reflection, Scheduler, SlowUpdate   ← core
├── core/           Skill, Edit, State, Trajectory
├── gate/           Validation gate
├── harness/        Task runners (spreadsheet, workspace CLI, …)
├── llm/            Provider clients (Mock, OpenAI, Kimi, …)
├── benchmarks/     Dataset adapters (SpreadsheetBench, baselines)
├── library/        Skill catalog & review
├── api/ + web/     Optional Web Console
└── cli.py

benchmarks/         YAML presets only — not part of the engine
  spreadsheet/
    _base/          Reusable config fragments (dataset, models, optimization)
    profiles/       Runnable presets (mock, official, kimi-code, …)
```

```
examples/demo_qa/   Runnable demo (+ config.paper.yaml, config.kimi.yaml)
.github/workflows/  CI pipeline
docs/PAPER_ALIGNMENT.md  Paper vs implementation tracker
docs/PRODUCT.md     Full product plan
```

## Paper-Aligned Optimization

```bash
# Uses merge → rank → refinement → dual slow-update (see docs/PAPER_ALIGNMENT.md)
skillopt optimize examples/demo_qa/config.paper.yaml

# Runtime benchmarks (openpyxl spreadsheet + OfficeQA oracle)
pip install -e ".[all]"
cd benchmarks/spreadsheet && skillopt optimize profiles/mock-spreadsheet.yaml && skillopt baselines profiles/mock-spreadsheet.yaml

# External baselines (TextGrad / GEPA / EvoSkill)
skillopt run-external-baselines profiles/mock-baselines.yaml
skillopt baselines profiles/mock-baselines.yaml --external

# Official SpreadsheetBench (after ./scripts/download_spreadsheetbench.sh)
skillopt optimize profiles/official-mock.yaml
skillopt import-spreadsheetbench data/spreadsheetbench -o tasks_official.yaml

cd benchmarks/office_qa && skillopt optimize config.yaml && skillopt baselines config.yaml
cd benchmarks/alfworld && skillopt optimize config.yaml && skillopt baselines config.yaml
```

## Kimi K2.6 (Kimi Code Plan API)

`sk-kimi-...` 是 **模型 API**（单轮对话），适合 `harness: spreadsheet` 的 JSON writes 模式：

```bash
skillopt evaluate initial_skill.md tasks.yaml --model kimi --harness spreadsheet
skillopt optimize examples/demo_qa/config.kimi.yaml
```

| Key prefix | Base URL | 用途 |
|------------|----------|------|
| `sk-kimi-...` | `https://api.kimi.com/coding/v1` | 模型 API（optimizer / 单轮 target） |
| Moonshot dev | `https://api.moonshot.cn/v1` | 开放平台按量计费 |

## Kimi Code CLI（终端 Agent）

[Kimi Code CLI](https://www.kimi.com/code/docs/kimi-code-cli/getting-started.html) 是**真 Agent**（读文件、跑命令、改 xlsx），对应 `harness: kimi_code`：

```bash
# 安装 + 登录（OAuth，需 Kimi Code 会员）
./scripts/install_kimi_cli.sh
kimi login

# Mock spreadsheet：Agent 改 task.xlsx，Kimi API 做 optimizer
cd benchmarks/spreadsheet
skillopt optimize profiles/mock-kimi-code.yaml

# 官方 SpreadsheetBench（Agent target + Kimi optimizer）
skillopt optimize profiles/official-kimi-code-smoke.yaml
```

架构：`harness: kimi_code` → `kimi --print --yolo -w <workspace> -p "..."`；`optimizer: kimi` → 仍走 API 优化 skill。

## Real LLM Setup (OpenAI / Anthropic / Azure)

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
# or
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
```

Edit config: `models.target: openai`, `models.optimizer: openai`

## License

MIT
