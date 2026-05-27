# 论文对齐进度

对照 [SkillOpt 论文](https://arxiv.org/abs/2605.23904) — **实现完成度 ~90%**

## 已完成

| 模块 | 实现 |
|------|------|
| 核心优化循环 | rollout → reflection → merge → rank → gate → slow/meta |
| SpreadsheetBench | 官方 adapter + answer xlsx 验证 + sample/912 下载脚本 |
| OfficeQA | oracle 文档 + strict scoring |
| ALFWorld | 文本世界 simulator harness + 6 任务预设 |
| Codex/Claude CLI | 可配置 CLI、`--print`/`--full-auto`、spreadsheet 评分 |
| Baselines | no_skill / initial / SkillOpt / TextGrad / GEPA / EvoSkill |
| Appendix A | 分角色 prompts + `insert_after` + nested JSON |
| Worker 池化 | `reflection_workers` + `merge_workers` |
| Kimi K2.6 | `profiles/mock-kimi-api.yaml` 等 |

## 需本机/生产环境验证

| 项目 | 说明 |
|------|------|
| Codex CLI | 本机未安装 `codex`；结构已就绪 |
| Claude Code 实跑 | 已适配 `claude --print --permission-mode bypassPermissions` |
| Kimi 全量 hard benchmark | 需 API 费用 |
| 真实 ALFWorld 环境 | 当前为论文兼容的 text-world mock（非原版 ALFWorld 二进制） |

## 命令速查

```bash
pip install -e ".[all]"

# 四大 benchmark
cd benchmarks/spreadsheet && skillopt optimize profiles/mock-spreadsheet.yaml && skillopt baselines profiles/mock-baselines.yaml --external
cd benchmarks/office_qa && skillopt optimize config.yaml
cd benchmarks/alfworld && skillopt optimize config.yaml
cd examples/demo_qa && skillopt optimize config.kimi.yaml

# 官方 SpreadsheetBench
./scripts/download_spreadsheetbench.sh sample
./scripts/download_spreadsheetbench.sh full   # 912 任务
cd benchmarks/spreadsheet && skillopt optimize profiles/official-mock.yaml

# Claude Code harness（本机需 claude CLI）
skillopt evaluate skill.md tasks.yaml --harness claude_code --model mock
```
