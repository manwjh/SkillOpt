# SkillOpt 产品方案

> 基于论文 [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904) 的落地产品规划

## 1. 产品定位

**一句话**：SkillOpt 是一个 Agent Skill 文本空间优化平台——冻结目标模型，通过可验证的迭代编辑，把自然语言 Skill 文档训练成可部署、可审计、可迁移的领域适配层。

**目标用户**：

| 角色 | 痛点 | SkillOpt 价值 |
|------|------|---------------|
| AI 应用开发者 | 闭源模型无法 fine-tune，prompt 调参不可复现 | 产出 compact `best_skill.md`，零推理额外成本 |
| Agent 平台团队 | Codex/Claude Code 等 harness 下 agent 表现不稳定 | Harness-agnostic 优化，跨环境迁移 |
| 领域专家 | 懂业务但不懂 prompt engineering | 从执行轨迹自动提炼程序性规则，人工可审阅 |

**与竞品差异**：

- vs Prompt 优化（TextGrad/GEPA）：优化**持久 Skill 文档**，非一次性 prompt
- vs Skill 进化（EvoSkill/Trace2Skill）：有 **validation gate + bounded edit + rejected buffer**
- vs Fine-tuning：不改权重，Skill 可跨模型/环境复用

---

## 2. 产品架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SkillOpt Platform                         │
├──────────────┬──────────────────────┬───────────────────────────┤
│   Console    │   Optimization API   │      Artifact Store       │
│  (Web/CLI)   │   (REST / Python SDK)│   (best_skill.md, logs)   │
└──────┬───────┴──────────┬───────────┴─────────────┬─────────────┘
       │                  │                         │
       ▼                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Optimization Engine                          │
│  ┌─────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Rollout │→ │ Reflection │→ │  Edit    │→ │ Validation    │  │
│  │ Batch   │  │ Minibatch  │  │  Engine  │  │ Gate          │  │
│  └─────────┘  └────────────┘  └──────────┘  └───────────────┘  │
│       ↑              ↑               ↑               ↓          │
│  ┌─────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Harness │  │ Rejected   │  │ LR       │  │ Slow/Meta     │  │
│  │ Adapter │  │ Buffer     │  │ Scheduler│  │ Update        │  │
│  └─────────┘  └────────────┘  └──────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
       │                                    │
       ▼                                    ▼
┌──────────────┐                    ┌─────────────────┐
│ Target Model │                    │ Optimizer Model │
│  (frozen)    │                    │  (teacher)      │
└──────────────┘                    └─────────────────┘
```

---

## 3. 核心功能模块

### 3.1 Skill 文档管理

- 支持 Markdown 格式的 Skill 文档
- 受保护的 `<!-- slow-update -->` 区域，step-level 编辑不可覆盖
- 版本历史与 diff 追踪
- 导出 `best_skill.md` 供部署

### 3.2 优化引擎（核心）

| 阶段 | 功能 | 论文对应 |
|------|------|----------|
| Forward | Rollout batch 收集轨迹证据 | §3.2 |
| Backward | Minibatch reflection 分析成败 | §3.3 |
| Update | Add/Delete/Replace 有界编辑 | §3.4 |
| Gate | Selection split 验证门控 | §3.5 |
| Memory | Rejected-edit buffer + slow/meta update | §3.5-3.6 |

**文本学习率调度**：constant / linear / cosine / autonomous，默认 cosine decay (L=4→2)

### 3.3 Harness 适配器

统一接口，支持多种执行环境：

```python
class HarnessAdapter(Protocol):
    def run(self, task, skill) -> Trajectory: ...
    def evaluate_batch(self, tasks, skill) -> float: ...
    def inject_skill(self, skill) -> None: ...
```

内置适配器：
- `DirectChatHarness` — 单轮/多轮 chat
- `CodexHarness` — Codex CLI 沙箱（Phase 2）
- `ClaudeCodeHarness` — Claude Code CLI（Phase 2）

### 3.4 数据集与评估

- Train / Selection / Test 三分法（默认 4:1:5 或 2:1:7）
- Selection 仅用于 gate，Test 仅用于最终报告
- 支持自定义 scorer（exact match、executable check、LLM judge）

---

## 4. 产品形态与路线图

### Phase 1 — MVP ✅

**目标**：验证核心优化 loop 可跑通

- [x] Python SDK 核心框架
- [x] Skill 文档 + Patch 编辑引擎
- [x] Validation gate + rejected buffer
- [x] Direct Chat harness
- [x] Mock LLM + Demo QA 任务
- [x] CLI 入口

### Phase 2 — 生产可用 ✅

- [x] OpenAI / Anthropic / Azure API 集成
- [x] Codex / Claude Code harness 适配（workspace + CLI fallback）
- [x] Web Console：FastAPI + 浏览器 Dashboard
- [x] 多 benchmark 预设（SpreadsheetBench、OfficeQA mock 子集）
- [x] 并行 reflection workers

### Phase 3 — 平台化 ✅

- [x] Skill Library：多 domain skill 管理与共享
- [x] Cross-model / cross-harness 迁移测试（`skillopt transfer`）
- [x] 团队协作：Skill 审核（`skillopt library review`）、A/B 对比
- [x] 成本追踪：tokens per point of gain
- [x] CI/CD 集成：GitHub Actions + `skillopt regression`

**交付物**：
```bash
pip install -e ".[all]"
skillopt optimize examples/demo_qa/config.yaml
skillopt serve                          # Web Console → :8080
skillopt regression best_skill.md tasks.yaml --min-score 0.8
```

---

## 5. 用户旅程

```
1. 创建项目
   └─ 选择 target model + harness + benchmark

2. 初始化 Skill
   └─ 空白 / 人工草稿 / 一次性 LLM 生成

3. 配置优化参数
   └─ epochs, batch size, learning rate, schedule

4. 启动优化
   └─ 实时查看: rollout score → reflection → edit proposals → gate decision

5. 审阅 & 部署
   └─ 导出 best_skill.md → 注入 Agent 上下文 → 零额外推理成本

6. （可选）迁移测试
   └─ 跨模型 / 跨 harness 验证 Skill 泛化性
```

---

## 6. 技术选型

| 层 | 选型 | 理由 |
|----|------|------|
| 语言 | Python 3.11+ | LLM 生态成熟 |
| 配置 | YAML + Pydantic | 类型安全、易读 |
| LLM | OpenAI SDK（抽象层） | 支持多 provider |
| CLI | Typer | 简洁、类型提示 |
| 存储 | 本地文件系统 | MVP 够用，Phase 2 接 S3/DB |
| Web | FastAPI + React | Phase 2 |

---

## 7. 关键指标（KPI）

**优化效果**：
- Test score lift over no-skill baseline
- Accepted edits count（compactness）
- Cost per point（training tokens / score gain）

**产品指标**：
- 优化任务完成率
- 平均优化轮次到收敛
- Skill 跨环境迁移成功率

**论文 benchmark 参考**（GPT-5.5, Direct Chat）：
- 52/52 cells best or tied-best
- 平均 +23.5 points over no-skill
- 1-4 accepted edits, 300-2000 tokens final skill

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 依赖可自动评分任务 | Phase 2 支持 LLM-as-judge + human review gate |
| 训练成本高 | 成本追踪、early stopping、skill 复用 |
| 优化器质量依赖 teacher 模型 | 支持 target-matched optimizer（论文验证 56-74% 增益） |
| Skill 过拟合训练分布 | 严格 held-out gate + 迁移测试 |

---

## 9. 定价思路（Phase 3）

| 层级 | 包含 | 定价逻辑 |
|------|------|----------|
| Free | CLI + Mock demo | 获客 |
| Pro | API 调用 + 1 并发优化 | 按 optimization run 计费 |
| Team | Web Console + Skill Library | 席位 + 存储 |
| Enterprise | 私有部署 + 自定义 harness | 定制 |

核心计费单位：**Optimization Run**（一次完整 train→gate→export 流程）+ **LLM API passthrough**

---

## 10. 快速开始

```bash
# 安装
pip install -e ".[all]"

# 跑 Demo
skillopt optimize examples/demo_qa/config.yaml

# Web Console
skillopt serve

# CI 回归测试
skillopt regression examples/demo_qa/artifacts/best_skill.md examples/demo_qa/tasks.yaml

# Skill Library
skillopt library add artifacts/best_skill.md "My Skill" qa --score 1.0
```
