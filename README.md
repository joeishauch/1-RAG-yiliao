# 智能分诊系统（基于 LangGraph）

面向「症状 → 科室 / 疾病 / 用药」就医辅助场景的智能问答系统。以 **LangGraph 多工具 ReAct 闭环**为核心，整合向量检索、知识图谱推理、多科室会诊，并在输出层加入**风险分级 + 人工审核（HITL）**的安全合规能力。

---

## ✨ 核心能力

### 四条问答链路

| 链路 | 用户意图示例 | 工具 | 图流程 | 风险 |
|---|---|---|---|---|
| 智能分诊 | 「咳嗽发烧该挂哪个科」 | `retrieve` | agent → call_tools → **consult 会诊** → **summarize 汇总** | 🔴 高 |
| 症状推疾病 | 「胸痛可能是什么病」 | `kg_query` | agent → call_tools → generate_kg | 🟡 中 |
| 用药禁忌 | 「阿司匹林有什么禁忌」 | `drug_taboo` | agent → call_tools → generate_drug | 🔴 高 |
| 疾病科普 | 「什么是高血压」 | `medical_qa` | agent → call_tools → generate_qa | 🟢 低 |

### 安全合规 + 人在回路（HITL）

- **病历脱敏**：入口正则掩码身份证 / 手机号 / 住址 / 姓名，单向 `***`，PII 永不落盘、不进持久化记忆。
- **拒答免责**：规则引擎（无 LLM）拦截危险输入（自杀 / 自残 / 伤人），出口检测肯定式诊断表述并追加免责提示。
- **风险分级**：高风险（分诊 / 用药）→ 阻断式强制医生审核；中低风险直出。
- **医生审核三态**：`approve`（通过）/ `revise`（改写）/ `reject`（驳回），基于 LangGraph `interrupt()` + `Command(resume=...)` 实现。
- **科室流转（转诊式审核）**：高风险问题先推给相关性最高的首选科室，该科不认可可 `transfer` 移交目标 / 备选科室，上限 2 次防踢皮球；每次移交落 `output/triage_feedback.jsonl` 纠错数据，反哺分诊库（数据飞轮）。
- **患者 / 医生职责分离**：高风险时患者端只显示「已提交 XX 科医生审核」，看不到草稿；独立医生审核端（`doctor_ui.py`，端口 7861）拉取待审队列并审核。患者端只看「推荐科室 + 就医建议」，**分诊依据仅医生可见**（信息隔离）；审核通过后答案底部显示「本建议已由 内科·王医生 审核」脱敏署名。
- **医生账号体系**：医生不能自助注册，管理员建号（`admin/admin123`）——账号为手机号、绑定姓名（追责）+ 科室（权限）、密码 sha256+盐哈希持久化；医生登录后只看自己科室队列。患者端账号统一手机号并持久化 `output/users.json`。
- **全程审计**：`output/audit.jsonl` 记录 `block` / `draft` / `review_decision` / `final` 事件，AI 输出与人工修改均可追溯。

---

## 🏗️ 架构

```
用户输入
   │
   ▼
入口边界：脱敏 → 危险拦截 → 风险路由
   │
   ▼
agent（ReAct，选择工具）──► call_tools（ParallelToolNode 并行执行）
   │                            │
   │        ┌───────────────────┼───────────────────────┐
   │        ▼                   ▼                       ▼
   │    retrieve            kg_query /             medical_qa /
   │    （分诊检索）         drug_taboo              （科普兜底）
   │        │                   │                       │
   │        ▼                   ▼                       ▼
   │    consult（会诊）     generate_kg /            generate_qa
   │        │              generate_drug                │
   │        ▼                   │                       │
   │    summarize               │                       │
   │        │                   │                       │
   ▼        ▼                   ▼                       ▼
            高风险 ──► review（interrupt 人工审核）──► END
            中低风险 ──────────────────────────────► END
```

- **短期记忆**：PostgreSQL `PostgresSaver`（checkpoint，支撑多轮会话与 resume）
- **长期记忆**：PostgreSQL `PostgresStore`（持久化用户画像）
- **分诊会诊**：单 LLM + 多科室视角并行（`ThreadPoolExecutor`），共享 vectorstore，按科室 `filter` 检索

---

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph（StateGraph / ReAct / interrupt 人工审核） |
| LLM | DeepSeek（chat）+ 通义 `text-embedding-v1`（embedding） |
| 检索 | ChromaDB 向量 + BM25 混合（jieba 分词）+ BGE-reranker 重排（RRF 融合） |
| 知识图谱 | networkx 内存图，症状 → 疾病 → 治疗 / 药物 / 科室 多跳推理 |
| 存储 | PostgreSQL（checkpoint + 长期记忆） |
| 服务 | FastAPI（OpenAI 兼容接口，端口 8012）+ Gradio 患者端（7860）/ 医生审核端（7861） |
| 配置 | pydantic-settings 集中管理（`.env`） |

---

## 📁 目录结构

```
.
├── main.py                  # FastAPI 服务 + HITL review 端点
├── ragAgent.py              # LangGraph 状态图（节点/路由/编译）
├── cli.py                   # 统一 CLI 入口（chat / serve / ui / eval）
├── webUI.py                 # Gradio 患者端（7860）
├── doctor_ui.py             # Gradio 医生审核端（7861）
├── app_launcher.py          # 一键启动器
├── utils/
│   ├── config.py            # pydantic-settings 配置中心
│   ├── llms.py              # LLM 初始化
│   ├── tools_config.py      # 四工具 + 检索/置信度/会诊逻辑
│   ├── kg_*.py              # 知识图谱（建图/查询/别名/症状模糊匹配）
│   ├── label_*.py           # 数据清洗（模板过滤/标签纠正/审核）
│   ├── privacy.py           # 入口脱敏
│   ├── safety.py            # 生成前/后规则校验（危险拦截/诊断检测）
│   └── audit.py             # 审计日志
├── prompts/                 # 6 个 prompt 模板（agent/consult/summarize/qa/kg/drug）
├── jsonl2chroma.py          # 数据集导入向量库
├── build_kg.py              # 知识图谱构建
├── extract_drug_contra.py   # 药物禁忌抽取
├── run_label_audit.py       # 标签审核
├── eval_triage_retrieval.py # 检索质量评估（Hit@K / MRR）
├── eval_triage_judge.py     # LLM-as-judge 分诊质量评估
├── e2e_regression.py        # 端到端四链路回归
├── verify_*.py              # 各模块独立验证脚本
├── drug_contraindications.json  # 药物禁忌知识（2709 条）
├── docker-compose.yml       # PostgreSQL 容器
└── requirements.txt
```

---

## 🚀 快速开始

### 0. 前置：环境与依赖

```bash
pip install -r requirements.txt
```

在 `.env` 中配置（参考 `.env.example`）：

```ini
DEEPSEEK_API_KEY=xxx          # chat LLM
DASHSCOPE_API_KEY=xxx         # embedding
```

启动 PostgreSQL 容器（checkpoint / 长期记忆依赖）：

```bash
docker-compose up -d
```

### 1. 统一 CLI 入口

```bash
python cli.py chat                    # 交互式问答（高风险场景触发终端人工审核）
python cli.py chat -v                 # 打印节点流转 + 工具调用
python cli.py serve                   # 启动 API（http://127.0.0.1:8012/docs）
python cli.py ui                      # 启动患者端（http://127.0.0.1:7860）
python doctor_ui.py                   # 启动医生审核端（http://127.0.0.1:7861，管理员 admin/admin123）
python cli.py eval retrieval          # 检索质量评估
python cli.py eval judge --limit 20   # LLM-as-judge 分诊质量评估
python cli.py eval e2e                # 端到端四链路回归
```

### 2. API 调用（含 HITL 三态）

**第 1 步：发问诊请求**（高风险触发待审核）

```bash
curl http://127.0.0.1:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"我咳嗽发烧，该挂哪个科？"}],"stream":false,"userId":"u001","conversationId":"c001"}'
```

返回 `status: pending_review` + `draft` + `risk_level: high`（草稿停在审核节点）。

**第 2 步：医生审核**（三态）

```bash
curl http://127.0.0.1:8012/v1/chat/review \
  -H "Content-Type: application/json" \
  -d '{"userId":"u001","conversationId":"c001","action":"approve"}'
# action 可选 approve / revise（附 revised_answer）/ reject
```

---

## 🧠 关键设计决策

- **分诊会诊而非单次生成**：`retrieve → consult → summarize`，多科室视角并行出「可能性 + 依据」，再汇总成分诊结论。输出用 **JSON Schema 结构化约束**（`recommend_departments` / `advice` / `basis`），患者端只看「推荐科室 + 就医建议」，`basis` 分诊依据仅医生可见。
- **砍掉 grade + rewrite 闭环**：实测原始查询直接检索反而比改写后更准（神经科学 80% vs 改写后 50%），查询改写对分诊是负收益。
- **置信度校准**：`score = cnt² / expected`（先验归一化，类似 TF-IDF），同时抑制大科基数优势与小样本 1 次命中噪声。
- **风险分级而非一刀切**：高风险阻断式审核、中风险直出、低风险直出（二期预留快速确认 / 抽样审计 / 置信度路由）。
- **脱敏放边界、审计放边界**：入口脱敏保证 PII 不进图；审计不进 state，避免与 `interrupt()` 幂等冲突、避免 resume 重复记录。
- **规则引擎 + prompt 软约束双保险**：`safety.py` 硬兜底（命中即拦）+ prompt 软约束（尽量不做），互补。

---

## 📊 评估结果

| 评估 | 指标 | 结果 |
|---|---|---|
| 检索质量（20 条症状用例） | Hit@1 / MRR | 85% / 0.925（纯向量模式） |
| LLM-as-judge 分诊（20 条） | 科室命中率 / Top-1 / 质量分 | 100% / 90% / 4.95（满分 5） |
| 端到端回归 | 四链路节点流转 | 全部跑通 |

> 检索评估含三档对比（vector / hybrid / hybrid_rerank），结论：明确症状上纯向量最准且快（重排 CPU 单条 5-6s），故默认 `vector`，混合检索 + 重排能力保留待 GPU 场景启用。

---

## 📦 数据说明

- 数据源：华佗系列（Huatuo26M-Lite / encyclopedia_qa / knowledge_graph_qa / consultation_qa）+ Chinese-medical-dialogue，共约 5.7GB。
- 分诊库 `medical_triage`：按科室均衡抽样（大科上限 3000 / 小科全取），清洗模板题与误标标签。
- 药物禁忌：从百科数据流式抽取，`drug_contraindications.json` 共 2709 条（药名 / 禁忌 / 来源）。

---

## ⚠️ 已知局限

- **急诊科数据缺口**：公开数据集急诊科样本极少（清洗后 13 条），急症场景分诊置信度天然偏低。
- **标签质量**：部分数据「头疼 + 失眠」被误标心理科（真实多为神经科），已用规则纠正 + qwen-max 独立裁判审核。
- **KG 排序缺共现权重**：知识图谱「症状 → 疾病」边缺少共现频次信号，端到端 top-1 可能命中罕见病，依赖生成层兜底纠正。

---

## 🧪 测试与验证

- `verify_*.py`：各模块独立验证（KG / 症状匹配 / 科普 / 用药 / 标签均衡），不依赖 PostgreSQL。
- `eval_triage_retrieval.py` / `eval_triage_judge.py` / `e2e_regression.py`：检索 / 分诊质量 / 端到端评估。
- 安全合规链路：危险拦截、脱敏、HITL 三态、审计留痕、开关（`HITL_ENABLED`）均已完成端到端验证。
