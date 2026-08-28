# 智能分诊系统（基于 LangGraph）

面向「症状 → 科室 / 疾病 / 用药」就医辅助场景的智能问答系统。以 **LangGraph 多工具 ReAct 闭环**为核心，整合向量检索、知识图谱推理、多科室会诊，并在输出层加入**风险分级 + 人工审核（HITL）**的安全合规能力。

---

## ✨ 核心能力

### 五条问答链路

| 链路 | 用户意图示例 | 工具 | 图流程 | 风险 |
|---|---|---|---|---|
| 智能分诊 | 「咳嗽发烧该挂哪个科」 | `retrieve` | agent → call_tools → **consult 会诊** → **summarize 汇总** | 🔴 高 |
| 症状推疾病 | 「胸痛可能是什么病」 | `kg_query` | agent → call_tools → generate_kg | 🟡 中 |
| 用药禁忌 | 「阿司匹林有什么禁忌」 | `drug_taboo` | agent → call_tools → generate_drug | 🔴 高 |
| 疾病科普 | 「什么是高血压」 | `medical_qa` | agent → call_tools → generate_qa | 🟢 低 |
| 医院实时查询 | 「耳鼻喉科还有号吗」 | `query_registration` 等（MCP 接入 HIS） | agent → call_tools → generate_his | 🟢 低 |

### MCP 工具生态（双方向）

- **方向 1（对外提供）**：把 `retrieve` / `kg_query` / `medical_qa` / `drug_taboo` 4 个领域工具标准化为 MCP Server（`mcp_server.py`），外部系统通过统一协议复用，零重构。
- **方向 2（对内消费）**：自研轻量 MCP → LangChain 适配器（`mcp_client.py`，MCP SDK + 后台 event loop 桥接 async→sync），把医院 HIS 系统（`hospital_mcp_server.py`，查号源 / 查报告）以 MCP 协议并入 LangGraph 工具链路，作为第 5 条链路独立路由到 `generate_his`。
- **工程健壮性**：HIS 工具模块级单例懒加载（持有 bridge 引用防 GC），加载失败自动降级不影响主链路；`MCP_HIS_ENABLED` 一键开关。

### 安全合规 + 人在回路（HITL）

- **病历脱敏**：入口正则掩码身份证 / 手机号 / 住址 / 姓名，单向 `***`，PII 永不落盘、不进持久化记忆。
- **拒答免责**：规则引擎（无 LLM）拦截危险输入（自杀 / 自残 / 伤人），出口检测肯定式诊断表述并追加免责提示。
- **风险分级**：高风险（分诊 / 用药）→ 阻断式强制医生审核；中低风险直出。
- **医生审核三态**：`approve`（通过）/ `revise`（改写）/ `reject`（驳回），基于 LangGraph `interrupt()` + `Command(resume=...)` 实现。
- **科室流转（转诊式审核）**：高风险问题先推给相关性最高的首选科室，该科不认可可 `transfer` 移交目标 / 备选科室，上限 2 次防踢皮球；每次移交落 `output/triage_feedback.jsonl` 纠错数据，反哺分诊库（数据飞轮）。
- **患者 / 医生职责分离**：高风险时患者端只显示「已提交 XX 科医生审核」，看不到草稿；独立医生审核端（`doctor_ui.py`，端口 7861）拉取待审队列并审核。患者端只看「推荐科室 + 就医建议」，**分诊依据仅医生可见**（信息隔离）；审核通过后答案底部显示「本建议已由 内科·王医生 审核」脱敏署名。
- **医生账号体系**：医生不能自助注册，管理员建号（`admin/admin123`）——账号为手机号、绑定姓名（追责）+ 科室（权限）、密码 sha256+盐哈希持久化；医生登录后只看自己科室队列。管理员端可**增删改查**全部医生名单（改姓名 / 职称 / 科室，不可改手机号——手机号即账号）；密码采用**一次性链接重置**（管理员只发 token、不生成也不接触明文，医生自助设密，本地模拟短信下发到 `output/reset_links.log`）。患者端账号统一手机号并持久化 `output/users.json`。
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
| Agent 协议 | MCP（Model Context Protocol）—— 4 工具暴露为 MCP Server + Agent 经 MCP 接入外部 HIS |
| 配置 | pydantic-settings 集中管理（`.env`） |

---

## 📁 目录结构

```
.
├── main.py                  # FastAPI 服务 + HITL review 端点
├── ragAgent.py              # LangGraph 状态图（节点/路由/编译）
├── cli.py                   # 统一 CLI 入口（chat / serve / ui / sync / dedup / eval / mcp）
├── mcp_server.py            # 方向1：4 领域工具 → MCP Server（对外提供）
├── hospital_mcp_server.py   # 方向2：模拟医院 HIS 的 MCP Server（外部系统）
├── mcp_client.py            # 方向2：MCP → LangChain 工具适配器（event loop 桥接）
├── verify_mcp.py            # MCP 双方向集成验证
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
├── prompts/                 # 7 个 prompt 模板（agent/consult/summarize/qa/kg/drug/his）
├── jsonl2chroma.py          # 数据集解析、质量门禁、切片与向量导入
├── doc_sync.py              # 文档变更感知、增量同步、manifest 与跨源去重
├── dedup.py                 # 确定性跨源精确去重与审计报告
├── quality_gate.py          # 数据质量门禁与拒绝记录
├── metrics.py               # embedding / sync / dedup 指标
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
python cli.py eval retrieval           # 检索质量评估
python cli.py eval judge --limit 20    # LLM-as-judge 分诊质量评估
python cli.py eval e2e                 # 端到端四链路回归
python cli.py mcp                      # 启动 MCP Server（方向1：对外提供 4 领域工具）
python cli.py mcp --his                # 启动医院 HIS MCP Server（方向2：外部系统 mock）
```

### 2. 数据处理命令与产物

数据处理链路统一遵循：

```text
原始 JSON/JSONL
    → source parser
    → 模板/标签清洗
    → QualityGate 数据质量门禁
    → 可选按句切片
    → 可选跨源精确去重
    → embedding
    → ChromaDB
```

> `DEDUP_ENABLED` 默认是 `false`。因此默认导入和同步行为不改变。只有显式开启，并且本次同时处理至少两个 source 时，才会执行跨源去重。去重发生在质量门禁和切片之后、embedding 之前，单位是最终写入 Chroma 的 chunk。

| 命令 / 文件 | 作用 | 是否写 Chroma | 主要 output 产物 |
|---|---|---:|---|
| `python jsonl2chroma.py --dry-run` | 解析全部配置 source，预览有效记录和样例，不导入 | 否 | `output/quality_report.json`、`output/rejected_chunks.jsonl` |
| `python jsonl2chroma.py --source huatuo_lite --limit 200` | 将指定 source 解析、清洗、切片并写入 Chroma | 是 | `output/quality_report.json`、`output/metrics.jsonl` |
| `python jsonl2chroma.py --clear --source huatuo_lite` | 清空指定目标 collection 后重建导入，适合数据规则变更后的重灌 | 是 | 同上；Chroma 数据位于 `chromaDB/` |
| `python cli.py sync --dry-run` | 检测源文件变化，预览将删除/写入的 chunk 数，不修改库 | 否 | `output/sync_manifest.json` 不应改变；质量报告会更新 |
| `python cli.py sync --source huatuo_encyclopedia` | 按 manifest 做变更感知同步；变化时 doc 级删除旧 chunk 后重建 | 是 | `output/sync_manifest.json`、`output/audit.jsonl`、`output/metrics.jsonl` |
| `python cli.py sync --watch --interval 60` | 守护模式定期执行变更检测和同步 | 可能 | 同步 manifest、审计和指标文件 |
| `python cli.py sync --migrate` | 为存量向量补写 `doc_id`，不重新 embedding | 仅更新 metadata | `output/sync_manifest.json`、`output/audit.jsonl` |
| `python cli.py sync --migrate-embedding` | 为存量 chunk 补写 `embedding_model` 标识，不重新 embedding | 仅更新 metadata | `output/audit.jsonl` |
| `python cli.py dedup --sources a,b --dry-run --limit 200` | 扫描多个 source，预览跨源精确去重结果 | 否 | `output/dedup_report.json`、`output/dedup_dropped.jsonl` |
| `python verify_b8_quality_gate.py` | 验证数据质量门禁规则和原子报告 | 否 | 终端验证通过 |
| `python verify_b9_dedup.py` | 验证归一化、score 选择、报告和阈值开关 | 否 | 终端验证通过 |
| `python cli.py quality --source huatuo_lite --limit 200` | 单独扫描指定 source 的质量问题，不触库 | 否 | `output/quality_report.json`、拒绝 JSONL |
| `python cli.py metrics` | 汇总 embedding、sync、collection、dedup 指标 | 否 | 读取 `output/metrics.jsonl` |
| `python cli.py metrics --diagnose` | 诊断指标中的耗时、快照和去重计数语义 | 否 | 终端诊断信息 |

### 3. 跨源去重用法

#### 3.1 只读预览（推荐）

```bash
python cli.py dedup \\
  --sources huatuo_encyclopedia,huatuo_knowledge_graph \\
  --dry-run \\
  --limit 200
```

该命令会读取两个 source，执行与实际导入相同的 parser、质量门禁和切片逻辑，然后输出：

- `input`：进入去重阶段的 chunk 总数；
- `kept`：去重后保留的 chunk 数；
- `dropped`：被判定为重复并丢弃的 chunk 数；
- `dropped_by_source`：各 source 被丢弃的数量；
- `output/dedup_report.json`：汇总报告；
- `output/dedup_dropped.jsonl`：逐条丢弃记录。

该命令**永远是分析模式**，即使不写 `--dry-run` 也不会初始化 embedding、打开 Chroma、删除 collection、写入向量或修改同步 manifest。

#### 3.2 开启实际导入/同步去重

在项目 `.env` 中显式配置：

```ini
DEDUP_ENABLED=true
DEDUP_REPORT_PATH=output/dedup_report.json
DEDUP_DROPPED_PATH=output/dedup_dropped.jsonl
```

然后同时选择至少两个 source：

```bash
python cli.py sync \\
  --source huatuo_encyclopedia \\
  --source huatuo_knowledge_graph \\
  --dry-run
```

确认 dry-run 报告符合预期后，再去掉 `--dry-run` 执行真实同步。真实同步只会将 `kept` chunk 送入 embedding 和 Chroma，`dropped` chunk 不产生向量。

#### 3.3 去重规则

| 规则 | 行为 |
|---|---|
| 文本归一化 | Unicode NFKC、大小写 `casefold`、常见中英文标点归一、连续空白折叠、首尾空白清除 |
| 精确重复 | 归一化后的 document 完全相同才算重复 |
| score 选择 | score 越高越优先；缺失、非法、NaN、Infinity 按 `0` |
| 分数相同 | 按 `SOURCES` 顺序和输入顺序保留先出现者 |
| 空文本 | 不使用空字符串作为去重 key，不把多个空文本错误合并；通常会先被质量门禁拒绝 |
| collection | 只在同一 collection 内去重；`medical_triage` 与 `medical_qa` 相互隔离 |
| 结果单位 | chunk，不是原始逻辑 record；因为 chunk 才是最终写入 Chroma 的单位 |
| 可解释性 | 每条 dropped 记录包含 source、score、normalized hash、`duplicate_of` 和 reason |
| 模糊相似 | 首版不使用 embedding、SimHash 或语义阈值；相似度扩展留作二期 |

例如百科和知识图谱都写入 `medical_qa` 时，归一化后的相同 chunk 只保留一份；但相同文本若分别属于 `medical_qa` 和 `medical_triage`，两份都会保留，因为两个 collection 服务不同检索链路。

#### 3.4 output 字段对应

`output/dedup_report.json` 是汇总报告，核心结构如下：

```json
{
  "rule_version": "b9-exact-1.0",
  "collection": "medical_qa",
  "stats": {
    "input_total": 400,
    "kept_total": 398,
    "dropped_total": 2,
    "empty_text_kept": 0,
    "duplicate_groups": 2,
    "kept_by_source": {
      "huatuo_encyclopedia": 250,
      "huatuo_knowledge_graph": 148
    },
    "dropped_by_source": {
      "huatuo_knowledge_graph": 2
    },
    "dropped_by_reason": {
      "lower_score": 2
    }
  },
  "groups": [
    {
      "normalized_hash": "...",
      "member_count": 2,
      "winner": {
        "index": 17,
        "id": "winner-chunk-id",
        "source": "huatuo_encyclopedia"
      },
      "winner_score": 0.0
    }
  ]
}
```

`output/dedup_dropped.jsonl` 每行对应一个被丢弃 chunk，示例：

```json
{
  "source": "huatuo_knowledge_graph",
  "score": 0.0,
  "winner_score": 0.0,
  "record_index": 217,
  "id": "dropped-chunk-id",
  "normalized_hash": "...",
  "duplicate_of": {
    "index": 17,
    "id": "winner-chunk-id",
    "source": "huatuo_encyclopedia"
  },
  "reason": "tie_first_seen"
}
```

这里的 `duplicate_of` 表示该条数据最终对应的保留项；`reason` 为 `lower_score` 或 `tie_first_seen`。报告和丢弃明细不保存完整医疗文本，使用 hash 做审计关联，避免不必要地重复落盘原文。

### 4. 数据处理相关 output 总表

| output 路径 | 生成模块 | 内容 | 是否原子写 |
|---|---|---|---:|
| `output/sync_manifest.json` | `doc_sync.py` | 每个 source 的文件 sha256、size、doc_id、chunk 数、chunk IDs、embedding 模型和最后状态 | 是（临时文件 + `os.replace`） |
| `output/quality_report.json` | `quality_gate.py` | 质量门禁规则版本、accepted/rejected、拒绝原因分布 | 是（临时文件 + `os.replace`） |
| `output/rejected_chunks.jsonl` | `quality_gate.py` | 质量门禁拒绝的 source、索引、原因和样本文本 | 否，带锁追加 |
| `output/dedup_report.json` | `dedup.py` | 去重总量、source/collection 分布、重复组和 winner | 是（临时文件 + `os.replace`） |
| `output/dedup_dropped.jsonl` | `dedup.py` | 每条 dropped 的 source、score、hash、winner 标识和 reason | 追加 |
| `output/metrics.jsonl` | `metrics.py` | embedding、sync、collection size、dedup 和审计事件指标 | 追加 |
| `output/audit.jsonl` | `utils/audit.py` | sync、dedup、质量/业务审计事件 | 带锁追加 |
| `chromaDB/` | ChromaDB | 实际向量、正文和 metadata | 向量库内部管理 |

> `output/` 和 `chromaDB/` 已加入 `.gitignore`，这些是运行时产物，不应提交到 Git；展示时可以展示 JSON 结构、终端指标和报告截图，不需要提交数据集或向量二进制。


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
- **MCP 双向集成而非单做**：MCP 是协议/基建，单做天花板低，焊进分诊项目——方向1（4 工具暴露成 MCP Server）+ 方向2（自研 client 适配器接入外部 HIS）。不用 `langchain-mcp-adapters`（会拉高 langchain-core、破坏 langgraph 0.2.74 旧生态），自写轻量 adapter 只依赖官方 mcp SDK；MCP 是 async 协议、LangGraph 工具同步执行，用后台线程常驻 event loop + `run_coroutine_threadsafe` 桥接。

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
- **密码重置 token 内存态**：一次性重置链接存在内存 `reset_tokens`，后端重启即失效（安全上属预期，生产换 Redis + 定时清理）；本地模拟短信下发到 `output/reset_links.log`，线上替换真实短信网关。

---

### 5. 数据处理方案验收结论

截至当前版本，数据处理方案的核心链路与工程化补强能力已完成，可以作为完整的数据处理方案进行介绍：

- 源文件变更感知、文件指纹、`doc_id`、doc 级删除重建、manifest、锁和 webhook 触发；
- 按句切片、overlap、chunk 追踪和本地 `bge-m3` embedding；
- 知识图谱缓存与向量同步、版本校验和 KG 产物失效；
- embedding 模型标记、模型迁移和维度不兼容治理；
- 药物禁忌文件变更检测与缓存失效；
- ChromaDB 快照、恢复和过期备份清理；
- embedding、同步、collection size、审计和去重指标；
- 结构、长度、metadata、标签和模板题质量门禁，拒绝记录与质量报告；
- 同 collection 内跨 source 精确去重、score 择优、丢弃审计和预览 CLI。

建议使用不修改数据库的命令：

```bash
python verify_b8_quality_gate.py
python verify_b9_dedup.py
python cli.py dedup --sources huatuo_encyclopedia,huatuo_knowledge_graph --dry-run --limit 200
python cli.py metrics --diagnose
```

> 真实启用需要在 `.env` 设置 `DEDUP_ENABLED=true`。不必为了展示而重灌全量向量库；可以直接展示 verifier、dry-run 报告、metrics 和审计设计。全量重建属于上线运维动作，应先备份并使用隔离 collection 做小批量验证。


- `verify_*.py`：各模块独立验证（KG / 症状匹配 / 科普 / 用药 / 标签均衡），不依赖 PostgreSQL。
- `verify_mcp.py`：MCP 双方向集成验证（方向1 连自建 server 调 retrieve + 方向2 bind HIS 工具看 LLM 调 query_registration）。
- `eval_triage_retrieval.py` / `eval_triage_judge.py` / `e2e_regression.py`：检索 / 分诊质量 / 端到端评估。
- 安全合规链路：危险拦截、脱敏、HITL 三态、审计留痕、开关（`HITL_ENABLED`）均已完成端到端验证。
