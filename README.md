# Multimodal-rag

基于 RAG (检索增强生成) 的设备/产品手册智能问答系统，支持 PDF 文档导入、多源检索、流式回答及图片关联展示。

## 核心特性

- **PDF 自动解析入库** — 上传 PDF 后自动完成：OCR/解析 → 图片提取与摘要 → 文档分块 → 向量化 → 存入 Milvus
- **多源混合检索** — 向量语义检索 (BGE-M3 dense+sparse) + HyDE 假设文档检索 + 网络搜索 (阿里百炼 MCP)
- **智能排序** — RRF 融合排序 + BGE-Reranker 交叉编码器精排 + Dynamic TopK 断崖检测
- **流式输出** — SSE 实时流式推送回答内容，支持逐 token 输出
- **图片关联** — 回答中自动关联文档原图，通过 MinIO 存储提供访问
- **会话管理** — MongoDB 持久化聊天历史，支持按 session 查看和清理

## 技术栈

| 层级        | 技术                             |
| --------- | ------------------------------ |
| Web 框架    | FastAPI + Uvicorn              |
| 工作流引擎     | LangGraph (StateGraph)         |
| Embedding | BGE-M3 (dense 1024维 + sparse ) |
| Reranker  | BGE-Reranker-Large             |
| 向量数据库     | Milvus v2.4.11                 |
| 关系存储      | MongoDB (聊天记录)                 |
| 对象存储      | MinIO (文档图片)                   |
| PDF 解析    | MinerU                         |
| Python    | 3.13                           |
| 包管理       | uv                             |

## 系统架构(导入+查询）

```
┌─────────────────────────────────────────────────────────────────┐
│                       导入服务 (port 8000)                       │
│                                                                 │
│  PDF 上传 → 文件检测 → PDF→MD → 图片提取/VLM摘要 → 文档分块     │
│         → 产品名识别 → BGE-M3 向量化 → Milvus 写入              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       查询服务 (port 8001)                       │
│                                                                 │
│  用户提问 → 产品名确认+查询改写 → [三路并行检索]                  │
│              ├─ 向量混合检索 (dense+sparse)                      │
│              ├─ HyDE 假设文档检索                                │
│              └─ 网络搜索 (阿里百炼 MCP)                          │
│         → RRF 融合排序 → Reranker 精排 → LLM 生成回答 (SSE)    │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.13 + [uv](https://github.com/astral-sh/uv) 包管理器
- Docker + Docker Compose
- MongoDB (默认端口 27017)
- MinIO (默认端口 9000)
- 阿里 DashScope API Key
- MinerU API Token (OpenXLab 平台获取)

### 安装步骤

1. **安装依赖**
   ```bash
   uv sync
   ```
2. **下载本地模型**
   ```bash
   # BGE-M3 Embedding 模型
   python app/tool/download_bgem3.py

   # BGE-Reranker 模型
   python app/tool/download_reranker.py
   ```
3. **启动 Milvus Docker 栈**
   ```bash
   cd milvus-docker
   docker-compose up -d
   ```
   启动 etcd + MinIO + Milvus + Attu 管理界面。
4. **配置环境变量**

   复制并编辑项目根目录的 `.env` 文件，填入以下关键配置：
   ```env
   # 阿里 DashScope (LLM)
   OPENAI_API_KEY=your_dashscope_api_key
   OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

   # MinerU (PDF 解析)
   MINERU_API_TOKEN=your_mineru_token

   # Milvus
   MILVUS_URL=http://127.0.0.1:19530

   # MongoDB
   MONGO_URL=mongodb://localhost:27017/

   # MinIO
   MINIO_ENDPOINT=http://127.0.0.1:9000
   MINIO_ACCESS_KEY=your_access_key
   MINIO_SECRET_KEY=your_secret_key

   # 本地模型路径
   BGE_M3_PATH=your_bgem3_model_path
   BGE_RERANKER_LARGE=your_reranker_model_path
   ```
5. **启动服务**
   ```bash
   # 导入服务 (端口 8000)
   python -m uvicorn app.import_process.api.import_server:app --host 127.0.0.1 --port 8000

   # 查询服务 (端口 8001)
   python -m uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8001
   ```

### 访问地址

| 页面   | 地址                                |
| ---- | --------------------------------- |
| 文档上传 | <http://localhost:8000/import>    |
| 智能问答 | <http://localhost:8001/chat.html> |

## API 接口

### 导入服务 (port 8000)

| 方法   | 路径                  | 说明               |
| ---- | ------------------- | ---------------- |
| GET  | `/import`           | 文件上传页面           |
| POST | `/upload`           | 上传 PDF 文件，触发导入流程 |
| GET  | `/status/{task_id}` | 查询导入任务状态         |

### 查询服务 (port 8001)

| 方法     | 路径                      | 说明                 |
| ------ | ----------------------- | ------------------ |
| GET    | `/health`               | 健康检查               |
| POST   | `/query`                | 提交查询 (支持同步/SSE 流式) |
| GET    | `/stream/{session_id}`  | SSE 事件流，实时推送回答     |
| GET    | `/history/{session_id}` | 获取聊天历史             |
| DELETE | `/history/{session_id}` | 清除聊天历史             |

## 项目结构

```
├── app/
│   ├── conf/                 # 配置层 (Dataclass)
│   ├── clients/              # 数据库客户端 (Milvus, MinIO, MongoDB, Neo4j)
│   ├── core/                 # 核心工具 (日志, Prompt 加载)
│   ├── lm/                   # 模型层 (LLM, Embedding, Reranker)
│   ├── import_process/       # 文档导入流水线
│   │   ├── api/              # FastAPI 入口 + 前端页面
│   │   └── agent/            # LangGraph 导入工作流 (7 节点)
│   ├── query_process/        # 查询问答流水线
│   │   ├── api/              # FastAPI 入口 + 前端页面
│   │   └── agent/            # LangGraph 查询工作流 (7 节点)
│   ├── utils/                # 通用工具 (SSE, 限流, 路径等)
│   └── tool/                 # 模型下载脚本
├── prompts/                  # LLM Prompt 模板 (.prompt)
├── doc/                      # 样例 PDF 文档 (~90 份产品手册)
├── milvus-docker/            # Milvus Docker 部署配置
├── pyproject.toml            # 项目依赖声明
└── .env                      # 环境变量配置
```

## 工作原理

### 导入流水线

```
PDF 上传 → 文件类型检测 → MinerU 解析 (PDF→MD)
  → 图片提取 + VLM 摘要 + MinIO 上传
  → 两阶段分块 (标题分割 + 递归字符分割)
  → LLM 产品名识别 → BGE-M3 向量化 (dense+sparse)
  → Milvus 写入 (幂等：同产品名先删后插)
```

### 查询流水线

```
用户提问 → LLM 提取产品名 + 查询改写
  → Milvus 产品名匹配 (命中则直接回答)
  → 未命中：三路并行检索
     ├─ 向量混合检索 (dense + sparse)
     ├─ HyDE 检索 (LLM 生成假设文档 → 向量检索)
     └─ 网络搜索 (阿里百炼 MCP)
  → RRF 融合排序 → BGE-Reranker 精排 + Dynamic TopK
  → LLM 生成回答 (SSE 流式输出) → MongoDB 保存历史
```

## 设计亮点

- **BGE-M3 双模态向量** — Dense (1024维, COSINE) 语义检索 + Sparse (IP) 关键词检索，L2 归一化后融合
- **Dynamic TopK 断崖检测** — Reranker 分数断崖式下降时自动截断，防止低质量文档污染上下文
- **HyDE** — 先让 LLM 生成假设性回答文档，再用假设文档的向量进行检索，提升语义匹配精度
- **幂等导入** — 同一产品名重复导入时自动清理旧数据，保证一致性
- **SSE 线程桥接** — 通过 queue.Queue + asyncio.run\_in\_executor() 实现 LangGraph 同步执行与 FastAPI 异步 SSE 的桥接

## 许可证

本项目为学习/研究用途。
