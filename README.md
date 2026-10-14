# Agentic RAG：三阶段级联问答

基于 LangGraph 的证据驱动问答系统。检索按 **Dense Retrieval → LinearRAG → LightRAG local** 的固定顺序进行。每阶段由 Judge 检查现有证据；证据足够时直接生成带来源引用的中文回答，证据不足时提取有效片段放入当前问题的 Evidence Working Memory，再进入下一阶段。

```text
问题 → Dense Retrieval → Judge ──证据充分──→ Answer
                         │
                         └─不足→ Evidence Pool → LinearRAG → Judge ──充分──→ Answer
                                                        │
                                                        └─不足→ Evidence Pool → LightRAG local → Judge / Answer
```

三种检索器通过 MCP 工具接入，LangGraph 负责阶段顺序、状态和提前停止。LightRAG 的 naive 模式承担稠密片段检索；LinearRAG 和 LightRAG local 负责后续检索。检索器运行在各自的 Python 环境中。Evidence Pool 只保存当前问题的结构化证据及来源，不做跨会话长期记忆。

## 仓库内容

- `src/agentic_rag/graph.py`：LangGraph 节点、条件边、阶段推进和证据池。
- `src/agentic_rag/state.py`：共享状态与证据类型。
- `src/agentic_rag/tools/`：三个 MCP 客户端和检索结果归一化。
- `src/agentic_rag/llm.py`：证据充分性判断、回答和界面翻译。
- `mcp_servers/`：LightRAG、LinearRAG 的 MCP 服务及 LinearRAG 工作进程。
- `src/agentic_rag/cli.py`：命令行入口。
- `streamlit_app.py`：Streamlit 页面，默认使用本机 8502 端口。
- `tests/`：不依赖真实索引的基础测试。

## 环境准备

1. 准备 Python 3.11 或更新版本作为主环境，并运行 `python -m pip install -e .`。
2. 分别准备可运行的 LightRAG 和 LinearRAG 项目及其 Python 环境，并事先构建相应数据集的索引。**这两个上游项目、模型、数据集和索引没有包含在本仓库中。** LinearRAG 使用其仓库内 `import/<dataset>/ner_results.json` 等索引文件。
3. 复制 `.env.example` 为 `.env`，填写项目路径、各子环境的 Python 可执行文件、模型地址与密钥。`.env` 已被 Git 忽略。
4. `LIGHTRAG_INDEX_ROOT` 指向含有以下目录的父目录。界面按所选数据集查找索引：

   ```text
   <LIGHTRAG_INDEX_ROOT>/
     hotpotqa/index/
     2wikimultihopqa/index/
     musique/index/
   ```

   `LIGHTRAG_ENV_FILE` 指向 LightRAG 已有的 embedding 配置文件；`LINEARRAG_ROOT` 指向 LinearRAG 仓库根目录。两者的依赖和索引格式应与各自项目代码匹配。

5. `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 配置兼容 OpenAI Chat Completions 的模型。Judge、中文回答以及 LightRAG local 都会用到模型；LightRAG 检索还需要其原项目的 embedding 设置。

## 运行

在仓库根目录执行：

```powershell
agentic-rag "Your question here"
```

命令行默认使用 HotpotQA 的 LightRAG 索引；可通过 `--index` 和 `--linear-dataset` 指定别的索引与数据集。`--max-stage dense` 或 `--max-stage linear` 用于只运行前一或前两阶段；`--retrieve-only` 用于查看原始检索状态。运行 `agentic-rag --help` 查看完整参数。

启动网页：

```powershell
streamlit run streamlit_app.py --server.port 8502
```

打开 `http://127.0.0.1:8502`。网页支持 HotpotQA、2WikiMultiHopQA、MuSiQue；对应的 LightRAG 与 LinearRAG 索引须提前准备。

## 测试

```powershell
python -m unittest discover -s tests -v
```

这些测试使用模拟 MCP 服务，不验证外部索引、模型或三个上游检索流程的实际效果。

## 公开仓库说明

本仓库只含问答编排代码、MCP 适配代码、测试和空白配置模板。密钥、真实 `.env`、本机路径、索引、缓存及实验结果应保留在本地。上传前可用 `git status --short` 与 `git diff --cached --name-only` 核对待提交文件。
