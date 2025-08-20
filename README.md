# deepSearch
---
## 📌 项目简介

本项目实现了一个工业级 **倒排索引搜索引擎**，并扩展了 **记忆模块 (Memory Module)**。
主要功能包括：

1. **并行构建倒排索引**（支持大规模文本集合）
2. **布尔检索、短语检索**
3. **BM25 排序机制**
4. **内存压缩与持久化存储**
5. **基于搜索结果的记忆抽取与压缩**
6. **结合记忆与最新搜索结果的 QA 问答模块**

## 📂 项目结构

```
deepsearch/
├─ app/
│  ├─ main.py
│  ├─ config.py
│  ├─ schemas.py
│  └─ deps.py
│
├─ deepsearch/
│  ├─ __init__.py
│  ├─ callbacks.py
│  ├─ utils.py
│  ├─ loaders.py
│  ├─ splitter.py
│  ├─ embeddings.py
│  ├─ vectorstore.py
│  ├─ bm25_index.py
│  ├─ hybrid_retriever.py
│  ├─ rag_chain.py
│  ├─ memory_store.py              # ⭐ 新增：记忆向量库（FAISS）
│  ├─ agent.py
│  └─ tools/
│     ├─ web_search.py
│     ├─ confidence.py
│     ├─ query_rewrite.py
│     └─ memory.py                 # ⭐ 新增：相关记忆提取/检索/压缩 工具
│
├─ prompts/
│  ├─ query_rewrite.md
│  ├─ rag_answer.md                # ⭐ 更新：支持 memory 段
│  ├─ confidence.md
│  ├─ memory_extract.md            # ⭐ 新增：从检索结果中提取可存储记忆
│  └─ memory_compress.md           # ⭐ 新增：压缩记忆为紧凑上下文
│
├─ scripts/
│  └─ build_indices.py
│
├─ data/
│  └─ pdf/
├─ model/
├─ stores/
│  ├─ faiss/
│  ├─ bm25/
│  └─ memory/                      # ⭐ 新增：记忆向量库持久化
└─ requirements.txt
```

---

## ⚙️ 安装与依赖

请确保 Python 版本 ≥ **3.9**。
安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

* `nltk` → 分词、停用词、词干化
* `loguru` → 日志打印
* `tqdm` → 进度条
* `scikit-learn` → 余弦相似度计算（记忆压缩）

---

## 🚀 使用方法

### 1. 构建索引

在 `data/` 文件夹中放置多个 `.txt` 文本文件，然后运行：

```bash
python src/main.py --mode build
```

输出：

* 生成 `index/main_index.json`（存储倒排索引）
* 打印日志（构建进度、词项数量、索引大小等）

---

### 2. 搜索

执行搜索命令：

```bash
python src/main.py --mode search --query "artificial intelligence in finance"
```

输出示例：

```
🔍 搜索查询: artificial intelligence in finance
✅ 匹配文档数: 12
🏆 Top 3 结果:
  [Doc 3] ... sentence snippet ...
  [Doc 8] ... sentence snippet ...
  [Doc 11] ... sentence snippet ...
```

---

### 3. 搜索 + 记忆抽取

运行带记忆功能的搜索：

```bash
python src/main.py --mode search --query "deep learning applications"
```

流程：

1. **先执行倒排索引搜索**
2. **调用记忆模块**：从 `memory_store.json` 提取历史相关信息
3. **压缩记忆**（相似性聚合）
4. **结合搜索结果 + 记忆进行 QA**

输出示例：

```
🔍 搜索查询: deep learning applications
✅ 匹配文档数: 8
🧠 提取相关记忆数: 3
📦 压缩后记忆数: 1
💡 综合回答:
  Deep learning has been applied in areas such as computer vision,
  natural language processing, and finance...
```

---

### 4. 查看记忆内容

运行：

```bash
python src/main.py --mode memory
```

输出示例：

```
🧠 当前记忆存储:
1. Deep learning is widely applied in NLP and finance.
2. Inverted index improves efficiency in text retrieval.
```

---

### 5. 清理索引与记忆

```bash
rm -rf index/*.json
```

---

## 🧪 测试与验证

可执行单元测试：

```bash
pytest tests/
```

测试覆盖：

* 倒排索引构建正确性
* BM25 排序是否符合预期
* 记忆抽取与压缩逻辑
* QA 模块输出合理性

---

## 🔑 核心设计亮点

* **工业级可扩展性**：支持并行构建 + 分段合并索引
* **可解释性**：布尔检索 + 短语匹配 + BM25 排序
* **记忆增强**：结合历史搜索上下文，提高 QA 连贯性
* **高鲁棒性**：边界检查、异常处理、日志追踪

---

## 📌 示例流程图

```
用户 Query → 搜索引擎 → 倒排索引结果
                         ↓
                    记忆模块提取
                         ↓
                   记忆压缩与合并
                         ↓
               QA 模块生成综合回答
```

---

## 📜 许可证

MIT License

---