# 初始化脚本

将评测项目的资产灌入 ragent 服务，准备好"被测系统"的状态。

## 前置

- ragent 服务已启动（默认 `http://localhost:9090/api/ragent`）
- 已有可登录的账号

## 环境变量

```bash
export RAGENT_BASE_URL=http://localhost:9090/api/ragent  # 可省略，默认值
export RAGENT_USERNAME=<your_username>
export RAGENT_PASSWORD=<your_password>
```

## Step 1：创建 4 个知识库

```bash
python eval/rag/init/create_kbs.py
```

成功后写入 `eval/rag/init/kb_ids.json`，供后续 Step 2（灌文档）使用。

**注意**：脚本不做幂等。如需重跑，先手动清理 ragent 里同名 KB。

## Step 2：批量上传知识库 md 并触发分块

依赖 Step 1 产出的 `kb_ids.json`。依赖 `requests`（`pip install requests`，或直接装 `langchain_openai` 会带上）。

```bash
# 先 dry-run 确认文件分布
python eval/rag/init/upload_docs.py --dry-run

# 冒烟：只跑前 3 个
python eval/rag/init/upload_docs.py --limit 3

# 全量
python eval/rag/init/upload_docs.py

# 如果遇到 upload 信号量限制（ragent.semaphore.document-upload.max-concurrent=10），
# 加 --sleep 控速
python eval/rag/init/upload_docs.py --sleep 0.2
```

**幂等性**：脚本会读 `doc_id_map.json` 跳过已上传文件，可断点续传。重灌请先删 `doc_id_map.json` 并清理 ragent 侧文档。

`doc_id_map.json` 写在 `eval/rag/dataset/doc_id_map.json`，但它不是共享数据集，而是当前 ragent 数据库的本地 ID 映射。重新初始化、换机器或换数据库后，ragent 内部文档 ID 会变化，应重新生成这份文件。

**异步分块**：`/chunk` 接口只是触发，不等完成。全部上传完后稍等几分钟，调
`GET /knowledge-base/{kb-id}/docs` 看 `chunkCount > 0` 才算真正入向量库。

## 清空所有 KB 和文档（重置）

破坏性脚本，默认 dry-run。

```bash
# 看会删什么
python eval/rag/init/reset_kbs.py

# 真删（删完会顺便清掉本地 kb_ids.json / doc_id_map.json）
python eval/rag/init/reset_kbs.py --yes

# 保留本地映射文件
python eval/rag/init/reset_kbs.py --yes --keep-local
```

行为：
1. 拉全所有 KB
2. 每个 KB 下文档逐个 DELETE（碰到 RUNNING 会重试，默认 3 次每次间隔 5s）
3. 文档清完后 DELETE KB
4. 默认删除本地 `kb_ids.json` 和 `doc_id_map.json`

## Step 3：构建并灌入意图树

依赖 Step 1（`kb_ids.json`）和 Step 2（`doc_id_map.json`）。

```bash
# 先 dry-run 看结构
python eval/rag/init/build_intent_tree.py --dry-run

# 真灌
python eval/rag/init/build_intent_tree.py
```

产出 `intent_ids.json`（intentCode → ragent 节点 id 映射）。

**意图树结构**：3 个 DOMAIN（SUPPORT/FEEDBACK/CHAT）+ 5 个 CATEGORY + 22 个 TOPIC（18 个 KB-kind + 4 个 SYSTEM-kind）。

**KB 归属**：数据驱动——每个 leaf 的 KB 取评估集 `expected_doc_ids` 投票多数派。F2/F3/C1/C2 强制为 SYSTEM-kind，不走 RAG。

**幂等**：基于 `intent_ids.json` 跳过已创建 intentCode。重灌请先调 ragent 的 `/intent-tree/{id}` DELETE 接口清空，或手动 truncate 表。
