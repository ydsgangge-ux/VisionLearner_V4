# storage.py — v3.0  SQLite + ChromaDB
"""
完全替代旧版 JSON 文件存储。

变化：
  JSONStorage  → SQLiteStorage   （结构化数据，精确查询）
  TF-IDF 向量  → ChromaDB         （语义向量检索）

对外接口：DataManager 和 VectorStore 的所有方法签名完全不变，
main.py / collector.py / web_server.py 一行都不需要改。

数据库文件：
  learning_data/visionlearner.db        ← SQLite 主库
  learning_data/vector_db/              ← ChromaDB 向量库
  learning_data/backups/                ← 备份目录（不变）
"""

import json
import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── ChromaDB / sentence-transformers 软依赖 ──────────────────
import sys

# ChromaDB 1.5.2+ 已支持 Python 3.14，但需要网络连接下载模型
# 如果没有 sentence-transformers，自动降级到 TF-IDF
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _deep_serialize(obj):
    """递归序列化，把 enum / dataclass 等转为可 JSON 存储的类型"""
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(i) for i in obj]
    if hasattr(obj, "value"):   # Enum
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return _deep_serialize(asdict(obj))
    return obj


# ══════════════════════════════════════════════════════════════
# SQLiteStorage：替代 JSONStorage
# ══════════════════════════════════════════════════════════════

class SQLiteStorage:
    """
    用 SQLite 替代散落的 JSON 文件。

    表结构：一张通用的 kv 表，用 (collection, id) 作主键，
    data 列存 JSON blob。对原有接口完全透明。

    额外能力：
      - FTS5 全文搜索（对话历史、知识节点）
      - 事务保护，写到一半断电不损坏
      - 单文件，备份只需复制一个 .db 文件
    """

    def __init__(self, data_dir: str = "./learning_data"):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "visionlearner.db"
        self._local = threading.local()   # 每线程独立连接
        self._init_schema()

    # ── 连接管理 ──

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # 写时不阻塞读
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        """事务上下文管理器"""
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        """建表，幂等操作"""
        with self._tx() as conn:
            # 主 KV 表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv (
                    collection TEXT NOT NULL,
                    id         TEXT NOT NULL,
                    data       TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (collection, id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kv_collection
                ON kv(collection)
            """)

            # 对话历史表（支持 FTS5 全文搜索）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dialog_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    goal_id    TEXT,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    entity     TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # FTS5 虚拟表（全文搜索）
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS dialog_fts
                USING fts5(content, entity, session_id, goal_id,
                           content='dialog_log',
                           content_rowid='id',
                           tokenize='unicode61')
            """)
            # 触发器：自动同步到 FTS
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS dialog_log_ai
                AFTER INSERT ON dialog_log BEGIN
                    INSERT INTO dialog_fts(rowid, content, entity, session_id, goal_id)
                    VALUES (new.id, new.content, new.entity,
                            new.session_id, new.goal_id);
                END
            """)

    # ── 核心 CRUD（兼容旧 JSONStorage 接口）──

    def save(self, collection: str, obj_id: str, data: Dict) -> None:
        data["_updated_at"] = datetime.now().isoformat()
        blob = json.dumps(data, ensure_ascii=False)
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO kv(collection, id, data, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(collection, id) DO UPDATE SET
                    data=excluded.data,
                    updated_at=excluded.updated_at
            """, (collection, obj_id, blob, data["_updated_at"]))

    def load(self, collection: str, obj_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT data FROM kv WHERE collection=? AND id=?",
            (collection, obj_id)
        ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def delete(self, collection: str, obj_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM kv WHERE collection=? AND id=?",
                (collection, obj_id)
            )
        return cur.rowcount > 0

    def list_ids(self, collection: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT id FROM kv WHERE collection=?", (collection,)
        ).fetchall()
        return [r[0] for r in rows]

    def load_all(self, collection: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT data FROM kv WHERE collection=?", (collection,)
        ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row[0]))
            except Exception:
                pass
        return results

    def count(self, collection: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM kv WHERE collection=?", (collection,)
        ).fetchone()
        return row[0] if row else 0

    # ── 对话历史（新能力）──

    def save_dialog(self, session_id: str, goal_id: str,
                    role: str, content: str, entity: str = "") -> int:
        """保存一条对话记录，返回行ID"""
        with self._tx() as conn:
            cur = conn.execute("""
                INSERT INTO dialog_log
                    (session_id, goal_id, role, content, entity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, goal_id, role, content, entity,
                  datetime.now().isoformat()))
        return cur.lastrowid

    def search_dialog(self, query: str, goal_id: str = "",
                      limit: int = 10) -> List[Dict]:
        """
        FTS5 全文搜索对话历史。
        可以找到历史上问过的问题和答案。
        """
        if goal_id:
            rows = self._conn.execute("""
                SELECT d.session_id, d.goal_id, d.role,
                       d.content, d.entity, d.created_at
                FROM dialog_log d
                JOIN dialog_fts f ON f.rowid = d.id
                WHERE dialog_fts MATCH ?
                  AND d.goal_id = ?
                ORDER BY rank
                LIMIT ?
            """, (query, goal_id, limit)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT d.session_id, d.goal_id, d.role,
                       d.content, d.entity, d.created_at
                FROM dialog_log d
                JOIN dialog_fts f ON f.rowid = d.id
                WHERE dialog_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()

        return [dict(r) for r in rows]

    def get_recent_dialog(self, session_id: str,
                          limit: int = 20) -> List[Dict]:
        """获取某 session 的最近对话"""
        rows = self._conn.execute("""
            SELECT role, content, entity, created_at
            FROM dialog_log
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
        return list(reversed([dict(r) for r in rows]))

    # ── 备份 ──

    def backup(self) -> str:
        backup_dir = self.root / "backups"
        backup_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"visionlearner_{ts}.db"
        # SQLite 热备份（安全）
        src_conn = sqlite3.connect(str(self.db_path))
        dst_conn = sqlite3.connect(str(dest))
        src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        log.info(f"备份完成：{dest}")
        return str(dest)

    def export_all(self, path: str = "./export.json") -> None:
        """导出所有数据到 JSON（兼容旧版 export）"""
        all_data: Dict[str, Dict] = {}
        rows = self._conn.execute(
            "SELECT collection, id, data FROM kv"
        ).fetchall()
        for row in rows:
            col, obj_id = row[0], row[1]
            if col not in all_data:
                all_data[col] = {}
            try:
                all_data[col][obj_id] = json.loads(row[2])
            except Exception:
                pass
        Path(path).write_text(
            json.dumps(all_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def import_all(self, path: str) -> int:
        """从 JSON 文件导入（兼容旧版 import）"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        count = 0
        for collection, items in data.items():
            for obj_id, item in items.items():
                self.save(collection, obj_id, item)
                count += 1
        return count

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ══════════════════════════════════════════════════════════════
# DataManager：接口完全兼容旧版
# ══════════════════════════════════════════════════════════════

class DataManager:
    """
    数据管理层。对上层（main.py/collector.py）完全透明：
    所有方法名、参数、返回值与旧版一致。
    底层从 JSONStorage 换成 SQLiteStorage。
    """

    def __init__(self, data_dir: str = "./learning_data"):
        self.storage = SQLiteStorage(data_dir)
        self.data_dir = data_dir
        self._session_id = f"session_{int(datetime.now().timestamp())}"
        self._ensure_system_record()

    def _ensure_system_record(self):
        existing = self.storage.load("system", "meta")
        if not existing:
            self.storage.save("system", "meta", {
                "id": "meta",
                "created_at": datetime.now().isoformat(),
                "version": "3.0.0",
                "total_sessions": 0,
                "total_learning_minutes": 0,
            })

    # ── 学习目标 ──

    def save_goal(self, goal) -> None:
        if hasattr(goal, "to_dict"):
            data = goal.to_dict()
        elif hasattr(goal, "__dataclass_fields__"):
            data = asdict(goal)
        else:
            data = goal
        data = _deep_serialize(data)
        self.storage.save("goals", data["id"], data)

    def load_goal(self, goal_id: str) -> Optional[Dict]:
        return self.storage.load("goals", goal_id)

    def load_all_goals(self) -> List[Dict]:
        return self.storage.load_all("goals")

    def delete_goal(self, goal_id: str) -> bool:
        return self.storage.delete("goals", goal_id)

    def get_active_goals(self) -> List[Dict]:
        return [g for g in self.load_all_goals()
                if g.get("status") == "active"]

    # ── 思维导图 ──

    def save_mindmap(self, mindmap_id: str, data: Dict) -> None:
        self.storage.save("mindmaps", mindmap_id, data)

    def load_mindmap(self, mindmap_id: str) -> Optional[Dict]:
        return self.storage.load("mindmaps", mindmap_id)

    def load_all_mindmaps(self) -> List[Dict]:
        return self.storage.load_all("mindmaps")

    # ── 知识节点 ──

    def save_knowledge_node(self, node) -> None:
        if hasattr(node, "to_dict"):
            data = node.to_dict()
        elif hasattr(node, "__dataclass_fields__"):
            data = asdict(node)
        else:
            data = node
        self.storage.save("knowledge", data["id"], data)

    def load_knowledge_node(self, node_id: str) -> Optional[Dict]:
        return self.storage.load("knowledge", node_id)

    def load_all_knowledge(self) -> List[Dict]:
        return self.storage.load_all("knowledge")

    def search_knowledge(self, keyword: str) -> List[Dict]:
        """全文搜索知识节点"""
        keyword = keyword.lower()
        results = []
        for node in self.load_all_knowledge():
            title   = node.get("title",   "").lower()
            content = str(node.get("content", "")).lower()
            tags    = " ".join(node.get("tags", [])).lower()
            if keyword in title or keyword in content or keyword in tags:
                results.append(node)
        return results

    # ── 学习计划 ──

    def save_plan(self, plan_id: str, plan: Dict) -> None:
        self.storage.save("plans", plan_id, plan)

    def load_plan(self, plan_id: str) -> Optional[Dict]:
        return self.storage.load("plans", plan_id)

    def save_learning_plan(self, goal_id: str, plan: dict) -> None:
        self.storage.save("plans", goal_id, plan)

    def load_learning_plan(self, goal_id: str) -> Optional[dict]:
        return self.storage.load("plans", goal_id)

    # ── 排课 ──

    def save_schedule(self, goal_id: str, schedule: dict) -> None:
        self.storage.save("schedules", goal_id, schedule)

    def load_schedule(self, goal_id: str) -> Optional[dict]:
        return self.storage.load("schedules", goal_id)

    # ── 进度 ──

    def record_progress(self, goal_id: str, progress_data: Dict) -> None:
        key = f"{goal_id}_history"
        existing = self.storage.load("progress", key) or {
            "id": key, "goal_id": goal_id, "history": []
        }
        existing["history"].append({
            **progress_data,
            "recorded_at": datetime.now().isoformat()
        })
        existing["history"] = existing["history"][-100:]
        self.storage.save("progress", key, existing)

    def load_progress_history(self, goal_id: str) -> List[Dict]:
        key = f"{goal_id}_history"
        data = self.storage.load("progress", key)
        return data.get("history", []) if data else []

    def save_progress_data(self, goal_id: str, progress: dict) -> None:
        self.storage.save("progress", f"{goal_id}_data", progress)

    def load_progress_data(self, goal_id: str) -> Optional[dict]:
        return self.storage.load("progress", f"{goal_id}_data")

    # ── 复习 ──

    def save_review(self, review: Dict) -> None:
        review_id = review.get(
            "id", f"review_{datetime.now().timestamp()}"
        )
        review["id"] = review_id
        self.storage.save("reviews", review_id, review)

    def get_due_reviews(self) -> List[Dict]:
        now = datetime.now().isoformat()
        due = [
            r for r in self.storage.load_all("reviews")
            if r.get("next_review_at", "") <= now
               and r.get("next_review_at", "")
        ]
        return sorted(due, key=lambda x: x.get("next_review_at", ""))

    # ── 会话 ──

    def start_session(self, goal_id: str,
                      session_type: str = "learning") -> Dict:
        session = {
            "id": f"session_{int(datetime.now().timestamp())}",
            "goal_id": goal_id,
            "type": session_type,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "duration_minutes": 0,
            "items_learned": 0,
            "notes": "",
        }
        self.storage.save("sessions", session["id"], session)
        return session

    def end_session(self, session_id: str,
                    summary: Dict = None) -> None:
        session = self.storage.load("sessions", session_id)
        if session:
            session["ended_at"] = datetime.now().isoformat()
            if session.get("started_at"):
                start = datetime.fromisoformat(session["started_at"])
                session["duration_minutes"] = int(
                    (datetime.now() - start).total_seconds() / 60
                )
            if summary:
                session.update(summary)
            self.storage.save("sessions", session_id, session)
            self._update_system_stats(session)

    def _update_system_stats(self, session: Dict) -> None:
        meta = self.storage.load("system", "meta") or {}
        meta["total_sessions"] = meta.get("total_sessions", 0) + 1
        meta["total_learning_minutes"] = (
            meta.get("total_learning_minutes", 0)
            + session.get("duration_minutes", 0)
        )
        meta["last_session_at"] = datetime.now().isoformat()
        self.storage.save("system", "meta", meta)

    # ── 对话历史（新能力）──

    def save_dialog_turn(self, goal_id: str, user_input: str,
                         response: str, entity: str = "") -> None:
        """保存一轮对话（user + assistant）"""
        sid = self._session_id
        self.storage.save_dialog(sid, goal_id, "user",
                                  user_input, entity)
        self.storage.save_dialog(sid, goal_id, "assistant",
                                  response[:500], entity)

    def search_dialog_history(self, query: str,
                               goal_id: str = "",
                               limit: int = 5) -> List[Dict]:
        """跨会话搜索历史对话"""
        return self.storage.search_dialog(query, goal_id, limit)

    def get_session_dialog(self, limit: int = 10) -> List[Dict]:
        """获取本次会话最近对话"""
        return self.storage.get_recent_dialog(
            self._session_id, limit
        )

    # ── 系统状态 ──

    def save_system_state(self, state: Dict) -> None:
        self.storage.save("system", "state", {
            **state, "saved_at": datetime.now().isoformat()
        })

    def load_system_state(self) -> Optional[Dict]:
        return self.storage.load("system", "state")

    def get_statistics(self) -> Dict:
        meta = self.storage.load("system", "meta") or {}
        return {
            "total_goals":          self.storage.count("goals"),
            "active_goals":         len(self.get_active_goals()),
            "total_knowledge_nodes": self.storage.count("knowledge"),
            "total_mindmaps":       self.storage.count("mindmaps"),
            "total_sessions":       meta.get("total_sessions", 0),
            "total_learning_minutes": meta.get("total_learning_minutes", 0),
            "last_session_at":      meta.get("last_session_at", ""),
            "data_dir":             self.data_dir,
            "db_size_mb": round(
                Path(self.storage.db_path).stat().st_size / 1024 / 1024, 2
            ) if Path(self.storage.db_path).exists() else 0,
        }

    def get_data_statistics(self) -> Dict:
        """兼容旧版本的方法名"""
        stats = self.get_statistics()
        return {
            "goals": stats["total_goals"],
            "mindmap_trees": self.storage.count("mindmap_trees"),
            "goal_units": self.storage.count("goal_units"),
            "plans": self.storage.count("plans"),
            "schedules": self.storage.count("schedules"),
            "progress": self.storage.count("progress"),
            "storage_size": f"{stats['db_size_mb']} MB" if stats['db_size_mb'] > 0 else "0 KB",
        }

    def backup(self) -> str:
        dest = self.storage.backup()
        print(f"OK Data backed up to {dest}")
        return dest

    def export(self, path: str = "./export.json") -> None:
        self.storage.export_all(path)

    def import_data(self, path: str) -> int:
        return self.storage.import_all(path)

    # ── 兼容旧版的其他方法 ──

    def _find_node_by_title(self, tree, title: str):
        from foundation import MindMapNode as FMN
        node_map: Dict = {}
        def build_map(node):
            node_map[node.id] = node
            for child in node.children:
                build_map(child)
        build_map(tree)
        return tree.find_by_title(title, node_map)

    def _tree_to_list(self, tree) -> List:
        result = []
        def add_node(node):
            result.append(node)
            for child in node.children:
                add_node(child)
        add_node(tree)
        return result


# ══════════════════════════════════════════════════════════════
# VectorStore：ChromaDB 替代 TF-IDF，接口完全兼容
# ══════════════════════════════════════════════════════════════

class VectorStore:
    """
    语义向量检索。
    ChromaDB 可用时用 ChromaDB，否则降级到 TF-IDF。
    对外接口与旧版完全一致。
    """

    def __init__(self, data_dir: str = "./learning_data"):
        self.data_dir = Path(data_dir)
        self._backend = self._init_backend()
        log.info(f"VectorStore 初始化：backend={self._backend}")

    def _init_backend(self) -> str:
        if HAS_CHROMADB:
            try:
                chroma_dir = self.data_dir / "vector_db" / "chroma"
                chroma_dir.mkdir(parents=True, exist_ok=True)
                self._chroma = chromadb.PersistentClient(
                    path=str(chroma_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                # 网络原因禁用 sentence-transformers，直接使用默认 embedding
                # 避免 huggingface.co 连接超时
                self._embed_fn = None
                log.info("向量后端：ChromaDB 默认 embedding")
                self._cols: Dict[str, Any] = {}
                return "chromadb"
            except Exception as e:
                log.warning(f"ChromaDB 初始化失败，降级 TF-IDF：{e}")

        # 降级到 TF-IDF
        return self._init_tfidf()

    # ── ChromaDB 方法 ──

    def _get_col(self, goal_id: str):
        """获取或创建 goal 对应的 ChromaDB collection"""
        if goal_id not in self._cols:
            kwargs = {"name": f"goal_{goal_id}",
                      "metadata": {"hnsw:space": "cosine"}}
            if self._embed_fn:
                kwargs["embedding_function"] = self._embed_fn
            self._cols[goal_id] = self._chroma.get_or_create_collection(
                **kwargs
            )
        return self._cols[goal_id]

    def _chroma_add(self, goal_id: str, doc_id: str,
                    text: str, metadata: Dict) -> None:
        col = self._get_col(goal_id)
        # 清理 metadata：ChromaDB 只接受 str/int/float/bool
        clean_meta = {
            k: str(v) for k, v in metadata.items()
            if v is not None
        }
        col.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[clean_meta],
        )

    def _chroma_search(self, query: str, goal_id: str,
                       top_k: int) -> List[Dict]:
        try:
            col = self._get_col(goal_id)
            count = col.count()
            if count == 0:
                return []
            k = min(top_k, count)
            res = col.query(query_texts=[query], n_results=k)
            results = []
            for i, doc in enumerate(res["documents"][0]):
                meta = res["metadatas"][0][i]
                dist = res["distances"][0][i]
                score = max(0.0, 1.0 - dist)   # cosine距离→相似度
                results.append({
                    "unit":       meta.get("unit", ""),
                    "node_title": meta.get("node_title", ""),
                    "content":    doc,
                    "score":      round(score, 4),
                    "goal_id":    goal_id,
                })
            return results
        except Exception as e:
            log.warning(f"ChromaDB 检索失败：{e}")
            return []

    # ── 公共接口（与旧版完全一致）──

    def add_unit_knowledge(self, goal_id: str, unit: str,
                           content_text: str,
                           node_title: str = "",
                           is_coarse: bool = False) -> None:
        """存入一条知识"""
        import hashlib
        doc_id = hashlib.md5(
            f"{goal_id}_{unit}_{node_title}".encode()
        ).hexdigest()[:16]

        text = f"{unit} {node_title}：{content_text}" \
               if node_title else f"{unit}：{content_text}"

        metadata = {
            "goal_id":    goal_id,
            "unit":       unit,
            "node_title": node_title,
            "is_coarse":  "1" if is_coarse else "0",
        }

        if self._backend == "chromadb":
            self._chroma_add(goal_id, doc_id, text, metadata)
        else:
            self._tfidf_add(goal_id, doc_id, text, metadata)

    def search(self, query: str, goal_id: str = "",
               top_k: int = 5) -> List[Dict]:
        """语义检索，返回最相关的知识列表"""
        if not query.strip():
            return []
        if self._backend == "chromadb":
            return self._chroma_search(query, goal_id, top_k)
        return self._tfidf_search(query, goal_id, top_k)

    def delete_unit(self, goal_id: str, unit: str) -> None:
        if self._backend == "chromadb":
            try:
                col = self._get_col(goal_id)
                existing = col.get(
                    where={"unit": unit}
                )
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
            except Exception as e:
                log.warning(f"ChromaDB 删除失败：{e}")
        else:
            self._tfidf_delete_unit(goal_id, unit)

    def count(self, goal_id: str = "") -> int:
        if self._backend == "chromadb":
            try:
                if goal_id:
                    return self._get_col(goal_id).count()
                total = 0
                for col_name in [
                    c.name for c in self._chroma.list_collections()
                ]:
                    try:
                        total += self._chroma.get_collection(
                            col_name
                        ).count()
                    except Exception:
                        pass
                return total
            except Exception:
                return 0
        if goal_id:
            return len(self._tfidf_index.get(goal_id, []))
        return sum(len(v) for v in self._tfidf_index.values())

    # ── TF-IDF 降级实现（完整保留）──

    def _init_tfidf(self) -> str:
        import math
        from collections import defaultdict
        self._math = math
        self._defaultdict = defaultdict
        self._tfidf_index: Dict[str, List[Dict]] = defaultdict(list)
        self._tfidf_idf:   Dict[str, Dict[str, float]] = {}
        self._tfidf_dirty: Dict[str, bool] = defaultdict(lambda: True)
        tfidf_path = self.data_dir / "vector_db" / "tfidf_index.json"
        tfidf_path.parent.mkdir(parents=True, exist_ok=True)
        self._tfidf_path = tfidf_path
        self._load_tfidf_index()
        return "tfidf"

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if "\u4e00" <= c <= "\u9fff":
                tokens.append(c)
                if (i + 1 < len(text) and
                        "\u4e00" <= text[i + 1] <= "\u9fff"):
                    tokens.append(text[i:i + 2])
            elif c.isalpha():
                j = i
                while j < len(text) and text[j].isalpha():
                    j += 1
                w = text[i:j].lower()
                if len(w) > 1:
                    tokens.append(w)
                i = j
                continue
            i += 1
        return tokens

    def _rebuild_idf(self, goal_id: str, corpus: List[Dict]) -> None:
        import math
        from collections import defaultdict
        N = len(corpus)
        df: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            for t in set(self._tokenize(doc["text"])):
                df[t] += 1
        self._tfidf_idf[goal_id] = {
            t: math.log((N + 1) / (cnt + 1)) + 1
            for t, cnt in df.items()
        }

    def _tfidf_add(self, goal_id: str, doc_id: str,
                   text: str, metadata: Dict) -> None:
        docs = self._tfidf_index[goal_id]
        for doc in docs:
            if doc["id"] == doc_id:
                doc["text"] = text
                doc["meta"] = metadata
                self._tfidf_dirty[goal_id] = True
                self._save_tfidf_index()
                return
        docs.append({"id": doc_id, "text": text, "meta": metadata})
        self._tfidf_dirty[goal_id] = True
        self._save_tfidf_index()

    def _tfidf_search(self, query: str, goal_id: str,
                      top_k: int) -> List[Dict]:
        import math
        from collections import defaultdict

        if goal_id and goal_id in self._tfidf_index:
            corpus = self._tfidf_index[goal_id]
        elif not goal_id:
            corpus = []
            for docs in self._tfidf_index.values():
                corpus.extend(docs)
        else:
            return []
        if not corpus:
            return []

        if self._tfidf_dirty.get(goal_id, True):
            self._rebuild_idf(goal_id, corpus)
            self._tfidf_dirty[goal_id] = False

        query_tokens = set(self._tokenize(query))
        idf = self._tfidf_idf.get(goal_id, {})
        N = len(corpus)

        scores = []
        for doc in corpus:
            tokens = self._tokenize(doc["text"])
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            total = len(tokens) or 1
            score = sum(
                tf[qt] / total * idf.get(qt, math.log(N + 1))
                for qt in query_tokens if qt in tf
            )
            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: -x[0])
        results = []
        for score, doc in scores[:top_k]:
            meta = doc.get("meta", {})
            results.append({
                "unit":       meta.get("unit", ""),
                "node_title": meta.get("node_title", ""),
                "content":    doc["text"],
                "score":      round(score, 4),
                "goal_id":    meta.get("goal_id", goal_id),
            })
        return results

    def _tfidf_delete_unit(self, goal_id: str, unit: str) -> None:
        if goal_id in self._tfidf_index:
            self._tfidf_index[goal_id] = [
                d for d in self._tfidf_index[goal_id]
                if d.get("meta", {}).get("unit") != unit
            ]
            self._tfidf_dirty[goal_id] = True
            self._save_tfidf_index()

    def _load_tfidf_index(self) -> None:
        if self._tfidf_path.exists():
            try:
                from collections import defaultdict
                data = json.loads(
                    self._tfidf_path.read_text(encoding="utf-8")
                )
                self._tfidf_index = defaultdict(list, data)
                log.info(f"TF-IDF 索引已加载："
                         f"{sum(len(v) for v in data.values())} 条")
            except Exception as e:
                log.warning(f"TF-IDF 索引加载失败：{e}")

    def _save_tfidf_index(self) -> None:
        try:
            self._tfidf_path.write_text(
                json.dumps(dict(self._tfidf_index),
                           ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"TF-IDF 索引保存失败：{e}")
