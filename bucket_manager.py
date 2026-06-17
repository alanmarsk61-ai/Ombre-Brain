# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 记忆桶的增删改查、多维索引搜索、激活更新。
#
# Storage backends:
#   - File-based (Markdown + YAML frontmatter) — default
#   - PostgreSQL — activated when DATABASE_URL env var is set
# ============================================================

import os
import json
import math
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter
from rapidfuzz import fuzz

from utils import generate_bucket_id, sanitize_name, safe_path, now_iso

logger = logging.getLogger("ombre_brain.bucket")

# Optional PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    HAS_PG = False

# ---------------------------------------------------------
# PostgreSQL schema — created on first use
# ---------------------------------------------------------
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS buckets (
    id TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tags JSONB DEFAULT '[]'::jsonb,
    domain JSONB DEFAULT '["未分类"]'::jsonb,
    valence REAL DEFAULT 0.5,
    arousal REAL DEFAULT 0.3,
    importance INTEGER DEFAULT 5,
    bucket_type TEXT DEFAULT 'dynamic',
    resolved BOOLEAN DEFAULT FALSE,
    pinned BOOLEAN DEFAULT FALSE,
    protected BOOLEAN DEFAULT FALSE,
    digested BOOLEAN DEFAULT FALSE,
    model_valence REAL,
    created TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    activation_count REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_buckets_type ON buckets(bucket_type);
CREATE INDEX IF NOT EXISTS idx_buckets_pinned ON buckets(pinned);
"""


class BucketManager:
    """Memory bucket manager with optional PostgreSQL persistence."""

    def __init__(self, config: dict, embedding_engine=None):
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)
        self.embedding_engine = embedding_engine

        # --- PostgreSQL setup ---
        self._db_url = config.get("database_url", os.environ.get("DATABASE_URL", ""))
        self._use_db = bool(self._db_url) and HAS_PG
        self._pg_conn = None

        if self._use_db:
            self._pg_connect()
            self._ensure_db()
            logger.info("BucketManager: using PostgreSQL backend ✅")
        else:
            os.makedirs(self.permanent_dir, exist_ok=True)
            os.makedirs(self.dynamic_dir, exist_ok=True)
            os.makedirs(self.archive_dir, exist_ok=True)
            os.makedirs(self.feel_dir, exist_ok=True)
            if self._db_url and not HAS_PG:
                logger.warning("DATABASE_URL set but psycopg2 not installed, falling back to files")
            else:
                logger.info("BucketManager: using file backend")

    # ---------------------------------------------------------
    # PostgreSQL connection management
    # ---------------------------------------------------------
    def _pg_connect(self):
        try:
            self._pg_conn = psycopg2.connect(self._db_url)
            self._pg_conn.autocommit = True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._use_db = False
            self._pg_conn = None

    def _ensure_db(self):
        if not self._pg_conn:
            return
        try:
            cur = self._pg_conn.cursor()
            cur.execute(PG_SCHEMA)
            cur.close()
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL schema: {e}")
            self._use_db = False

    def _pg_dict(self, row):
        """Convert a DB row tuple to a bucket dict compatible with file format."""
        if not row:
            return None
        cols = [
            "id", "name", "content", "tags", "domain", "valence", "arousal",
            "importance", "bucket_type", "resolved", "pinned", "protected",
            "digested", "model_valence", "created", "last_active", "activation_count"
        ]
        d = dict(zip(cols, row))
        # Parse JSONB fields
        for field in ("tags", "domain"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        # Ensure defaults
        if not d.get("domain"):
            d["domain"] = ["未分类"]
        if not d.get("tags"):
            d["tags"] = []
        # Convert timestamps to ISO format strings
        for ts_field in ("created", "last_active"):
            val = d.get(ts_field)
            if isinstance(val, datetime):
                d[ts_field] = val.isoformat()
        return d

    def _row_to_bucket(self, row):
        """Convert DB row to the bucket dict format used by search/list_all."""
        meta = self._pg_dict(row)
        if not meta:
            return None
        content = meta.pop("content", "")
        bucket_id = meta.get("id", "")
        return {"id": bucket_id, "metadata": meta, "content": content, "path": None}

    # ---------------------------------------------------------
    # Create bucket
    # ---------------------------------------------------------
    async def create(self, content, tags=None, importance=5, domain=None,
                     valence=0.5, arousal=0.3, bucket_type="dynamic",
                     name=None, pinned=False, protected=False) -> str:
        bucket_id = generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id
        if bucket_type == "feel":
            domain = domain if domain is not None else []
        else:
            domain = domain or ["未分类"]
        tags = tags or []
        if pinned or protected:
            importance = 10

        if self._use_db:
            return await self._pg_create(
                bucket_id, bucket_name, content, tags, domain,
                valence, arousal, importance, bucket_type, pinned, protected
            )
        return await self._file_create(
            bucket_id, bucket_name, content, tags, domain,
            valence, arousal, importance, bucket_type, pinned, protected
        )

    async def _pg_create(self, bucket_id, bucket_name, content, tags, domain,
                         valence, arousal, importance, bucket_type, pinned, protected):
        try:
            cur = self._pg_conn.cursor()
            cur.execute("""
                INSERT INTO buckets (id, name, content, tags, domain, valence, arousal,
                    importance, bucket_type, pinned, protected, created, last_active)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                bucket_id, bucket_name, content,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(domain, ensure_ascii=False),
                max(0.0, min(1.0, valence)),
                max(0.0, min(1.0, arousal)),
                max(1, min(10, importance)),
                bucket_type, pinned, protected
            ))
            cur.close()
            logger.info(f"[DB] Created bucket: {bucket_id} ({bucket_name})")
            return bucket_id
        except Exception as e:
            logger.error(f"[DB] Create failed: {e}")
            raise

    async def _file_create(self, bucket_id, bucket_name, content, tags, domain,
                           valence, arousal, importance, bucket_type, pinned, protected):
        metadata = {
            "id": bucket_id, "name": bucket_name, "tags": tags, "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type, "created": now_iso(),
            "last_active": now_iso(), "activation_count": 0,
        }
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True

        post = frontmatter.Post(content, **metadata)

        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
            if pinned and bucket_type != "permanent":
                metadata["type"] = "permanent"
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        else:
            type_dir = self.dynamic_dir

        if bucket_type == "feel":
            primary_domain = "沉淀物"
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"

        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file: {file_path}: {e}")
            raise

        logger.info(f"Created bucket: {bucket_id} → {primary_domain}/")
        return bucket_id

    # ---------------------------------------------------------
    # Get bucket
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        if self._use_db:
            return await self._pg_get(bucket_id)
        return self._file_get(bucket_id)

    async def _pg_get(self, bucket_id):
        try:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM buckets WHERE id = %s", (bucket_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                return self._row_to_bucket(row)
        except Exception as e:
            logger.warning(f"[DB] Get failed: {e}")
        return None

    def _file_get(self, bucket_id):
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    # ---------------------------------------------------------
    # Update bucket
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, **kwargs) -> bool:
        if self._use_db:
            return await self._pg_update(bucket_id, **kwargs)
        return await self._file_update(bucket_id, **kwargs)

    async def _pg_update(self, bucket_id, **kwargs):
        try:
            build = []
            vals = []
            for key, val in kwargs.items():
                if key == "content":
                    build.append("content = %s"); vals.append(val)
                elif key == "tags":
                    build.append("tags = %s::jsonb"); vals.append(json.dumps(val, ensure_ascii=False))
                elif key == "importance":
                    build.append("importance = %s"); vals.append(max(1, min(10, int(val))))
                elif key == "domain":
                    build.append("domain = %s::jsonb"); vals.append(json.dumps(val, ensure_ascii=False))
                elif key == "valence":
                    build.append("valence = %s"); vals.append(max(0.0, min(1.0, float(val))))
                elif key == "arousal":
                    build.append("arousal = %s"); vals.append(max(0.0, min(1.0, float(val))))
                elif key == "name":
                    build.append("name = %s"); vals.append(sanitize_name(val))
                elif key == "resolved":
                    build.append("resolved = %s"); vals.append(bool(val))
                elif key == "pinned":
                    build.append("pinned = %s"); vals.append(bool(val))
                    if val:
                        build.append("importance = 10")
                elif key == "digested":
                    build.append("digested = %s"); vals.append(bool(val))
                elif key == "model_valence":
                    build.append("model_valence = %s"); vals.append(max(0.0, min(1.0, float(val))))

            build.append("last_active = NOW()")
            vals.append(bucket_id)  # for WHERE clause

            cur = self._pg_conn.cursor()
            cur.execute(f"UPDATE buckets SET {', '.join(build)} WHERE id = %s", vals)
            updated = cur.rowcount > 0
            cur.close()
            logger.info(f"[DB] Updated bucket: {bucket_id}")
            return updated
        except Exception as e:
            logger.warning(f"[DB] Update failed: {e}")
            return False

    async def _file_update(self, bucket_id, **kwargs):
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for update: {file_path}: {e}")
            return False

        is_pinned = post.get("pinned", False) or post.get("protected", False)
        if is_pinned:
            kwargs.pop("importance", None)

        if "content" in kwargs:
            post.content = kwargs["content"]
        if "tags" in kwargs:
            post["tags"] = kwargs["tags"]
        if "importance" in kwargs:
            post["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            post["domain"] = kwargs["domain"]
        if "valence" in kwargs:
            post["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            post["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            post["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            post["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            post["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                post["importance"] = 10
        if "digested" in kwargs:
            post["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            post["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))

        post["last_active"] = now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket update: {file_path}: {e}")
            return False

        domain = post.get("domain", ["未分类"])
        if kwargs.get("pinned") and post.get("type") != "permanent":
            post["type"] = "permanent"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            self._move_bucket(file_path, self.permanent_dir, domain)

        logger.info(f"Updated bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Delete bucket
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str) -> bool:
        if self._use_db:
            return await self._pg_delete(bucket_id)
        return await self._file_delete(bucket_id)

    async def _pg_delete(self, bucket_id):
        try:
            cur = self._pg_conn.cursor()
            cur.execute("DELETE FROM buckets WHERE id = %s", (bucket_id,))
            deleted = cur.rowcount > 0
            cur.close()
            logger.info(f"[DB] Deleted bucket: {bucket_id}")
            return deleted
        except Exception as e:
            logger.error(f"[DB] Delete failed: {e}")
            return False

    async def _file_delete(self, bucket_id):
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Failed to delete bucket file: {file_path}: {e}")
            return False
        logger.info(f"Deleted bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Touch bucket (refresh activation)
    # ---------------------------------------------------------
    async def touch(self, bucket_id: str) -> None:
        if self._use_db:
            await self._pg_touch(bucket_id)
        else:
            await self._file_touch(bucket_id)

    async def _pg_touch(self, bucket_id):
        try:
            cur = self._pg_conn.cursor()
            cur.execute("""
                UPDATE buckets SET last_active = NOW(),
                activation_count = activation_count + 1
                WHERE id = %s
            """, (bucket_id,))
            cur.close()
        except Exception as e:
            logger.warning(f"[DB] Touch failed: {e}")

    async def _file_touch(self, bucket_id):
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return
        try:
            post = frontmatter.load(file_path)
            post["last_active"] = now_iso()
            post["activation_count"] = post.get("activation_count", 0) + 1
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except Exception as e:
            logger.warning(f"Failed to touch bucket: {bucket_id}: {e}")

    async def _time_ripple(self, source_id, reference_time, hours=48.0):
        """File-only ripple; DB mode skips this for simplicity."""
        if self._use_db:
            return
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return
        rippled = 0
        max_ripple = 5
        for bucket in all_buckets:
            if rippled >= max_ripple:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue
            created_str = meta.get("created", meta.get("last_active", ""))
            try:
                created = datetime.fromisoformat(str(created_str))
                delta_hours = abs((reference_time - created).total_seconds()) / 3600
            except (ValueError, TypeError):
                continue
            if delta_hours <= hours:
                file_path = self._find_bucket_file(bucket["id"])
                if not file_path:
                    continue
                try:
                    post = frontmatter.load(file_path)
                    post["activation_count"] = round(post.get("activation_count", 1) + 0.3, 1)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter.dumps(post))
                    rippled += 1
                except Exception:
                    continue

    # ---------------------------------------------------------
    # Search (multi-dimensional)
    # ---------------------------------------------------------
    async def search(self, query, limit=None, domain_filter=None,
                     query_valence=None, query_arousal=None) -> list[dict]:
        if not query or not query.strip():
            return []
        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=False)
        if not all_buckets:
            return []

        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
            if not candidates:
                candidates = all_buckets
        else:
            candidates = all_buckets

        if self.embedding_engine and self.embedding_engine.enabled:
            try:
                vector_results = await self.embedding_engine.search_similar(query, top_k=50)
                if vector_results:
                    vector_ids = {bid for bid, _ in vector_results}
                    emb_candidates = [b for b in candidates if b["id"] in vector_ids]
                    if emb_candidates:
                        candidates = emb_candidates
            except Exception as e:
                logger.warning(f"Embedding pre-filter failed: {e}")

        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})
            try:
                topic_score = self._calc_topic_score(query, bucket)
                emotion_score = self._calc_emotion_score(query_valence, query_arousal, meta)
                time_score = self._calc_time_score(meta)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0
                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0
                if normalized >= self.fuzzy_threshold:
                    if meta.get("resolved", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(f"Scoring failed for bucket {bucket.get('id', '?')}: {e}")
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _calc_topic_score(self, query, bucket):
        meta = bucket.get("metadata", {})
        name_score = fuzz.partial_ratio(query, meta.get("name", "")) * 3
        domain_score = max(
            (fuzz.partial_ratio(query, d) for d in meta.get("domain", [])),
            default=0,
        ) * 2.5
        tag_score = max(
            (fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])),
            default=0,
        ) * 2
        content_score = fuzz.partial_ratio(query, bucket.get("content", "")[:1000]) * self.content_weight
        return (name_score + domain_score + tag_score + content_score) / (100 * (3 + 2.5 + 2 + self.content_weight))

    def _calc_emotion_score(self, q_valence, q_arousal, meta):
        if q_valence is None or q_arousal is None:
            return 0.5
        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    def _calc_time_score(self, meta):
        last_active_str = meta.get("last_active", meta.get("created", ""))
        try:
            # Handle both ISO string and datetime object from PostgreSQL
            val = last_active_str
            if hasattr(val, 'isoformat'):
                last_active = val
            elif isinstance(val, str) and val:
                last_active = datetime.fromisoformat(val.replace('Z', '+00:00'))
            else:
                last_active = datetime.now()
            days = max(0.0, (datetime.now(last_active.tzinfo if last_active.tzinfo else None) - last_active.replace(tzinfo=None)).total_seconds() / 86400)
        except (ValueError, TypeError, AttributeError):
            days = 30
        return math.exp(-0.02 * days)

    # ---------------------------------------------------------
    # List all buckets
    # ---------------------------------------------------------
    async def list_all(self, include_archive=False) -> list[dict]:
        if self._use_db:
            return await self._pg_list_all(include_archive)
        return self._file_list_all(include_archive)

    async def _pg_list_all(self, include_archive):
        try:
            cur = self._pg_conn.cursor()
            if include_archive:
                cur.execute("SELECT * FROM buckets ORDER BY last_active DESC")
            else:
                cur.execute(
                    "SELECT * FROM buckets WHERE bucket_type != 'archived' ORDER BY last_active DESC"
                )
            rows = cur.fetchall()
            cur.close()
            return [self._row_to_bucket(r) for r in rows if r]
        except Exception as e:
            logger.error(f"[DB] list_all failed: {e}")
            return []

    def _file_list_all(self, include_archive):
        buckets = []
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    bucket = self._load_bucket(file_path)
                    if bucket:
                        buckets.append(bucket)
        return buckets

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        if self._use_db:
            return await self._pg_get_stats()
        return self._file_get_stats()

    async def _pg_get_stats(self):
        try:
            cur = self._pg_conn.cursor()
            stats = {"permanent_count": 0, "dynamic_count": 0, "archive_count": 0,
                     "feel_count": 0, "total_size_kb": 0.0, "domains": {}}
            cur.execute("SELECT bucket_type, COUNT(*) FROM buckets GROUP BY bucket_type")
            for btype, cnt in cur.fetchall():
                key = f"{btype}_count"
                if key in stats:
                    stats[key] = cnt
            cur.close()
            return stats
        except Exception as e:
            logger.error(f"[DB] get_stats failed: {e}")
            return {"permanent_count": 0, "dynamic_count": 0, "archive_count": 0,
                    "feel_count": 0, "total_size_kb": 0.0, "domains": {}}

    def _file_get_stats(self):
        stats = {"permanent_count": 0, "dynamic_count": 0, "archive_count": 0,
                 "feel_count": 0, "total_size_kb": 0.0, "domains": {}}
        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1
        return stats

    # ---------------------------------------------------------
    # Archive bucket
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str) -> bool:
        if self._use_db:
            return await self._pg_archive(bucket_id)
        return await self._file_archive(bucket_id)

    async def _pg_archive(self, bucket_id):
        try:
            cur = self._pg_conn.cursor()
            cur.execute("UPDATE buckets SET bucket_type = 'archived' WHERE id = %s", (bucket_id,))
            updated = cur.rowcount > 0
            cur.close()
            return updated
        except Exception as e:
            logger.error(f"[DB] Archive failed: {e}")
            return False

    async def _file_archive(self, bucket_id):
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            post = frontmatter.load(file_path)
            domain = post.get("domain", ["未分类"])
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)
            dest = safe_path(archive_subdir, os.path.basename(file_path))
            post["type"] = "archived"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(f"Failed to archive bucket: {bucket_id}: {e}")
            return False
        logger.info(f"Archived bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Move bucket (file-only)
    # ---------------------------------------------------------
    def _move_bucket(self, file_path, target_type_dir, domain=None):
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.rename(file_path, new_path)
            logger.info(f"Moved bucket: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Find bucket file (file-only)
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id):
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    name_part = fname[:-3]
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    # ---------------------------------------------------------
    # Load bucket from file (file-only)
    # ---------------------------------------------------------
    def _load_bucket(self, file_path):
        try:
            post = frontmatter.load(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
            }
        except Exception as e:
            logger.warning(f"Failed to load bucket file: {file_path}: {e}")
            return None
