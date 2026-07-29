"""Persistent storage for Article Mode.

Storage sits behind the :class:`ArticleRepository` interface so it can later be
swapped (Postgres, a hosted service, …) without touching the ingestion worker,
API or pipeline. The default implementation is a dependency-free SQLite database
using only the Python standard library, with WAL mode, transactions and indexes.

Each row keeps a small set of *indexed* columns for querying plus a JSON ``data``
blob holding the full Pydantic model. Reads reconstruct the typed model, so the
rest of the codebase always works with models, never loose dicts.

Deterministic integrity checks here (schema, foreign-key-ish links, dedupe keys)
protect the application; they are not editorial gates.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import date
from typing import Iterable, List, Optional

from loguru import logger

from app.config import config
from app.models.article import (
    ArticleCluster,
    ArticleRecord,
    ArticleStatus,
    PollRun,
    StoryAssessment,
    TopicSubscription,
    _iso,
    utcnow,
)
from app.utils import utils


def default_database_path() -> str:
    """Resolve the SQLite path from config, defaulting under ``storage/article``."""
    configured = str(config.app.get("article_database_path", "") or "").strip()
    if configured:
        return configured
    return os.path.join(utils.storage_dir("article", create=True), "articles.db")


class ArticleRepository(ABC):
    """Abstract persistence interface for Article Mode."""

    # subscriptions -------------------------------------------------------
    @abstractmethod
    def upsert_subscription(self, subscription: TopicSubscription) -> TopicSubscription: ...

    @abstractmethod
    def get_subscription(self, subscription_id: str) -> Optional[TopicSubscription]: ...

    @abstractmethod
    def list_subscriptions(self, enabled_only: bool = False) -> List[TopicSubscription]: ...

    @abstractmethod
    def delete_subscription(self, subscription_id: str) -> bool: ...

    # poll runs -----------------------------------------------------------
    @abstractmethod
    def save_poll_run(self, poll_run: PollRun) -> None: ...

    @abstractmethod
    def list_poll_runs(
        self, subscription_id: Optional[str] = None, limit: int = 20
    ) -> List[PollRun]: ...

    # articles ------------------------------------------------------------
    @abstractmethod
    def save_article(self, article: ArticleRecord) -> None: ...

    @abstractmethod
    def get_article(self, article_id: str) -> Optional[ArticleRecord]: ...

    @abstractmethod
    def list_articles(
        self,
        subscription_id: Optional[str] = None,
        status: Optional[str] = None,
        domain: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ArticleRecord]: ...

    @abstractmethod
    def find_article_by_hash(self, text_hash: str) -> Optional[ArticleRecord]: ...

    @abstractmethod
    def find_article_by_canonical(self, canonical_url: str) -> Optional[ArticleRecord]: ...

    # clusters ------------------------------------------------------------
    @abstractmethod
    def save_cluster(self, cluster: ArticleCluster) -> None: ...

    @abstractmethod
    def get_cluster(self, cluster_id: str) -> Optional[ArticleCluster]: ...

    @abstractmethod
    def list_clusters(self, subscription_id: Optional[str] = None) -> List[ArticleCluster]: ...

    # assessments ---------------------------------------------------------
    @abstractmethod
    def save_assessment(self, cluster_id: str, assessment: StoryAssessment) -> None: ...

    @abstractmethod
    def get_assessment(self, cluster_id: str) -> Optional[StoryAssessment]: ...

    # generation bookkeeping ---------------------------------------------
    @abstractmethod
    def record_generation(
        self, cluster_id: str, task_id: str, published: bool = False
    ) -> None: ...

    @abstractmethod
    def has_generated_cluster(self, cluster_id: str) -> bool: ...

    @abstractmethod
    def count_generations(self, on_day: Optional[date] = None) -> int: ...

    @abstractmethod
    def count_publications(self, on_day: Optional[date] = None) -> int: ...


class SqliteArticleRepository(ArticleRepository):
    """Standard-library SQLite implementation with WAL mode and indexes."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or default_database_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
        # A process-wide lock serializes writers. SQLite handles cross-process
        # locking via WAL, but the in-process lock avoids "database is locked"
        # churn when the worker and API run in the same process.
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.database_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    enabled INTEGER,
                    last_polled_at TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS poll_runs (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT,
                    started_at TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS clusters (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT,
                    normalized_title TEXT,
                    created_at TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT,
                    cluster_id TEXT,
                    domain TEXT,
                    canonical_url TEXT,
                    text_hash TEXT,
                    status TEXT,
                    published_at TEXT,
                    retrieved_at TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assessments (
                    cluster_id TEXT PRIMARY KEY,
                    story_score REAL,
                    confidence REAL,
                    created_at TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id TEXT,
                    task_id TEXT,
                    published INTEGER DEFAULT 0,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_articles_subscription
                    ON articles(subscription_id);
                CREATE INDEX IF NOT EXISTS idx_articles_cluster
                    ON articles(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);
                CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
                CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(text_hash);
                CREATE INDEX IF NOT EXISTS idx_articles_canonical
                    ON articles(canonical_url);
                CREATE INDEX IF NOT EXISTS idx_clusters_subscription
                    ON clusters(subscription_id);
                CREATE INDEX IF NOT EXISTS idx_poll_runs_subscription
                    ON poll_runs(subscription_id);
                CREATE INDEX IF NOT EXISTS idx_generations_cluster
                    ON generations(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_generations_created
                    ON generations(created_at);
                CREATE INDEX IF NOT EXISTS idx_generations_cluster_task
                    ON generations(cluster_id, task_id);
                """
            )

    # -- subscriptions ----------------------------------------------------
    def upsert_subscription(self, subscription: TopicSubscription) -> TopicSubscription:
        subscription.updated_at = utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO subscriptions (id, name, enabled, last_polled_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    enabled=excluded.enabled,
                    last_polled_at=excluded.last_polled_at,
                    data=excluded.data
                """,
                (
                    subscription.id,
                    subscription.name,
                    1 if subscription.enabled else 0,
                    _iso(subscription.last_polled_at),
                    subscription.model_dump_json(),
                ),
            )
        return subscription

    def get_subscription(self, subscription_id: str) -> Optional[TopicSubscription]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM subscriptions WHERE id=?", (subscription_id,)
            ).fetchone()
        return TopicSubscription.model_validate_json(row["data"]) if row else None

    def list_subscriptions(self, enabled_only: bool = False) -> List[TopicSubscription]:
        query = "SELECT data FROM subscriptions"
        params: tuple = ()
        if enabled_only:
            query += " WHERE enabled=1"
        query += " ORDER BY name COLLATE NOCASE"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [TopicSubscription.model_validate_json(r["data"]) for r in rows]

    def delete_subscription(self, subscription_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM subscriptions WHERE id=?", (subscription_id,)
            )
            return cursor.rowcount > 0

    # -- poll runs --------------------------------------------------------
    def save_poll_run(self, poll_run: PollRun) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO poll_runs (id, subscription_id, started_at, data)"
                " VALUES (?, ?, ?, ?)",
                (
                    poll_run.id,
                    poll_run.subscription_id,
                    _iso(poll_run.started_at),
                    poll_run.model_dump_json(),
                ),
            )

    def list_poll_runs(
        self, subscription_id: Optional[str] = None, limit: int = 20
    ) -> List[PollRun]:
        query = "SELECT data FROM poll_runs"
        params: List[object] = []
        if subscription_id:
            query += " WHERE subscription_id=?"
            params.append(subscription_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [PollRun.model_validate_json(r["data"]) for r in rows]

    # -- articles ---------------------------------------------------------
    def save_article(self, article: ArticleRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO articles
                    (id, subscription_id, cluster_id, domain, canonical_url,
                     text_hash, status, published_at, retrieved_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.id,
                    article.subscription_id,
                    article.cluster_id,
                    article.domain,
                    article.canonical_url,
                    article.text_hash,
                    article.status.value,
                    _iso(article.published_at),
                    _iso(article.retrieved_at),
                    article.model_dump_json(),
                ),
            )

    def get_article(self, article_id: str) -> Optional[ArticleRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM articles WHERE id=?", (article_id,)
            ).fetchone()
        return ArticleRecord.model_validate_json(row["data"]) if row else None

    def list_articles(
        self,
        subscription_id: Optional[str] = None,
        status: Optional[str] = None,
        domain: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ArticleRecord]:
        clauses: List[str] = []
        params: List[object] = []
        if subscription_id:
            clauses.append("subscription_id=?")
            params.append(subscription_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        if domain:
            clauses.append("domain=?")
            params.append(domain)
        if cluster_id:
            clauses.append("cluster_id=?")
            params.append(cluster_id)
        query = "SELECT data FROM articles"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(published_at, retrieved_at) DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [ArticleRecord.model_validate_json(r["data"]) for r in rows]

    def find_article_by_hash(self, text_hash: str) -> Optional[ArticleRecord]:
        if not text_hash:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM articles WHERE text_hash=? LIMIT 1", (text_hash,)
            ).fetchone()
        return ArticleRecord.model_validate_json(row["data"]) if row else None

    def find_article_by_canonical(self, canonical_url: str) -> Optional[ArticleRecord]:
        if not canonical_url:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM articles WHERE canonical_url=? LIMIT 1",
                (canonical_url,),
            ).fetchone()
        return ArticleRecord.model_validate_json(row["data"]) if row else None

    # -- clusters ---------------------------------------------------------
    def save_cluster(self, cluster: ArticleCluster) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO clusters"
                " (id, subscription_id, normalized_title, created_at, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    cluster.id,
                    cluster.subscription_id,
                    cluster.normalized_title,
                    _iso(cluster.created_at),
                    cluster.model_dump_json(),
                ),
            )

    def get_cluster(self, cluster_id: str) -> Optional[ArticleCluster]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM clusters WHERE id=?", (cluster_id,)
            ).fetchone()
        return ArticleCluster.model_validate_json(row["data"]) if row else None

    def list_clusters(self, subscription_id: Optional[str] = None) -> List[ArticleCluster]:
        query = "SELECT data FROM clusters"
        params: tuple = ()
        if subscription_id:
            query += " WHERE subscription_id=?"
            params = (subscription_id,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [ArticleCluster.model_validate_json(r["data"]) for r in rows]

    # -- assessments ------------------------------------------------------
    def save_assessment(self, cluster_id: str, assessment: StoryAssessment) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO assessments"
                " (cluster_id, story_score, confidence, created_at, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    cluster_id,
                    float(assessment.story_score),
                    float(assessment.confidence),
                    _iso(utcnow()),
                    assessment.model_dump_json(),
                ),
            )

    def get_assessment(self, cluster_id: str) -> Optional[StoryAssessment]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM assessments WHERE cluster_id=?", (cluster_id,)
            ).fetchone()
        return StoryAssessment.model_validate_json(row["data"]) if row else None

    # -- generation bookkeeping ------------------------------------------
    def record_generation(
        self, cluster_id: str, task_id: str, published: bool = False
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO generations (cluster_id, task_id, published, created_at)
                SELECT ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM generations WHERE cluster_id=? AND task_id=?
                )
                """,
                (
                    cluster_id,
                    task_id,
                    1 if published else 0,
                    _iso(utcnow()),
                    cluster_id,
                    task_id,
                ),
            )

    def has_generated_cluster(self, cluster_id: str) -> bool:
        if not cluster_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM generations WHERE cluster_id=? LIMIT 1", (cluster_id,)
            ).fetchone()
        return row is not None

    def _count(self, column_filter: str, on_day: Optional[date]) -> int:
        day = (on_day or utcnow().date()).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM generations"
                f" WHERE {column_filter} AND substr(created_at, 1, 10)=?",
                (day,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def count_generations(self, on_day: Optional[date] = None) -> int:
        return self._count("1=1", on_day)

    def count_publications(self, on_day: Optional[date] = None) -> int:
        return self._count("published=1", on_day)


_default_repository: Optional[ArticleRepository] = None
_default_lock = threading.Lock()


def get_repository() -> ArticleRepository:
    """Return the process-wide default repository (lazy singleton)."""
    global _default_repository
    if _default_repository is None:
        with _default_lock:
            if _default_repository is None:
                _default_repository = SqliteArticleRepository()
                logger.info(
                    "article repository ready: "
                    f"{_default_repository.database_path}"  # type: ignore[attr-defined]
                )
    return _default_repository


def set_repository(repository: ArticleRepository) -> None:
    """Override the default repository (used by tests and alternate backends)."""
    global _default_repository
    with _default_lock:
        _default_repository = repository


def iter_due_subscriptions(
    repository: ArticleRepository,
) -> Iterable[TopicSubscription]:
    """Yield enabled subscriptions that are due to be polled."""
    now = utcnow()
    for subscription in repository.list_subscriptions(enabled_only=True):
        if subscription.is_due(now):
            yield subscription


def mark_articles_status(
    repository: ArticleRepository, article_ids: Iterable[str], status: ArticleStatus
) -> None:
    for article_id in article_ids:
        article = repository.get_article(article_id)
        if article:
            article.status = status
            repository.save_article(article)
