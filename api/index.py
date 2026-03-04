import os
import json
import asyncio
import logging
import time
import re
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_MODEL = os.environ.get(
    "NVIDIA_CHAT_MODEL", "meta/llama-3.1-8b-instruct"
)
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
SUPABASE_PG_URL = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_PG_URL", "")

# Keep neo4j+s:// protocol — Aura requires routing protocol
NEO4J_URI_BOLT = NEO4J_URI

# Sync env vars for drivers that read them directly
os.environ["NEO4J_URL"] = NEO4J_URI_BOLT
os.environ["NEO4J_URI"] = NEO4J_URI_BOLT
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

_aura_match = re.match(r"neo4j\+s(?:sc)?://([^.]+)\.databases\.neo4j\.io", NEO4J_URI)
os.environ["NEO4J_DATABASE"] = _aura_match.group(1) if _aura_match else "neo4j"

# ── Parse PG URL → env vars for LightRAG's ClientManager ───────────
if SUPABASE_PG_URL:
    try:
        parsed = urlparse(SUPABASE_PG_URL)
        os.environ["POSTGRES_USER"] = unquote(parsed.username or "")
        os.environ["POSTGRES_PASSWORD"] = unquote(parsed.password or "")
        os.environ["POSTGRES_HOST"] = parsed.hostname or ""
        os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
        os.environ["POSTGRES_DATABASE"] = unquote(parsed.path.lstrip("/"))
    except Exception:
        pass

WORKING_DIR = "/tmp/lightrag_cache"
os.makedirs(WORKING_DIR, exist_ok=True)

# ── MONKEY-PATCH: asyncpg + PgBouncer (Supavisor) ───────────────────
import asyncpg
from lightrag.kg.postgres_impl import PostgreSQLDB

_original_initdb = PostgreSQLDB.initdb


async def _patched_initdb(self):
    try:
        self.pool = await asyncpg.create_pool(
            user=self.user,
            password=self.password,
            database=self.database,
            host=self.host,
            port=self.port,
            min_size=1,
            max_size=self.max,
            statement_cache_size=0,
            server_settings={"jit": "off"},
        )
        print(
            f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database}"
        )
    except Exception as e:
        print(
            f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}"
        )
        raise


PostgreSQLDB.initdb = _patched_initdb

# ── LLM / Embedding wrappers ────────────────────────────────────────
logging.getLogger("lightrag").setLevel(logging.WARNING)
logging.getLogger("lightrag.kg.shared_storage").setLevel(logging.ERROR)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg import shared_storage as shared_storage_module

_original_direct_log = shared_storage_module.direct_log


def _quiet_direct_log(message, level="INFO", enable_output=True):
    msg = str(message)
    if level == "INFO" and ("Shared-Data" in msg or "Pipeline namespace initialized" in msg):
        return
    _original_direct_log(message, level=level, enable_output=enable_output)


shared_storage_module.direct_log = _quiet_direct_log


async def nvidia_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    llm_max_tokens = kwargs.pop("max_tokens", 900)
    llm_temperature = kwargs.pop("temperature", 0.25)
    llm_top_p = kwargs.pop("top_p", 0.9)
    kwargs.pop("history_messages", None)
    return await openai_complete_if_cache(
        NVIDIA_CHAT_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        temperature=llm_temperature,
        top_p=llm_top_p,
        max_tokens=llm_max_tokens,
        openai_client_configs={"timeout": 35.0},
        **kwargs,
    )


async def nvidia_embed(texts):
    # NVIDIA's nv-embedqa-e5-v5 is asymmetric and expects input_type.
    from openai import AsyncOpenAI
    import numpy as np

    client = AsyncOpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    response = await client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage"},
    )
    return np.array([dp.embedding for dp in response.data])


ASSISTANT_SYSTEM_PROMPT = (
    "You are Satvik Mudgal replying in first person. "
    "Sound like a real engineer talking naturally: direct, warm, and specific. "
    "Use contractions and short-to-medium sentences, and reference concrete projects or tools when possible. "
    "Avoid robotic framing like 'based on the knowledge base' or generic disclaimers. "
    "Never invent project details, timelines, or technologies. "
    "If context is missing, explicitly say you do not have that detail in your profile yet."
)


def clean_answer(text):
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?is)\n*#+\s*references.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*[-*]\s*\[(?:KG|SOURCE)\].*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:source|references?)\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(
        r"(?i)^based on (the )?(knowledge base|provided context)[:,]?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _tokenize(text):
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "your", "have", "about",
        "what", "when", "where", "which", "into", "during", "work", "worked", "tell",
        "me", "you", "are", "was", "how", "did", "use", "used", "on", "in", "of",
    }
    raw_tokens = re.findall(r"[a-z0-9\+\#-]{3,}", text.lower())
    return [tok for tok in raw_tokens if tok not in stop]


def _rank_profile_docs(question, schema, limit=3):
    docs = (schema or {}).get("documents", [])
    q_tokens = _tokenize(question)
    ranked = []
    for doc in docs:
        title = re.sub(r"[^a-z0-9]+", " ", (doc.get("title") or "").lower())
        tags = [str(t).lower() for t in (doc.get("tags") or [])]
        content = re.sub(r"[^a-z0-9]+", " ", (doc.get("content") or "").lower())
        score = 0
        for tok in q_tokens:
            if tok in title:
                score += 5
            if any(tok in tag for tag in tags):
                score += 3
            if tok in content:
                score += 1
        if score > 0:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


def _to_first_person(text):
    if not text:
        return ""
    out = text
    out = re.sub(r"\bSatvik\b", "I", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhe has\b", "I have", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhe is\b", "I am", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhis\b", "my", out, flags=re.IGNORECASE)
    return out.strip()


def _normalize_title(title):
    out = (title or "").strip()
    out = re.sub(r"^project\s*:\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^section\s*\d+[:\-]?\s*", "", out, flags=re.IGNORECASE)
    return out.strip()


def _best_doc_label(doc):
    title = _normalize_title(doc.get("title", "this project"))
    content = doc.get("content", "")
    if title.lower() in {"projects expanded", "projects - expanded", "projects — expanded"}:
        m = re.search(r"project:\s*([a-z0-9][a-z0-9 \-_/&]+)", content, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            first_token = raw.split()[0] if raw.split() else raw
            key = re.sub(r"[^a-z0-9]+", "", first_token.lower())
            if key == "healthjini":
                return "HealthJini"
            if key == "vaani":
                return "Vaani"
            if key == "pm":
                return "PM GatiShakti"
            return first_token.title()
    return title


def _answer_from_profile_fastpath(question, schema):
    ranked = _rank_profile_docs(question, schema, limit=3)
    if not ranked or ranked[0][0] < 3:
        return None

    parts = []
    primary = ranked[0][1]
    primary_title = _best_doc_label(primary)
    primary_summary = _to_first_person(primary.get("metadata", {}).get("summary", ""))
    if primary_summary:
        parts.append(f"I've worked on {primary_title}. {primary_summary}")
    else:
        parts.append(f"I've worked on {primary_title}.")

    for _, doc in ranked[1:]:
        title = _best_doc_label(doc)
        summary = _to_first_person(doc.get("metadata", {}).get("summary", ""))
        if summary:
            parts.append(f"I've also done {title}. {summary}")

    answer = " ".join(parts).strip()
    return clean_answer(answer) if answer else None


def _schema_graph_overlay(schema, max_docs=18):
    nodes = {}
    links = []
    seen = set()

    root_id = "satvik_profile_root"
    nodes[root_id] = {
        "id": root_id,
        "name": "Satvik Mudgal",
        "type": "Person",
        "desc": "Structured profile root node",
    }

    docs = (schema or {}).get("documents", [])[:max_docs]
    for doc in docs:
        doc_id = f"profile_{doc.get('id')}"
        doc_name = doc.get("title") or "Profile Section"
        doc_type = (doc.get("category") or "general").title()
        doc_desc = doc.get("metadata", {}).get("summary", "")
        nodes[doc_id] = {"id": doc_id, "name": doc_name, "type": doc_type, "desc": doc_desc}
        edge_key = (root_id, doc_id, "profile")
        if edge_key not in seen:
            seen.add(edge_key)
            links.append({"source": root_id, "target": doc_id, "label": "profile", "weight": 1.0})

        for tag in (doc.get("tags") or [])[:4]:
            tag_id = f"tag_{re.sub(r'[^a-z0-9]+', '_', tag.lower()).strip('_')}"
            if not tag_id:
                continue
            if tag_id not in nodes:
                nodes[tag_id] = {
                    "id": tag_id,
                    "name": tag,
                    "type": "Technology",
                    "desc": "",
                }
            tkey = (doc_id, tag_id, "uses")
            if tkey not in seen:
                seen.add(tkey)
                links.append({"source": doc_id, "target": tag_id, "label": "uses", "weight": 0.8})

    return list(nodes.values()), links


# ── Structured profile dataset ──────────────────────────────────────
PROFILE_SCHEMA_VERSION = 1
PROFILE_CACHE_JSON_PATH = os.path.join(WORKING_DIR, "satvik_profile.schema.json")
PROFILE_CACHE_RAG_PATH = os.path.join(WORKING_DIR, "satvik_profile.rag.txt")
PROFILE_META_TABLE = "LIGHTRAG_APP_META"

profile_cache = {"source_hash": "", "schema": None, "rag_text": ""}
profile_cache_lock = asyncio.Lock()
meta_store_supported = True


def _satvik_data_path():
    return os.path.join(os.path.dirname(__file__), "satvik_data.txt")


def _is_section_heading(line):
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    if re.match(r"^section\s+\d+[:\- ]", s, re.IGNORECASE):
        return True
    if len(s) <= 90 and s == s.upper():
        alpha_chars = [c for c in s if c.isalpha()]
        if len(alpha_chars) >= 6 and len(s.split()) <= 12:
            return True
    return False


def _categorize_section(title, content):
    hay = f"{title} {content}".lower()
    rules = [
        ("identity", ("identity", "name", "contact", "portfolio", "linkedin", "github", "email", "location")),
        ("education", ("education", "university", "college", "b.tech", "vit", "ieee")),
        ("experience", ("experience", "engineer", "intern", "nuvo", "meril", "bhashini", "indiaai", "iiit", "nic")),
        ("projects", ("project", "pipeline", "healthjini", "gatishakti", "vaani", "translation", "system")),
        ("skills", ("skills", "technical stack", "languages", "frameworks", "devops", "ml", "ai")),
        ("personality", ("personality", "belief", "communication style", "hobby", "behavior")),
    ]
    for category, terms in rules:
        if any(t in hay for t in terms):
            return category
    return "general"


def _extract_tags(title, content, limit=10):
    corpus = f"{title} {content}"
    tags = []
    seen = set()
    known = [
        "Python",
        "FastAPI",
        "NLP",
        "ASR",
        "NMT",
        "TTS",
        "Neo4j",
        "PostgreSQL",
        "SQLAlchemy",
        "Docker",
        "Vercel",
        "LangChain",
        "LightRAG",
        "Transformers",
        "Embeddings",
        "Speech",
        "Healthcare",
        "PM GatiShakti",
        "HealthJini",
    ]
    for item in known:
        if item.lower() in corpus.lower():
            key = item.lower()
            if key not in seen:
                seen.add(key)
                tags.append(item)
            if len(tags) >= limit:
                return tags
    for token in re.findall(r"\b[A-Z][A-Za-z0-9\+\#\.-]{2,}\b", corpus):
        key = token.lower()
        if key not in seen:
            seen.add(key)
            tags.append(token)
        if len(tags) >= limit:
            break
    return tags


def _split_profile_sections(raw_text):
    lines = raw_text.replace("\r\n", "\n").split("\n")
    sections = []
    title = "Overview"
    current = []

    def flush():
        body = "\n".join(current).strip()
        if body and len(body) > 20:
            normalized = re.sub(r"\n{3,}", "\n\n", body).strip()
            sections.append((title, normalized))

    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"[=\-]{5,}", stripped):
            continue
        if _is_section_heading(stripped):
            candidate = stripped.lstrip("#").strip(" :-")
            lowered = candidate.lower()
            if lowered in {
                "satvik mudgal — digital twin knowledge base",
                "satvik mudgal - digital twin knowledge base",
                "satvik mudgal — personal knowledge base",
                "satvik mudgal - personal knowledge base",
            }:
                continue
            flush()
            title = candidate or "Section"
            current = []
            continue
        if not stripped:
            if current and current[-1] != "":
                current.append("")
            continue
        cleaned = (
            stripped.replace("→", "-")
            .replace("•", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        current.append(cleaned)
    flush()

    if not sections and raw_text.strip():
        sections = [("Overview", raw_text.strip())]
    return sections


def _build_profile_schema(raw_text, source_path):
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    sections = _split_profile_sections(raw_text)
    docs = []
    for idx, (title, content) in enumerate(sections, start=1):
        raw_lines = [ln.strip(" -\t") for ln in content.splitlines() if ln.strip()]
        candidate_lines = []
        for ln in raw_lines:
            if len(ln) < 3:
                continue
            if ln.isupper() and len(ln.split()) <= 3:
                continue
            candidate_lines.append(ln)
            if len(candidate_lines) >= 2:
                break
        one_line = re.sub(r"\s+", " ", " ".join(candidate_lines) or content).strip()
        summary = one_line[:200]
        if len(one_line) > 200:
            summary = summary.rsplit(" ", 1)[0].rstrip() + "..."
        summary = re.sub(r"(?i)^project:\s*[a-z0-9_-]+\s*", "", summary).strip()
        category = _categorize_section(title, content)
        tags = _extract_tags(title, content, limit=10)
        doc_hash = hashlib.md5(f"{title}|{content[:160]}".encode("utf-8")).hexdigest()[:10]
        docs.append(
            {
                "id": f"satvik_{idx:03d}_{doc_hash}",
                "title": title,
                "category": category,
                "tags": tags,
                "content": content,
                "metadata": {
                    "priority": 2 if category in {"identity", "experience", "projects"} else 1,
                    "summary": summary,
                },
            }
        )
    schema = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": source_path, "sha256": source_hash},
        "stats": {
            "section_count": len(docs),
            "character_count": len(raw_text),
        },
        "documents": docs,
    }
    return schema


def _compact_for_index(content, max_chars=700):
    one_line = re.sub(r"\s+", " ", content).strip()
    if len(one_line) <= max_chars:
        return one_line
    sentences = re.split(r"(?<=[.!?])\s+", one_line)
    if sentences:
        compact = " ".join(sentences[:3]).strip()
        if compact:
            return compact[:max_chars].rstrip()
    return one_line[:max_chars].rstrip()


def _render_schema_for_rag(schema):
    blocks = []
    for doc in schema.get("documents", []):
        tags = ", ".join(doc.get("tags") or []) or "general"
        content = _compact_for_index(doc.get("content", ""), max_chars=700)
        block = (
            f"<profile_document id=\"{doc['id']}\" category=\"{doc['category']}\">\n"
            f"<title>{doc['title']}</title>\n"
            f"<tags>{tags}</tags>\n"
            f"<summary>{doc.get('metadata', {}).get('summary', '')}</summary>\n"
            "<content>\n"
            f"{content}\n"
            "</content>\n"
            "</profile_document>"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _render_fast_delta(schema):
    rows = []
    for doc in schema.get("documents", []):
        tags = ", ".join((doc.get("tags") or [])[:4]) or "general"
        summary = doc.get("metadata", {}).get("summary", "")
        rows.append(
            f"[{doc.get('category', 'general')}] {doc.get('title', 'Section')} | tags: {tags} | {summary}"
        )
    joined = "\n".join(rows)
    return joined[:4000]


def _load_profile_payload():
    source_path = _satvik_data_path()
    if not os.path.exists(source_path):
        return {"schema": None, "rag_text": "", "source_hash": "", "section_count": 0}

    with open(source_path, encoding="utf-8") as f:
        raw_text = f.read().strip()
    if not raw_text:
        return {"schema": None, "rag_text": "", "source_hash": "", "section_count": 0}

    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if profile_cache["source_hash"] == source_hash and profile_cache["schema"]:
        schema = profile_cache["schema"]
        rag_text = profile_cache["rag_text"]
    else:
        schema = _build_profile_schema(raw_text, source_path)
        rag_text = _render_schema_for_rag(schema)
        profile_cache["source_hash"] = source_hash
        profile_cache["schema"] = schema
        profile_cache["rag_text"] = rag_text
        try:
            with open(PROFILE_CACHE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, ensure_ascii=True)
            with open(PROFILE_CACHE_RAG_PATH, "w", encoding="utf-8") as f:
                f.write(rag_text)
        except Exception:
            # Cache files are optional.
            pass

    return {
        "schema": schema,
        "rag_text": rag_text,
        "delta_text": _render_fast_delta(schema) if schema else "",
        "source_hash": source_hash,
        "section_count": schema.get("stats", {}).get("section_count", 0) if schema else 0,
    }


# ── Runtime state ───────────────────────────────────────────────────
rag = None
rag_init_lock = asyncio.Lock()

warmup_task = None
warmup_task_lock = asyncio.Lock()
warmup_state = {
    "status": "idle",  # idle | warming | ready | error
    "message": "Waiting to start",
    "progress": 0,
    "updated_at": int(time.time()),
    "source_hash": "",
    "section_count": 0,
}
shared_storage_initialized = False


def _warmup_snapshot():
    return {
        "status": warmup_state["status"],
        "message": warmup_state["message"],
        "progress": warmup_state["progress"],
        "updated_at": warmup_state["updated_at"],
        "source_hash": warmup_state.get("source_hash", ""),
        "section_count": warmup_state.get("section_count", 0),
    }


def set_warmup_state(status, message, progress, source_hash="", section_count=0):
    warmup_state["status"] = status
    warmup_state["message"] = message
    warmup_state["progress"] = max(0, min(100, int(progress)))
    warmup_state["updated_at"] = int(time.time())
    if source_hash:
        warmup_state["source_hash"] = source_hash
    if section_count:
        warmup_state["section_count"] = section_count


async def _workspace_has_index(r):
    try:
        db = r.doc_status.db
        if not db or not db.pool:
            return False
        async with db.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1",
                db.workspace,
            )
            return int(count or 0) > 0
    except Exception:
        return False


async def _get_indexed_profile_hash(r):
    global meta_store_supported
    if not meta_store_supported:
        return None
    try:
        db = r.doc_status.db
        if not db or not db.pool:
            return None
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PROFILE_META_TABLE} (
                    workspace TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL,
                    section_count INTEGER NOT NULL DEFAULT 0,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            row = await conn.fetchrow(
                f"SELECT source_hash FROM {PROFILE_META_TABLE} WHERE workspace=$1",
                db.workspace,
            )
            return row["source_hash"] if row else None
    except Exception:
        meta_store_supported = False
        return None


async def _set_indexed_profile_hash(r, source_hash, section_count):
    global meta_store_supported
    if not meta_store_supported:
        return
    try:
        db = r.doc_status.db
        if not db or not db.pool:
            return
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PROFILE_META_TABLE} (
                    workspace TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL,
                    section_count INTEGER NOT NULL DEFAULT 0,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                f"""
                INSERT INTO {PROFILE_META_TABLE}
                    (workspace, source_hash, section_count, schema_version, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (workspace)
                DO UPDATE SET
                    source_hash = EXCLUDED.source_hash,
                    section_count = EXCLUDED.section_count,
                    schema_version = EXCLUDED.schema_version,
                    updated_at = NOW()
                """,
                db.workspace,
                source_hash,
                int(section_count or 0),
                PROFILE_SCHEMA_VERSION,
            )
    except Exception:
        meta_store_supported = False


async def _run_warmup(force_reindex=False):
    try:
        set_warmup_state("warming", "Preparing profile schema", 12)
        async with profile_cache_lock:
            payload = _load_profile_payload()
        rag_text = payload["rag_text"]
        delta_text = payload["delta_text"]
        source_hash = payload["source_hash"]
        section_count = payload["section_count"]

        if not rag_text:
            set_warmup_state("error", "satvik_data.txt is missing or empty", 100)
            return

        set_warmup_state(
            "warming",
            "Connecting retrieval stores",
            28,
            source_hash=source_hash,
            section_count=section_count,
        )
        r = await get_rag()

        set_warmup_state(
            "warming",
            "Checking index freshness",
            52,
            source_hash=source_hash,
            section_count=section_count,
        )
        has_index = await _workspace_has_index(r)
        indexed_hash = await _get_indexed_profile_hash(r)

        if not has_index:
            set_warmup_state(
                "warming",
                "Building knowledge graph index",
                74,
                source_hash=source_hash,
                section_count=section_count,
            )
            await asyncio.wait_for(r.ainsert(rag_text), timeout=55.0)
            await _set_indexed_profile_hash(r, source_hash, section_count)
            set_warmup_state(
                "ready",
                f"RAG pipeline ready ({section_count} structured sections)",
                100,
                source_hash=source_hash,
                section_count=section_count,
            )
            return

        # Fast startup path for warm instances with an existing index.
        if force_reindex:
            set_warmup_state(
                "warming",
                "Applying forced profile refresh",
                78,
                source_hash=source_hash,
                section_count=section_count,
            )
            await asyncio.wait_for(r.ainsert(delta_text), timeout=30.0)
            await _set_indexed_profile_hash(r, source_hash, section_count)
            set_warmup_state(
                "ready",
                f"RAG pipeline ready ({section_count} structured sections)",
                100,
                source_hash=source_hash,
                section_count=section_count,
            )
            return

        # Keep chat fast: rely on existing index + structured profile fast-path.
        if meta_store_supported and indexed_hash != source_hash:
            await _set_indexed_profile_hash(r, source_hash, section_count)
            message = "RAG ready using hot index + updated structured profile"
        else:
            message = "RAG pipeline ready"
        set_warmup_state(
            "ready",
            message,
            100,
            source_hash=source_hash,
            section_count=section_count,
        )
    except asyncio.TimeoutError:
        set_warmup_state("error", "Warmup timed out - please retry init", 100)
    except Exception as e:
        set_warmup_state("error", f"Warmup failed: {e}", 100)


async def ensure_warmup_started(force_reindex=False):
    global warmup_task
    if warmup_state["status"] == "ready" and not force_reindex:
        return
    if warmup_task and not warmup_task.done() and not force_reindex:
        return
    async with warmup_task_lock:
        if warmup_state["status"] == "ready" and not force_reindex:
            return
        if warmup_task and not warmup_task.done() and not force_reindex:
            return
        warmup_task = asyncio.create_task(_run_warmup(force_reindex=force_reindex))


async def get_rag():
    global rag
    if rag is None:
        async with rag_init_lock:
            if rag is None:
                instance = LightRAG(
                    working_dir=WORKING_DIR,
                    chunk_token_size=300,
                    llm_model_func=nvidia_llm,
                    embedding_func=EmbeddingFunc(
                        embedding_dim=1024,
                        max_token_size=512,
                        func=nvidia_embed,
                    ),
                    graph_storage="Neo4JStorage",
                    vector_storage="PGVectorStorage",
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    addon_params={
                        "neo4j_url": NEO4J_URI_BOLT,
                        "neo4j_auth": (NEO4J_USER, NEO4J_PASSWORD),
                        "connection_string": SUPABASE_PG_URL,
                    },
                    vector_db_storage_cls_kwargs={"connection_string": SUPABASE_PG_URL},
                )
                await instance.initialize_storages()
                rag = instance
    return rag


# ── FastAPI lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

    global shared_storage_initialized
    if not shared_storage_initialized:
        initialize_share_data(workers=1)
        await initialize_pipeline_status()
        shared_storage_initialized = True
    # Do not auto-trigger warmup here; /api/init is the explicit bootstrap call.
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Q(BaseModel):
    question: str
    mode: str = "hybrid"


# ── Endpoints ───────────────────────────────────────────────────────
@app.get("/api/ping")
def ping():
    return {
        "status": "alive",
        "version": "2.2",
        "neo4j_uri": NEO4J_URI_BOLT,
        "neo4j_user": NEO4J_USER,
        "pg_host": os.environ.get("POSTGRES_HOST", "NOT SET"),
        "pipeline": _warmup_snapshot(),
    }


@app.get("/api/readiness")
async def readiness():
    # Lightweight status endpoint: no auto-warmup to avoid noisy repeated calls.
    snapshot = _warmup_snapshot()
    snapshot["running"] = bool(warmup_task and not warmup_task.done())
    return snapshot


@app.get("/api/init")
async def init_pipeline(force: bool = False):
    await ensure_warmup_started(force_reindex=force)
    if warmup_task:
        try:
            await warmup_task
        except Exception as e:
            set_warmup_state("error", f"Warmup crashed: {e}", 100)
    snapshot = _warmup_snapshot()
    snapshot["running"] = bool(warmup_task and not warmup_task.done())
    return snapshot


@app.get("/api/profile-schema")
async def profile_schema():
    async with profile_cache_lock:
        payload = _load_profile_payload()
    schema = payload.get("schema") or {}
    docs = schema.get("documents", [])
    return {
        "schema_version": schema.get("schema_version"),
        "generated_at": schema.get("generated_at"),
        "source": schema.get("source", {}),
        "stats": schema.get("stats", {}),
        "sample_documents": docs[:4],
    }


@app.get("/api/debug")
async def debug():
    r = await get_rag()
    out = {}

    try:
        db = r.doc_status.db
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1",
                db.workspace,
            )
            out["doc_status_rows"] = [dict(rw) for rw in rows]
            out["workspace"] = db.workspace
            out["chunk_count"] = await conn.fetchval(
                "SELECT COUNT(*) FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1",
                db.workspace,
            )
            try:
                out["vector_count"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM LIGHTRAG_VDB_ENTITY WHERE workspace=$1",
                    db.workspace,
                )
            except Exception as e:
                out["vector_error"] = str(e)
            try:
                row = await conn.fetchrow(
                    f"SELECT source_hash, section_count, updated_at FROM {PROFILE_META_TABLE} WHERE workspace=$1",
                    db.workspace,
                )
                out["profile_meta"] = dict(row) if row else None
            except Exception as e:
                out["profile_meta_error"] = str(e)
    except Exception as e:
        out["postgres_error"] = str(e)

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=os.environ.get("NEO4J_DATABASE")) as s:
            count = s.run("MATCH (n:base) RETURN count(n) AS c").single()["c"]
            out["neo4j_node_count"] = count
        driver.close()
    except Exception as e:
        out["neo4j_error"] = str(e)

    out["warmup"] = _warmup_snapshot()
    out["meta_store_supported"] = meta_store_supported
    return out


@app.post("/api/ask")
async def ask(req: Q):
    try:
        if warmup_state["status"] in {"idle", "error"}:
            await ensure_warmup_started()
        if warmup_state["status"] != "ready":
            snapshot = _warmup_snapshot()
            return {
                "error": "RAG pipeline is still warming up. Please wait a few seconds.",
                "mode_used": "warmup",
                "pipeline": snapshot,
            }

        async with profile_cache_lock:
            payload = _load_profile_payload()
        fast_answer = _answer_from_profile_fastpath(req.question, payload.get("schema"))
        if fast_answer:
            return {"answer": fast_answer, "mode_used": "profile_fastpath"}

        r = await get_rag()
        mode = req.mode if req.mode in {"local", "hybrid", "global"} else "hybrid"
        result = await asyncio.wait_for(
            r.aquery(
                req.question,
                param=QueryParam(
                    mode=mode,
                    response_type="Natural conversational answer in first person (4-7 sentences)",
                    top_k=10,
                    max_token_for_text_unit=1200,
                    max_token_for_local_context=1400,
                    max_token_for_global_context=1400,
                ),
                system_prompt=ASSISTANT_SYSTEM_PROMPT,
            ),
            timeout=16.0,
        )

        text_result = clean_answer(str(result or ""))
        if text_result and text_result.lower() not in {"none", ""}:
            return {"answer": text_result, "mode_used": mode}
        return {
            "answer": (
                "I do not have that exact detail cached yet, but I can answer if you ask a bit more specifically "
                "(for example: project name, timeline, or tool stack)."
            ),
            "mode_used": mode,
        }
    except asyncio.TimeoutError:
        return {"error": "Query timed out. Please try again in a few seconds.", "mode_used": "timeout"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/ask")
async def ask_get():
    return {"error": "Use POST /api/ask with JSON body: {\"question\": \"...\", \"mode\": \"hybrid\"}"}


@app.get("/api/graph")
async def graph():
    from collections import Counter
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(NEO4J_URI_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=os.environ.get("NEO4J_DATABASE")) as s:
            res = s.run(
                "MATCH (n:base) WITH n LIMIT 220 "
                "OPTIONAL MATCH (n)-[r]->(m:base) "
                "RETURN n.entity_id AS sname, n.entity_type AS stype, "
                "n.description AS sdesc, m.entity_id AS tname, "
                "r.keywords AS rlabel, r.weight AS w"
            )
            nodes = {}
            links = []
            seen = set()
            for rec in res:
                sname = rec["sname"]
                if sname and sname not in nodes:
                    nodes[sname] = {
                        "id": sname,
                        "name": sname,
                        "type": rec["stype"] or "Unknown",
                        "desc": rec["sdesc"] or "",
                    }
                tname = rec["tname"]
                if tname is not None:
                    if tname not in nodes:
                        nodes[tname] = {"id": tname, "name": tname, "type": "Unknown", "desc": ""}
                    if sname:
                        key = (sname, tname)
                        if key not in seen:
                            seen.add(key)
                            links.append(
                                {
                                    "source": sname,
                                    "target": tname,
                                    "label": (rec["rlabel"] or "").split(",")[0],
                                    "weight": float(rec["w"] or 1),
                                }
                            )
        driver.close()
        async with profile_cache_lock:
            payload = _load_profile_payload()
        overlay_nodes, overlay_links = _schema_graph_overlay(payload.get("schema"))
        for node in overlay_nodes:
            if node["id"] not in nodes:
                nodes[node["id"]] = node
        existing = {(lk["source"], lk["target"], lk.get("label", "")) for lk in links}
        for lk in overlay_links:
            key = (lk["source"], lk["target"], lk.get("label", ""))
            if key not in existing:
                existing.add(key)
                links.append(lk)

        type_counts = Counter((n.get("type") or "unknown").lower() for n in nodes.values())
        return {
            "nodes": list(nodes.values()),
            "links": links,
            "meta": {
                "node_count": len(nodes),
                "edge_count": len(links),
                "type_counts": dict(type_counts),
            },
        }
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}
