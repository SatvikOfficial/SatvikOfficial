import os, json, asyncio, logging, time, re
from urllib.parse import urlparse, unquote
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "meta/llama-3.1-8b-instruct")
NEO4J_URI       = os.environ.get("NEO4J_URI", "")
NEO4J_USER      = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD  = os.environ.get("NEO4J_PASSWORD", "")
SUPABASE_PG_URL = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_PG_URL", "")

# Keep neo4j+s:// protocol — Aura requires routing protocol
# (bolt+s:// can connect but can't discover the database name)
NEO4J_URI_BOLT = NEO4J_URI  # keep original protocol

# Sync env vars for drivers that read them directly
os.environ["NEO4J_URL"]      = NEO4J_URI_BOLT
os.environ["NEO4J_URI"]      = NEO4J_URI_BOLT  # neo4j_impl reads this
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
# Aura free tier database is named after the instance ID (e.g. "66adaa8b")
# Extract it from the URI: neo4j+s://66adaa8b.databases.neo4j.io
import re as _re
_aura_match = _re.match(r'neo4j\+s(?:sc)?://([^.]+)\.databases\.neo4j\.io', NEO4J_URI)
os.environ["NEO4J_DATABASE"] = _aura_match.group(1) if _aura_match else "neo4j"

# ── Parse PG URL → individual env vars for LightRAG's ClientManager ─
if SUPABASE_PG_URL:
    try:
        parsed = urlparse(SUPABASE_PG_URL)
        os.environ["POSTGRES_USER"]     = unquote(parsed.username or "")
        os.environ["POSTGRES_PASSWORD"] = unquote(parsed.password or "")
        os.environ["POSTGRES_HOST"]     = parsed.hostname or ""
        os.environ["POSTGRES_PORT"]     = str(parsed.port or 5432)
        os.environ["POSTGRES_DATABASE"] = unquote(parsed.path.lstrip("/"))
    except Exception:
        pass

WORKING_DIR = "/tmp/lightrag_cache"
os.makedirs(WORKING_DIR, exist_ok=True)

# ── MONKEY-PATCH: Make asyncpg work with PgBouncer (Supavisor) ──────
# PgBouncer in transaction-pooling mode does NOT support prepared statements.
# asyncpg uses prepared statements by default, causing silent connection failures.
# We patch PostgreSQLDB.initdb to add statement_cache_size=0.
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
            statement_cache_size=0,          # PgBouncer compat
            server_settings={"jit": "off"},  # avoid JIT issues on poolers
        )
        print(f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database}")
    except Exception as e:
        print(f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}")
        raise

PostgreSQLDB.initdb = _patched_initdb

# ── LLM / Embedding wrappers ───────────────────────────────────────
# Keep LightRAG's internal informational logs out of request logs.
logging.getLogger("lightrag").setLevel(logging.WARNING)
logging.getLogger("lightrag.kg.shared_storage").setLevel(logging.ERROR)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg import shared_storage as shared_storage_module

_original_direct_log = shared_storage_module.direct_log


def _quiet_direct_log(message, level="INFO", enable_output: bool = True):
    msg = str(message)
    if level == "INFO" and (
        "Shared-Data" in msg or "Pipeline namespace initialized" in msg
    ):
        return
    _original_direct_log(message, level=level, enable_output=enable_output)


shared_storage_module.direct_log = _quiet_direct_log

async def nvidia_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    llm_max_tokens = kwargs.pop("max_tokens", 1024)
    llm_temperature = kwargs.pop("temperature", 0.2)
    llm_top_p = kwargs.pop("top_p", 0.9)
    kwargs.pop("history_messages", None)
    return await openai_complete_if_cache(
        NVIDIA_CHAT_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        temperature=llm_temperature, top_p=llm_top_p, max_tokens=llm_max_tokens,
        openai_client_configs={"timeout": 45.0},
        **kwargs,
    )

async def nvidia_embed(texts):
    # NVIDIA's nv-embedqa-e5-v5 is an asymmetric model requiring 'input_type'
    # LightRAG's openai_embed doesn't support extra_body, so we call directly
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
    "You are Satvik Mudgal speaking in first person. "
    "Answer as me using 'I' and 'my', with a confident and friendly tone. "
    "Be specific and concise, focus on practical details, and avoid generic filler. "
    "Do not mention 'knowledge base', 'sources', or 'references' unless the user explicitly asks. "
    "If a detail is missing, say you don't have that detail currently instead of guessing."
)


def clean_answer(text: str) -> str:
    cleaned = (text or "").strip()
    # Drop common autogenerated reference tails.
    cleaned = re.sub(r"(?is)\n*#+\s*references.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*[-*]\s*\[KG\].*$", "", cleaned)
    # Remove stale template prefix.
    cleaned = re.sub(r"(?i)^based on (the )?knowledge base[:,]?\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ── Runtime state ───────────────────────────────────────────────────
rag = None
rag_init_lock = asyncio.Lock()

warmup_task = None
warmup_task_lock = asyncio.Lock()
warmup_state = {
    "status": "idle",          # idle | warming | ready | error
    "message": "Waiting to start",
    "progress": 0,
    "updated_at": int(time.time()),
}
shared_storage_initialized = False


def _warmup_snapshot():
    return {
        "status": warmup_state["status"],
        "message": warmup_state["message"],
        "progress": warmup_state["progress"],
        "updated_at": warmup_state["updated_at"],
    }


def set_warmup_state(status: str, message: str, progress: int):
    warmup_state["status"] = status
    warmup_state["message"] = message
    warmup_state["progress"] = max(0, min(100, int(progress)))
    warmup_state["updated_at"] = int(time.time())


def _satvik_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "satvik_data.txt")


def _read_satvik_data() -> str:
    data_path = _satvik_data_path()
    if not os.path.exists(data_path):
        return ""
    with open(data_path, encoding="utf-8") as f:
        return f.read()


async def _workspace_has_index(r: LightRAG) -> bool:
    """Fast readiness check: if chunks exist, retrieval can run."""
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


async def _run_warmup(force_reindex: bool = False):
    try:
        set_warmup_state("warming", "Booting retrieval pipeline", 10)
        r = await get_rag()
        set_warmup_state("warming", "Connecting graph/vector stores", 35)

        has_index = await _workspace_has_index(r)
        if force_reindex or not has_index:
            data = _read_satvik_data()
            if not data:
                set_warmup_state("error", "satvik_data.txt is missing", 100)
                return
            set_warmup_state("warming", "Building and validating knowledge index", 65)
            await asyncio.wait_for(r.ainsert(data), timeout=55.0)

        set_warmup_state("ready", "RAG pipeline ready", 100)
    except asyncio.TimeoutError:
        set_warmup_state("error", "Warmup timed out - retrying on next check", 100)
    except Exception as e:
        set_warmup_state("error", f"Warmup failed: {e}", 100)


async def ensure_warmup_started(force_reindex: bool = False):
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
                    chunk_token_size=256,  # nv-embedqa-e5-v5 max is 512 tokens
                    llm_model_func=nvidia_llm,
                    embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=512, func=nvidia_embed),
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
                # Explicitly await storage initialization instead of letting it run
                # as a fire-and-forget background task (which silently fails on Vercel)
                await instance.initialize_storages()
                rag = instance
    return rag

# ── FastAPI lifespan (critical for LightRAG shared storage) ─────────
@asynccontextmanager
async def lifespan(app):
    from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
    global shared_storage_initialized
    if not shared_storage_initialized:
        initialize_share_data(workers=1)
        await initialize_pipeline_status()
        shared_storage_initialized = True
    await ensure_warmup_started()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str
    mode: str = "local"

# ── Endpoints ───────────────────────────────────────────────────────
@app.get("/api/ping")
def ping():
    return {
        "status": "alive",
        "version": "2.1",
        "neo4j_uri": NEO4J_URI_BOLT,
        "neo4j_user": NEO4J_USER,
        "pg_host": os.environ.get("POSTGRES_HOST", "NOT SET"),
        "pipeline": _warmup_snapshot(),
    }

@app.get("/api/readiness")
async def readiness():
    await ensure_warmup_started()
    if warmup_task and warmup_task.done():
        try:
            warmup_task.result()
        except Exception as e:
            set_warmup_state("error", f"Warmup crashed: {e}", 100)
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
    return _warmup_snapshot()

@app.get("/api/debug")
async def debug():
    r = await get_rag()
    out = {}
    
    # Check doc status in Postgres
    try:
        db = r.doc_status.db
        async with db.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1", db.workspace)
            out["doc_status_rows"] = [dict(r) for r in rows]
            out["workspace"] = db.workspace
            
            # Check if any chunks exist
            chunks = await conn.fetchval("SELECT COUNT(*) FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1", db.workspace)
            out["chunk_count"] = chunks
            
            # Check vectors
            try:
                vecs = await conn.fetchval("SELECT COUNT(*) FROM LIGHTRAG_VDB_ENTITY WHERE workspace=$1", db.workspace)
                out["vector_count"] = vecs
            except Exception as e:
                out["vector_error"] = str(e)
    except Exception as e:
        out["postgres_error"] = str(e)
    
    # Check Neo4j node count
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=os.environ.get("NEO4J_DATABASE")) as s:
            count = s.run("MATCH (n:base) RETURN count(n) AS c").single()["c"]
            out["neo4j_node_count"] = count
        driver.close()
    except Exception as e:
        out["neo4j_error"] = str(e)
    
    return out

@app.post("/api/ask")
async def ask(req: Q):
    try:
        await ensure_warmup_started()
        if warmup_state["status"] != "ready":
            snapshot = _warmup_snapshot()
            return {
                "error": "RAG pipeline is still warming up. Please wait a moment.",
                "mode_used": "warmup",
                "pipeline": snapshot,
            }

        r = await get_rag()

        async def _query_mode(mode: str, timeout_s: float):
            try:
                result = await asyncio.wait_for(
                    r.aquery(
                        req.question,
                        param=QueryParam(
                            mode=mode,
                            response_type="Single Paragraph",
                            top_k=20,
                            max_token_for_text_unit=1800,
                            max_token_for_local_context=2200,
                            max_token_for_global_context=2200,
                        ),
                        system_prompt=ASSISTANT_SYSTEM_PROMPT,
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                return None
            except AttributeError as e:
                if "workspace" in str(e):
                    return None
                raise
            text_result = clean_answer(str(result or ""))
            if text_result and text_result.lower() not in ("none", ""):
                return text_result
            return None

        async def _run_query():
            # Local is typically fastest, then hybrid for richer retrieval.
            for mode, timeout_s in (("local", 10.0), ("hybrid", 14.0), ("global", 8.0)):
                text = await _query_mode(mode, timeout_s)
                if text:
                    return {"answer": text, "mode_used": mode}
            return {
                "answer": "I do not have that detail in my current profile context yet. Ask me again in a more specific way and I will answer directly.",
                "mode_used": "none",
            }

        # Keep total request under serverless timeout ceilings.
        try:
            return await asyncio.wait_for(_run_query(), timeout=30.0)
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
    from neo4j import GraphDatabase
    from collections import Counter
    try:
        driver = GraphDatabase.driver(NEO4J_URI_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=os.environ.get("NEO4J_DATABASE")) as s:
            res = s.run(
                "MATCH (n:base) WITH n LIMIT 150 "
                "OPTIONAL MATCH (n)-[r]->(m:base) "
                "RETURN n.entity_id AS sname, n.entity_type AS stype, "
                "n.description AS sdesc, m.entity_id AS tname, "
                "r.keywords AS rlabel, r.weight AS w"
            )
            nodes, links, seen = {}, [], set()
            for rec in res:
                sname = rec["sname"]
                if sname and sname not in nodes:
                    nodes[sname] = {"id": sname, "name": sname, "type": rec["stype"] or "Unknown", "desc": rec["sdesc"] or ""}
                tname = rec["tname"]
                if tname is not None:
                    if tname not in nodes:
                        nodes[tname] = {"id": tname, "name": tname, "type": "Unknown", "desc": ""}
                    if sname:
                        key = (sname, tname)
                        if key not in seen:
                            seen.add(key)
                            links.append({"source": sname, "target": tname, "label": (rec["rlabel"] or "").split(",")[0], "weight": float(rec["w"] or 1)})
        driver.close()
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
