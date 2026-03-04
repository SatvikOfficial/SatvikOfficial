import os, json, asyncio
from urllib.parse import urlparse, unquote
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
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
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def nvidia_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    kwargs.pop("history_messages", None)
    return await openai_complete_if_cache(
        "z-ai/glm-4", prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        temperature=1, top_p=1, max_tokens=16384,
        openai_client_configs={"timeout": 30.0},
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

# ── RAG singleton ───────────────────────────────────────────────────
rag = None

async def get_rag():
    global rag
    if rag is None:
        rag = LightRAG(
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
        await rag.initialize_storages()
    return rag

# ── FastAPI lifespan (critical for LightRAG shared storage) ─────────
@asynccontextmanager
async def lifespan(app):
    from lightrag.kg.shared_storage import (
        initialize_share_data,
        initialize_pipeline_status,
        finalize_share_data,
    )
    initialize_share_data(workers=1)
    await initialize_pipeline_status()
    # Do NOT pre-warm get_rag() here — it's too slow for Vercel cold starts.
    # It will initialize lazily on the first /api/ask or /api/init request.
    yield
    finalize_share_data()

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
    }

@app.get("/api/init")
async def init_pipeline():
    try:
        r = await get_rag()
        data_path = os.path.join(os.path.dirname(__file__), "satvik_data.txt")
        if not os.path.exists(data_path):
            return {"status": "partial", "message": "Data file missing."}

        with open(data_path) as f:
            content = f.read()

        # Force re-ingestion by clearing stale doc tracking rows.
        # LightRAG writes the doc ID before processing — if a previous run
        # crashed mid-way, the doc is "seen" but the graph is empty.
        try:
            db = r.doc_status.db
            if db and db.pool:
                async with db.pool.acquire() as conn:
                    deleted = await conn.fetchval(
                        "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 RETURNING count(*)",
                        db.workspace
                    )
                    print(f"Cleared {deleted} stale doc_status rows for workspace={db.workspace}")
        except Exception as clear_err:
            print(f"Note: Could not clear doc_status: {clear_err}")

        # Add timeout for serverless (Vercel Hobby max 60s)
        try:
            await asyncio.wait_for(r.ainsert(content), timeout=55.0)
            return {"status": "success", "message": "Data ingested into graph."}
        except asyncio.TimeoutError:
            return {"status": "partial", "message": "Ingestion timed out. Please visit /api/init again to complete."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

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
        r = await get_rag()
        
        async def _run_query():
            # Try hybrid first, fall back to local if empty
            for mode in ["hybrid", "local"]:
                result = await r.aquery(req.question, param=QueryParam(mode=mode))
                text_result = str(result or "").strip()
                if text_result and text_result.lower() not in ("none", ""):
                    return {"answer": text_result, "mode_used": mode}
            return {"answer": "I could not find a specific answer. Try asking about Satvik's work experience, projects, skills, or education.", "mode_used": "none"}
        
        # Timeout to prevent endless buffering on Vercel (Hobby max 60s)
        try:
            return await asyncio.wait_for(_run_query(), timeout=50.0)
        except asyncio.TimeoutError:
            return {"error": "Query timed out. Please try again.", "mode_used": "timeout"}
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
        return {"nodes": list(nodes.values()), "links": links}
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}
