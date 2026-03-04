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
        "z-ai/glm5", prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        temperature=1, top_p=1, max_tokens=16384,
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
            llm_model_func=nvidia_llm,
            embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=nvidia_embed),
            graph_storage="Neo4JStorage",
            vector_storage="PGVectorStorage",
            kv_storage="PGKVStorage",
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
        if os.path.exists(data_path):
            with open(data_path) as f:
                content = f.read()
            # Clear any previously-failed document status so it gets re-processed
            try:
                if r.doc_status and r.doc_status.db:
                    await r.doc_status.db.execute(
                        f"DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1",
                        {"workspace": r.doc_status.db.workspace}
                    )
                    print("Cleared old doc_status entries for re-processing")
            except Exception as clear_err:
                print(f"Note: Could not clear doc_status: {clear_err}")
            await r.ainsert(content)
            return {"status": "success", "message": "Pipeline initialized and data ingested."}
        return {"status": "partial", "message": "Pipeline initialized but data file missing."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/ask")
async def ask(req: Q):
    async def stream():
        try:
            yield f"data: {json.dumps({'token': 'Thinking...'})}\n\n"
            await asyncio.sleep(0.1)
            r = await get_rag()
            query_mode = req.mode if req.mode in ["local", "global", "hybrid"] else "local"
            result = await r.aquery(req.question, param=QueryParam(mode=query_mode))
            text_result = str(result or "").strip()
            if not text_result:
                yield f"data: {json.dumps({'token': 'I could not find a specific answer. Make sure /api/init has been run.'})}\n\n"
            else:
                for i, word in enumerate(text_result.split(" ")):
                    token = word + (" " if i < len(text_result.split(" ")) - 1 else "")
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

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
