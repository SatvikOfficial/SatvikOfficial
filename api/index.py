import os, json, asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Load local environment
load_dotenv()

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEO4J_URI       = os.environ.get("NEO4J_URI", "")
# Try both common names for the user variable
NEO4J_USER      = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD  = os.environ.get("NEO4J_PASSWORD", "")
SUPABASE_PG_URL = os.environ.get("SUPABASE_PG_URL", "")

# Sync back to os.environ for drivers that check it directly
os.environ["NEO4J_URL"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

if SUPABASE_PG_URL:
    try:
        from urllib.parse import unquote
        parsed = urlparse(SUPABASE_PG_URL)
        os.environ["POSTGRES_USER"] = unquote(parsed.username or "")
        os.environ["POSTGRES_PASSWORD"] = unquote(parsed.password or "")
        os.environ["POSTGRES_HOST"] = parsed.hostname or ""
        os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
        os.environ["POSTGRES_DATABASE"] = unquote(parsed.path.lstrip("/"))
    except: pass

WORKING_DIR     = "/tmp/lightrag_cache"
os.makedirs(WORKING_DIR, exist_ok=True)

async def nvidia_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    # Completely sanitize history_messages to prevent KeyError
    hist = []
    if history_messages is not None:
        hist = history_messages
    elif "history_messages" in kwargs:
        hist = kwargs.pop("history_messages")
    
    return await openai_complete_if_cache(
        "z-ai/glm5", prompt,
        system_prompt=system_prompt,
        history_messages=hist,
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        temperature=1,
        top_p=1,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
        **kwargs,
    )

async def nvidia_embed(texts):
    return await openai_embed(
        texts, model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL,
    )

rag = None

async def get_rag():
    global rag
    if rag is None:
        try:
            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=nvidia_llm,
                embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=nvidia_embed),
                graph_storage="Neo4JStorage",
                vector_storage="PGVectorStorage",
                kv_storage="PGKVStorage",
                addon_params={
                    "neo4j_url": NEO4J_URI, 
                    "neo4j_auth": (NEO4J_USER, NEO4J_PASSWORD),
                    "connection_string": SUPABASE_PG_URL
                },
                vector_db_storage_cls_kwargs={"connection_string": SUPABASE_PG_URL},
            )
        except Exception as e:
            print(f"CRITICAL RAG INIT ERROR: {e}")
            raise e
    return rag

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str
    mode: str = "local" # Default to 'local' for speed on Vercel

@app.get("/api/ping")
def ping():
    return {"status": "alive", "version": "1.6", "neo4j_user": NEO4J_USER}

@app.get("/api/init")
async def init_pipeline():
    try:
        r = await get_rag()
        data_path = os.path.join(os.path.dirname(__file__), "satvik_data.txt")
        if os.path.exists(data_path):
            with open(data_path) as f:
                await r.ainsert(f.read())
            return {"status": "success", "message": "Pipeline initialized and data ingested."}
        return {"status": "partial", "message": "Pipeline initialized but data file missing."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/ask")
async def ask(req: Q):
    async def stream():
        try:
            # Yield initial token so Vercel doesn't timeout immediately
            yield f"data: {json.dumps({'token': 'Thinking...'})}\n\n"
            await asyncio.sleep(0.1)
            
            r = await get_rag()
            # Use 'local' mode if not specified, it's faster
            query_mode = req.mode if req.mode in ["local", "global", "hybrid"] else "local"
            result = await r.aquery(req.question, param=QueryParam(mode=query_mode))
            
            # Clear "Thinking..." with a space
            yield f"data: {json.dumps({'token': '\b\b\b\b\b\b\b\b\b\b\b'})}\n\n" 
            
            text_result = str(result or "").strip()
            if not text_result:
                yield f"data: {json.dumps({'token': 'I couldn\'t find a specific answer. Make sure /api/init has been run.'})}\n\n"
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
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            res = s.run("MATCH (n) WITH n LIMIT 150 OPTIONAL MATCH (n)-[r]->(m) RETURN id(n) AS sid, n.id AS sname, n.entity_type AS stype, n.description AS sdesc, id(m) AS tid, m.id AS tname, r.keywords AS rlabel, r.weight AS w")
            nodes, links, seen = {}, [], set()
            for rec in res:
                if rec["sid"] not in nodes:
                    nodes[rec["sid"]] = {"id": rec["sid"], "name": rec["sname"] or f"node_{rec['sid']}", "type": rec["stype"] or "Unknown", "desc": rec["sdesc"] or ""}
                if rec["tid"] is not None:
                    if rec["tid"] not in nodes:
                        nodes[rec["tid"]] = {"id": rec["tid"], "name": rec["tname"] or f"node_{rec['tid']}", "type": "Unknown", "desc": ""}
                    key = (rec["sid"], rec["tid"])
                    if key not in seen:
                        seen.add(key)
                        links.append({"source": rec["sid"], "target": rec["tid"], "label": (rec["rlabel"] or "").split(",")[0], "weight": float(rec["w"] or 1)})
        driver.close()
        return {"nodes": list(nodes.values()), "links": links}
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}
