import os, json, asyncio
from contextlib import asynccontextmanager
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
NEO4J_USER      = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD  = os.environ.get("NEO4J_PASSWORD", "")
SUPABASE_PG_URL = os.environ.get("SUPABASE_PG_URL", "")
WORKING_DIR     = "/tmp/lightrag_cache"
os.makedirs(WORKING_DIR, exist_ok=True)

async def nvidia_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    # Use GLM5 with thinking enabled as requested
    return await openai_complete_if_cache(
        "z-ai/glm5", prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
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

async def init_rag():
    global rag
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=nvidia_llm,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=nvidia_embed),
        graph_storage="Neo4JStorage",
        addon_params={"neo4j_url": NEO4J_URI, "neo4j_auth": (NEO4J_USER, NEO4J_PASSWORD)},
        vector_storage="PGVectorStorage",
        kv_storage="PGKVStorage",
        vector_db_storage_cls_kwargs={"connection_string": SUPABASE_PG_URL},
        kv_storage_cls_kwargs={"connection_string": SUPABASE_PG_URL},
    )
    
    # Check if we need to ingest data
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
        if count == 0:
            data_path = os.path.join(os.path.dirname(__file__), "satvik_data.txt")
            if os.path.exists(data_path):
                with open(data_path) as f:
                    await rag.ainsert(f.read())
    except Exception as e:
        print(f"Graph check/ingestion error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_rag()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str
    mode: str = "hybrid"

@app.post("/ask")
async def ask(req: Q):
    if not rag: raise HTTPException(503, "RAG initializing")
    async def stream():
        result = await rag.aquery(req.question, param=QueryParam(mode=req.mode))
        for i, word in enumerate(result.split(" ")):
            yield f"data: {json.dumps({'token': word + (' ' if i < len(result.split())-1 else '')})}\n\n"
            await asyncio.sleep(0.015) # Faster streaming
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/graph")
async def graph():
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            res = s.run("""
                MATCH (n) WITH n LIMIT 150
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN id(n) AS sid, n.id AS sname, n.entity_type AS stype, n.description AS sdesc,
                       id(m) AS tid, m.id AS tname, r.keywords AS rlabel, r.weight AS w
            """)
            nodes, links, seen = {}, [], set()
            for rec in res:
                if rec["sid"] not in nodes:
                    nodes[rec["sid"]] = {"id": rec["sid"], "name": rec["sname"] or f"node_{rec['sid']}",
                                          "type": rec["stype"] or "Unknown", "desc": rec["sdesc"] or ""}
                if rec["tid"] is not None:
                    if rec["tid"] not in nodes:
                        nodes[rec["tid"]] = {"id": rec["tid"], "name": rec["tname"] or f"node_{rec['tid']}",
                                              "type": "Unknown", "desc": ""}
                    key = (rec["sid"], rec["tid"])
                    if key not in seen:
                        seen.add(key)
                        links.append({"source": rec["sid"], "target": rec["tid"],
                                      "label": (rec["rlabel"] or "").split(",")[0], "weight": float(rec["w"] or 1)})
        driver.close()
        return {"nodes": list(nodes.values()), "links": links}
    except Exception as e:
        return {"error": str(e), "nodes": [], "links": []}

@app.get("/health")
def health():
    return {"ok": True, "rag": rag is not None}
