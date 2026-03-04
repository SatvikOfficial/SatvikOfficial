import asyncio
import os
import traceback
from api.index import get_rag

async def main():
    from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
    initialize_share_data(workers=1)
    await initialize_pipeline_status()
    print("Initializing RAG...")
    try:
        r = await get_rag()
        print(f"Reading small test string...")
        content = "Satvik Mudgal is an AI Engineer at Nuvo AI. He lives in Bengaluru."

        print("Clearing doc status...")
        try:
            db = r.doc_status.db
            if db and db.pool:
                async with db.pool.acquire() as conn:
                    await conn.execute("DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1", db.workspace)
                    print("Cleared!")
        except Exception as clear_err:
            print(f"Clear err: {clear_err}")

        print("Inserting document...")
        await r.ainsert(content)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
