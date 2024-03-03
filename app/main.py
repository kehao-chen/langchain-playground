from fastapi import FastAPI

from .dependencies import get_rag_chain

app = FastAPI()


@app.get("/")
async def root():
    chain = get_rag_chain()
    message = chain.invoke("how can langsmith help with testing?")
    return {"message": message}
