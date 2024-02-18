from fastapi import FastAPI

from .dependencies import llm

app = FastAPI()


@app.get("/")
async def root():
    message = llm.invoke("What does the fox say?")
    return {"message": message}
