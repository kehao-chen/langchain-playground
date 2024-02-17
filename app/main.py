from fastapi import FastAPI

from .dependencies import llm
from .routers import demo

app = FastAPI()
app.include_router(demo.router)


@app.get("/")
async def root():
    message = llm.invoke("What does the fox say?")
    return {"message": message}
