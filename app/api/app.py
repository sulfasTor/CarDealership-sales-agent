from fastapi import FastAPI
from app.llm.llm import init_agent, run_query
from pydantic import BaseModel

app = FastAPI()

global agent
agent = init_agent()

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(q: Query):
    output = run_query(agent, q.text)
    return {"answer": output}
