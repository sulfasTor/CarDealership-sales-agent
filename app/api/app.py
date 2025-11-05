from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from app.llm.llm import init_agent, llm_run_query
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

global agent
agent = init_agent()


@app.post("/ask", response_class=Response)
async def ask(request: Request):
    form = await request.form()
    body = form.get("Body", "")
    sender = form.get("From", "")
    wa_id = form.get("WaId", "")

    output = llm_run_query(agent, body)

    resp = MessagingResponse()
    resp.message(output)

    return Response(content=str(resp), media_type="application/xml")
