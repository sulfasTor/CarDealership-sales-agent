import logging
from fastapi import FastAPI, Request, Response
from app.llm.llm import init_agent, llm_run_query
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

global agent
agent = init_agent()


@app.post("/ask", response_class=Response)
async def ask(request: Request):
    logger.info('POST /ask')
    form = await request.form()
    body = form.get("Body", "")
    sender = form.get("From", "")

    logger.info(f'Got request from {sender}')

    output = llm_run_query(agent, str(body), str(sender))

    logger.info('LLM answered')

    resp = MessagingResponse()
    resp.message(output)

    return Response(content=str(resp), media_type="application/xml")
