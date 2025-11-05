run:
	 uv run uvicorn app.api.app:app --reload

ngrok:
	ngrok http 8000