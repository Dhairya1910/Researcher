import sys
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Adjust path so we can import Backend modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Backend.Services.Agent_Utils import Agent

app = FastAPI(title="Research AI - Jarvis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = {
    "model": "mistral-medium-latest",
    "mode": "General",
    "current_agent_mode": None,
    "agent": None,
    "file_stored": False,
    "file_type": "text",
    "uploaded_filename": None,
}


def get_or_create_agent() -> Agent:
    """Create agent if needed, or recreate on mode/model change."""
    mode = session["mode"]
    model = session["model"]

    if (
        session["agent"] is None
        or session["current_agent_mode"] != mode
    ):
        print(f"[server] Creating new Agent: model={model}, role={mode}")
        session["agent"] = Agent(
            model_name=model,
            agent_role=mode,
        )
        session["current_agent_mode"] = mode

    return session["agent"]


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path, media_type="text/html")


@app.post("/api/config")
async def update_config(request: Request):
    """Update model and mode settings."""
    data = await request.json()

    if "model" in data:
        session["model"] = data["model"]
    if "mode" in data:
        session["mode"] = data["mode"]
        # Force Research mode to use magistral
        if data["mode"] == "Research":
            session["model"] = "magistral-medium-latest"

    return {
        "status": "ok",
        "model": session["model"],
        "mode": session["mode"],
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF/image file to use as context."""
    agent = get_or_create_agent()
    content = await file.read()
    content_type = file.content_type or ""
    filename = file.filename or "unknown"

    f_type = "text"
    if "pdf" in content_type:
        f_type = "pdf"
        agent.convert_and_store_to_vect_db(content)
        session["file_stored"] = True
    elif "image" in content_type:
        f_type = "image"

    session["file_type"] = f_type
    session["uploaded_filename"] = filename

    return {
        "status": "ok",
        "filename": filename,
        "file_type": f_type,
        "stored": session["file_stored"],
    }


@app.post("/api/clear-file")
async def clear_file():
    """Remove uploaded file and clear vectorstore."""
    if session["agent"] and session["file_stored"]:
        session["agent"].clear_vectorstore()
    session["file_stored"] = False
    session["file_type"] = "text"
    session["uploaded_filename"] = None
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request):
    """Stream agent response via SSE."""
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"error": "Empty message"}

    agent = get_or_create_agent()
    f_type = session["file_type"]

    def generate():
        try:
            stream = agent.Invoke_agent(message, f_type)
            for chunk in stream:
                payload = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            error_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
async def clear_conversation():
    """Clear conversation — reset agent to force new memory."""
    if session["agent"] and session["file_stored"]:
        session["agent"].clear_vectorstore()
    session["agent"] = None
    session["current_agent_mode"] = None
    session["file_stored"] = False
    session["file_type"] = "text"
    session["uploaded_filename"] = None
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    return {
        "status": "ready",
        "model": session["model"],
        "mode": session["mode"],
        "file_stored": session["file_stored"],
        "uploaded_filename": session["uploaded_filename"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
