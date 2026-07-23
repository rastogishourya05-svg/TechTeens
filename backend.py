"""
FastAPI backend for MedAssist — wraps the LangChain agent so a Flutter app
(or any HTTP client) can interact with it over REST.
Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
Endpoints:
    POST /chat              — send a message, get a reply + structured tool data
    POST /session/new       — create a fresh conversation session
    DELETE /session/{id}     — delete a session
    GET  /health            — liveness check
"""
from __future__ import annotations
import asyncio
import uuid
import time
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# ---------------------------------------------------------------------------
# Import everything from the existing agent module
# ---------------------------------------------------------------------------
from agent2 import create_agent, chat
# ---------------------------------------------------------------------------
# Session store — keeps per-user conversation history + agent executor
# ---------------------------------------------------------------------------
class Session:
    """Holds one user's conversation state."""
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.history: list = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()
_sessions: dict[str, Session] = {}
# How long (seconds) before an idle session is eligible for cleanup.
SESSION_TTL = 60 * 60  # 1 hour
def _prune_stale_sessions() -> None:
    """Remove sessions that have been idle longer than SESSION_TTL."""
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in stale:
        del _sessions[sid]
def _get_or_create_session(session_id: str | None) -> tuple[str, Session]:
    """Return (session_id, Session). Creates a new one if id is None or unknown."""
    _prune_stale_sessions()
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        session.last_active = time.time()
        return session_id, session
    # Create a new session
    new_id = session_id or uuid.uuid4().hex
    agent_executor = create_agent()
    session = Session(agent_executor)
    _sessions[new_id] = session
    return new_id, session
# ---------------------------------------------------------------------------
# Pydantic models (request / response schemas)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    session_id: str | None = Field(
        default=None,
        description="Session ID. Omit or pass null to start a new conversation.",
    )
class ToolResult(BaseModel):
    tool: str
    input: dict[str, Any]
    data: Any
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_results: list[ToolResult] = []
class SessionResponse(BaseModel):
    session_id: str
    message: str = "Session created"
class HealthResponse(BaseModel):
    status: str = "ok"
    active_sessions: int = 0
# ---------------------------------------------------------------------------
# App factory with lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    print("🏥 MedAssist API starting …")
    yield
    print("🏥 MedAssist API shutting down — clearing sessions.")
    _sessions.clear()
app = FastAPI(
    title="MedAssist API",
    description="Medical information assistant powered by LangChain + Groq",
    version="1.0.0",
    lifespan=lifespan,
)
# ---------------------------------------------------------------------------
# CORS — allow Flutter apps (web, mobile emulator, desktop) to connect
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Send a message and receive MedAssist's reply along with any structured
    tool results (parsed JSON from each tool the agent called this turn).
    """
    session_id, session = _get_or_create_session(req.session_id)
    try:
        # chat() is a blocking, synchronous call (it hits the Groq API under the
        # hood). Running it directly in this async endpoint would block FastAPI's
        # event loop and serialize every concurrent user's requests. Offload it
        # to a worker thread instead so other requests can proceed in parallel.
        result = await asyncio.to_thread(
            chat,
            user_input=req.message,
            agent_executor=session.agent_executor,
            session_history=session.history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    # Update session history from the returned state
    session.history = result.get("history", session.history)
    tool_results = [
        ToolResult(tool=tr["tool"], input=tr["input"], data=tr["data"])
        for tr in result.get("tool_results", [])
    ]
    return ChatResponse(
        session_id=session_id,
        reply=result.get("reply", "No response generated."),
        tool_results=tool_results,
    )
@app.post("/session/new", response_model=SessionResponse)
async def new_session():
    """Create a brand-new conversation session and return its ID."""
    session_id, _ = _get_or_create_session(None)
    return SessionResponse(session_id=session_id)
@app.delete("/session/{session_id}", response_model=SessionResponse)
async def delete_session(session_id: str):
    """Delete an existing session (clear its history and free resources)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return SessionResponse(session_id=session_id, message="Session deleted")
@app.get("/health", response_model=HealthResponse)
async def health():
    """Simple liveness / readiness probe."""
    return HealthResponse(status="ok", active_sessions=len(_sessions))