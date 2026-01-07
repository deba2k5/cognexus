"""
AWE API Server
==============
FastAPI server exposing AWE functionality as REST API endpoints.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# Configuration
# =============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# LLM Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Ollama Configuration (fallback)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TOT_ENABLED = os.getenv("TOT_ENABLED", "true").lower() == "true"


# =============================================================================
# Data Models
# =============================================================================

class ExplorationRequest(BaseModel):
    """Request to start a new exploration task."""
    url: str = Field(..., description="Starting URL for exploration")
    objective: str = Field(..., description="What to extract/explore")
    target_fields: Optional[List[str]] = Field(
        default=None,
        description="Specific fields to extract"
    )
    max_pages: Optional[int] = Field(default=100, description="Maximum pages to explore")
    timeout: Optional[int] = Field(default=300, description="Timeout in seconds")


class ExplorationStatus(BaseModel):
    """Status of an exploration task."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    pages_visited: int
    items_extracted: int
    current_url: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ExplorationResult(BaseModel):
    """Result of a completed exploration."""
    task_id: str
    status: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    patterns_learned: int
    duration_seconds: float


class DemoRequest(BaseModel):
    """Request for demo exploration."""
    url: str
    quick_mode: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model: str
    model_provider: str
    tot_enabled: bool


# =============================================================================
# In-memory task storage (replace with Redis/DB in production)
# =============================================================================

tasks: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# App Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"🚀 AWE API Server starting on {API_HOST}:{API_PORT}")
    print(f"   Model: {MODEL_NAME} ({MODEL_PROVIDER})")
    print(f"   ToT Enabled: {TOT_ENABLED}")
    yield
    print("👋 AWE API Server shutting down")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="AWE - Agentic Web Explorer API",
    description="Production-grade multi-agent framework for autonomous web exploration and data extraction.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        tot_enabled=TOT_ENABLED,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        tot_enabled=TOT_ENABLED,
    )


@app.post("/explore", response_model=ExplorationStatus)
async def start_exploration(
    request: ExplorationRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new exploration task.
    
    The exploration runs in the background. Use the returned task_id
    to poll for status and results.
    """
    task_id = str(uuid4())
    
    # Store task info
    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "pages_visited": 0,
        "items_extracted": 0,
        "current_url": request.url,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "request": request.model_dump(),
        "data": [],
    }
    
    # Start background exploration
    background_tasks.add_task(run_exploration, task_id, request)
    
    return ExplorationStatus(
        task_id=task_id,
        status="pending",
        progress=0,
        pages_visited=0,
        items_extracted=0,
        current_url=request.url,
        started_at=tasks[task_id]["started_at"],
    )


async def run_exploration(task_id: str, request: ExplorationRequest):
    """Background task to run the exploration."""
    try:
        tasks[task_id]["status"] = "running"
        
        # Import AWE components
        try:
            from . import WebExplorer, ExplorationGoal
            
            goal = ExplorationGoal(
                objective=request.objective,
                target_fields=request.target_fields or [],
                start_url=request.url,
                constraints={
                    "max_pages": request.max_pages,
                    "timeout": request.timeout,
                }
            )
            
            async with WebExplorer(
                model=MODEL_NAME,
                tot_enabled=TOT_ENABLED
            ) as explorer:
                result = await explorer.explore(goal)
                
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["progress"] = 100
                tasks[task_id]["data"] = result.items
                tasks[task_id]["pages_visited"] = result.pages_visited
                tasks[task_id]["items_extracted"] = len(result.items)
                
        except ImportError:
            # AWE not available, simulate exploration for demo
            await simulate_exploration(task_id, request)
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()


async def simulate_exploration(task_id: str, request: ExplorationRequest):
    """Simulate exploration for demo purposes."""
    import random
    
    # Simulate progress
    for i in range(10):
        await asyncio.sleep(0.5)
        tasks[task_id]["progress"] = (i + 1) * 10
        tasks[task_id]["pages_visited"] = i + 1
        tasks[task_id]["items_extracted"] = random.randint(1, 5) * (i + 1)
        tasks[task_id]["current_url"] = f"{request.url}/page-{i+1}"
    
    # Generate sample data
    sample_data = [
        {
            "name": "Dr. Jane Smith",
            "title": "Professor of Computer Science",
            "email": "jane.smith@example.edu",
            "department": "Computer Science",
            "research_areas": ["Machine Learning", "NLP", "AI Ethics"],
        },
        {
            "name": "Dr. John Doe",
            "title": "Associate Professor",
            "email": "john.doe@example.edu",
            "department": "Computer Science",
            "research_areas": ["Distributed Systems", "Cloud Computing"],
        },
        {
            "name": "Dr. Alice Johnson",
            "title": "Assistant Professor",
            "email": "alice.j@example.edu",
            "department": "Computer Science",
            "research_areas": ["Computer Vision", "Robotics"],
        },
    ]
    
    tasks[task_id]["status"] = "completed"
    tasks[task_id]["progress"] = 100
    tasks[task_id]["data"] = sample_data
    tasks[task_id]["items_extracted"] = len(sample_data)


@app.get("/explore/{task_id}", response_model=ExplorationStatus)
async def get_exploration_status(task_id: str):
    """Get the status of an exploration task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return ExplorationStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        pages_visited=task["pages_visited"],
        items_extracted=task["items_extracted"],
        current_url=task.get("current_url"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        error=task.get("error"),
    )


@app.get("/explore/{task_id}/results", response_model=ExplorationResult)
async def get_exploration_results(task_id: str):
    """Get the results of a completed exploration."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Task is still {task['status']}. Wait for completion."
        )
    
    started = datetime.fromisoformat(task["started_at"])
    completed = datetime.fromisoformat(task["completed_at"]) if task["completed_at"] else datetime.utcnow()
    duration = (completed - started).total_seconds()
    
    return ExplorationResult(
        task_id=task_id,
        status=task["status"],
        data=task.get("data", []),
        metadata={
            "url": task["request"]["url"],
            "objective": task["request"]["objective"],
            "pages_visited": task["pages_visited"],
        },
        patterns_learned=1,  # Placeholder
        duration_seconds=duration,
    )


@app.post("/demo")
async def run_live_extraction(request: DemoRequest):
    """
    Run a LIVE extraction from the provided URL.
    Uses ToT reasoning when enabled for better accuracy with SLMs.
    """
    try:
        objective = "Extract all relevant structured information from this page including names, titles, descriptions, prices, ratings, quotes, or any other key data points"
        
        if TOT_ENABLED:
            # Use Tree of Thought extraction
            from tot_extractor import extract_from_url_with_tot
            
            result = await extract_from_url_with_tot(
                url=request.url,
                objective=objective,
                use_tot=True,
            )
            
            metadata = result.get("metadata", {})
            
            return {
                "status": "success",
                "message": "ToT extraction completed",
                "url": result["url"],
                "data": result["data"],
                "stats": {
                    "pages_visited": 1,
                    "items_extracted": len(result["data"]),
                    "duration_ms": int(metadata.get("duration_seconds", 0) * 1000),
                    "model": metadata.get("model", MODEL_NAME),
                    "tot_enabled": True,
                    "thoughts_generated": metadata.get("thoughts_generated", 0),
                    "thoughts_tried": metadata.get("thoughts_tried", 0),
                },
                "tot_info": {
                    "best_strategy": metadata.get("best_strategy"),
                    "all_strategies": metadata.get("all_strategies", []),
                },
            }
        else:
            # Use simple extraction
            from extractor import extract_from_url
            
            result = await extract_from_url(
                url=request.url,
                objective=objective,
                target_fields=None,
            )
            
            return {
                "status": "success",
                "message": "Live extraction completed",
                "url": result["url"],
                "data": result["data"],
                "stats": {
                    "pages_visited": 1,
                    "items_extracted": len(result["data"]),
                    "duration_ms": int(result["metadata"]["duration_seconds"] * 1000),
                    "model": result["metadata"]["model"],
                    "tokens_used": result["metadata"]["tokens_used"],
                    "tot_enabled": False,
                },
                "links": result.get("links", []),
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "url": request.url,
            "data": [],
            "stats": {
                "pages_visited": 0,
                "items_extracted": 0,
                "duration_ms": 0,
                "tot_enabled": TOT_ENABLED,
            }
        }


@app.get("/config")
async def get_config():
    """Get current server configuration (non-sensitive)."""
    return {
        "model": MODEL_NAME,
        "model_provider": MODEL_PROVIDER,
        "tot_enabled": TOT_ENABLED,
        "ollama_url": OLLAMA_BASE_URL,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
