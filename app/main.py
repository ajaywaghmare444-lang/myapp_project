import logging
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
import uvicorn

from app.config import settings
from app.api.endpoints.agent import router as agent_router
from app.services.mcp_service import mcp_service

# Configure structured production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the Atlassian MCP server process and establish client session
    logger.info("Initializing application lifespan...")
    await mcp_service.start()
    yield
    # Shutdown: Stop and clean up the MCP client subprocess
    logger.info("Tearing down application lifespan...")
    await mcp_service.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A FastAPI backend hosting an intelligent agent integrated with OpenAI GPT LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up CORS middleware to restrict cross-origin requests in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global unhandled exception handler to protect backend responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected server error occurred. Please try again later."
        }
    )

# Include the agent endpoints router
app.include_router(agent_router, prefix=settings.API_PREFIX)

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to ensure the service is running.
    """
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "model_configured": settings.MODEL_NAME,
        "api_key_configured": settings.OPENAI_API_KEY is not None and len(settings.OPENAI_API_KEY) > 0 and settings.OPENAI_API_KEY != "your_openai_api_key_here"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
