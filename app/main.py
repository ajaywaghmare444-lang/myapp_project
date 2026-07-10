from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.api.endpoints.agent import router as agent_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A FastAPI backend hosting an intelligent agent integrated with OpenAI GPT LLM.",
    version="1.0.0",
)

# Set up CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
