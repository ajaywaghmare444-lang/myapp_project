from fastapi import APIRouter, HTTPException, Depends
from app.schemas.agent import QuestionRequest, AnswerResponse
from app.services.llm_service import LLMService, get_llm_service

router = APIRouter(prefix="/agent", tags=["Agent"])

@router.post("/ask", response_model=AnswerResponse)
async def ask_agent(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Send a prompt/question to the agent and receive a structured answer using OpenAI GPT LLM.
    """
    try:
        answer = await llm_service.ask_question(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature
        )
        return AnswerResponse(
            answer=answer,
            model_used=llm_service.model_name
        )
    except ValueError as ve:
        # e.g., missing API key or bad inputs
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # General LLM or execution error
        raise HTTPException(status_code=500, detail=f"LLM Agent processing failed: {str(e)}")
