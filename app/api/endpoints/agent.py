from fastapi import APIRouter, HTTPException, Depends
from app.schemas.agent import QuestionRequest, AnswerResponse, CreateStoryRequest, CreateStoryResponse
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
        result = await llm_service.ask_question(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature
        )
        if isinstance(result, dict):
            return AnswerResponse(
                answer=result.get("answer", ""),
                model_used=llm_service.model_name,
                token_usage=result.get("token_usage"),
                agent_history=result.get("agent_history", [])
            )
        else:
            return AnswerResponse(
                answer=str(result),
                model_used=llm_service.model_name
            )
    except ValueError as ve:
        # e.g., missing API key or bad inputs
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # General LLM or execution error
        raise HTTPException(status_code=500, detail=f"LLM Agent processing failed: {str(e)}")

@router.post("/create_story", response_model=CreateStoryResponse)
async def create_story(
    request: CreateStoryRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Instruct the MAF agent to create a user story in JIRA.
    """
    try:
        result = await llm_service.create_story_in_jira(
            project_key=request.project_key,
            summary=request.summary,
            description=request.description,
            priority=request.priority,
            story_points=request.story_points
        )
        
        # Parse result to extract issue key
        import re
        project_prefix = request.project_key.upper()
        match = re.search(rf"\b({project_prefix}-\d+)\b", result)
        if match:
            issue_key = match.group(1)
        else:
            # Fallback to general JIRA key regex: [A-Z]+-\d+
            match_any = re.search(r"\b([A-Z]+-\d+)\b", result)
            if match_any:
                issue_key = match_any.group(1)
            else:
                issue_key = f"Unknown ({result[:30]}...)"
        
        from app.config import settings
        base_url = settings.ATLASSIAN_BASE_URL or "https://atlassian.net"
        if not base_url.endswith("/"):
            base_url += "/"
        
        issue_url = f"{base_url}browse/{issue_key}" if "Unknown" not in issue_key else base_url
        
        return CreateStoryResponse(
            issue_key=issue_key,
            issue_url=issue_url,
            status="success"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create story: {str(e)}")

