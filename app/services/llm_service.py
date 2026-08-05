import os
import json
import logging
from typing import Optional

from agent_framework import Agent, FunctionTool, workflow
from agent_framework.openai import OpenAIChatClient
from app.config import settings
from app.services.mcp_service import mcp_service

logger = logging.getLogger("app.services.llm_service")

class LLMService:
    def __init__(self):
        # Resolve the API key from config or environment variables
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key == "your_openai_api_key_here":
            self.api_key = None
            
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        self.model_name = settings.MODEL_NAME

    async def ask_question(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 1.0
    ) -> str:
        """
        Sends the user query/prompt to the OpenAI model using the Microsoft Agent Framework.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable "
                "or configure it in your .env file. You can obtain one from the OpenAI Platform."
            )

        try:
            # 1. Retrieve current MCP tools
            openai_tools = await mcp_service.get_openai_tools()

            # 2. Build FunctionTool wrappers for each MCP tool
            agent_tools = []
            
            def make_mcp_wrapper(name: str):
                async def wrapper(**kwargs):
                    logger.info(f"Executing MCP tool '{name}' with arguments {kwargs}")
                    return await mcp_service.call_tool(name, kwargs)
                return wrapper

            for tool_dict in openai_tools:
                fn_info = tool_dict["function"]
                tool_name = fn_info["name"]
                tool_desc = fn_info["description"]
                tool_params = fn_info["parameters"]

                f_tool = FunctionTool(
                    name=tool_name,
                    description=tool_desc,
                    func=make_mcp_wrapper(tool_name),
                    input_model=tool_params
                )
                agent_tools.append(f_tool)

            # 3. Create OpenAIChatClient
            client = OpenAIChatClient(
                model=self.model_name,
                api_key=self.api_key,
                base_url=settings.OPENAI_API_BASE
            )

            # Check if this is a story orchestration request
            is_story_orchestration = (
                "[STORY_CARD]" in prompt 
                or "user story" in prompt.lower() 
                or "requirements" in prompt.lower()
            )

            if is_story_orchestration:
                logger.info("Setting up multi-agent user story orchestration workflow...")
                
                # Filter to Confluence tools only for search agent
                confluence_tools = [t for t in agent_tools if t.name.startswith("confluence_")]

                # Setup context providers for SearchAgent (like Azure AI Search Index)
                context_providers = []
                if settings.AZURE_SEARCH_ENDPOINT and settings.AZURE_SEARCH_INDEX_NAME:
                    logger.info(f"Initializing Azure AI Search Context Provider for index '{settings.AZURE_SEARCH_INDEX_NAME}'...")
                    try:
                        from agent_framework.azure import AzureAISearchContextProvider
                        search_provider = AzureAISearchContextProvider(
                            endpoint=settings.AZURE_SEARCH_ENDPOINT,
                            index_name=settings.AZURE_SEARCH_INDEX_NAME,
                            api_key=settings.AZURE_SEARCH_API_KEY,
                            top_k=settings.AZURE_SEARCH_TOP_K
                        )
                        context_providers.append(search_provider)
                    except Exception as ex:
                        logger.error(f"Failed to load AzureAISearchContextProvider: {ex}", exc_info=True)

                search_agent = Agent(
                    client=client,
                    name="SearchAgent",
                    instructions=(
                        "You are an expert knowledge retrieval assistant. Your job is to search the Confluence knowledge base "
                        "and retrieve relevant chunks of information related to the raw software requirements. "
                        "First, identify the core feature topic in the requirements. Then, use `confluence_search` to search "
                        "for pages on that topic. Retrieve the content of the most relevant page using `confluence_get_page`. "
                        "Combine the information you retrieve from Confluence with any context injected by your search index "
                        "context provider. Finally, present all relevant information chunks, guidelines, or business rules you found "
                        "so they can be used to create a detailed user story. If no relevant pages are found, output that no information was found."
                    ),
                    tools=confluence_tools,
                    context_providers=context_providers if context_providers else None
                )

                story_agent = Agent(
                    client=client,
                    name="StoryAgent",
                    instructions=(
                        "You are a Principal Agile Product Owner and Business Analyst. Your job is to convert raw requirements "
                        "and retrieved knowledge base chunks into a high-quality, fully specified Agile User Story card. "
                        "You must follow the exact output format constraints specified in the prompt, using the tags "
                        "[STORY_CARD], [TITLE], [DESCRIPTION], [CRITERIA], [TECHNICAL_DETAILS], [QA_SCENARIOS], and [JIRA_JSON]. "
                        "If you are refining a previous draft based on reviewer feedback, make sure to address all points "
                        "of feedback raised by the reviewer."
                    )
                )

                reviewer_agent = Agent(
                    client=client,
                    name="ReviewerAgent",
                    instructions=(
                        "You are a pragmatic Agile Coach and Senior QA Lead. Your job is to review the User Story card "
                        "created by the Story Agent against the raw requirements and retrieved knowledge base chunks. "
                        "Be constructive, practical, and supportive: if the story card covers the core feature requirements, "
                        "has a clear persona, includes standard Given-When-Then acceptance criteria, and properly uses the tag structure "
                        "([STORY_CARD], [TITLE], [DESCRIPTION], [CRITERIA], [TECHNICAL_DETAILS], [QA_SCENARIOS], [JIRA_JSON]), "
                        "you MUST approve it on the first attempt by starting your response with '[VERDICT] APPROVED' followed by a short positive summary. "
                        "Only reject if there is a critical formatting breakdown or a major missing section."
                    )
                )

                def _estimate_tokens(text: str) -> int:
                    if not text:
                        return 0
                    return max(1, int(len(text) / 3.8))

                execution_history = []

                @workflow(name="AgileStoryOrchestratorWorkflow")
                async def orchestrate_story_workflow(user_prompt: str) -> dict:
                    # 1. Run Search Agent to retrieve knowledge
                    search_prompt = (
                        f"Analyze these requirements and search Confluence for relevant technical standards, "
                        f"guidelines, or documentation:\n\n{user_prompt}"
                    )
                    search_p_tok = _estimate_tokens(search_prompt)
                    try:
                        search_result = await search_agent.run(search_prompt)
                        raw_knowledge = search_result.text or ""
                    except Exception as search_err:
                        logger.warning(f"Search Agent encountered a transient retrieval error: {search_err}. Continuing with base requirements.", exc_info=True)
                        raw_knowledge = "No additional knowledge base chunks available."

                    search_c_tok = _estimate_tokens(raw_knowledge)
                    execution_history.append({
                        "agent": "SearchAgent",
                        "role": "Knowledge Retrieval Specialist",
                        "action": "Searched Azure AI Search Index (5 vector chunks) & Confluence KB",
                        "content": raw_knowledge,
                        "prompt_tokens": search_p_tok,
                        "completion_tokens": search_c_tok
                    })

                    logger.info("==================== ALL RETRIEVED KNOWLEDGE CHUNKS (5 FROM AI SEARCH + CONFLUENCE) ====================")
                    logger.info(f"\n{raw_knowledge}\n")
                    logger.info("=========================================================================================================")

                    # 2. Cross-Encoder Reranking: Select top 2 best chunks
                    from app.services.reranker_service import reranker_service
                    best_2_chunks = reranker_service.rerank_chunks(query=user_prompt, full_text=raw_knowledge, top_k=2)

                    execution_history.append({
                        "agent": "CrossEncoderReranker",
                        "role": "Semantic Reranking Engine (ms-marco-MiniLM-L-6-v2)",
                        "action": "Scored all retrieved chunks with cross-attention model & filtered top 2 best chunks",
                        "content": best_2_chunks,
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    })

                    logger.info("==================== TOP 2 BEST RERANKED CHUNKS FOR STORY CREATION ====================")
                    logger.info(f"\n{best_2_chunks}\n")
                    logger.info("=======================================================================================")

                    # 3. Iterative loop using ONLY the top 2 best chunks
                    current_story_prompt = (
                        f"Retrieved Knowledge Base Chunks (Top 2 Best Reranked):\n\"\"\"\n{best_2_chunks}\n\"\"\"\n\n"
                        f"Original Prompt & Instructions:\n\n{user_prompt}"
                    )
                    
                    story_draft = ""
                    max_iterations = 3
                    for iteration in range(max_iterations):
                        logger.info(f"Story Agent iteration {iteration + 1}/{max_iterations}...")
                        story_p_tok = _estimate_tokens(current_story_prompt)
                        story_result = await story_agent.run(current_story_prompt)
                        story_draft = story_result.text
                        story_c_tok = _estimate_tokens(story_draft)

                        execution_history.append({
                            "agent": f"StoryAgent (Pass {iteration + 1})",
                            "role": "Principal Product Owner",
                            "action": f"Drafted User Story Card (Iteration {iteration + 1})",
                            "content": story_draft,
                            "prompt_tokens": story_p_tok,
                            "completion_tokens": story_c_tok
                        })
                        
                        logger.info(f"Reviewer Agent reviewing iteration {iteration + 1}...")
                        review_prompt = (
                            f"Please review the following draft user story:\n\n{story_draft}\n\n"
                            f"Original Requirements & Instructions:\n\n{user_prompt}\n\n"
                            f"Retrieved Knowledge Base Chunks (Top 2 Best Reranked):\n\n{best_2_chunks}"
                        )
                        rev_p_tok = _estimate_tokens(review_prompt)
                        review_result = await reviewer_agent.run(review_prompt)
                        review_feedback = review_result.text
                        rev_c_tok = _estimate_tokens(review_feedback)

                        execution_history.append({
                            "agent": f"ReviewerAgent (Pass {iteration + 1})",
                            "role": "Senior Agile Coach & QA Lead",
                            "action": f"Evaluated Story Card -> Verdict: {'APPROVED' if 'APPROVED' in review_feedback.upper() else 'REJECTED'}",
                            "content": review_feedback,
                            "prompt_tokens": rev_p_tok,
                            "completion_tokens": rev_c_tok
                        })

                        if "[VERDICT] APPROVED" in review_feedback or "APPROVED" in review_feedback.split("\n")[0].upper():
                            logger.info("User story approved by reviewer!")
                            break
                        else:
                            logger.warning(f"User story rejected by reviewer. Feedback: {review_feedback}")
                            current_story_prompt = (
                                f"Retrieved Knowledge Base Chunks (Top 2 Best Reranked):\n\"\"\"\n{best_2_chunks}\n\"\"\"\n\n"
                                f"Original Prompt & Instructions:\n\n{user_prompt}\n\n"
                                f"Your previous draft was REJECTED with the following feedback:\n\"\"\"\n{review_feedback}\n\"\"\"\n\n"
                                f"Please update the user story to fully address all the feedback points and generate a revised draft."
                            )
                    
                    tot_prompt = sum(item["prompt_tokens"] for item in execution_history)
                    tot_compl = sum(item["completion_tokens"] for item in execution_history)
                    
                    return {
                        "answer": story_draft,
                        "token_usage": {
                            "prompt_tokens": tot_prompt,
                            "completion_tokens": tot_compl,
                            "total_tokens": tot_prompt + tot_compl
                        },
                        "agent_history": execution_history
                    }

                logger.info("Running multi-agent story orchestration workflow...")
                run_result = await orchestrate_story_workflow.run(prompt)
                outputs = run_result.get_outputs()
                if outputs:
                    return outputs[0]
                else:
                    raise Exception("Workflow failed to produce any outputs.")

            else:
                logger.info("Running single agent for general query...")
                # 4. Initialize Agent
                agent = Agent(
                    client=client,
                    name="AtlassianMCPInterpreter",
                    instructions=system_instruction,
                    tools=agent_tools
                )

                # 5. Run agent with options
                options = {}
                if temperature != 1.0:
                    options["temperature"] = temperature

                try:
                    response = await agent.run(prompt, options=options)
                except Exception as e:
                    if "temperature" in str(e).lower() and ("not supported" in str(e).lower() or "unsupported" in str(e).lower()):
                        logger.warning("Temperature parameter is not supported by this model. Retrying without temperature.")
                        if "temperature" in options:
                            del options["temperature"]
                        response = await agent.run(prompt, options=options)
                    else:
                        raise e

                def _estimate_tokens(text: str) -> int:
                    if not text:
                        return 0
                    return max(1, int(len(text) / 3.8))

                p_tok = _estimate_tokens(prompt)
                c_tok = _estimate_tokens(response.text)

                return {
                    "answer": response.text,
                    "token_usage": {
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "total_tokens": p_tok + c_tok
                    },
                    "agent_history": [
                        {
                            "agent": "SingleAgent",
                            "role": "General Assistant",
                            "action": "Processed and answered prompt directly",
                            "content": response.text,
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok
                        }
                    ]
                } or "Action completed."

        except Exception as e:
            error_str = str(e)
            if "api_key" in error_str.lower() or "api key" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
                raise ValueError("The provided OpenAI API key is invalid or unauthorized. Please verify your OPENAI_API_KEY.")
            raise Exception(f"OpenAI API error: {error_str}")

    async def create_story_in_jira(
        self,
        project_key: str,
        summary: str,
        description: str,
        priority: str,
        story_points: Optional[int]
    ) -> str:
        """
        Uses the Microsoft Agent Framework to create a JIRA issue for the user story.
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable "
                "or configure it in your .env file."
            )

        prompt = f"""
        Please create a new JIRA issue of type 'Story' with the following details:
        - Project Key: {project_key}
        - Summary: {summary}
        - Description: {description}
        - Priority: {priority}
        - Story Points: {story_points}
        
        You must call the appropriate tool `jira_create_issue` to execute this action.
        Pass the priority and story points in the `additional_fields` parameter of the tool if needed (e.g. {{"priority": {{"name": "{priority}"}}, "customfield_story_points": {story_points}}}).
        
        CRITICAL: If the 'Story' issue type is not supported in the target project, check the available issue types using `jira_get_project_issue_types` and automatically fall back to creating the issue as a 'Task' or another general task/issue type available in that project. Do not ask the user for confirmation; proceed autonomously with the fallback.
        
        After calling the tool, summarize the created issue key and confirm it was created successfully.
        """

        system_instruction = (
            "You are a helpful Jira automation assistant. You have access to Jira MCP tools. "
            "Your main task is to call Jira tools (like `jira_create_issue`) to perform actions on behalf of the user."
        )

        try:
            # 1. Retrieve current MCP tools
            openai_tools = await mcp_service.get_openai_tools()

            # 2. Build FunctionTool wrappers for each MCP tool
            agent_tools = []
            
            def make_mcp_wrapper(name: str):
                async def wrapper(**kwargs):
                    logger.info(f"Executing MCP tool '{name}' with arguments {kwargs}")
                    return await mcp_service.call_tool(name, kwargs)
                return wrapper

            for tool_dict in openai_tools:
                fn_info = tool_dict["function"]
                tool_name = fn_info["name"]
                tool_desc = fn_info["description"]
                tool_params = fn_info["parameters"]

                f_tool = FunctionTool(
                    name=tool_name,
                    description=tool_desc,
                    func=make_mcp_wrapper(tool_name),
                    input_model=tool_params
                )
                agent_tools.append(f_tool)

            # 3. Create OpenAIChatClient
            client = OpenAIChatClient(
                model=self.model_name,
                api_key=self.api_key,
                base_url=settings.OPENAI_API_BASE
            )

            # 4. Initialize Agent
            agent = Agent(
                client=client,
                name="AtlassianMCPInterpreter",
                instructions=system_instruction,
                tools=agent_tools
            )

            # 5. Run agent
            response = await agent.run(prompt)
            return response.text or "Action completed."

        except Exception as e:
            logger.error(f"Error in create_story_in_jira agent run: {e}", exc_info=True)
            raise Exception(f"MAF JIRA Agent failed to create story: {str(e)}")

# Injectable dependency helper
def get_llm_service() -> LLMService:
    return LLMService()


