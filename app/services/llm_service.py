import os
import json
import logging
from typing import Optional

from agent_framework import Agent, FunctionTool
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

            return response.text or "Action completed."

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


