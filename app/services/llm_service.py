import os
import json
import logging
from typing import Optional

from openai import AsyncOpenAI
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
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable "
                    "or configure it in your .env file. You can obtain one from the OpenAI Platform."
                )
            # Configure production client with standard timeouts and auto-retries
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=settings.OPENAI_API_BASE,
                timeout=30.0,
                max_retries=3
            )
        return self._client

    async def ask_question(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 1
    ) -> str:
        """
        Sends the user query/prompt to the OpenAI model and handles Atlassian MCP tool execution in a loop.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        # Accessing self.client triggers API key verification
        client = self.client

        # Initialize messages list for LLM context
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        # Prevent infinite loop cases (safety ceiling)
        max_iterations = 8

        try:
            for iteration in range(max_iterations):
                # Retrieve mapped tools from the active Atlassian MCP service
                openai_tools = await mcp_service.get_openai_tools()

                request_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                )
                if temperature != 1.0:
                    request_kwargs["temperature"] = temperature
                if openai_tools:
                    request_kwargs["tools"] = openai_tools
                    request_kwargs["tool_choice"] = "auto"

                # Request response/actions from LLM
                response = await client.chat.completions.create(**request_kwargs)
                
                if not response.choices or not response.choices[0].message:
                    return "The agent did not generate any text response."
                
                assistant_message = response.choices[0].message
                
                # Format assistant message for history
                message_dict = {
                    "role": "assistant",
                    "content": assistant_message.content
                }
                if assistant_message.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                
                messages.append(message_dict)

                # If no tool calls are requested, we're done and can return the response text
                if not assistant_message.tool_calls:
                    return assistant_message.content or "Action completed."

                # Execute requested tools
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        logger.error(f"Error parsing tool arguments for {tool_name}: {e}")
                        args = {}

                    logger.info(f"Executing MCP tool call: {tool_name} with args {args}")
                    try:
                        tool_result = await mcp_service.call_tool(tool_name, args)
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                        tool_result = f"Error executing tool '{tool_name}': {e}"

                    # Append tool result to context
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result
                    })

            logger.warning(f"Safety turn limit of {max_iterations} reached in ask_question. Halting loop.")
            return f"Agent execution paused after reaching safety limit of {max_iterations} turns. Please refine your query."

        except Exception as e:
            error_str = str(e)
            if "api_key" in error_str.lower() or "api key" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
                raise ValueError("The provided OpenAI API key is invalid or unauthorized. Please verify your OPENAI_API_KEY.")
            raise Exception(f"OpenAI API error: {error_str}")

# Injectable dependency helper
def get_llm_service() -> LLMService:
    return LLMService()
