import os
from typing import Optional

from openai import AsyncOpenAI
from app.config import settings

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
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=settings.OPENAI_API_BASE
            )
        return self._client

    async def ask_question(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.7
    ) -> str:
        """
        Sends the user query/prompt to the OpenAI model and returns the text response.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        # Accessing self.client triggers API key verification
        client = self.client

        try:
            # Call openai chat completion endpoint asynchronously
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            
            if not response.choices or not response.choices[0].message.content:
                return "The agent did not generate any text response."
            
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "api_key" in error_str.lower() or "api key" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
                raise ValueError("The provided OpenAI API key is invalid or unauthorized. Please verify your OPENAI_API_KEY.")
            raise Exception(f"OpenAI API error: {error_str}")

# Injectable dependency helper
def get_llm_service() -> LLMService:
    return LLMService()
