import os
import sys
import logging
import asyncio
from typing import Optional, List, Dict, Any
# pyrefly: ignore [missing-import]
from mcp import ClientSession, StdioServerParameters
# pyrefly: ignore [missing-import]
from mcp.client.stdio import stdio_client
from app.config import settings

logger = logging.getLogger("app.services.mcp_service")

class MCPService:
    def __init__(self):
        self._session: Optional[ClientSession] = None
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            if self._session is not None:
                return

            if not settings.ATLASSIAN_BASE_URL or not settings.ATLASSIAN_EMAIL or not settings.ATLASSIAN_API_TOKEN:
                logger.warning("Atlassian credentials not configured. Atlassian MCP client will remain inactive.")
                return

            # Resolve the path to the executable to be robust on Windows/Linux inside the virtualenv
            bindir = os.path.dirname(sys.executable)
            command = os.path.join(bindir, "mcp-atlassian")
            if sys.platform == "win32":
                command_win = f"{command}.exe"
                if os.path.exists(command_win):
                    command = command_win
                else:
                    command = "mcp-atlassian"
            else:
                if not os.path.exists(command):
                    command = "mcp-atlassian"

            base_url = settings.ATLASSIAN_BASE_URL.rstrip("/")
            confluence_url = f"{base_url}/wiki" if "atlassian.net" in base_url else base_url

            server_params = StdioServerParameters(
                command=command,
                args=[],
                env={
                    "JIRA_URL": base_url,
                    "JIRA_USERNAME": settings.ATLASSIAN_EMAIL,
                    "JIRA_API_TOKEN": settings.ATLASSIAN_API_TOKEN,
                    "CONFLUENCE_URL": confluence_url,
                    "CONFLUENCE_USERNAME": settings.ATLASSIAN_EMAIL,
                    "CONFLUENCE_API_TOKEN": settings.ATLASSIAN_API_TOKEN,
                    "PATH": os.environ.get("PATH", "")
                }
            )

            logger.info(f"Starting Atlassian MCP server subprocess via {command}...")
            try:
                # Initialize stdio context manager
                self._client_context = stdio_client(server_params)
                self._read_stream, self._write_stream = await self._client_context.__aenter__()
                
                # Create and enter ClientSession context
                self._session = ClientSession(self._read_stream, self._write_stream)
                await self._session.__aenter__()
                
                # Initialize standard MCP handshake
                await self._session.initialize()
                logger.info("Successfully initialized Atlassian MCP Client and server subprocess.")
            except Exception as e:
                logger.error(f"Failed to start/connect to Atlassian MCP server: {e}", exc_info=True)
                await self._reset_internal_state()

    async def _reset_internal_state(self):
        self._session = None
        self._client_context = None
        self._read_stream = None
        self._write_stream = None

    async def _ensure_session(self):
        if self._session is None:
            logger.warning("Atlassian MCP session is currently down. Triggering startup...")
            await self.start()
            if self._session is None:
                raise RuntimeError("Atlassian credentials are not configured or the server process failed to launch.")

    async def stop(self):
        async with self._lock:
            if self._session is not None:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error exiting MCP session during shutdown: {e}")
                self._session = None
            if self._client_context is not None:
                try:
                    await self._client_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error exiting MCP stdio client context during shutdown: {e}")
                self._client_context = None
            await self._reset_internal_state()
            logger.info("Atlassian MCP Client stopped.")

    async def get_openai_tools(self) -> List[Dict[str, Any]]:
        try:
            await self._ensure_session()
        except Exception as e:
            logger.warning(f"Could not load Atlassian tools: {e}")
            return []
        
        try:
            tools_resp = await self._session.list_tools()
            openai_tools = []
            for tool in tools_resp.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            return openai_tools
        except Exception as e:
            logger.error(f"Failed to fetch tools from Atlassian MCP server: {e}", exc_info=True)
            return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            await self._ensure_session()
        except Exception as e:
            raise RuntimeError(f"Atlassian MCP Service is not active: {e}")
        
        try:
            res = await self._session.call_tool(name, arguments)
            text_content = ""
            for block in res.content:
                if block.type == "text":
                    text_content += block.text
            return text_content
        except (ConnectionError, BrokenPipeError, RuntimeError) as e:
            logger.warning(f"Connection lost to Atlassian MCP server during '{name}' execution: {e}. Attempting self-healing restart...")
            async with self._lock:
                await self._reset_internal_state()
            try:
                await self._ensure_session()
                logger.info(f"Re-executing MCP tool '{name}' after successful self-healing restart...")
                res = await self._session.call_tool(name, arguments)
                text_content = ""
                for block in res.content:
                    if block.type == "text":
                        text_content += block.text
                return text_content
            except Exception as restart_exc:
                logger.error(f"Self-healing restart failed during tool execution: {restart_exc}", exc_info=True)
                raise RuntimeError(f"Connection lost and auto-restart failed for tool '{name}': {restart_exc}")
        except Exception as e:
            raise RuntimeError(f"Error calling MCP tool '{name}': {e}")

mcp_service = MCPService()
