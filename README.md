# FastAPI Agent Backend with OpenAI GPT LLM

This is a premium, lightweight, and extensible FastAPI backend that implements an LLM-powered agent using the official `openai` SDK.

## Features

- **FastAPI Framework**: High performance, type-safe API with automatic OpenAPI documentation.
- **Official OpenAI SDK**: Integrates the `openai` SDK using modern chat completion patterns (e.g. `gpt-4o-mini`).
- **Asynchronous Execution**: Native async execution using `AsyncOpenAI`, preventing event loop blocking.
- **Robust Error Handling**: Gracefully handles missing, unauthorized, or invalid API key situations.
- **Pydantic Validation**: Strictly validates request and response payloads.

---

## Project Structure

```
myapp_project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint, middleware, and route registration
│   ├── config.py            # Environment configuration settings (pydantic-settings)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── agent.py         # Request and Response Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm_service.py   # LLM interaction wrapper utilizing openai
│   └── api/
│       ├── __init__.py
│       └── endpoints/
│           ├── __init__.py
│           └── agent.py     # Agent API route handler
├── requirements.txt         # Project package requirements
├── .env                     # Local environment variables (API Keys, Server Settings)
├── .env.example             # Example environment variable file
└── README.md                # This setup & walkthrough guide
```

---

## Getting Started

### 1. Prerequisites

Make sure you have **Python 3.10+** installed on your machine.

### 2. Set Up Virtual Environment

Open your terminal in the project root (`c:\myapp_project`) and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (CMD):
.\venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API Key from [OpenAI Platform](https://platform.openai.com/).
   ```env
   OPENAI_API_KEY=sk-...
   ```

---

## Running the Server

Start the FastAPI application with:

```bash
uvicorn app.main:app --reload
```

The server will start on `http://127.0.0.1:8000`.

---

## Testing the API

### 1. Swagger UI Docs (Recommended)
FastAPI automatically generates interactive documentation. Visit:
- **Interactive docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. cURL Example
You can send a POST request to ask the agent a question:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agent/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is the meaning of life?",
       "system_instruction": "Answer in a philosophical yet concise way in under 2 sentences.",
       "temperature": 0.7
     }'
```

#### Example Response:
```json
{
  "answer": "The meaning of life is a subjective quest for purpose, connection, and understanding, defined by how we choose to live and impact others. In a grander sense, it is to experience existence consciously and create our own value in a vast universe.",
  "model_used": "gpt-4o-mini",
  "status": "success"
}
```

### 3. Health Check
To check the status of your backend:
```bash
curl http://127.0.0.1:8000/health
```
