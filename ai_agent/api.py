
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agent import BayesianCreditRiskAgent


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(
        min_length=1,
        max_length=4000
    )


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000
    )

    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=6
    )


class ChatResponse(BaseModel):
    answer: str
    model: str
    tools_used: list[dict]


@lru_cache(maxsize=1)
def get_agent():
    return BayesianCreditRiskAgent()


app = FastAPI(
    title="Bayesian Credit Risk Intelligence API",
    description=(
        "Evidence-grounded Bayesian credit risk assistant"
    ),
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Bayesian Credit Risk Intelligence"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await run_in_threadpool(
            get_agent().chat,
            request.question,
            [
                item.model_dump()
                for item in request.history
            ]
        )

        return ChatResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service could not complete the request. "
                "Check the backend logs."
            )
        )


@app.get("/")
def root():
    return {
        "application": "Bayesian Credit Risk Intelligence",
        "agent": "Google Gemini",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "chat": "/chat"
    }