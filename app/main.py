import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from langchain_core.language_models import BaseChatModel

from app.agent import (
    AgentResponseError,
    AgentStorageError,
    AgentTimeoutError,
    AgentUnavailableError,
    PreferenceAgent,
    build_ollama_model,
)
from app.config import Settings
from app.schemas import ChatRequest, ChatResponse
from app.storage import PreferenceStorage, StorageError

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    storage: PreferenceStorage | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_storage = storage or PreferenceStorage(resolved_settings.sqlite_path)
    resolved_model = model or build_ollama_model(resolved_settings)
    agent = PreferenceAgent(resolved_model, resolved_storage, resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            resolved_storage.initialize()
        except StorageError:
            logger.exception("Application startup failed: preference storage is unavailable")
            raise
        yield

    application = FastAPI(
        title="Preference Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.preference_storage = resolved_storage
    application.state.preference_agent = agent

    @application.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            response = await agent.chat(request.user_id, request.message)
        except AgentTimeoutError as exc:
            logger.warning("The local model request timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Локальная модель не ответила вовремя.",
            ) from exc
        except AgentStorageError as exc:
            logger.exception("Preference storage operation failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Хранилище предпочтений временно недоступно.",
            ) from exc
        except AgentResponseError as exc:
            logger.warning("The agent returned an invalid result: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Локальная модель вернула некорректный ответ.",
            ) from exc
        except AgentUnavailableError as exc:
            logger.exception("The local model request failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Локальная модель недоступна.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error while handling a chat request")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Внутренняя ошибка сервиса.",
            ) from exc

        return ChatResponse(response=response)

    return application


app = create_app()
