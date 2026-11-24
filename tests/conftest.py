from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from langchain.messages import AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _response_index: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def bound_tools(self) -> list[Any]:
        return self._bound_tools

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        del tool_choice, kwargs
        self._bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._calls.append(list(messages))
        if self._response_index >= len(self.responses):
            raise RuntimeError("No scripted response remains")
        response = self.responses[self._response_index]
        self._response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FailingChatModel(ScriptedChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise RuntimeError("sensitive provider failure")


class UnavailableChatModel(ScriptedChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
        raise httpx.ConnectError("sensitive connection details", request=request)


class TimedOutChatModel(ScriptedChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
        raise httpx.ReadTimeout("sensitive timeout details", request=request)


@pytest.fixture
def scripted_model_factory():
    def factory(*responses: AIMessage) -> ScriptedChatModel:
        return ScriptedChatModel(responses=list(responses))

    return factory
