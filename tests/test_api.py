import pytest
from fastapi.testclient import TestClient
from langchain.messages import AIMessage

from app.config import Settings
from app.main import create_app
from tests.conftest import FailingChatModel, TimedOutChatModel, UnavailableChatModel


def make_client(tmp_path, model) -> TestClient:
    settings = Settings(
        sqlite_path=tmp_path / "preferences.db",
        ollama_timeout_seconds=5,
    )
    return TestClient(create_app(settings, model=model))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": "user-a"},
        {"message": "hello"},
        {"user_id": 123, "message": "hello"},
        {"user_id": "user-a", "message": 123},
        {"user_id": "", "message": "hello"},
        {"user_id": "   ", "message": "hello"},
        {"user_id": "user-a", "message": ""},
        {"user_id": "user-a", "message": "   "},
        {"user_id": "u" * 129, "message": "hello"},
        {"user_id": "user-a", "message": "m" * 4_001},
        {"user_id": "user-a", "message": "hello", "unexpected": True},
    ],
)
def test_rejects_invalid_requests(payload, scripted_model_factory, tmp_path) -> None:
    model = scripted_model_factory(AIMessage(content="unused"))
    with make_client(tmp_path, model) as client:
        response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert model.calls == []


def test_handles_unavailable_ollama_without_leaking_details(tmp_path) -> None:
    with make_client(tmp_path, UnavailableChatModel()) as client:
        response = client.post(
            "/chat",
            json={"user_id": "user-a", "message": "Привет"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Локальная модель недоступна."}
    assert "sensitive connection details" not in response.text


def test_handles_model_timeout_without_leaking_details(tmp_path) -> None:
    with make_client(tmp_path, TimedOutChatModel()) as client:
        response = client.post(
            "/chat",
            json={"user_id": "user-a", "message": "Привет"},
        )

    assert response.status_code == 504
    assert response.json() == {"detail": "Локальная модель не ответила вовремя."}
    assert "sensitive timeout details" not in response.text


def test_invalid_model_completion_returns_bad_gateway(scripted_model_factory, tmp_path) -> None:
    model = scripted_model_factory(AIMessage(content=""))
    with make_client(tmp_path, model) as client:
        response = client.post(
            "/chat",
            json={"user_id": "user-a", "message": "Привет"},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Локальная модель вернула некорректный ответ."}


def test_unexpected_code_error_returns_internal_server_error(tmp_path) -> None:
    with make_client(tmp_path, FailingChatModel()) as client:
        response = client.post(
            "/chat",
            json={"user_id": "user-a", "message": "Привет"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Внутренняя ошибка сервиса."}
    assert "sensitive provider failure" not in response.text
