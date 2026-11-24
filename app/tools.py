import json
from dataclasses import dataclass
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from pydantic import Field, StringConstraints

from app.storage import PreferenceStorage, StorageError


@dataclass(frozen=True, slots=True)
class AgentContext:
    user_id: str
    storage: PreferenceStorage


PreferenceCategory = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    ),
    Field(
        description=(
            "Stable semantic category in lowercase English snake_case; reuse an existing "
            "equivalent category instead of creating a synonym"
        )
    ),
]
PreferenceValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    Field(
        min_length=1,
        max_length=500,
        description="Current preference value explicitly stated by the user",
    ),
]


@tool
def save_preference(
    category: PreferenceCategory,
    value: PreferenceValue,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Save or update one explicit, durable preference of the current user.

    Use this whenever the current user states, changes, or confirms a lasting preference.
    Call get_preferences first so an existing semantically equivalent category is reused.
    The current user identity is injected by the server and cannot be selected by the model.
    """
    try:
        preference = runtime.context.storage.save_preference(
            user_id=runtime.context.user_id,
            category=category,
            value=value.strip(),
        )
    except StorageError:
        return json.dumps({"ok": False, "error": "storage_unavailable"})

    return json.dumps(
        {
            "ok": True,
            "status": "saved",
            "category": preference.category,
            "value": preference.value,
        },
        ensure_ascii=False,
    )


@tool
def get_preferences(runtime: ToolRuntime[AgentContext]) -> str:
    """Return all durable preferences saved for the current user only.

    Always use this before preference-based advice, before answering what the user likes or
    prefers, and before save_preference. An empty result is the only reliable indication that
    the current user's preference is unknown. The tool takes no user identity argument.
    """
    try:
        preferences = runtime.context.storage.get_preferences(runtime.context.user_id)
    except StorageError:
        return json.dumps({"ok": False, "error": "storage_unavailable"})

    return json.dumps(
        {
            "ok": True,
            "preferences": [
                {"category": preference.category, "value": preference.value}
                for preference in preferences
            ],
        },
        ensure_ascii=False,
    )
