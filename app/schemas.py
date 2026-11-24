from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4_000)


class ChatResponse(BaseModel):
    response: str
