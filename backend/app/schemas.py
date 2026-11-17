import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def lowercase_email(cls, v: str) -> str:
        return v.lower()

    @field_validator("password")
    @classmethod
    def bcrypt_byte_limit(cls, v: str) -> str:
        # bcrypt only looks at the first 72 *bytes*; reject rather than silently truncate.
        if len(v.encode()) > 72:
            raise ValueError("Password must be at most 72 bytes")
        return v


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error: str | None
    num_chunks: int
    num_pages: int | None
    created_at: datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    # Optionally restrict the search to specific documents.
    document_ids: list[uuid.UUID] | None = None


class SourceOut(BaseModel):
    n: int  # citation number used in the answer, e.g. [1]
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    score: float  # cosine similarity, higher = more similar
    snippet: str
