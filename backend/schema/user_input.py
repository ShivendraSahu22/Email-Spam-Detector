from pydantic import BaseModel, Field, field_validator
from typing import Annotated


class UserInput(BaseModel):
    message: Annotated[
        str,
        Field(description="The message to be analyzed for prediction")
    ]

    @field_validator("message")
    @classmethod
    def validate_message(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be empty or whitespace.")
        return value