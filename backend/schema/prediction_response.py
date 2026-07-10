from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="The predicted output based on the input message",
        example="spam"
    )