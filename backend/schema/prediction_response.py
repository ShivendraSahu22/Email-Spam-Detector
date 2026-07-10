from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="The predicted class",
        examples=["NOT SPAM"]
    )

    confidence: float = Field(
        ...,
        description="Prediction confidence",
        examples=[0.98]
    )