from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict_message, model
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):
    try:
        return predict_message(data.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))