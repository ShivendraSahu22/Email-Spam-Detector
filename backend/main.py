from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict_output, model
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model = PredictionResponse)
def predict(data: UserInput):

    try:

        prediction = predict_output(data)

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e