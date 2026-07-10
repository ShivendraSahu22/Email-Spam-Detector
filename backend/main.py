from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_output, model
from schema.user_input import UserInput

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
        return {"error": str(e)}