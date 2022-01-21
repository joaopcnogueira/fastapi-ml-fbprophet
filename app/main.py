from fastapi import FastAPI, HTTPException, Response, status
from app import schemas

from model import train, predict, convert

app = FastAPI()


@app.post("/predict", response_model=schemas.StockOut, status_code=200)
def get_prediction(payload: schemas.StockIn):
    ticker = payload.ticker

    prediction_list = predict(ticker)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found. Please train it using the train endpoint. After that, you will be able to make predictions.")

    response_object = {"ticker": ticker, "forecast": convert(prediction_list)}
    return response_object

@app.post("/train")
def train_model(payload: schemas.StockIn):
    print(payload.ticker)
    ticker = payload.ticker
    train(ticker=ticker)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
