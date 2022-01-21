from pydantic import BaseModel

# pydantic models
class StockIn(BaseModel):
    ticker: str

class StockOut(StockIn):
    forecast: dict
