from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import json
import dill


# load the model
with open('model.pkl', 'rb') as f:
    model = dill.load(f)


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    print(item.json())
    item_df = pd.DataFrame(json.loads(item.json()), index=[0])
    y_pred = model.predict(item_df)
    return y_pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    print([json.loads(item.json()) for item in items])
    items_df = pd.DataFrame.from_dict([json.loads(item.json()) for item in items], orient='columns')
    y_pred = model.predict(items_df)
    return y_pred.tolist()


@app.post("/csv")
def read_csv(file: UploadFile = File(...)):
    # read file
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    # get predictions
    y_pred = model.predict(df)

    df['selling_price'] = y_pred

    # send result
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")

    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
