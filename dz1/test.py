import dill
from sklearn.metrics import r2_score, mean_squared_error as MSE
import pandas as pd
import json
import requests
from io import BytesIO


df_test = pd.read_csv('cars_test.csv')
with open('model.pkl', 'rb') as f:
    model = dill.load(f)


def test_predict_item():
    url = 'http://localhost:8000/predict_item'
    data = json.loads(df_test.drop(columns=['selling_price']).iloc[0].to_json())
    response = requests.post(url, json=data)
    print(response.json())
    y_pred = model.predict(df_test.drop(columns=['selling_price']).head(1))
    assert([float(response.json())] == y_pred)


def test_predict_items():
    url = 'http://localhost:8000/predict_items'
    data = json.loads(df_test.drop(columns=['selling_price']).head(2).to_json(orient='records'))
    response = requests.post(url, json=data)
    print(response.json())
    y_pred = model.predict(df_test.drop(columns=['selling_price']).head(2))
    assert((response.json()==y_pred).all())


def test_csv():
    url = 'http://localhost:8000/csv'
    with open("10.csv", "rb") as f:
        file_data = {"file": f}
        response = requests.post(url, files=file_data)
        df_response = pd.read_csv(BytesIO(response.content), index_col=0)
        print(df_response)
        y_pred = model.predict(df_response.drop(columns=['selling_price']))
        assert((df_response['selling_price'].values==y_pred).all())


test_predict_item()
test_predict_items()
test_csv()
