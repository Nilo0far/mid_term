
import pandas as pd
import requests

url = "http://127.0.0.1:9697/predict"

data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"

df = pd.read_csv(data_url)

booking = df.iloc[0].drop("is_canceled").to_dict()

booking = {
    k: None if pd.isna(v) else v
    for k, v in booking.items()
}

response = requests.post(url, json=booking, timeout=30)

print("Status code:", response.status_code)
print("Prediction:", response.json())
