# Hotel Booking Cancellation Prediction

## Project Overview

This project predicts whether a hotel reservation will be canceled using machine learning.

The project was developed as part of the **Machine Learning Zoomcamp Midterm Project**.

The goal is to help hotels identify bookings with a high risk of cancellation.

## Dataset

The dataset contains 119,390 hotel booking records, including information about:

- Hotel type
- Lead time
- Number of guests
- Booking dates
- Market segment
- Deposit type
- Customer type
- Average daily rate (ADR)

**Target variable:** `is_canceled`

- `0`: Booking was not canceled
- `1`: Booking was canceled

**Dataset source:**

https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv

## Machine Learning Approach

The project includes:

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature importance analysis
4. Training and comparing multiple classification models
5. Hyperparameter tuning
6. Final model evaluation

The models evaluated include:

- Logistic Regression
- Random Forest
- XGBoost

**Final model:** XGBoost

**Test ROC AUC:** Approximately 0.9350

## Project Structure

```text
mid_term/
├── README.md
├── .gitignore
└── mid_term/
    ├── mid_term.ipynb
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    ├── Dockerfile
    ├── model.json
    └── dv.bin
```

## Installation

Clone the repository and navigate to the application directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run:

```bash
python train.py
```

This script downloads the dataset, preprocesses the data, trains the final XGBoost model, and saves:

- `model.json`: Trained XGBoost model
- `dv.bin`: Fitted DictVectorizer

## Running the Prediction API

Start the Flask application:

```bash
python predict.py
```

The API will be available at:

```text
http://127.0.0.1:9696/predict
```

The `/predict` endpoint accepts a JSON object containing hotel booking features and returns:

- `cancellation_probability`: Predicted cancellation probability
- `is_canceled`: Predicted cancellation decision using a threshold of 0.5

## Running with Docker

Navigate to the directory containing the Dockerfile.

Build the Docker image:

```bash
docker build -t hotel-cancellation .
```

Run the container:

```bash
docker run --rm -p 9697:9696 hotel-cancellation
```

The Dockerized prediction API will be available at:

```text
http://127.0.0.1:9697/predict
```

## Example Prediction

A successful API response looks like:

```json
{
  "cancellation_probability": 0.30729952454566956,
  "is_canceled": false
}
```

This means the model estimates approximately a 30.73% probability of cancellation for the example booking.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- Gunicorn
- Docker
- Jupyter Notebook

## Conclusion

The final XGBoost model achieved a test ROC AUC of approximately 0.9350.

The model is deployed as a REST API using Flask and can also run inside a Docker container.