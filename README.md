# Hotel Booking Cancellation Prediction

## Project Overview

This project predicts whether a hotel reservation will be canceled using machine learning.

It was developed as part of the **Machine Learning Zoomcamp Midterm Project**.

The goal is to help hotels identify bookings with a high risk of cancellation, allowing them to better understand booking behavior and support operational planning.

## Dataset

The project uses the Hotel Booking Demand dataset, containing **119,390 hotel booking records**.

The dataset includes information such as:

- Hotel type
- Lead time
- Arrival date
- Number of guests
- Meal type
- Country
- Market segment
- Distribution channel
- Deposit type
- Customer type
- Average daily rate (ADR)

**Target variable:** `is_canceled`

- `0`: The booking was not canceled.
- `1`: The booking was canceled.

**Dataset source:**

https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv

The dataset is downloaded automatically when running the training script.

## Machine Learning Approach

The machine learning workflow includes:

1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Missing value handling
4. Feature importance analysis
5. Categorical feature encoding using DictVectorizer
6. Training and comparing classification models
7. Hyperparameter tuning
8. Final model evaluation

### Models Evaluated

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### Final Model

**Selected model:** XGBoost

**Test ROC AUC:** Approximately **0.9350**

The final model was trained using the combined training and validation datasets and evaluated on a held-out test dataset.

The trained model and preprocessing encoder are saved as:

- `model.json`: Trained XGBoost model
- `dv.bin`: Fitted DictVectorizer

## Project Structure

```text
mid_term/
├── README.md
├── .gitignore
└── mid_term/
    ├── mid_term.ipynb
    ├── train.py
    ├── predict.py
    ├── test.py
    ├── requirements.txt
    ├── Dockerfile
    ├── model.json
    └── dv.bin
```

## Requirements

- Python 3.12 (used in the Docker image)
- Docker (for containerized deployment)

The main Python libraries are:

- pandas
- numpy
- scikit-learn
- xgboost
- flask
- gunicorn
- requests

## Installation

Clone the repository:

```bash
git clone https://github.com/Nilo0far/mid_term.git
```

Navigate to the application directory:

```bash
cd mid_term/mid_term
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model from scratch, run:

```bash
python train.py
```

The training script:

1. Downloads the hotel booking dataset.
2. Removes selected columns.
3. Splits the dataset into training, validation, and test sets.
4. Preprocesses numerical and categorical features.
5. Trains the final XGBoost classification model.
6. Saves the trained model and DictVectorizer.

The generated files are:

```text
model.json
dv.bin
```

The complete exploratory analysis, model comparison, and experimentation are available in `mid_term.ipynb`.

## Running the Prediction API Locally

Start the Flask application:

```bash
python predict.py
```

The API will run at:

```text
http://127.0.0.1:9696/predict
```

The `/predict` endpoint accepts a POST request containing hotel booking information in JSON format.

The API returns:

- `cancellation_probability`: Predicted probability of cancellation
- `is_canceled`: Boolean cancellation prediction using a threshold of 0.5

## Running the API with Docker

Docker allows the application to run in a reproducible environment without manually installing the Python dependencies.

### Step 1: Build the Docker Image

Navigate to the directory containing the Dockerfile:

```bash
cd mid_term/mid_term
```

If you already followed the installation instructions above, you are already in this directory.

Build the image:

```bash
docker build -t hotel-cancellation .
```

### Step 2: Run the Docker Container

Run:

```bash
docker run --rm -p 9697:9696 hotel-cancellation
```

The application runs inside the container on port `9696` and is accessible from the host on port `9697`.

The prediction endpoint is:

```text
http://127.0.0.1:9697/predict
```

Keep the Docker container running while testing the API.

## Testing the Docker API

A test script named `test.py` is provided to demonstrate how to send a booking to the prediction service.

### Step 1: Start Docker

Make sure the Docker container is running:

```bash
docker run --rm -p 9697:9696 hotel-cancellation
```

### Step 2: Run the Test Script

Open another terminal and navigate to the application directory.

Run:

```bash
python test.py
```

The test script:

1. Downloads the original hotel booking dataset.
2. Selects one booking record.
3. Removes the target variable (`is_canceled`).
4. Converts missing values into JSON-compatible values.
5. Sends a POST request to the Docker API.
6. Prints the prediction result.

### Expected Output

```text
Status code: 200
Prediction: {'cancellation_probability': 0.30729952454566956, 'is_canceled': False}
```

The result indicates that the model predicts approximately a **30.73% probability of cancellation** for the sample booking.

Since the probability is below the default threshold of 0.5, the predicted cancellation decision is `False`.

**Note:** The test script uses port `9697`, so the Docker container must be running with the port mapping shown above.

## API Response Format

The API returns a JSON response containing the predicted probability and classification result.

Example:

```json
{
  "cancellation_probability": 0.30729952454566956,
  "is_canceled": false
}
```

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
- Git and GitHub

## Results and Conclusion

The project compares multiple machine learning classification algorithms for predicting hotel booking cancellations.

The final XGBoost model achieved a **test ROC AUC of approximately 0.9350**.

The model is exported as a Flask REST API and can be deployed locally using Docker and Gunicorn.

The successful Docker API test confirms that the saved model can be loaded and used to generate predictions in a containerized environment.

## Repository

GitHub: https://github.com/Nilo0far/mid_term