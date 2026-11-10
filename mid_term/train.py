import pandas as pd
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Load dataset
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"

df = pd.read_csv(url)

print("Dataset loaded:", df.shape)

# Remove columns that are not used for prediction
columns_to_drop = [
    'company',
    'reservation_status',
    'reservation_status_date',
    'assigned_room_type',
    'booking_changes',
    'days_in_waiting_list'
]

df_model = df.drop(columns=columns_to_drop)

print("Dataset after dropping columns:", df_model.shape)

# Split dataset into train, validation and test
df_full_train, df_test = train_test_split(
    df_model,
    test_size=0.2,
    random_state=42,
    stratify=df_model['is_canceled']
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=42,
    stratify=df_full_train['is_canceled']
)

print("Train:", df_train.shape)
print("Validation:", df_val.shape)
print("Test:", df_test.shape)

# Combine train and validation
df_full_train = pd.concat([df_train, df_val])

# Separate features and target
X_full_train = df_full_train.drop(columns=['is_canceled']).copy()
y_full_train = df_full_train['is_canceled'].values

print("Full Train Features:", X_full_train.shape)
print("Full Train Target:", y_full_train.shape)

# Select booking features
booking_features = [
    'hotel',
    'lead_time',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type',
    'deposit_type',
    'agent',
    'customer_type',
    'adr'
]

X_full_train = X_full_train[booking_features].copy()

# Handle missing values
X_full_train['children'] = X_full_train['children'].fillna(0)
X_full_train['country'] = X_full_train['country'].fillna('unknown')
X_full_train['agent'] = (
    X_full_train['agent']
    .fillna(-1)
    .astype(int)
    .astype(str)
)

print("Selected features:", X_full_train.shape)
print("Missing values:", X_full_train.isnull().sum().sum())

# Convert features to dictionaries
train_dicts = X_full_train.to_dict(orient='records')

# Encode categorical features
dv = DictVectorizer(sparse=False)

X_train_encoded = dv.fit_transform(train_dicts)

print("Encoded features:", X_train_encoded.shape)

# Create XGBoost DMatrix
dtrain = xgb.DMatrix(
    X_train_encoded,
    label=y_full_train
)

# Final model parameters
params = {
    'eta': 0.3,
    'max_depth': 7,
    'min_child_weight': 5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42,
    'nthread': 4
}

# Train final model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=265
)

print("Model trained successfully!")
print("Boosting rounds:", model.num_boosted_rounds())

# Save the trained model
model.save_model('model.json')

# Save the DictVectorizer
with open('dv.bin', 'wb') as f:
    pickle.dump(dv, f)

print("Model and DictVectorizer saved successfully!")