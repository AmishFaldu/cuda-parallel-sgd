import time
import pandas as pd
from sklearn.linear_model import SGDRegressor

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('./data/train_truncated.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Train a linear regression model
print("Training the linear regression model...")
model = SGDRegressor(penalty=None, max_iter=18, tol=1e-3, random_state=42)

# Measure the time taken to fit the model
print("Fitting the model...")
start_time = time.time()
model.fit(df.drop(columns=['fare_amount']), df['fare_amount'])
end_time = time.time()
print(f"Model fitting took {end_time - start_time:.2f} seconds")

print(f"Model iterations: {model.n_iter_}")
