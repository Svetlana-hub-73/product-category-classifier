import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Create a folder for models
os.makedirs("models", exist_ok=True)

# Loading data
df = pd.read_csv("data/raw/products.csv")

# Cleaning the speakers
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Product Title", "Category Label"])
df["Product Title"] = df["Product Title"].str.lower()
df = df.drop_duplicates()

# Split into characteristics and target variable
X = df["Product Title"]
y = df["Category Label"]


# Text vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# Save the model and vectorizer
joblib.dump(model, "models/product_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved successfully!")
