import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Создаем папку для моделей
os.makedirs("models", exist_ok=True)

# Загружаем данные
df = pd.read_csv("data/raw/products.csv")

# Чистим колонки
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Product Title", "Category Label"])
df["Product Title"] = df["Product Title"].str.lower()
df = df.drop_duplicates()

# Разделение на признаки и целевую переменную
X = df["Product Title"]
y = df["Category Label"]

# Векторизация текста
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# Сохраняем модель и векторизатор
joblib.dump(model, "models/product_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Модель и векторизатор успешно сохранены!")
