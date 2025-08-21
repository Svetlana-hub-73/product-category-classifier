import os
import joblib

# Проверка наличия модели
if not os.path.exists("models/product_classifier.pkl") or not os.path.exists("models/vectorizer.pkl"):
    print("Ошибка: модель или векторизатор не найдены. Сначала запустите train_model.py")
    exit()

# Загрузка
model = joblib.load("models/product_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_product_category(product_name):
    X = vectorizer.transform([product_name])
    return model.predict(X)[0]

# Интерактивный режим
while True:
    product_name = input("Введите название продукта (или 'exit' для выхода): ")
    if product_name.lower() == "exit":
        break
    category = predict_product_category(product_name)
    print(f"Предсказанная категория: {category}\n")
