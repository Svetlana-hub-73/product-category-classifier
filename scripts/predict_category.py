import os
import joblib

# Checking for a model
if not os.path.exists("models/product_classifier.pkl") or not os.path.exists("models/vectorizer.pkl"):
    print("mistake: No model or vectorizer found. Will start first train_model.py")
    exit()

# loading
model = joblib.load("models/product_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_product_category(product_name):
    X = vectorizer.transform([product_name])
    return model.predict(X)[0]

# interactive mode
while True:
    product_name = input("Enter product name (or'exit 'to exit): ")
    if product_name.lower() == "exit":
        break
    category = predict_product_category(product_name)
    print(f"Predicted category: {category}\n")
