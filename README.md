# product-category-classifier
Sklearn project: automatic classification of goods into categories (30k + products). 


This project is a system for classifying goods into categories using machine learning (Scikit-Learn).

# # → → How to start a project
```bash
git clone https://github.com/Svetlana-hub-73/product-category-classifier.git
cd product-category-classifier
pip install -r requirements.txt


# Класификатор категорија производа 🛒

## 📌 Опис пројекта
Овај пројекат је развијен са циљем да аутоматски класификује производе по категоријама на основу њихових описа.  
Модел је трениран на скупу података `products.csv`, који садржи називе, описе и категорије производа.

---

## 📂 Структура пројекта
product-category-classifier/
├── data/ # Скуп података
│ └── products.csv
├── notebooks/ # Jupyter бележнице са анализом
│ └── product_category_classifier.ipynb
├── scripts/ # Скрипте за тренирање и тестирање модела
│ ├── train_model.py
│ └── predict_category.py
├── models/ # Сачувани модели (опционо)
└── README.md # Упутство за покретање пројекта



---

## 🚀 Како покренути пројекат

### 1. Клонирати репозиторијум
```bash
git clone https://github.com/Svetlana-hub-73/product-category-classifier.git
cd product-category-classifier


#Инсталирати зависности
pip install -r requirements.txt

# Тренирати модел
python scripts/train_model.py

# Тестирати модел
python scripts/predict_category.py
#### Функционалности пројекта

#Учитавање и анализа података

#Обрада текста (TF-IDF / CountVectorizer)

#Тренирање модела машинског учења

#Визуелизација резултата

# Интерактивно тестирање модела


###Коришћене библиотеке

#pandas

#numpy

#scikit-learn

#matplotlib / seaborn

#joblib 

### Аутор
# https://github.com/Svetlana-hub-73/product-category-classifier/tree/main
