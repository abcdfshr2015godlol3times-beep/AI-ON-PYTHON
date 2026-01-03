from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Данные для обучения
texts = ["Привет, как дела?", "Отличная погода сегодня", "Мне нужна помощь"]
labels = ["приветствие", "комментарий", "запрос"]

# Векторизация текста
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Обучение модели
model = MultinomialNB()
model.fit(X, labels)

# Прогноз
test_text = ["Привет"]
test_vector = vectorizer.transform(test_text)
print("Предсказание:", model.predict(test_vector)[0])
