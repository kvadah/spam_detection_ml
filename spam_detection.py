import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam.csv', usecols=['label', 'text'], encoding='latin-1')

df.dropna(subset=['label', 'text'], inplace=True)

df['label'] = df['label'].str.strip().str.lower()

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df.dropna(subset=['label'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.00011)

vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_features, y_train)


y_pred = model.predict(X_test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))

user_input = input("\n📨 Enter a message to classify as Spam or Ham:\n> ")
user_vec = vectorizer.transform([user_input])
# Predict
user_pred = model.predict(user_vec)
print("\n🔍 Prediction:", "Spam 🚫" if user_pred[0] == 1 else "Ham ✅")

