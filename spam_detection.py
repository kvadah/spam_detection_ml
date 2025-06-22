import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset and specify only required columns
df = pd.read_csv('spam.csv', usecols=['label', 'text'], encoding='latin-1')

# 2. Drop rows with missing values in 'label' or 'text'
df.dropna(subset=['label', 'text'], inplace=True)

# 3. Remove whitespace and lowercase labels
df['label'] = df['label'].str.strip().str.lower()

# 4. Encode label: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 5. Remove rows where label mapping failed (NaN)
df.dropna(subset=['label'], inplace=True)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.00011)

# 7. Vectorize text
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# 8. Train the model
model = MultinomialNB()
model.fit(X_train_features, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Test custom message
test_msg = ["click the"]
test_vec = vectorizer.transform(test_msg)
pred = model.predict(test_vec)
print("\nCustom prediction:", "Spam" if pred[0] == 1 else "Ham")
