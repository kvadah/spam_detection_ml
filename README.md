# SMS Spam Detection 

This project is a machine learning-based SMS spam classifier that uses Natural Language Processing  techniques and a Naive Bayes algorithm to determine whether an SMS message is "spam" or "ham" (not spam).

---

#Dataset

We use the [SMS Spam Collection Dataset](https://github.com/vinit9638/SMS-scam-detection-dataset) which contains **138,000+ SMS messages** labeled as spam or ham. The dataset includes real-world text messages, making it ideal for building a robust spam filter.

**CSV Format:**


---

## 🚀 Features

- Text preprocessing using `CountVectorizer`
- Binary classification using `Multinomial Naive Bayes`
- Evaluation with accuracy and classification report
- Custom prediction support
- Clean and reproducible code structure

---

## 🧠 Machine Learning Flow

1. Load and clean the dataset (`pandas`)
2. Convert text to numeric using `CountVectorizer`
3. Train `MultinomialNB` model from `scikit-learn`
4. Evaluate using accuracy and precision/recall/F1-score
5. Predict new messages




