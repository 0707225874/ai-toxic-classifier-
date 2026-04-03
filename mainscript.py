import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Split features and labels
X = df['text']
y = df['label']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   # looks at single words + pairs of words
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. Predict
y_pred = model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))