import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("training_data_lowercase.csv", sep="\t", header=None, names=["label", "text"])

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    words = text.split()  # Simple split for tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Convert labels to integers
data['label'] = data['label'].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.3, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Simplified SVM Model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_tfidf, y_train)

# Predictions
svm_train_pred = svm.predict(X_train_tfidf)
svm_test_pred = svm.predict(X_test_tfidf)

# Evaluation
svm_train_accuracy = accuracy_score(y_train, svm_train_pred)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
svm_report = classification_report(y_test, svm_test_pred)

# Output Results
print("SVM Results:")
print(f"Training Accuracy: {svm_train_accuracy:.2f}")
print(f"Testing Accuracy: {svm_test_accuracy:.2f}")
print("Classification Report:")
print(svm_report)

# Graphical Representation
plt.figure(figsize=(8, 6))
bars = ['Training Accuracy', 'Testing Accuracy']
accuracy_values = [svm_train_accuracy, svm_test_accuracy]

plt.bar(bars, accuracy_values, color=['blue', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('SVM Training vs Testing Accuracy')
plt.tight_layout()
plt.show()
