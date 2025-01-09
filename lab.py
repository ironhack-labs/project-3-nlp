import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Read txt file and process it into a DataFrame
data_rows = []
with open('TRAINING_DATA.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            label, text = parts
            data_rows.append({'label': label, 'text': text})


# Create DataFrame
df = pd.DataFrame(data_rows)

# Basic data analysis
print("\nDataset Overview:")
print("-" * 50)
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")


def preprocess_text(text):
    # Convert to lowercase (after preserving Ñ)
    #text = text.replace('Ñ', 'ñ')  # Preserve Ñ before lowercase
    text = text.lower()

    # Remove special characters and digits, keeping Spanish characters
    text = re.sub(r'[^a-záéíóúñüÁÉÍÓÚÑÜ\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Calculate text statistics
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print("\nText Statistics:")
print("-" * 50)
print(f"Average text length: {df['text_length'].mean():.2f} characters")
print(f"Average word count: {df['word_count'].mean():.2f} words")

# Prepare for ML
X = df['processed_text']
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
print("\nTraining SVM Classifier...")
svm_classifier = LinearSVC(random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
print("\nModel Evaluation:")
print("-" * 50)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Perform cross-validation
cv_scores = cross_val_score(svm_classifier, X_train_tfidf, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance analysis
feature_names = tfidf.get_feature_names_out()
coefficients = svm_classifier.coef_.toarray()
top_features = 10

for class_idx in range(coefficients.shape[0]):
    print(f"\nTop {top_features} most important words for class {class_idx}:")
    top_coef_indices = coefficients[class_idx].argsort()[-top_features:][::-1]
    for idx in top_coef_indices:
        print(f"{feature_names[idx]}: {coefficients[class_idx][idx]:.4f}")

