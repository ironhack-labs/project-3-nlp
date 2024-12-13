README 

# Project 2: Fake News Detector 
![Fake News](images/fakeimg.jpg)
## Overview
The Fake News Detector is a machine learning project using Natural Language Processing (NLP). This AI model classifies news articles as either True or Fake. By leveraging a robust pipeline and comparing various machine learning models, the best-performing model is selected to ensure high accuracy and efficiency.

## Key Features
### Model Benchmarking
Multiple machine learning models were tested and compared to identify the most effective solution:

📊 Naive Bayes

🌳 Random Forest

📈 Logistic Regression

💻 Passive-Aggressive Classifier

⚙️ Support Vector Machine (SVM)

### Best Model

![Best Model Highlight](images/best.png)


### Hyperparameter Tuning

While advanced hyperparameter tuning techniques like Grid Search and Random Search were explored, they proved computationally expensive with marginal benefits. A basic SVM configuration was retained for optimal performance.

### NLP Pipeline

The system implements a comprehensive NLP pipeline to preprocess and analyze text:

* Text Cleaning

* Tokenization

* TF-IDF Vectorization




## 📊 Results

### SVM Performance Metrics

#### Accuracy
| Metric                | Value  |
|-----------------------|--------|
| **Training Accuracy** | 0.97   |
| **Testing Accuracy**  | 0.94   |

#### Classification Report
| Label       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **0 (True)**| 0.95      | 0.94   | 0.94     | 5295    |
| **1 (Fake)**| 0.93      | 0.95   | 0.94     | 4951    |
| **Overall Accuracy** | **0.94** |

#### Averages
| Metric       | Value  |
|--------------|--------|
| **Macro Avg.**   | 0.94   |
| **Weighted Avg.**| 0.94   |

![svm](images/SVM.png)


## Unlabeled Dataset Predictions
### Predicted Label Distribution

| Label  | Count |
|--------|-------|
| **True** | 5244  |
| **Fake** | 4740  |

![Unlabeled Predictions](images/pred.png)


## 🔍 Limitations & Future Improvements

### Computational Efficiency

Advanced hyperparameter tuning like Grid Search and Random Search was computationally expensive and yielded negligible improvements.
Future efforts could leverage cloud-based or distributed computing for such tasks.

### Advanced Models

Incorporating deep learning architectures like Transformers (e.g., BERT) could further enhance accuracy and generalizability.

### Dataset Diversity

Expanding the dataset to include more diverse and global sources would improve the model's robustness and applicability.




 