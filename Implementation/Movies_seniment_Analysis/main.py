import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Download NLTK stopwords data
nltk.download('stopwords')


# Step 2: Read the data
def read_data(folder):
    texts = []
    labels = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
            labels.append(1 if folder == 'positive' else 0)
    return texts, labels


# Step 3: Preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization (split text into words)
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Step 4: Apply TF-IDF Vectorization and train a classifier
def train_classifier(pos_folder, neg_folder):
    # Read the data
    pos_texts, pos_labels = read_data(pos_folder)
    neg_texts, neg_labels = read_data(neg_folder)
    all_texts = pos_texts + neg_texts
    all_labels = pos_labels + neg_labels

    # Print #numbers of pos and neg samples
    print("Number of positive samples:", len(pos_texts))
    print("Number of negative samples:", len(neg_texts))

    # Preprocess all texts
    preprocessed_texts = [preprocess_text(text) for text in all_texts]

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the preprocessed data
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
    print("TF-IDF Matrix shape: ",tfidf_matrix.shape)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, all_labels, test_size=0.2, random_state=42)

    # Initialize and train the classifier (Logistic Regression)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return classifier, accuracy, report, conf_matrix


# Visualize confusion matrix
def visualize_confusion_matrix(conf_matrix):
    labels = ['Negative', 'Positive']
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Example usage
pos_folder = "positive"
neg_folder = "negative"
classifier, accuracy, report, conf_matrix = train_classifier(pos_folder, neg_folder)
print("Classifier Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Visualize the confusion matrix
visualize_confusion_matrix(conf_matrix)
