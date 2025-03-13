import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Load the dataset
data = pd.read_csv('environmental_sentiments (1).csv')
data['clean_text'] = data['text'].apply(clean_text)

# Encode the labels: 'negative' -> 0, 'neutral' -> 1, 'positive' -> 2
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
data['sentiment'] = data['sentiment'].map(label_mapping)

# Split the data into training and testing sets
X = data['clean_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)

# SVM model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_test_tfidf)

# LSTM model
# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
lstm_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(3, activation='softmax'))

# Compile the LSTM model
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the LSTM model
lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Get the class predictions for the LSTM model
lstm_preds_prob = lstm_model.predict(X_test_pad)
lstm_preds = np.argmax(lstm_preds_prob, axis=1)

# Evaluate the models and print the results
lr_accuracy = accuracy_score(y_test, lr_preds)
svm_accuracy = accuracy_score(y_test, svm_preds)
lstm_accuracy = accuracy_score(y_test, lstm_preds)

print("Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_preds))

print("SVM Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_preds))

print("LSTM Accuracy:", lstm_accuracy)
print(classification_report(y_test, lstm_preds))

# Plotting the accuracies of the models
accuracies = [lr_accuracy, svm_accuracy, lstm_accuracy]
models = ['Logistic Regression', 'SVM', 'LSTM']

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)
plt.show()

# Plotting the confusion matrices for each model
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Plot confusion matrices for each model
plot_confusion_matrix(y_test, lr_preds, 'Confusion Matrix for Logistic Regression')
plot_confusion_matrix(y_test, svm_preds, 'Confusion Matrix for SVM')
plot_confusion_matrix(y_test, lstm_preds, 'Confusion Matrix for LSTM')

# Function to predict sentiment using the three models
def predict_sentiment(opinions):
    opinions_clean = [clean_text(opinion) for opinion in opinions]
    opinions_tfidf = vectorizer.transform(opinions_clean)
    opinions_seq = tokenizer.texts_to_sequences(opinions_clean)
    opinions_pad = pad_sequences(opinions_seq, maxlen=100)
    
    lr_predictions = lr_model.predict(opinions_tfidf)
    svm_predictions = svm_model.predict(opinions_tfidf)
    lstm_predictions_prob = lstm_model.predict(opinions_pad)
    lstm_predictions = np.argmax(lstm_predictions_prob, axis=1)
    
    return lr_predictions, svm_predictions, lstm_predictions

# Example usage
new_opinions = [
    "I love the new environmental policies.",
    "I'm not sure if these climate regulations are effective.",
    "The government's efforts to tackle climate change are insufficient."
]

# Predict sentiments for new opinions using all three models
lr_preds, svm_preds, lstm_preds = predict_sentiment(new_opinions)
print("Logistic Regression Predictions:", lr_preds)
print("SVM Predictions:", svm_preds)
print("LSTM Predictions:", lstm_preds)
