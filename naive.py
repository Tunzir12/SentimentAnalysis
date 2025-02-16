# Import necessary libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load Dataset (Sports & Outdoors category)
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Sports_and_Outdoors", trust_remote_code=True)
df = pd.DataFrame(dataset["full"])

# 🚀 STEP 1: DATA PREPROCESSING

# Remove duplicates and missing values
df = df.drop_duplicates()
df = df.dropna(subset=["reviewText", "starRating"])

# Convert star ratings to sentiment labels
def categorize_sentiment(rating):
    if rating >= 4:
        return 1  # Positive
    elif rating == 3:
        return 0  # Neutral
    else:
        return -1  # Negative

df["sentiment"] = df["starRating"].astype(int).apply(categorize_sentiment)

# Clean text function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

df["cleaned_text"] = df["reviewText"].astype(str).apply(clean_text)

# Split dataset into Train (70%), Validation (15%), Test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Apply text cleaning to all sets
train_df["cleaned_text"] = train_df["reviewText"].astype(str).apply(clean_text)
val_df["cleaned_text"] = val_df["reviewText"].astype(str).apply(clean_text)
test_df["cleaned_text"] = test_df["reviewText"].astype(str).apply(clean_text)

# 🚀 STEP 2: NAIVE BAYES + TF-IDF

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X_train_tfidf = vectorizer.fit_transform(train_df["cleaned_text"])
X_val_tfidf = vectorizer.transform(val_df["cleaned_text"])
X_test_tfidf = vectorizer.transform(test_df["cleaned_text"])

# Extract labels
y_train = train_df["sentiment"]
y_val = val_df["sentiment"]
y_test = test_df["sentiment"]

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Evaluate Naive Bayes Model
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# 🚀 STEP 3: FINE-TUNING ROBERTA

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, padding="max_length", max_length=512)

# Convert Pandas DataFrames to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df[["cleaned_text", "sentiment"]])
val_dataset = Dataset.from_pandas(val_df[["cleaned_text", "sentiment"]])
test_dataset = Dataset.from_pandas(test_df[["cleaned_text", "sentiment"]])

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir="./roberta_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Fine-tune the RoBERTa model
trainer.train()

# Evaluate RoBERTa on test set
results = trainer.evaluate(test_dataset)
print("RoBERTa Test Results:", results)

# 🚀 STEP 4: MODEL COMPARISON

# Compute Naive Bayes accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# Compute RoBERTa accuracy
def get_roberta_predictions(model, dataset):
    predictions = trainer.predict(dataset).predictions
    return np.argmax(predictions, axis=1)

y_pred_roberta = get_roberta_predictions(model, test_dataset)
roberta_accuracy = accuracy_score(y_test, y_pred_roberta)

# Print comparison
print("\nModel Comparison:")
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print(f"RoBERTa Accuracy: {roberta_accuracy:.4f}")

# Visualize Performance
models = ["Naive Bayes", "RoBERTa"]
accuracies = [nb_accuracy, roberta_accuracy]

plt.bar(models, accuracies, color=["blue", "red"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()
