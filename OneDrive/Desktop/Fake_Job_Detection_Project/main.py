import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("fake_job_postings.csv")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df['fraudulent'].value_counts())

# Keep required columns
df = df[['description', 'fraudulent']]
df.dropna(inplace=True)

# Define input and output
X = df['description']
y = df['fraudulent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical form
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text converted to TF-IDF successfully!")

# Train model
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Function to predict new job description
def predict_job(description):
    transformed = vectorizer.transform([description])
    prediction = model.predict(transformed)
    
    if prediction[0] == 1:
        return "Fake Job Posting"
    else:
        return "Real Job Posting"


# Test with sample input
sample_job = "Earn $5000 weekly from home. No experience required. Limited seats available!"

result = predict_job(sample_job)
print("\nSample Prediction Result:", result)

import pickle

# Save model
with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")