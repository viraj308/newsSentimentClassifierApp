
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import string

# Load the training data
data = pd.read_excel('training data/train.xlsx')

# Preprocess the text data


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    preprocessed_text = text.translate(
        str.maketrans('', '', string.punctuation))

    return preprocessed_text


data['News Headline'] = data['News Headline'].apply(preprocess_text)

# Split the data into features (X) and labels (y)
X = data['News Headline']
y = data['Sentiment']

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

joblib.dump(lr_model, 'pretrained_model.pkl')
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Predict sentiment on the test set
y_pred = lr_model.predict(X_test)

# Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
