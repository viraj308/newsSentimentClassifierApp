import joblib

# Preprocessing function (similar to what you used during training)
from model.model import preprocess_text

# Load the trained model
model = joblib.load('pretrained_model.pkl')

# Load the feature vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict sentiment for custom news input


def predict_sentiment(news_input):
    # Preprocess the input text
    preprocessed_news = preprocess_text(news_input)

    # apply the vectorizer
    tfidf_vectors = tfidf_vectorizer.transform([preprocessed_news])

    # Use the model to make predictions
    sentiment = model.predict(tfidf_vectors)

    return sentiment


if __name__ == "__main__":
    news_input = input("Enter the news headline: ")
    predicted_sentiment = predict_sentiment(news_input)
    print(predicted_sentiment)
