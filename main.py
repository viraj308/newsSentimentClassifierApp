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

    ascii_art = """ _                    _ _ _                   _               _  __ _           
    | |                  | | (_)                 | |             (_)/ _(_)          
    | |__   ___  __ _  __| | |_ _ __   ___    ___| | __ _ ___ ___ _| |_ _  ___ _ __ 
    | '_ \ / _ \/ _` |/ _` | | | '_ \ / _ \  / __| |/ _` / __/ __| |  _| |/ _ \ '__|
    | | | |  __/ (_| | (_| | | | | | |  __/ | (__| | (_| \__ \__ \ | | | |  __/ |   
    |_| |_|\___|\__,_|\__,_|_|_|_| |_|\___|  \___|_|\__,_|___/___/_|_| |_|\___|_|   
    """
    print("----------------------------------------------------------------------------------------------")
    print(ascii_art)
    print("----------------------------------------------------------------------------------------------")
    print("The data achieved 73% accuracy with linear regression")
    print(
        "Input news headline for classification: [neutral, negative, positive]")
    news_input = input("\nEnter the news headline: ")
    predicted_sentiment = predict_sentiment(news_input)
    print(predicted_sentiment)
