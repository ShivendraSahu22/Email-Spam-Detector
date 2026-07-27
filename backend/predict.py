import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack

# Download stopwords if not available
nltk.download("stopwords", quiet=True)

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str):
    """
    Clean text exactly the same way as during training.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove everything except letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


def create_features(message: str):
    """
    Create engineered features using the ORIGINAL message.
    """

    df = pd.DataFrame({"Message": [message]})

    df["message_length"] = df["Message"].apply(len)

    df["word_count"] = df["Message"].apply(
        lambda x: len(x.split())
    )

    df["has_currency"] = df["Message"].apply(
        lambda x: 1 if re.search(r"[$£€¥₹]", x) else 0
    )

    df["has_numbers"] = df["Message"].apply(
        lambda x: 1 if re.search(r"\d", x) else 0
    )

    df["has_special_chars"] = df["Message"].apply(
        lambda x: 1 if re.search(r"[!@#$%^&*()]", x) else 0
    )

    df["has_urgent_words"] = df["Message"].apply(
        lambda x: 1
        if re.search(
            r"\b(urgent|free|prize|winner|cash|guarantee)\b",
            x.lower(),
        )
        else 0
    )

    return df


def predict_message(message: str):
    """
    Predict whether a message is spam.
    """

    # Clean message (same as training)
    cleaned_message = clean_text(message)

    # Original message features
    df = create_features(message)

    # TF-IDF features (cleaned text)
    text_features = vectorizer.transform([cleaned_message])

    # Engineered features
    extra_features = df[
        [
            "message_length",
            "word_count",
            "has_currency",
            "has_numbers",
            "has_special_chars",
            "has_urgent_words",
        ]
    ]

    # Combine
    X = hstack((text_features, extra_features))

    # Prediction
    prediction = model.predict(X)[0]

    # Confidence
    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())

    # Label
    label = "SPAM" if prediction == 1 else "NOT SPAM"

    return {
        "prediction": label,
        "confidence": round(confidence, 4) if confidence else None,
    }


# ------------------------------
# Local Testing
# ------------------------------

if __name__ == "__main__":

    while True:

        message = input("\nEnter message (type 'exit' to quit): ")

        if message.lower() == "exit":
            break

        result = predict_message(message)

        print("\nPrediction :", result["prediction"])
        print("Confidence :", result["confidence"])
    
    