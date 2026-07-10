import joblib
import pandas as pd
import re
from scipy.sparse import hstack

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def create_features(message):
    """Create the same features used during training."""

    df = pd.DataFrame({"Message": [message]})

    df["message_length"] = df["Message"].apply(len)
    df["word_count"] = df["Message"].apply(lambda x: len(x.split()))
    df["has_currency"] = df["Message"].apply(
        lambda x: 1 if re.search(r'[$£€¥]', x) else 0
    )
    df["has_numbers"] = df["Message"].apply(
        lambda x: 1 if re.search(r'\d', x) else 0
    )
    df["has_special_chars"] = df["Message"].apply(
        lambda x: 1 if re.search(r'[!@#$%^&*()]', x) else 0
    )
    df["has_urgent_words"] = df["Message"].apply(
        lambda x: 1 if re.search(
            r'\b(urgent|free|prize|winner|cash|guarantee)\b',
            x.lower()
        ) else 0
    )

    return df


def predict_message(message):
    # Create additional features
    df = create_features(message)

    # Convert text to TF-IDF
    text_features = vectorizer.transform(df["Message"])

    # Numerical features
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

    # Combine TF-IDF and numerical features
    X = hstack((text_features, extra_features))

    # Predict
    prediction = model.predict(X)[0]

    # Probability (if supported)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X).max()

    return prediction, probability


if __name__ == "__main__":

    while True:
        message = input("\nEnter Message (or type 'exit'): ")

        if message.lower() == "exit":
            break

        prediction, probability = predict_message(message)

        if prediction in [1, "spam", "Spam"]:
            label = "SPAM"
        else:
            label = "HAM"

        print(f"\nPrediction : {label}")

        if probability is not None:
            print(f"Confidence : {probability:.2%}")