import joblib
import pandas as pd
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def create_features(message: str):
    df = pd.DataFrame({"Message": [message]})

    df["message_length"] = df["Message"].apply(len)
    df["word_count"] = df["Message"].apply(lambda x: len(x.split()))
    df["has_currency"] = df["Message"].apply(
        lambda x: 1 if re.search(r"[$£€¥]", x) else 0
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
    # Create features
    df = create_features(message)

    # TF-IDF features
    text_features = vectorizer.transform(df["Message"])

    # Additional features
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

    # Combine features
    X = hstack((text_features, extra_features))

    # Predict
    prediction = model.predict(X)[0]

    # Confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())

    # Label
    label = "SPAM" if prediction == 1 else "NOT SPAM"

    return {
        "prediction": label,
        "confidence": confidence,
    }