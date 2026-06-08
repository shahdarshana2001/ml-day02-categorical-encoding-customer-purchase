from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "customer_purchase.csv"
RESULTS_FILE = BASE_DIR / "results.txt"


def load_data():
    df = pd.read_csv(DATA_FILE)
    print("Dataset Loaded Successfully")
    print(df.head())
    return df


def label_encoding_model(df):
    label_df = df.copy()

    encoder = LabelEncoder()

    categorical_columns = ["Gender", "City", "Profession"]

    for col in categorical_columns:
        label_df[col] = encoder.fit_transform(label_df[col])

    X = label_df[["Age", "Gender", "City", "Profession"]]
    y = label_df["Purchased"]

    model = LogisticRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    return accuracy, confusion_matrix(y, predictions), classification_report(y, predictions)


def one_hot_encoding_model(df):
    X = df[["Age", "Gender", "City", "Profession"]]
    y = df["Purchased"]

    X_encoded = pd.get_dummies(
        X,
        columns=["Gender", "City", "Profession"],
        drop_first=True
    )

    model = LogisticRegression()
    model.fit(X_encoded, y)

    predictions = model.predict(X_encoded)
    accuracy = accuracy_score(y, predictions)

    return accuracy, confusion_matrix(y, predictions), classification_report(y, predictions), X_encoded


def save_results(label_results, one_hot_results):
    label_accuracy, label_cm, label_report = label_results
    one_hot_accuracy, one_hot_cm, one_hot_report, encoded_data = one_hot_results

    results = f"""
Customer Purchase Prediction - Categorical Encoding Results

Label Encoding Accuracy:
{label_accuracy:.4f}

Label Encoding Confusion Matrix:
{label_cm}

Label Encoding Classification Report:
{label_report}

One-Hot Encoding Accuracy:
{one_hot_accuracy:.4f}

One-Hot Encoding Confusion Matrix:
{one_hot_cm}

One-Hot Encoding Classification Report:
{one_hot_report}

One-Hot Encoded Columns:
{list(encoded_data.columns)}

Observations:
1. Label Encoding converts categories into numbers like 0, 1, 2.
2. Label Encoding can create false order between categories.
3. One-Hot Encoding creates separate columns for each category.
4. One-Hot Encoding is usually better for nominal data like Gender, City, and Profession.
5. Logistic Regression is used here because Purchased is a binary output: 0 or 1.
"""

    RESULTS_FILE.write_text(results)
    print(results)
    print("Results saved successfully.")


def main():
    df = load_data()

    label_results = label_encoding_model(df)
    one_hot_results = one_hot_encoding_model(df)

    save_results(label_results, one_hot_results)


if __name__ == "__main__":
    main()