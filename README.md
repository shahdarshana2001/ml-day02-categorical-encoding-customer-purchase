# Customer Purchase Prediction

This project explores how categorical data can be converted into numerical values before training a machine learning model.

The dataset contains customer information such as age, gender, city, and profession. Since machine learning models cannot work directly with text data, I used two common encoding techniques:

* Label Encoding
* One-Hot Encoding

After encoding the data, I trained a Logistic Regression model and compared the results.

## What I Learned

* How to work with categorical data
* Difference between Label Encoding and One-Hot Encoding
* How Logistic Regression is used for classification problems
* How to evaluate a model using accuracy and classification metrics

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* uv

## Project Structure

* `data/` – Dataset
* `src/` – Python code
* `results.txt` – Model results and observations

## How to Run

```bash
uv run python src/main.py
```

## Project Goal

The goal of this project is to understand how different encoding techniques affect machine learning models and to practice building a simple classification pipeline.
