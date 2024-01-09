import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the training data
data = pd.read_csv("emails_dataset.csv")
print(data.head())

df = pd.DataFrame(data)
print(df)

# Fill NaN values in the "Message" column with an empty string (you can choose a different strategy based on your data)
df["Message"].fillna("", inplace=True)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Message"])
print(X)

# Drop rows with NaN values in both X and y
df.dropna(subset=["Classification"], inplace=True)
y = df["Classification"]

# Ensure that X and y have the same number of rows
X = X[: len(y)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


#Testing our model
#load the test dataset
test_data = pd.read_csv("test_emails.csv")
print(data.head())
X_new = vectorizer.transform(test_data["Messages"])
new_predictions = clf.predict(X_new)
results_df = pd.DataFrame(
    {"Messages": test_data["Messages"], "Prediction": new_predictions}
)
print(results_df)
