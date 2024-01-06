
# Email Classification using Naive Bayes

## Overview
This repository contains a simple implementation of an email classification model using the Naive Bayes algorithm. The model is trained on a labeled dataset of emails, distinguishing between different classifications such as spam and ham. Additionally, the trained model is tested on a separate dataset to evaluate its performance.

## Installation
To run the code, make sure you have the required Python libraries installed. You can install them using the following:

```bash
pip install numpy pandas scikit-learne
```
## Usage
Clone the repository:

```bash
git clone https://github.com/yourusername/email-classification.git
cd email-classification
```
Download the training dataset (emails_dataset.csv) and test dataset (test_emails.csv) and place them in the project folder.

Run the provided Python script:

```bash
python email_classification.py
```
The script will load the training data, preprocess it, train a Naive Bayes classifier, and then evaluate its performance on a test dataset. The classification report will be printed, showing metrics such as precision, recall, and F1-score.

## Code Explanation
The code is structured as follows:

**Loading and Preprocessing Data:** The training data is loaded from the emails_dataset.csv file, and any NaN values in the "Message" column are filled with empty strings.

**Feature Extraction:** The CountVectorizer is used to convert the text data into a bag-of-words representation. This is essential for training the Naive Bayes classifier.

**Train-Test Split:** The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

**Model Training:** A Multinomial Naive Bayes classifier is trained using the training data.

**Model Evaluation:** The model is evaluated on the test set, and a classification report is printed, providing insights into its performance.

**Testing the Model:** The trained model is applied to a new dataset (test_emails.csv), and the predictions are displayed alongside the original messages.

## Results
The classification report printed after model evaluation provides information about the precision, recall, and F1-score for each class. Additionally, the predictions on the test dataset are presented in a DataFrame for further analysis.

Feel free to modify the script, tweak parameters, or use your own datasets to experiment with different configurations.
