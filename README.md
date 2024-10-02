# Spam Detector

This project uses a machine learning approach to classify email as spam or not spam. Two models are created and compared: a **Logistic Regression model** and a **Random Forest Classifier model**. The performance of these models is evaluated using a dataset sourced from the UCI Machine Learning Library.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Installation](#installation)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

This project aims to detect whether an email is spam or not based on various features extracted from the email's content. The features include frequency of specific words, characters, and other heuristics derived from the text of the emails.

The two models used for classification are:
1. **Logistic Regression**: A simple linear model that predicts the probability of spam based on the relationship between features.
2. **Random Forest Classifier**: An ensemble model that creates multiple decision trees and aggregates their predictions for a more accurate classification.

## Data Source

The dataset used for this project is the **Spambase Dataset** from the UCI Machine Learning Library. It contains information about spam and non-spam emails with extracted features from the email text.

- Dataset link: [Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase)
- Data used in this project is located at: [Spam Data CSV](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)

## Installation

To run this project, follow these steps:

1. Clone the repository or download the project files.
2. Install the required Python libraries using `pip`:
   ```bash
   pip install pandas scikit-learn
3. Load and execute the Jupyter notebook, ensuring all the necessary dependencies are installed.

## Models
**Logistic Regression Model**

A logistic regression model was built and trained using the scaled data. The model was evaluated using the test data.

**Random Forest Classifier Model**

A random forest classifier was also trained using the same dataset. The model consists of 128 decision trees.

## Data Preprocessing

Before training the models, the data was split into training and testing sets. The features were then scaled using StandardScaler to standardize the inputs for both models.

## Evaluation

The models were evaluated using the accuracy score metric. 
Here's how accuracy was calculated:

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, predictions)

The accuracy score reflects how well the models classify spam and non-spam emails compared to the actual labels.

## Results
After running both models, here are the results:

Logistic Regression Model:
- Training Accuracy: 92.58%
- Testing Accuracy: 92.27%

Random Forest Model:
- Training Accuracy: 99.94%
- Testing Accuracy: 95.83%
## Conclusion
The Random Forest Classifier outperformed the Logistic Regression Model on this dataset. The higher complexity of Random Forest and its ability to handle non-linear relationships helped it achieve a better accuracy score. However, Logistic Regression remains a valuable model due to its simplicity and interpretability.

For future improvements, other classification techniques like Support Vector Machines or Gradient Boosted Trees could be explored to further improve performance.

## License
This project is licensed under the MIT License.

## References
Class material was the main source of code for this project. ChatGPT was used to assist with the readme file.
