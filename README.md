# salesPrediction_LogisticRegression



## Logistic Regression:
In brief, Logistic Regression is a statistical method used for binary classification problems, where the goal is to predict one of two possible outcomes (usually labeled as 0 and 1) based on one or more predictor variables. It's called "logistic" because it models the probability that a given input belongs to a specific class using the logistic function (also known as the sigmoid function). Logistic Regression is a type of generalized linear model that works well for problems where the dependent variable is binary.

### Binary classification:
Binary classification is a type of supervised machine learning problem where the goal is to categorize items into one of two classes or categories typically, these are represented as 0 (negative class) and 1 (positive class), or equivalently, as "negative" and "positive.".  The main objective of binary classification is to learn a model or algorithm that can automatically assign one of these two labels to each input data point based on its features or attributes.

### logistic function:
The logistic function, also known as the sigmoid function, is a mathematical function that maps any real-valued number to a value between 0 and 1. It has an S-shaped curve and is commonly used in logistic regression and other machine learning models to model and estimate probabilities, especially in binary classification problems.

The logistic function is defined as:
    f(x) = 1/(1+(e^(-x)))
In this equation:
• f(x) represents the output of the logistic function for a given input x.
• e is the base of the natural logarithm (approximately equal to 2.71828).

### Generalized Linear Model (GLM):
A Generalized Linear Model (GLM) is a statistical modeling framework that extends the concepts of linear regression to handle a broader range of data types and distributional assumptions. GLMs are a class of models used for regression and classification tasks, and they are particularly useful when the dependent variable (the target variable which is been trying to predict) does not follow a normal distribution or when we want to model the relationship between variables with different types of probability distributions.




## Here is a, Python Jupyter Notebook that demonstrates the use of Logistic Regression for a binary classification problem, predicting whether a new customer will make a purchase or not based on their age and salary. 

Here is a breakdown of the code:-
===================================

### Importing Libraries

### Choose Dataset file from Local Directory

### Load Dataset:
The dataset is loaded using Pandas' pd.read_csv() function from the uploaded CSV file named 'ad_dataset.csv.'

### Summarize Dataset
The code prints the shape (number of rows and columns) of the loaded dataset and displays the first 5 rows of the dataset using dataset.shape and dataset.head(5).

### Segregate Dataset into X and Y:
The dataset is split into input features (X) and the output variable (Y). X contains all columns except the last one, and Y contains the last column of the dataset.

### Splitting Dataset into Train & Test:
The dataset is further split into training and testing sets using train_test_split from sklearn.model_selection. It divides the data into 75% for training (X_train, y_train) and 25% for testing (X_test, y_test).

### Feature Scaling:
Standardization is applied to the input features using StandardScaler from sklearn.preprocessing. It scales the features to have a mean of 0 and a standard deviation of 1 for better model performance.

### Training:
The Logistic Regression model is created using LogisticRegression from sklearn.linear_model. It's trained on the standardized training data using the model.fit() method.

### Predicting for a New Customer:
The code allows the user to input the age and salary of a new customer. It then creates a new customer data point, standardizes it using the same scaler used for the training data, and predicts whether the customer will buy or not based on the trained logistic regression model.

### Prediction for all Test Data:
The model is used to make predictions on the test data (X_test), and the predictions are concatenated with the actual test labels (y_test) for evaluation purposes.

### Evaluating Model - CONFUSION MATRIX:
The code calculates a confusion matrix and accuracy score to evaluate the model's performance on the test data. It uses confusion_matrix and accuracy_score from sklearn.metrics to do this. The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives, while the accuracy score indicates how accurate the model's predictions are.
