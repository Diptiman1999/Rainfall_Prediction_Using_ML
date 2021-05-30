# Implementation of Multiple Linear Regression
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# Importing the dataset
df = pd.read_csv('Rainfall2.csv')
print(df.describe().T)
print(df.corr())

values = df.values
# specify columns to plot for Column 1 to 9
groups = [x for x in range(3, 10)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# specify columns to plot for columns 10 to 14
groups = [x for x in range(10, 15)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# specify columns to plot for columns 15 to 20
groups = [x for x in range(15, 21)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# Partitioning to Dependent and Independent variables
# Independent Variable
X = df.iloc[:, 0:-3].values

# Dependent Variable
Y = df.iloc[:, -3].values


# Splitting the dataset to training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Fitting the Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set result
y_prediction = regressor.predict(X_test)

# Plotting the Graph
plt.plot(Y_test, color='black', label=' Actual Rainfall')
plt.plot(y_prediction, color='green', label='Predicted Rainfall')
plt.title(' Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel(' Rainfall')
plt.legend()
plt.show()

# For retrieve the intercept:
print("Intercepts = ", regressor.intercept_)

# For retrieving the slope:
print("Coefficient= ", regressor.coef_)

# Calculating the Errors
print("Score= ", regressor.score(X_test, Y_test))
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_prediction))
print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, y_prediction))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))
