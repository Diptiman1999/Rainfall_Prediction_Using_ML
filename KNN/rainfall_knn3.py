# Implementation of KNN Algorithm
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Importing the dataset
df = pd.read_csv('Rainfall 3.csv')
print(df.describe().T)
print(df.corr())

values = df.values
# specify columns to plot for Column 2 to 5
groups = [x for x in range(2, 6)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# specify columns to plot for columns 6 to 9
groups = [x for x in range(6, 10)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# specify columns to plot for columns 10 to 13
groups = [x for x in range(10, 14)]
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
X = df.iloc[:, 2:-2].values

# Dependent Variable
Y = df.iloc[:, -2].values

# Importing the split and KNeighbors Regressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
r_square_list = []
for i in range(1, 100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=i)
    neigh = KNeighborsRegressor(n_neighbors=8)
    neigh.fit(X_train, Y_train)

    y_predicted = neigh.predict(X_test)
    r_square = neigh.score(X_test, Y_test)
    r_square_list.append(r_square)

plt.plot(Y_test, color='black', label=' Actual Rainfall')
plt.plot(y_predicted, color='green', label='Predicted rainfall')
plt.title(' Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel(' rainfall')
plt.legend()
plt.show()

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_predicted))
print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, y_predicted))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, y_predicted)))
