# Implementation of Neural Network with Back Propagation
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Importing the dataset
df = pd.read_csv('Rainfall.csv')

values = df.values
# specify columns to plot for Column 1 to 9
groups = [x for x in range(1, 10)]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.5, loc='left')
    i += 1
plt.show()

# specify columns to plot for columns 10 to 19
groups = [x for x in range(10, 20)]
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
X = df.iloc[:, 0:-2].values

# Dependent Variable
Y = df.iloc[:, -2].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=100)

# ['identity', 'logistic', 'relu', 'softmax', 'tanh'].
nn = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                  learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True,
                  random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                  early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X_train, Y_train)
y_predicted = nn.predict(X_test)

# Plotting graph between Actual Vs Predicted
plt.plot(Y_test, color='black', label=' Actual Rainfall')
plt.plot(y_predicted, color='green', label='Predicted Rainfall')
plt.title(' Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel(' Rainfall')
plt.legend()
plt.show()


print("score= ", nn.score(X_test, Y_test))
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_predicted))
print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, y_predicted))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, y_predicted)))
