# Implementing LSTM
# Importing Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
import matplotlib.pyplot as plt

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

X = X.astype('float32')
Y = Y.astype('float32')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=40)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# design network
model = Sequential()
model.add(LSTM(80, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, Y_train, epochs=200, batch_size=72, validation_data=(X_test, Y_test), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
pyplot.show()

# make a prediction
y_predicted = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

# Plotting the graph between Actual And Predicted
plt.plot(Y_test, color='black', label=' Actual Rainfall')
plt.plot(y_predicted, color='green', label='Predicted Rainfall')
plt.title(' Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel(' Rainfall')
plt.legend()
plt.show()

# Calculating the Errors
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_predicted))
print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, y_predicted))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, y_predicted)))
