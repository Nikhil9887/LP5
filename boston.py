import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('HousingData.csv')
df

df.isnull().sum()

df.describe()

for i in df.columns:
    mean_value = df[i].mean()
    df[i].fillna(mean_value, inplace=True)

df.isnull().sum()

df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x = df.drop('MEDV', axis=1)
y = df['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_shape=(13, ), activation='relu', name='dense_1'))
model.add(Dense(64, activation='relu', name='dense_2'))
model.add(Dense(1, activation='linear', name='dense_output'))

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])
model.summary()

model.fit(x_train_scaled, y_train, epochs=50)

loss_nn, mse_nn, mae_nn = model.evaluate(x_test_scaled, y_test)

print('Mean absolute error on test data: ', mae_nn)
print('Mean squared error on test data: ', mse_nn)

y_pred = model.predict(x_test_scaled)
y_test = np.array(y_test).reshape(-1,1)
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_test[i][0]:.2f}")
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
