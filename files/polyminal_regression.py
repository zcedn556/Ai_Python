import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv('./csv/fuel.csv')

X = df[['speed_kmh']]
y = df['fuel']

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X,y)

y_pred = model.predict(X)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)  

X_range = np.linspace(X.min().values[0], X.max().values[0], 200).reshape(-1, 1)
y_curve = model.predict(X_range)

test1 = pd.DataFrame({'speed_kmh': [35]})
test2 = pd.DataFrame({'speed_kmh': [95]})
test3 = pd.DataFrame({'speed_kmh': [140]})

plt.figure()
plt.scatter(X, y, color='green')
plt.plot(X_range, y_curve, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('speed=fuel')
plt.show()

print(model.predict(test1))
print(model.predict(test2))
print(model.predict(test3))

