import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./csv/energy_usage.csv")

X = df[['temperature','humidity','hour', 'is_weekend']]
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X,y)

test = pd.DataFrame([{
    'temperature': 11,
    'humidity': 67,
    'hour': 10,
    'is_weekend':1,
}])

predicted_energy = model.predict(test)
print(f"Енергія прогноз: {predicted_energy[0]:,.2f} $")

y_pred = model.predict(X_test)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

plt.scatter(y_test, y_pred)
plt.xlabel("Справжня")
plt.ylabel("Передбачення")
plt.title("Справжня проти передбачення")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()