from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

model = keras.Sequential(
    [
        layers.Dense(4, activation='relu', input_shape=(1,)),
        layers.Dense(2, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer='adam',
    loss='mse',
)

df = pd.read_csv('./csv/fuel.csv')

x = df['speed_kmh']
y = df['fuel']

model.fit(x,y, epochs=30)

prediction = model.predict(np.array([[35], [95], [140]]))
print(prediction)

model.summary()
