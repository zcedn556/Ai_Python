from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

model = keras.Sequential(
    [
        layers.Dense(8, activation='relu', input_shape=(3,)),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer='adam',
    loss='mse',
)

df = pd.read_csv('./csv/fuel.csv')
df = pd.get_dummies(df, columns=['engine_type'],drop_first=True)

x = df[['speed_kmh', 'hour', 'engine_type_petrol']].astype(float).values
y = df['fuel'].astype(float).values

model.fit(x,y, epochs=150)

prediction = model.predict(np.array([[35, 12, 1]], dtype=float))
print(prediction)

model.summary()
