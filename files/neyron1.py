from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=999)

model = tf.keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(13,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(3),
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"],)

model.fit(X_train,y_train, epochs=30, batch_size=8)

model.evaluate(X_test, y_test)

test = np.array([[13.2, 2.77, 2.51, 18.5, 100, 2.8, 3.1, 0.30, 2.5, 5.0, 1.0, 3.0, 1050]])
test = scaler.transform(test)

pred_class = np.argmax(model.predict(test), axis=1)
print(pred_class)