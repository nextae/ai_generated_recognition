import keras.optimizers
from keras import models, layers, metrics
from data_preprocessing import load_data
import pickle

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    keras.optimizers.Adam(),
    "binary_crossentropy",
    ["accuracy", keras.metrics.Recall(), keras.metrics.Precision()]
)

x_train, x_valid = load_data()

history = model.fit(x_train, validation_data=x_valid, epochs=10)

model.save("model1.h5")
with open("history.pkl", 'wb') as f:
    pickle.dump(history, f)
