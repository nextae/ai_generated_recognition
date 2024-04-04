import tensorflow as tf
import cv2
import numpy as np

# I = cv2.imread("datasets/CIFAKE/test/FAKE/11 (9).jpg")
I = cv2.imread("datasets/google_graphics/fake/ai_cake.png")
I = cv2.resize(I, (32, 32), interpolation=cv2.INTER_AREA)

model = tf.keras.models.load_model("model1.h5")
print(round(model.predict(np.array([I]))[0][0], 2))
