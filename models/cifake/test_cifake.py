import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()

# Load image
I = cv2.imread("../../datasets/google_graphics/fake/ai_animal.jpg")
I_oryg = I.copy()
I = cv2.resize(I, (32, 32), interpolation=cv2.INTER_AREA)

# Load model
model = tf.keras.models.load_model("saved_models/model1.h5")

# Predict and print result
preds = model.predict(np.array([I]))
print(round(preds[0][0], 4))

# Heatmap below
x = np.array([I])
model_output = model.output[:, np.argmax(preds[0])]

last_conv_layer = model.get_layer('conv2d_1')
grads = tf.gradients(model_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (I_oryg.shape[1], I_oryg.shape[0]))
I_oryg = cv2.cvtColor(I_oryg, cv2.COLOR_BGR2GRAY)
I_oryg = cv2.cvtColor(I_oryg, cv2.COLOR_GRAY2BGR)

heatmap = np.uint8(255 * heatmap)

# Show heatmap on top of image

superimposed_img = heatmap * 0.3 + I_oryg * 0.7
superimposed_img = superimposed_img.astype('uint8')

cv2.imshow("Im", superimposed_img)
cv2.waitKey()
cv2.destroyAllWindows()