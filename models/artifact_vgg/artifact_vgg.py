import tensorflow as tf
import keras
from keras import layers
from utils.data_preprocessing import load_data_artifact


def create_model():
    backbone = keras.applications.VGG19(False, 'imagenet', input_shape=(200, 200, 3))

    for layer in backbone.layers:
        layer.trainable = False

    flatten = layers.Flatten()(backbone.output)
    dense = layers.Dense(256, activation='elu')(flatten)
    norm = layers.BatchNormalization()(dense)

    dense = layers.Dense(256, activation='elu')(norm)
    norm = layers.BatchNormalization()(dense)

    dense = layers.Dense(256, activation='elu')(norm)
    norm = layers.BatchNormalization()(dense)

    dense = layers.Dense(1, activation='sigmoid')(norm)

    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model = tf.keras.Model(inputs=backbone.input, outputs=dense)
    model.compile(
      optimizer=adam,
      loss='binary_crossentropy',
      metrics=['accuracy', 'mse', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.summary()
    return model


def train_model(model: tf.keras.Model, train_data, validation_data, model_name):
    # train_data = train_data.map(tf.keras.applications.vgg19.preprocess_input)
    # validation_data = validation_data.map(tf.keras.applications.vgg19.preprocess_input)

    saving_models = keras.callbacks.ModelCheckpoint("./saved_models/checkpoints/"+model_name+".{epoch:02d}.h5")
    model.fit(train_data, validation_data=validation_data, epochs=15, callbacks=[saving_models])


model = create_model()
train, valid = load_data_artifact("../../datasets/artifact")
train = train.take(100000)
valid = valid.take(100000)
train_model(model, train, valid, "artifact-1")
