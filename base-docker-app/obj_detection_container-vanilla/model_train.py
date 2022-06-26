# trains and saves the model

from preprocess import preprocess
from tensorflow.keras import layers, models
import tensorflow as tf
import pathlib

import argparse

parser = argparse.ArgumentParser(description='Train a speech recognition model for a given dataset.')
parser.add_argument('path', type=str, help="The path to the folder containing the training files.")


args = parser.parse_args()
path = args.path
data_dir = pathlib.Path(path)

# preprocess the speech_file
train_ds, test_ds, val_ds, spectrogram_ds, commands = preprocess(data_dir)
    
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

print('Model summary:', model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# train the model
EPOCHS = 10000
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

# save the trained model
model.save('saved_model/speech_rec_model')