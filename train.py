import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from nets.MobileNetV2.mobilenetv2_tf import MobileNetV2
from nets.ReXNet.rexnetv2_tf import ReXNet

from create_data import download_data

ds_train, ds_test = download_data(dataset_name='cifar100', batch_size=128)


model = ReXNet(classes=100)
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

#keras.utils.plot_model(model, "rexnet.png")

checkpoint_path = "./checkpoints/rexnet.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


hist = model.fit(ds_train, epochs=10, validation_data=ds_test,
                    callbacks=[cp_callback])


loss, acc = model.evaluate(ds_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Save history
import pickle

with open('rexnet_hist.pkl', 'wb') as f:
    pickle.dump(hist)