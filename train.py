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

train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
hist = model.fit(ds_train, epochs=10, validation_data=ds_test,
                    callbacks=[cp_callback])


loss, acc = model.evaluate(ds_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Save history
import pickle

with open('rexnet_hist.pkl', 'wb') as f:
    pickle.dump(hist)