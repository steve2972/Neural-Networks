import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from nets.MobileNetV2.mobilenetv2_tf import MobileNetV2
from nets.ReXNet.rexnetv2_tf import ReXNet

from create_data import download_data

ds_train, ds_test = download_data(dataset_name='cifar100', batch_size=128)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model = MobileNetV2(classes=100)
model.compile(
    optimizer=optimizer,  # Optimizer
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

#keras.utils.plot_model(model, "rexnet.png")


import pickle

train_loss_results = []
train_accuracy_results = []

num_epochs = 15

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss_object(y_true=y, y_pred=y_)

for epoch in range(num_epochs):
    print("Starting Epoch {:03d}".format(epoch))
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in ds_train:
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
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                            epoch_loss_avg.result(),
                                                            epoch_accuracy.result()))
        
#hist = model.fit(ds_train, epochs=20, validation_data=ds_test, verbose=1)


with open('train_loss_rexnet.pkl') as f:
    pickle.dump(train_loss_results)
with open('test_loss_rexnet.pkl') as f:
    pickle.dump(test_loss_results)