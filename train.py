import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from nets.MobileNetV2.mobilenetv2_tf import MobileNetV2
from nets.ReXNet.rexnetv2_tf import ReXNet

from create_data import download_data

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x, y, training):
    y_ = model(x, training=training)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss_object(y_true=y, y_pred=y_)

def train(model, train_data, num_epochs):
    for epoch in range(num_epochs):
        print("Starting Epoch {:02d}\n".format(epoch))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_data:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch)
        print("\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}\n".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


#keras.utils.plot_model(model, "rexnet.png")
def main(*args, **kwargs):
    ds_train, ds_test = download_data(dataset_name='cifar100', batch_size=128)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = MobileNetV2(classes=100)
    model.compile(
        optimizer=optimizer,  # Optimizer
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    train(model, ds_train, 20)
    model.evaluate(ds_test)
    model.save_weights('./checkpoints/MobileNetV2_epoch20')
