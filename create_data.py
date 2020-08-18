import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224

# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.cast(image, tf.float32) / 255.
    return image, label


def download_data(dataset_name="cifar100", batch_size=128):
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name, split=["train", "test"], with_info=True, as_supervised=True, data_dir='./data/tfds_datasets'
    )
    global NUM_CLASSES
    NUM_CLASSES = ds_info.features["label"].num_classes

    # Resize the dataset into 256x256 images
    size = (IMG_SIZE, IMG_SIZE)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

    # Preprocess the data so that the images are in range [0,1]
    # and labels are one-hot encoded
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    return ds_train, ds_test