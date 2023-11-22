import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt


def check_images(train_img, classes, img_index):
    """ Checks the image at the argument index."""

    plt.imshow(train_img[img_index], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_lab[img_index][0]])

    plt.show()

    return 0


def build_model():
    """ Builds the model with the necessary layers."""

    model = models.Sequential()

    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)
    ))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10))


