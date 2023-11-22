from cifar_X_functions import *


def main():
    """ Driver function."""

    (train_img, train_lab), (test_img, test_lab) = datasets.cifar10.load_data()

    train_img, test_img = train_img / 255.0, test_img / 255.0

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    check_images(train_img, class_names, img_index=8)

    model = build_model()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_img,
        train_lab,
        epochs=4,
        validation_data=(test_img, test_lab)
    )

    test_loss, test_acc = model.evaluate(test_img, test_lab, verbose=1)

    return 0


if __name__ == '__main__':
    main()
