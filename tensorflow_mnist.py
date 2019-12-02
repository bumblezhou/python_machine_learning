import tensorflow as tf


def main():
    mnist = tf.keras.datasets.mnist
    # On windows, place mnist.npz to folder: C:\Users\<UserName>\.keras\datasets
    # On Linux, place mnist.npz to folder: ~/.keras/datasets/
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
    x_train, x_test = x_train / 1024, x_test / 1024
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == '__main__':
    main()
