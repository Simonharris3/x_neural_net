import keras
import tensorflow as tf
import numpy

num_classes = 10
input_shape = (28*28, 1)


def learn():
    mnist = tf.keras.datasets.mnist.load_data()

    train_input = mnist[0][0].reshape(mnist[0][0].shape[0], 28*28, 1)
    train_input = train_input.astype("float32") / 255

    train_labels = keras.utils.to_categorical(mnist[0][1], num_classes)

    test_input = mnist[1][0].reshape(mnist[1][0].shape[0], 28*28, 1)
    test_input = test_input.astype("float32") / 255

    test_labels = keras.utils.to_categorical(mnist[1][1], num_classes)

    input_layer = keras.Input(shape=input_shape)
    l1 = keras.layers.Dense(5, activation="relu")
    flatten = keras.layers.Flatten()
    l2 = keras.layers.Dense(num_classes, activation="sigmoid")
    model = keras.models.Sequential([input_layer, l1, flatten, l2])
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_input, train_labels, batch_size=100, epochs=10)

    model_l0 = tf.keras.models.Sequential([input_layer])
    model_l1 = tf.keras.models.Sequential([l1])
    model_l2 = tf.keras.models.Sequential([flatten])
    model_l3 = tf.keras.models.Sequential([l2])
    # print("TESTING")
    # print(model(numpy.array([train_input[0]])).shape)
    correlated_l0_neurons = []
    for i in range(10):
        correlated_l0_neurons.append([])

    for i in range(len(train_input)):
        l0 = model_l0(train_input)
        l1 = model_l1(l0)
        l2 = model_l2(l1)
        output = model_l3(l2)

        correlated_l0_neurons[max(output)].append(max(l0))

    score = model.evaluate(test_input, test_labels)


if __name__ == '__main__':
    learn()
