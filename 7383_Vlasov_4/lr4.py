import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from PIL import Image
import numpy as np

EPOHS = 5
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


def buildModel(optimizer):
    model = Sequential()
    model.add(Dense(IMAGE_HEIGHT * IMAGE_WIDTH, activation='relu', input_shape=(IMAGE_HEIGHT * IMAGE_WIDTH,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def loadData():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, IMAGE_HEIGHT * IMAGE_WIDTH))
    train_images = train_images / 255.0
    test_images = test_images.reshape((10000, IMAGE_HEIGHT * IMAGE_WIDTH))
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


def predict(path, model):
    return np.argmax(model.predict(loadImage(path)))


def loadImage(path):
    image = Image.open(path)
    image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    image = np.dot(np.asarray(image), np.array([1 / 3, 1 / 3, 1 / 3]))
    image /= 255
    image = 1 - image
    image = image.reshape((1, IMAGE_HEIGHT * IMAGE_WIDTH))
    return image


def trainModel(optimizer, epohs):
    optimizerConf = optimizer.get_config()
    print(
        "Researching with optimizer %s with learning rate %s" % (optimizerConf["name"], optimizerConf["learning_rate"]))

    model = buildModel(optimizer)
    history = model.fit(train_images, train_labels, epochs=epohs,
                        batch_size=128, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    plt.title('Training and test accuracy')
    plt.plot(history.history['accuracy'], 'r', label='train')
    plt.plot(history.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.savefig("Graphics/%s%s_acc.png" % (optimizerConf["name"], optimizerConf["learning_rate"]), format='png')
    plt.clf()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.savefig("Graphics/%s%s_loss.png" % (optimizerConf["name"], optimizerConf["learning_rate"]), format='png')
    plt.clf()

    result["%s%s" % (optimizerConf["name"], optimizerConf["learning_rate"])] = test_acc
    return model


train_images, train_labels, test_images, test_labels = loadData()

result = dict()

for learning_rate in [0.001, 0.01, 0.1]:
    trainModel(optimizers.Adagrad(learning_rate=learning_rate), EPOHS)
    trainModel(optimizers.Adam(learning_rate=learning_rate), EPOHS)
    trainModel(optimizers.RMSprop(learning_rate=learning_rate), EPOHS)
    trainModel(optimizers.SGD(learning_rate=learning_rate), EPOHS)

for res in result:
    print("%s: %s" % (res, result[res]))
