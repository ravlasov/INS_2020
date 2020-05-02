import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.datasets import imdb
import string

EPOCHS = 2
BATCH_SIZE = 500
TEST_DIMENSIONS = [10, 50, 100, 500, 1000, 5000, 10000]
CUSTOM_REVIEWS = [
    "amazing",
    "nothing special"]


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def buildModel(dimensions=10000):
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(dimensions,)))
    model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.35, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def loadData(dimension=10000):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, dimension=dimension)
    targets = np.array(targets).astype("float32")
    return data[10000:], targets[10000:], data[:10000], targets[:10000]


def testDimensions():
    ev_accuracy = dict()
    ev_loss = dict()
    for dim in TEST_DIMENSIONS:
        print("testing %d dimensions" % dim)
        train_x, train_y, test_x, test_y = loadData(dim)
        model = buildModel(dim)
        results = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x, test_y))
        ev_loss["%s" % dim], ev_accuracy["%s" % dim] = model.evaluate(test_x, test_y)
        plt.title('Training and test accuracy')
        plt.plot(results.history['accuracy'], 'r', label='train')
        plt.plot(results.history['val_accuracy'], 'b', label='test')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig("Graphics/%s_acc.png" % dim, format='png')
        plt.clf()

        plt.title('Training and test loss')
        plt.plot(results.history['loss'], 'r', label='train')
        plt.plot(results.history['val_loss'], 'b', label='test')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("Graphics/%s_loss.png" % dim, format='png')
        plt.clf()

    for entry in ev_accuracy:
        print("%s: %s" % (entry, ev_accuracy[entry]))
        plt.barh(entry, ev_accuracy[entry])
    plt.title('Models accuracy summarize')
    plt.ylabel('accuracy')
    plt.xlabel('dimensions')
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.gca().invert_yaxis()
    plt.savefig('Graphics/summarize.png', format='png')
    plt.clf()


def predict(review, model, dimensions=10000):
    punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    review = review.lower().translate(punctuation).split(" ")
    indexes = imdb.get_word_index()
    encoded = []
    for w in review:
        if w in indexes and indexes[w] < dimensions:
            encoded.append(indexes[w])
    review = vectorize([np.array(encoded)], dimensions)
    return model.predict(review)[0][0]


def testCustomReview():
    dims = 6000
    train_x, train_y, test_x, test_y = loadData(dims)
    model = buildModel(dims)
    model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x, test_y))
    for review in CUSTOM_REVIEWS:
        print('"%s" for %s%% is a good review' % (str(review), predict(review, model, dims) * 100))


testCustomReview()
#testDimensions()
