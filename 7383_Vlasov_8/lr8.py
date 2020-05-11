import os
import requests
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import LSTM, Dropout, Dense
from keras.utils import np_utils

BOOK_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
BOOK_PATH = "./alice's_adventures_in_wonderland.txt"
MODEL_PATH = "./bestModel.hdf5"

EPOCHS = 100
BATCH_SIZE = 256

char_to_int = None
int_to_char = None
n_vocab = None


class RuntimeTestCallback(Callback):
    def __init__(self, pattern, frequency=1):
        super(RuntimeTestCallback, self).__init__()
        self.seed = pattern
        self.frequency = frequency

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch + 1) % self.frequency == 0:
            res = generateText(self.seed, self.model, 100)
            with open("generated_on_{}.txt".format(epoch), "w") as f:
                f.write(res)


def loadData():
    global char_to_int
    global int_to_char
    global n_vocab
    if not os.path.exists(BOOK_PATH):
        f = requests.get(BOOK_URL)
        open(BOOK_PATH, 'wb').write(f.content)
    raw_text = open(BOOK_PATH).read()
    raw_text = raw_text.lower()
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    dataX = []
    dataY = []
    seq_length = 100
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    Y = np_utils.to_categorical(dataY)
    return X, Y, dataX, dataY


def buildModel(shapeIn, shapeOut):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shapeIn[1], shapeIn[2])))
    model.add(Dropout(0.2))
    model.add(Dense(shapeOut[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def trainModel():
    start = np.random.randint(0, len(dataX) - 1)
    seed = dataX[start]
    print("Seed: \"", ''.join([int_to_char[value] for value in seed]), "\"")
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    earlyStopping = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=2)
    customCallback = RuntimeTestCallback(seed, 1)
    callbacks_list = [checkpoint, earlyStopping, customCallback]
    model = buildModel(X.shape, Y.shape)
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)


def generateText(pattern, model, length=100):
    pattern = pattern.copy()
    res = ""
    for i in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        res += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return res


X, Y, dataX, dataY = loadData()

trainModel()

start = np.random.randint(0, len(dataX) - 1)
seed = dataX[start]
print("Seed: \"", ''.join([int_to_char[value] for value in seed]), "\"")
model = buildModel(X.shape, Y.shape)
model.load_weights(MODEL_PATH)
text = generateText(seed, model, 500)
with open("generatedText.txt", "w") as f:
    f.write(text)
