from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from var4 import gen_data
from callback import SaveModelCallback

EPOCHS = 5
DATA_LENGTH = 1000
IMAGE_SIZE = 50
TEST_SIZE = 0.2
SAVE_MODEL_ON_EPOCHS = [0, 1, 3]
PREFIX = "myModel"

def buildModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(IMAGE_SIZE//2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def loadData():
    data, labels = gen_data(DATA_LENGTH, IMAGE_SIZE)
    data = data.reshape(data.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    encoder = LabelEncoder()
    encoder.fit(labels.ravel())
    labels = encoder.transform(labels.ravel())
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=TEST_SIZE)
    return train_data, test_data, train_labels, test_labels

train_data, test_data, train_labels, test_labels = loadData()
model = buildModel()
model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=10, validation_data=(test_data, test_labels),
          callbacks=[SaveModelCallback(SAVE_MODEL_ON_EPOCHS, PREFIX)])
ev = model.evaluate(test_data, test_labels)
print("Model accuracy: %s" % (ev[1]))