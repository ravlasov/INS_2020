import csv
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential

X_MIN = 0
X_MAX = 10
E_MIN = 0
E_MAX = 0.3
ROUND = 3
INPUT_DIMENSIONS = 6
TRAIN_SIZE = 300
TEST_SIZE = 50



def save_to_csv(path, data):
    with open(path, 'w', newline='') as file:
        output = csv.writer(file, delimiter=',')
        for i in data:
            output.writerow(np.round(i, decimals=ROUND))


def generate_dataset(size):
    dataset = []
    target = []
    for i in range(size):
        X = np.random.normal(X_MIN, X_MAX)
        e = np.random.normal(E_MIN, E_MAX)
        tmp = []
        tmp.append(np.cos(X) + e)
        tmp.append(-X + e)
        tmp.append(np.sin(X) * X + e)
        tmp.append(X ** 2 + e)
        tmp.append(-np.fabs(X) + 4)
        tmp.append(X - (X ** 2) / 5 + e)
        dataset.append(tmp)
        target.append([np.sqrt(np.fabs(X)) + e])
    return np.round(np.array(dataset), decimals=ROUND), np.round(np.array(target), decimals=ROUND)


def build_model():
    model = Sequential()
    model.add(Dense(60, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1))
    return model


def create_objects():
    main_input = Input(shape=(INPUT_DIMENSIONS,), name='main_input')
    encoded = Dense(60, activation='relu')(main_input)
    encoded = Dense(60, activation='relu')(encoded)
    encoded = Dense(6, activation='linear')(encoded)

    input_encoded = Input(shape=(6,), name='input_encoded')
    decoded = Dense(35, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(INPUT_DIMENSIONS, name="out_aux")(decoded)

    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")

    return encoded, decoded, predicted, main_input


def generate_data():
    x_train, y_train = generate_dataset(TRAIN_SIZE)
    x_test, y_test = generate_dataset(TEST_SIZE)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train -= y_mean
    y_train /= y_std
    y_test -= y_mean
    y_test /= y_std

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = generate_data()

encoded, decoded, full_model, main_input = create_objects()

keras_model = build_model()
keras_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
keras_model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_test, y_test))

full_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
full_model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_test, y_test))

encoded_data = encoded.predict(x_test)
decoded_data = decoded.predict(encoded_data)
regression = full_model.predict(x_test)

save_to_csv('./x_train.csv', x_train)
save_to_csv('./y_train.csv', y_train)
save_to_csv('./x_test.csv', x_test)
save_to_csv('./y_test.csv', y_test)
save_to_csv('./encoded.csv', encoded_data)
save_to_csv('./decoded.csv', decoded_data)
save_to_csv('./regression.csv', regression)

decoded.save('decoder.h5')
encoded.save('encoder.h5')
full_model.save('regression.h5')
