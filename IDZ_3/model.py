import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adadelta
from settings import *
from tqdm.keras import TqdmCallback
from utils import *


class RecognitionModel:
    model = None

    def __init__(self):
        input_layer = Input(shape=(None, None, 3))
        hidden_layer = ZeroPadding2D((2, 2))(input_layer)
        hidden_layer = Conv2D(32, (3, 3), activation='relu', padding="same")(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = ZeroPadding2D((2, 2))(hidden_layer)
        hidden_layer = Conv2D(64, (3, 3), activation='relu', padding="same")(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = ZeroPadding2D((2, 2))(hidden_layer)
        hidden_layer = Conv2D(128, (3, 3), activation='relu', padding="same")(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = ZeroPadding2D((2, 2))(hidden_layer)
        hidden_layer = Conv2D(128, (3, 3), activation='relu', padding="same")(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = Conv2D(256, (1, 1))(hidden_layer)
        hidden_layer = GlobalAveragePooling2D()(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)
        hidden_layer = Dense(8192, activation="relu")(hidden_layer)
        output_layer = Dense(62, activation="softmax")(hidden_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        # for layer in self.model.layers[:3]:
        # layer.trainable = False
        self.model.compile(optimizer=Adadelta(learning_rate=0.5), loss="categorical_crossentropy", metrics=["accuracy"])

    def predict(self, image):
        ans = self.model.predict(image)
        return get_character(np.argmax(ans))

    def get_model(self):
        return self.model

    def load_model(self, path):
        self.model.load_weights(path)

    def fit_model(self, X_train, Y_train):
        order = get_random_order(len(X_train))
        val = int(len(X_train) * VALIDATION)
        self.model.fit_generator(generate(X_train, Y_train, order[val:]), steps_per_epoch=len(X_train[val:]),
                                           nb_epoch=EPOCHS_MAX, validation_steps=len(X_train[:val]),
                                           validation_data=generate(X_train, Y_train, order[:val]), verbose=0,
                                           callbacks=self.get_callbacks())

    def fit_model_low_RAM(self, X_train, Y_train):
        print("WARNING! Using low RAM mode will lead to a large number of read operations from your disk.")
        order = get_random_order(len(X_train))
        val = int(len(X_train) * VALIDATION)
        self.model.fit_generator(generate_low_RAM(X_train, Y_train, order[val:]),
                                           steps_per_epoch=len(X_train[val:]),
                                           nb_epoch=EPOCHS_MAX, validation_steps=len(X_train[:val]),
                                           validation_data=generate_low_RAM(X_train, Y_train, order[:val]), verbose=0,
                                           callbacks=self.get_callbacks())

    def evaluate(self, X_train, Y_train):
        order = get_random_order(len(X_train))
        loss, acc = self.model.evaluate_generator(generate(X_train, Y_train, order), steps=len(X_train), verbose=1)
        return loss, acc

    def evaluate_low_RAM(self, X_train, Y_train):
        order = get_random_order(len(X_train))
        loss, acc = self.model.evaluate_generator(generate_low_RAM(X_train, Y_train, order), steps=len(X_train),
                                                  verbose=1)
        return loss, acc

    def get_callbacks(self):
        callbacks = []
        callbacks.append(TqdmCallback(verbose=1))
        callbacks.append(ModelCheckpoint(SAVE_PATH_PREFIX + MODEL_PATH, monitor='val_loss',
                                         save_best_only=True, mode='min', verbose=1))
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.3, patience=3, min_lr=0.001,
                                           verbose=1))
        return callbacks
