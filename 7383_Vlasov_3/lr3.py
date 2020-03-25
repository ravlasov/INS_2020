from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

k = 8
num_val_samples = len(train_data) // k
num_epochs = 25
mae_histories = []
for i in range(k):
    print(i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets))
    mae = history.history['mae']
    v_mae = history.history['val_mae']
    x = range(1, num_epochs + 1)
    mae_histories.append(v_mae)
    plt.figure(i + 1)
    plt.plot(x, mae, label='Training MAE')
    plt.plot(x, v_mae, label='Validation MAE')
    plt.title('absolute error')
    plt.ylabel('absolute error')
    plt.xlabel('epochs')
    plt.legend()

average_mae_history = [np.mean([x[i] for x in mae_histories]) for i in range(num_epochs)]
plt.figure(0)
plt.plot(range(1, num_epochs + 1), average_mae_history)
plt.xlabel('epochs')
plt.ylabel("mean absolute error")
figs = [plt.figure(n) for n in plt.get_fignums()]
for i in range(len(figs)):
        figs[i].savefig("./Graphics/%d.png" %(i), format='png')
