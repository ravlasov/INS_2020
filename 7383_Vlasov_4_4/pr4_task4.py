import numpy as np
import math
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def logic_func(a, b, c):
    return (a or b) and (b or c)


def correct_result(dataset):
    return np.array([int(logic_func(x[0], x[1], x[2])) for x in dataset])


def element_wise_operations_predict(dataset, weights):
    dataset = dataset.copy()
    relu = lambda x: max(x, 0)
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    activation = [relu for _ in range(len(weights) - 1)]
    activation.append(sigmoid)
    for w in range(len(weights)):
        res = np.zeros((len(dataset), len(weights[w][1])))
        for i in range(len(dataset)):
            for j in range(len(weights[w][1])):
                sum = 0
                for k in range(len(dataset[i])):
                    sum += dataset[i][k] * weights[w][0][k][j]
                res[i][j] = activation[w](sum + weights[w][1][j])
        dataset = res
    return dataset


def numpy_operations_predict(dataset, weights):
    dataset = dataset.copy()
    relu = lambda x: np.maximum(x, 0)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    activation = [relu for _ in range(len(weights) - 1)]
    activation.append(sigmoid)
    for i in range(len(weights)):
        dataset = activation[i](np.dot(dataset, weights[i][0]) + weights[i][1])
    return dataset


def compare_predicts(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    correct_res = correct_result(dataset)
    element_wise_res = element_wise_operations_predict(dataset, weights)
    numpy_res = numpy_operations_predict(dataset, weights)
    model_res = model.predict(dataset)
    print("Correct result:\n", correct_res)
    print("Element-wise operations predict result:\n", element_wise_res)
    print("Numpy operations predict result:\n", numpy_res)
    print("Model predict result:\n", model_res)


dataset = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=3))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
compare_predicts(model, dataset)
model.fit(dataset, correct_result(dataset), epochs=300, batch_size=1)
compare_predicts(model, dataset)
