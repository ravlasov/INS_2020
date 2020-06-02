import os
import random
import re
import tarfile

import numpy as np
import requests
from PIL import Image
from settings import *
from tqdm import tqdm


# Скачивание файла
def download(url, path):
    chunk_size = 1024
    file = requests.get(url, stream=True)
    total_size = int(file.headers['content-length'])
    with open(path, 'wb') as f:
        for data in tqdm(iterable=file.iter_content(chunk_size=chunk_size), total=total_size // chunk_size + 1,
                         unit='KB'):
            f.write(data)


# Распаковка архива
def extract_file(path, to_dir='.'):
    with tarfile.open(name=path) as tar:
        for data in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(member=data, path=to_dir)


# Скачивание и подготовка всех необходимых ресурсов
def download_resources():
    for resource in RESOURCES:
        path = "%s%s" % (DOWNLOAD_DIRECTORY, resource)
        print("Downloading %s to %s" % (RESOURCES.get(resource), path), flush=True)
        download(url=RESOURCES.get(resource), path=path)
        print("\nExtracting archive...", flush=True)
        extract_file(path=path, to_dir=DOWNLOAD_DIRECTORY)
        os.remove(path)
        print("\nDone", flush=True)


# Проверка наличия ресурсов
def check_resources():
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.mkdir(DOWNLOAD_DIRECTORY)
        return False
    for list in LISTS:
        if not os.path.exists(DOWNLOAD_DIRECTORY + list[0]):
            return False
    return True


# Извлечение меток из файла .m
def get_labels_from_file(list):
    path = DOWNLOAD_DIRECTORY + list[0]
    with open(path) as f:
        content = f.read()
    labels = re.search(LIST_BORDERS[0][0], content, re.DOTALL).group().replace(LIST_BORDERS[0][1], "").replace(
        LIST_BORDERS[0][2], "").replace(";", "").split("\n")
    return labels


# Извлечение имен из файла .m
def get_names_from_file(list):
    path = DOWNLOAD_DIRECTORY + list[0]
    with open(path) as f:
        content = f.read()
    names = re.search(LIST_BORDERS[1][0], content, re.DOTALL).group().replace(LIST_BORDERS[1][1], "").replace(
        LIST_BORDERS[1][2], "").replace(";", "").replace("'", "").split("\n")
    for i in range(len(names)):
        names[i] = DOWNLOAD_DIRECTORY + list[1] + names[i] + IMAGES_FORMAT
    return names


# Загрузка изображения в память
def load_image(path):
    image = Image.open(path).convert('RGB')
    image = np.asarray(image)
    return image


# Перемешивание изображений
def get_random_order(length):
    order = list(range(length))
    random.shuffle(order)
    return order


# Генератор изображений
def generate(X, Y, order):
    while 1:
        for i in order:
            img = np.reshape(X[i], (1, *(X[i].shape)))
            l = np.zeros(62)
            l[int(Y[i]) - 1] = 1
            l = l.reshape((1, 62))
            yield (img, l)


# Генератор изображений (низкое потребление ОЗУ)
def generate_low_RAM(X, Y, order):
    while 1:
        for i in order:
            img = load_image(X[i])
            img = np.reshape(img, (1, *(img.shape)))
            l = np.zeros(62)
            l[int(Y[i]) - 1] = 1
            l = l.reshape((1, 62))
            yield (img, l)


# Загрузка данных в память, возвращает список картинок и меток
def load_train_resources():
    if not check_resources():
        download_resources()
    allLabels = list()
    allNames = list()
    for l in LISTS:
        allLabels += get_labels_from_file(l)
        allNames += get_names_from_file(l)
    images = []
    labels = []
    for i in tqdm(range(len(allNames)), desc="Loading images: "):
        images.append(load_image(allNames[i]))
        labels.append(allLabels[i])
    return images, labels


# Загрузка данных (низкое потребление ОЗУ), возвращающая список путей до изображений и меток
def load_train_resources_low_RAM():
    if not check_resources():
        download_resources()
    allLabels = list()
    allNames = list()
    for l in LISTS:
        allLabels += get_labels_from_file(l)
        allNames += get_names_from_file(l)
    return allNames, allLabels


# Восстановление символа его номеру
def get_character(num):
    if num < 10:
        return chr(ord('0') + num)
    elif num < 36:
        return chr(ord('A') + num - 10)
    else:
        return chr(ord('a') + num - 36)
