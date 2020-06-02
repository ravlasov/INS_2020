# Ссылки на архивы
RESOURCES = {
    "listsTXT.tar.gz": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ListsTXT.tgz",
    "img.tar.gz": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz",
    "hnd.tar.gz": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz",
    "fnt.tar.gz": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
}

# Связь между списком изображений и папкой их хранения
LISTS = [
    ["list_English_Fnt.m", "English/Fnt/"],
    ["list_English_Hnd.m", "English/Hnd/"],
    ["list_English_Img.m", "English/Img/"]
]

LIST_PREFIXES = [
    ""
]

# Регулярные выражения для считывания нужных данных из .m файлов
LIST_BORDERS = [[r'list.ALLlabels = \[.*?\n\]', "list.ALLlabels = [", "\n]"],
                 [r'list.ALLnames = \[.*?\n\]', "list.ALLnames = [", "\n]"]]

SAVE_PATH_PREFIX = ""
# Директория для загрузки датасета
DOWNLOAD_DIRECTORY = "./cache/"
IMAGES_FORMAT = ".png"

EPOCHS_MAX = 100
MODEL_PATH = "trained_model.hdf5"
BACKUP_PATH = "backup_model.hdf5"
VALIDATION = 0.15
