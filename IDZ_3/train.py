import os
from utils import *
from settings import *
from model import RecognitionModel

model = RecognitionModel()

if os.path.exists(SAVE_PATH_PREFIX + BACKUP_PATH):
    print("Loading existing model...")
    model.load_model(SAVE_PATH_PREFIX + BACKUP_PATH)

images, targets = load_train_resources_low_RAM()
model.fit_model_low_RAM(images, targets)
