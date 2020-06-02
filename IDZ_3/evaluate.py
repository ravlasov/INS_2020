import os
from utils import *
from settings import *
from model import RecognitionModel

model = RecognitionModel()

if os.path.exists(SAVE_PATH_PREFIX + BACKUP_PATH):
    print("Loading existing model...")
    model.load_model(SAVE_PATH_PREFIX + BACKUP_PATH)
else:
    print("Warning! Model must be at '%s'. Evaluating brand new model" % (SAVE_PATH_PREFIX + BACKUP_PATH))

images, targets = load_train_resources_low_RAM()
print(model.evaluate_low_RAM(images, targets))
