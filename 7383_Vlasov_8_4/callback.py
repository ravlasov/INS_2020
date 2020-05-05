import tensorflow as tf
import datetime

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, prefix="model"):
        super(SaveModelCallback, self).__init__()
        self.epochs = epochs
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:
            self.model.save("%s_%s_%s.h5" % (datetime.datetime.now().strftime("%Y-%m-%d"), self.prefix, epoch))