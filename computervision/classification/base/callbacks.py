import keras
import time
import datetime

class LogTimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class LogEpochTime(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print("Started training at: ", datetime.datetime.now())

    def on_epoch_begin(self, batch, logs={}):
        print(f"Epoch started on batch {batch} at: ", datetime.datetime.now())

    def on_epoch_end(self, batch, logs={}):
        print(f"Epoch finished with batch {batch} at: ", datetime.datetime.now())
