from abc import ABC, abstractclassmethod
from keras.callbacks import Callback
from keras.models import Model
from typing import Callable, Generator, Optional, Sequence, Tuple
import numpy as np

from computervision.data.base_data import GenericObject

class BaseModelTrainer(ABC):
    def __init__(self, model: Model, training_comment: str) -> None:
        super().__init__()

        self._model = model
        self._training_comment = training_comment

    @property
    def model_name(self):
        return self._model.name

    @property
    def model(self):
        return self._model

    @abstractclassmethod
    def preprocess_training_data(self, 
        train_objs: Tuple[str, GenericObject], validation_objs: Tuple[str, GenericObject], 
        transform_train_imgs_func: Optional[Callable] = None, transform_validation_imgs_func: Optional[Callable] = None) -> Tuple[
            Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None],
            Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None]]:
        pass

    def train(self, 
        train_generator: Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None],
        validation_generator: Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None],
        optimizer: str, loss: str, metrics: Sequence[str],
        train_steps: int, validation_steps: int, epochs: int, callbacks: Optional[Sequence[Callback]]):

        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        h = self._model.fit(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, validation_steps=validation_steps, epochs=epochs, callbacks=callbacks, verbose=2)

        # Best validation model
        best_idx = int(np.argmax(h.history['val_accuracy']))
        best_value = np.max(h.history['val_accuracy'])
        print('Best validation model: epoch ' + str(best_idx+1), ' - val_accuracy ' + str(best_value))
