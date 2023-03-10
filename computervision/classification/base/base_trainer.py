from abc import ABC, abstractclassmethod
from keras.callbacks import Callback
from keras.models import Model
from typing import Callable, Generator, Optional, Sequence, Tuple
import numpy as np

from computervision.classification.data.base_data import GenericObject, generator_images

class BaseModelTrainer(ABC):
    def __init__(self, model: Model, training_comment: str) -> None:
        super().__init__()

        self._model = model
        self._training_comment = training_comment

    @property
    def model(self):
        return self._model

    def preprocess_training_data(self, 
        train_objs: Tuple[str, GenericObject], validation_objs: Tuple[str, GenericObject], batch_size: Optional[int] = 16,
        transform_train_imgs_func: Optional[Callable] = None, transform_validation_imgs_func: Optional[Callable] = None):
        
        # Generators
        train_generator = generator_images(train_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=True)
        valid_generator = generator_images(validation_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=False)

        return train_generator, valid_generator

    def train(self, 
        train_generator,
        validation_generator,
        optimizer: str, loss: str, metrics: Sequence[str],
        train_steps: int, validation_steps: int, epochs: int, callbacks: Optional[Sequence[Callback]]):

        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        h = self._model.fit(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, validation_steps=validation_steps, epochs=epochs, callbacks=callbacks, verbose=1)

        # Best validation model
        best_idx = int(np.argmax(h.history['val_accuracy']))
        best_value = np.max(h.history['val_accuracy'])
        print('Best validation model: epoch ' + str(best_idx+1), ' - val_accuracy ' + str(best_value))
