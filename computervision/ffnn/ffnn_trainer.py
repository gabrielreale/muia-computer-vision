from keras.models import Model
from typing import Callable, Generator, Optional, Sequence, Tuple
import numpy as np

from computervision.base.base_trainer import BaseModelTrainer
from computervision.data.base_data import GenericObject, generator_images

class FFNNModelTrainer(BaseModelTrainer):
    def __init__(self, model: Model, training_comment: str) -> None:
        super().__init__(model, training_comment)

    def preprocess_training_data(self, 
        train_objs: Tuple[str, GenericObject], validation_objs: Tuple[str, GenericObject], batch_size: Optional[int] = 16,
        transform_train_imgs_func: Optional[Callable] = None, transform_validation_imgs_func: Optional[Callable] = None) -> Tuple[
            Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None],
            Generator[Tuple[np.ndarray[np.float32], np.ndarray[np.float32]], None, None]]:
        
        # Generators
        train_generator = generator_images(train_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=True)
        valid_generator = generator_images(validation_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=False)

        return train_generator, valid_generator
