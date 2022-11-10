from datetime import datetime
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from typing import Callable, Generator, Optional, Sequence, Tuple
import numpy as np

from computervision.base.base_trainer import BaseModelTrainer
from computervision.data.base_data import GenericObject, generator_images
from computervision.ffnn.ffnn_params_parser import FFNNParamsParser

class FFNNModelTrainer(BaseModelTrainer):
    def __init__(self, ffnn_params_parser: FFNNParamsParser, num_categories: str, training_comment: str) -> None:
        
        number_of_hidden_layers = ffnn_params_parser.get_number_of_dense_hidden_layers()
        num_of_layers = 1 + number_of_hidden_layers
        neurons_per_hidden_layer = ffnn_params_parser.get_neurons_per_hidden_layer()
        activation = ffnn_params_parser.get_activation_name()
        dropout = ffnn_params_parser.get_dropout_value()
        optimizer_str = ffnn_params_parser.get_optimizer_str()
        lr=ffnn_params_parser.get_learning_rate()
        batch_size=ffnn_params_parser.get_batch_size()
        data_augmentation_flag = ffnn_params_parser.get_data_augmentation_flag()
        
        # Get name
        data_augmentation_str = "DATAAUGM_" if data_augmentation_flag else ""
        self.model_name = f"FFNN_{data_augmentation_str}opt_{optimizer_str}_lr_{str(lr).replace('.', '')}_lyrs_{num_of_layers}_batch_size_{batch_size}_time_{datetime.now().strftime('%Y%m%d%H%M')}"
    
        print('Load model')
        model = Sequential(name=self.model_name)
        model.add(Flatten(input_shape=(224, 224, 3)))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        for i in range(number_of_hidden_layers):
            model.add(Dense(neurons_per_hidden_layer[i]))
            model.add(Activation(activation))
            model.add(Dropout(dropout))
        model.add(Dense(num_categories))
        model.add(Activation('softmax'))
        model.summary()
    
        super().__init__(model, training_comment)

    def preprocess_training_data(self, 
        train_objs: Tuple[str, GenericObject], validation_objs: Tuple[str, GenericObject], batch_size: Optional[int] = 16,
        transform_train_imgs_func: Optional[Callable] = None, transform_validation_imgs_func: Optional[Callable] = None):
        
        # Generators
        train_generator = generator_images(train_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=True)
        valid_generator = generator_images(validation_objs, batch_size, transform=transform_train_imgs_func, do_shuffle=False)

        return train_generator, valid_generator
