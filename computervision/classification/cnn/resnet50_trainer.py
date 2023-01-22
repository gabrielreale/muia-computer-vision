from datetime import datetime
import tensorflow as tf
import tensorflow.keras as K

from computervision.classification.base.base_trainer import BaseModelTrainer
from computervision.classification.ffnn.ffnn_params_parser import FFNNParamsParser

class ResNet50ModelTrainer(BaseModelTrainer):
    def __init__(self, ffnn_params_parser: FFNNParamsParser, num_categories: str, training_comment: str) -> None:
        
        optimizer_str = ffnn_params_parser.get_optimizer_str()
        lr=ffnn_params_parser.get_learning_rate()
        batch_size=ffnn_params_parser.get_batch_size()
        data_augmentation_flag = ffnn_params_parser.get_data_augmentation_flag()
        
        # Get name
        data_augmentation_str = "DATAAUGM_" if data_augmentation_flag else ""
        self.model_name = f"ResNet50_{data_augmentation_str}opt_{optimizer_str}_lr_{str(lr).replace('.', '')}_batch_size_{batch_size}_time_{datetime.now().strftime('%Y%m%d%H%M')}"
    
        print('Load model')
        res_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        # Freeze all but last 5 layers
        for layer in res_model.layers[:-5]:
            layer.trainable = False

        # Create final model
        model = K.models.Sequential()
        model.add(res_model)
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(num_categories, activation='softmax'))
        model.summary()
    
        super().__init__(model, training_comment)
