from datetime import datetime
import tensorflow as tf

from computervision.classification.base.base_trainer import BaseModelTrainer
from computervision.classification.ffnn.ffnn_params_parser import FFNNParamsParser

class VGG19ModelTrainer(BaseModelTrainer):
    def __init__(self, ffnn_params_parser: FFNNParamsParser, num_categories: str, training_comment: str) -> None:
        
        optimizer_str = ffnn_params_parser.get_optimizer_str()
        lr=ffnn_params_parser.get_learning_rate()
        batch_size=ffnn_params_parser.get_batch_size()
        data_augmentation_flag = ffnn_params_parser.get_data_augmentation_flag()
        
        # Get name
        data_augmentation_str = "DATAAUGM_" if data_augmentation_flag else ""
        self.model_name = f"VGG19_{data_augmentation_str}opt_{optimizer_str}_lr_{str(lr).replace('.', '')}_batch_size_{batch_size}_time_{datetime.now().strftime('%Y%m%d%H%M')}"
    
        print('Load model')
        model = tf.keras.applications.VGG19(weights=None, input_shape=(224, 224, 3), classes=num_categories)
        model.summary()
    
        super().__init__(model, training_comment)
