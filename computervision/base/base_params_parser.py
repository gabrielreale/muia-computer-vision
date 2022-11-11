import json
from typing import Mapping


class BaseParamsParser():
    def __init__(self, params_json_file: str) -> None:
        with open(params_json_file) as ifs:
            self.params = json.load(ifs)
        ifs.close()

    def get_model_type(self) -> str:
        return self.params["model_type"]

    def get_training_comment(self) -> str:
        return self.params["training_comment"]

    def get_oversample_data_flag(self) -> bool:
        return self.params["oversample"]

    def get_data_augmentation_flag(self) -> bool:
        return self.params["data_augmentation"]
    
    def get_categories_to_oversample_and_size_multiplier(self) -> Mapping[str, float]:
        return self.params["categories_to_oversample_and_size_multiplier"]

    def get_max_number_of_samples_by_category(self) -> int:
        return self.params["max_number_of_samples_by_category"]

    def get_activation_name(self) -> str:
        return self.params["activation"]

    def get_dropout_value(self) -> float:
        return self.params["dropout"]

    def get_optimizer_str(self) -> str:
        return self.params["optimizer_str"]

    def get_learning_rate(self) -> float:
        return self.params["learning_rate"]

    def get_momentum(self) -> float:
        return self.params["momentum"]

    def get_epochs(self) -> int:
        return self.params["epochs"]

    def get_batch_size(self) -> int:
        return self.params["batch_size"]
 
    def get_loss_function(self) -> str:
        return self.params["loss"]
