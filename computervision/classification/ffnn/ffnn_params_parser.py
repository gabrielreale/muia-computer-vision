from typing import Sequence
from computervision.classification.base.base_params_parser import BaseParamsParser

class FFNNParamsParser(BaseParamsParser):
    def __init__(self, params_json_file: str) -> None:
        super().__init__(params_json_file)

    def get_number_of_dense_hidden_layers(self) -> int:
        return self.params["number_of_hidden_layers"]

    def get_neurons_per_hidden_layer(self) -> Sequence[int]:
        return self.params["neurons_per_hidden_layer"]
