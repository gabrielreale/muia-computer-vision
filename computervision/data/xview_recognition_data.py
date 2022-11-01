import os 
import json
import numpy as np
from typing import Any, Optional, Sequence, Tuple

from computervision.data.base_data import GenericObject, GenericImage

def get_categories():
    return {
        13: 'CARGO_PLANE', 
        15: 'HELICOPTER', 
        18: 'SMALL_CAR', 
        19: 'BUS',
        23: 'TRUCK', 
        41: 'MOTORBOAT', 
        47: 'FISHING_VESSEL', 
        60: 'DUMP_TRUCK', 
        64: 'EXCAVATOR', 
        73: 'BUILDING', 
        86: 'STORAGE_TANK', 
        91: 'SHIPPING_CONTAINER'}


def get_image_objects_list_from_file(database_json_file: str, dataset_base_dir: str, maximum_training_samples_per_category: Optional[int] = 5000) -> Tuple[Sequence[GenericImage], int]:
    categories = get_categories()
    json_data = load_json_database(database_json_file)

    counts = dict.fromkeys(categories.values(), 0)
    anns = []
    for json_img, json_ann in zip(json_data['images'], json_data['annotations']):
        image = GenericImage(os.path.join(dataset_base_dir, json_img['file_name']))
        image.tile = np.array([0, 0, json_img['width'], json_img['height']])
        obj = GenericObject()
        obj.id = json_ann['id']
        obj.bb = (int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
        obj.category = list(categories.values())[json_ann['category_id']-1]
        # Resampling strategy to reduce training time
        if counts[obj.category] >= maximum_training_samples_per_category:
            continue
        counts[obj.category] += 1
        image.add_object(obj)
        anns.append(image)
    print(counts)

    return anns, counts


def load_json_database(json_file: str) -> Any:
    with open(json_file) as ifs:
        json_data = json.load(ifs)
    ifs.close()

    return json_data

if __name__ == "__main__":
    print("HELP")