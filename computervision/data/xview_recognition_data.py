import os 
import json
import random
import numpy as np
from typing import Any, Mapping, Optional, Sequence, Tuple

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


def get_image_objects_list_from_file(
    database_json_file: str, 
    dataset_base_dir: str, 
    maximum_training_samples_per_category: Optional[int] = 5000) -> Tuple[Sequence[GenericImage], int]:
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

def oversample_image_objects(
    img_objs: Sequence[GenericImage],
    categories_to_oversample_and_size_multiplier: Mapping[str, float],
    seed_value: Optional[int] = 11) -> Tuple[Sequence[GenericImage], int]:
    print("Annotations initial size: ", len(img_objs))
    
    categories = get_categories()
    img_objs_per_category = dict.fromkeys(categories.values(), None)
    # Create a dictionary holding sequences of the same class
    for img_obj in img_objs:
        current_img_cat = img_obj.objects[0].category
        if img_objs_per_category[current_img_cat] is None:
            img_objs_per_category[current_img_cat] = [img_obj] 
        else:
            img_objs_per_category[current_img_cat].append(img_obj)

    print("img_objs_per_category: ", img_objs_per_category.keys())

    random.seed(a=seed_value)
    for category_to_oversample in categories_to_oversample_and_size_multiplier.keys():
        # Get images of category
        imgs_to_oversample = img_objs_per_category[category_to_oversample]
        print(category_to_oversample)
        print("imgs_to_oversample: ", len(imgs_to_oversample))
        # Get the multiplier 
        category_size_multiplier = categories_to_oversample_and_size_multiplier[category_to_oversample]
        print("category_size_multiplier: ", category_size_multiplier)
        # Obtain the oversampled sequence length
        output_size = len(imgs_to_oversample) * category_size_multiplier
        print("output_size: ", output_size)
        # Randomly choose k elemnts of the current list with replcament
        oversampled_imgs = random.choices(imgs_to_oversample, weights=None, cum_weights=None, k=output_size) # with replacement
        print("oversampled_imgs: ", len(oversampled_imgs))
        # Overwrite dictionary
        img_objs_per_category[category_to_oversample] = oversampled_imgs
        
    # Generate the output list
    counts = dict.fromkeys(categories.values(), 0)
    anns = []
    for category in img_objs_per_category.keys():
        counts[category] = len(img_objs_per_category[category])
        anns += img_objs_per_category[category] # Concatenate list

    # Shuffle the items in the list
    random.shuffle(anns)
    print("Annotations size: ", len(anns))
    print(counts)
    return anns, counts   

def load_json_database(json_file: str) -> Any:
    with open(json_file) as ifs:
        json_data = json.load(ifs)
    ifs.close()

    return json_data

if __name__ == "__main__":
    print("HELP")