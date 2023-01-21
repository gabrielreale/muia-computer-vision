from PIL import Image
from sklearn.model_selection import train_test_split
from typing import NamedTuple
import numpy as np
import numpy as np
import os
import pandas as pd
import shutil
import uuid
import warnings

XVIEW_CATEGORIES = {13: 'CARGO_PLANE', 15: 'HELICOPTER', 18: 'SMALL_CAR', 19: 'BUS', 23: 'TRUCK', 41: 'MOTORBOAT', 47: 'FISHING_VESSEL', 60: 'DUMP_TRUCK', 64: 'EXCAVATOR', 73: 'BUILDING', 86: 'STORAGE_TANK', 91: 'SHIPPING_CONTAINER'}
XVIEW_2_YOLO_CATEGORIES = {'CARGO_PLANE': 0, 'HELICOPTER': 1, 'SMALL_CAR': 2, 'BUS': 3, 'TRUCK': 4, 'MOTORBOAT': 5, 'FISHING_VESSEL': 6, 'DUMP_TRUCK': 7, 'EXCAVATOR': 8, 'BUILDING': 9, 'STORAGE_TANK': 10, 'SHIPPING_CONTAINER': 11}

class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.category = -1
        self.score = -1

class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)


def load_filename(path, line, main_imgs_dir):
    parts = line.strip().split(';')
    image = GenericImage(path + parts[0])
    num_predictions = int(parts[1])
    img_filepath = os.path.join(main_imgs_dir, image.filename)
    width, height = Image.open(img_filepath).size
    image.tile = np.array([0, 0, width, height])
    image.width = width
    image.height = height
    image.gsd = 0.3
    for idx in range(0, num_predictions):
        obj = GenericObject()
        obj.id = int(parts[(3*idx)+2])
        pts = parts[(3*idx)+3].split(',')
        obj.bb = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
        cat = int(parts[(3*idx)+4])
        if cat not in XVIEW_CATEGORIES.keys():
            continue
        obj.category = XVIEW_CATEGORIES[cat]
        obj.score = 1.0
        image.add_object(obj)
    return image


def load_database_txt(anns_file, main_imgs_dir):
    pos = anns_file.rfind('/') + 1
    path = anns_file[:pos]
    with open(anns_file, 'r', encoding='utf-8') as ifs:
        lines = ifs.readlines()
        anns = []
        for i in range(len(lines)):
            parts = lines[i].strip().split(';')
            if parts[0] == '#' or parts[0] == '@':
                continue
            anns.append(load_filename(path, lines[i], main_imgs_dir))
    ifs.close()
    return anns


def get_label_str_in_yolo_format(obj: GenericObject, img_w: int, img_h: int):
    xvview_format_class_name = obj.category
    yolo_class_id = XVIEW_2_YOLO_CATEGORIES[xvview_format_class_name]
                
    # Get Bounding Box region
    x1, y1, x2, y2 = obj.bb
    w, h = x2 - x1, y2 - y1
    x = x1
    y = y1
    
    # Finding midpoints
    x_centre = (x + (x+w))/2
    y_centre = (y + (y+h))/2
    
    # Normalization
    x_centre = x_centre / img_w
    y_centre = y_centre / img_h
    w = w / img_w
    h = h / img_h
    
    # Limiting upto fix number of decimal places
    x_centre = format(x_centre, '.6f')
    y_centre = format(y_centre, '.6f')
    w = format(w, '.6f')
    h = format(h, '.6f')

    return f"{yolo_class_id} {x_centre} {y_centre} {w} {h}\n"


def move_xview_image_to_yolo_dir(image_id: str, xview_stage_images_dir: str, yolo_images_dir_path: str):
    # Get tif image
    file_name = f"{image_id}.tif"
    tif_file_path = os.path.join(xview_stage_images_dir, file_name)

    # Get output image path
    output_img_path = os.path.join(yolo_images_dir_path, file_name)

    # Copy to output path
    shutil.copyfile(tif_file_path, output_img_path)


def xview_data_dir_to_yolo_data_dir(xview_main_imgs_dir: str, output_yolo_main_dir: str):
    # Create output directory
    os.makedirs(output_yolo_main_dir, exist_ok=True)
    if len(os.listdir(output_yolo_main_dir)) > 0:
        raise ValueError(f"Output directory is not empty: {output_yolo_main_dir}. \nPlease clean the directory before running this code.")
    
    for stage_str in ['train', 'test']:
        yolo_labels_dir_path = os.path.join(output_yolo_main_dir, stage_str, 'labels')
        yolo_images_dir_path = os.path.join(output_yolo_main_dir, stage_str, 'images')
        os.makedirs(yolo_labels_dir_path, exist_ok=True)
        os.makedirs(yolo_images_dir_path, exist_ok=True)

        anns_file = os.path.join(xview_dir, f"xview_ann_{stage_str}.txt")
        print('Open annotations file: ' + str(anns_file))
        if os.path.isfile(anns_file):
            anns = load_database_txt(anns_file, xview_dir)
        else:
            raise ValueError('Annotations file does not exist')
        print(f'Number of {stage_str} images: {len(anns)}')
        
        xview_stage_images_dir = os.path.join(xview_main_imgs_dir, stage_str)

        # Get via_csv_row attributes
        for ann in anns:
            # Get image ID
            image_id = os.path.splitext(os.path.basename(ann.filename))[0]

            # Move the scan image contents to YOLO output format
            move_xview_image_to_yolo_dir(image_id, xview_stage_images_dir, yolo_images_dir_path)

            if image_id == "38":
                print(f"{ann.width}, {ann.height}")

            # Iterate over each object
            for obj in ann.objects:
                # Transform label in corresponding YOLO format
                yolo_label_str = get_label_str_in_yolo_format(obj, ann.width, ann.height)
                        
                # Writing current object label in yolo format
                label_path = os.path.join(yolo_labels_dir_path, f"{image_id}.txt") 
                with open(label_path, "a") as file_object:
                    file_object.write(yolo_label_str)


def yolo_train_dev_split(yolo_train_dir: str, dev_size=0.2, random_state=1):

    yolo_train_imgs_dir = os.path.join(yolo_train_dir, "images")
    yolo_train_imgs_listdir = os.listdir(yolo_train_imgs_dir)

    yolo_train_imgs_filenames, yolo_val_imgs_filenames = train_test_split(yolo_train_imgs_listdir, test_size=dev_size, random_state=random_state, shuffle=True)

    # Create val folders
    yolo_valid_labels_dir_path = os.path.join(output_yolo_main_dir, 'valid', 'labels')
    yolo_valid_images_dir_path = os.path.join(output_yolo_main_dir, 'valid', 'images')
    os.makedirs(yolo_valid_labels_dir_path, exist_ok=True)
    os.makedirs(yolo_valid_images_dir_path, exist_ok=True)
    
    # Only move val files to the corresponding file folders
    for yolo_val_img_filename in yolo_val_imgs_filenames:
        image_id = os.path.splitext(os.path.basename(yolo_val_img_filename))[0]

        # Get src path
        yolo_val_img_src_filepath = os.path.join(yolo_train_imgs_dir, yolo_val_img_filename)
        yolo_val_lbl_src_filepath = os.path.join(yolo_train_dir, "labels", f"{image_id}.txt")
        # Get dst path
        yolo_val_img_dst_filepath = os.path.join(yolo_valid_images_dir_path, yolo_val_img_filename)
        yolo_val_lbl_dst_filepath = os.path.join(yolo_valid_labels_dir_path, f"{image_id}.txt")

        # Copy to output path
        shutil.move(yolo_val_img_src_filepath, yolo_val_img_dst_filepath)
        shutil.move(yolo_val_lbl_src_filepath, yolo_val_lbl_dst_filepath)

if __name__ == "__main__":
    # Load database
    xview_dir = r"C:\Gabriel\Edu\MUIA\ComputerVision\datasets\xview_detection"
    output_yolo_main_dir = r"C:\Gabriel\Edu\MUIA\ComputerVision\datasets\xview_detection_yolo"
    # xview_data_dir_to_yolo_data_dir(xview_dir, output_yolo_main_dir)
    yolo_train_dev_split(os.path.join(output_yolo_main_dir, "train"))
