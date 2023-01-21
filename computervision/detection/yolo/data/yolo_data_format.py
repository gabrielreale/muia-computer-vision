import os
import cv2
import matplotlib.pyplot as plt
import random

XVIEW_2_YOLO_CATEGORIES = {'CARGO_PLANE': 0, 'HELICOPTER': 1, 'SMALL_CAR': 2, 'BUS': 3, 'TRUCK': 4, 'MOTORBOAT': 5, 'FISHING_VESSEL': 6, 'DUMP_TRUCK': 7, 'EXCAVATOR': 8, 'BUILDING': 9, 'STORAGE_TANK': 10, 'SHIPPING_CONTAINER': 11}
YOLO_2_XVIEW_CATEGORIES = {v: k for k, v in XVIEW_2_YOLO_CATEGORIES.items()} # Reverse keys and values


def add_box_to_image_with_start_end_pts(x1, x2, image, color=None, label=None, line_thickness=None):
    """Refactored from https://github.com/waittim/draw-YOLO-box/blob/main/draw_box.py

    Args:
        x1 (_type_): [x1,y1] Start point of the bounding box
        x2 (_type_): [x2,y2] End point of the bounding box
        image (_type_): _description_
        color (_type_, optional): _description_. Defaults to None.
        label (_type_, optional): _description_. Defaults to None.
        line_thickness (_type_, optional): _description_. Defaults to None.
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1 = (int(x1[0]), int(x1[1]))
    c2 = (int(x2[0]), int(x2[1]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def add_box_to_image_with_center_point_and_dims(x_center, bb_width, bb_height, image, color=None, label=None, line_thickness=None):
    x_c = x_center[0]
    y_c = x_center[1]

    if isinstance(x_c, float) or isinstance(y_c, float) or isinstance(bb_width, float) or isinstance(bb_height, float):
        # Values are normalized so we need to transform them back
        img_w = image.shape[1]
        img_h = image.shape[0]

        x_c = x_c * img_w
        y_c = y_c * img_h
        bb_width = bb_width * img_w
        bb_height = bb_height * img_h
        print(f"{img_w}, {img_h}, {x_c}, {y_c}, {bb_width}, {bb_height}")

    x1 = round(x_c-bb_width/2)
    y1 = round(y_c-bb_height/2)
    x2 = round(x_c+bb_width/2)
    y2 = round(y_c+bb_height/2)
    
    add_box_to_image_with_start_end_pts([x1,y1], [x2,y2], image, color=color, label=label, line_thickness=line_thickness)


def draw_box_on_image(image_name, classes, colors, label_folder, raw_images_folder, save_images_folder ):
    """Refactored from https://github.com/waittim/draw-YOLO-box/blob/main/draw_box.py

    Args:
        image_name (_type_): _description_
        classes (_type_): _description_
        colors (_type_): _description_
        label_folder (_type_): _description_
        raw_images_folder (_type_): _description_
        save_images_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    txt_path  = os.path.join(label_folder,'%s.txt'%(image_name))
    print(image_name)
    if image_name == '.DS_Store':
        return 0
    image_path = os.path.join( raw_images_folder,'%s.jpg'%(image_name))
    
    save_file_path = os.path.join(save_images_folder,'%s.jpg'%(image_name))
    
    source_file = open(txt_path) if os.path.exists(txt_path) else []
    image = cv2.imread(image_path)
    try:
        height, width, channels = image.shape
    except:
        print('no shape info.')
        return 0

    box_number = 0
    for line in source_file:
        staff = line.split()
        class_idx = int(staff[0])

        x_center, y_center, w, h = float(staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)     
        
        add_box_to_image_with_start_end_pts([x1,y1], [x2,y2], image, color=colors[class_idx], label=classes[class_idx], line_thickness=None)

        cv2.imwrite(save_file_path,image) 

        box_number += 1
    return box_number
    

class YoloLabel():
    def __init__(self, img_label_file: str) -> None:
        self.anns = []

        with open(img_label_file, 'r') as fl:
            data = fl.readlines()

            for dt in data:
                # Split string to float
                cat_id, x, y, w, h = map(float, dt.split(' '))
                self.anns.append(
                    (int(cat_id), {
                        "x_center": float(x), 
                        "y_center": float(y), 
                        "bb_width": float(w), 
                        "bb_height": float(h), 
                }))

    def get_num_of_classes(self) -> int:
        return len(self.anns)


class YoloImage():
    def __init__(self, img_file: str) -> None:
        self.image = cv2.imread(img_file)
        

if __name__ == "__main__":
    img_id = "38"
    yolo_main_dir = r"C:\Gabriel\Edu\MUIA\ComputerVision\datasets\xview_detection_yolo6\train"

    img_file_path = os.path.join(yolo_main_dir, "images", f"{img_id}.tif")
    lbl_file_path = os.path.join(yolo_main_dir, "labels", f"{img_id}.txt")

    # Load
    yolo_image = YoloImage(img_file_path)
    yolo_label = YoloLabel(lbl_file_path)

    anns = yolo_label.anns
    img = yolo_image.image
    for class_id, bbox in anns:
        x_center = bbox["x_center"], bbox["y_center"]
        bb_width = bbox["bb_width"] 
        bb_height = bbox["bb_height"] 
        class_name = YOLO_2_XVIEW_CATEGORIES[class_id]

        add_box_to_image_with_center_point_and_dims(x_center, bb_width, bb_height, image=img, label=class_name, line_thickness=1)

    plt.imshow(img)
    plt.show()
