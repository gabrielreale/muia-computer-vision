import os
import json
import gc
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

from computervision.data.xview_recognition_data import get_categories
from computervision.data.base_data import GenericImage, GenericObject, load_geoimage, draw_confusion_matrix


if __name__ == "__main__":
    input_args = sys.argv

    if len(input_args) <= 1 or input_args[1] is None:
        raise ValueError(r"Wrong app usage: python file.py 'model_name'")
    else:
        model_name = input_args[1]

    print("Running a A100 32G job")
    rand_seed = 11
    dataset_dirpath = 'datasets/xview_recognition'
    log_dir = 'log/tensorboard'
    results_dir = 'log/results'
    models_dir = 'models/xview_recognition'
    categories = get_categories()

    # Load database
    json_file = os.path.join(dataset_dirpath, 'xview_ann_test.json')
    with open(json_file) as ifs:
        json_data = json.load(ifs)
    ifs.close()

    anns = []
    for json_img, json_ann in zip(json_data['images'], json_data['annotations']):
        image = GenericImage(os.path.join(dataset_dirpath, json_img['file_name']))
        image.tile = np.array([0, 0, json_img['width'], json_img['height']])
        obj = GenericObject()
        obj.id = json_ann['id']
        obj.bb = (int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
        obj.category = list(categories.values())[json_ann['category_id']-1]
        image.add_object(obj)
        anns.append(image)

    model_path = os.path.join(models_dir, model_name, model_name+'.hdf5')
    print("Model path: ", model_path)

    # Load architecture
    print('Load model')
    model = load_model(model_path)
    model.summary()
    
    y_true, y_pred = [], []
    print(f"Total num of images: {len(anns)}")
    i = 0
    for ann in anns:
        print(f"Predicting image {i}")
        # Load image
        image = load_geoimage(ann.filename)
        for obj_pred in ann.objects:
            # Generate prediction
            warped_image = np.expand_dims(image, 0)
            # Following https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
            print("Converting to tensor")
            tensor = tf.convert_to_tensor(warped_image, dtype=tf.float32)
            predictions = model.predict(tensor)
            # Save prediction
            pred_category = list(categories.values())[np.argmax(predictions)]
            pred_score = np.max(predictions)
            y_true.append(obj_pred.category)
            y_pred.append(pred_category)

        # print(tf.config.experimental.get_memory_info('GPU:0'))
        # Following https://github.com/keras-team/keras/issues/5337
        print("Running GC...")
        gc.collect()
        # print(tf.config.experimental.get_memory_info('GPU:0'))

        i += 1

    print("Predicting confusion matrix.")
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(categories.values()))
    draw_confusion_matrix(cm, categories, normalize=True)
    results_model_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_model_dir, exist_ok=True)
    fig_path = os.path.join(results_model_dir, 'confusion_matrix.jpg')
    plt.savefig(fig_path)

    print("Computing accurracy matrix.")
    # Compute the accuracy
    correct_samples_class = np.diag(cm).astype(float)
    total_samples_class = np.sum(cm, axis=1).astype(float)
    total_predicts_class = np.sum(cm, axis=0).astype(float)
    print('Mean Accuracy: %.3f%%' % (np.sum(correct_samples_class) / np.sum(total_samples_class) * 100))
    acc = correct_samples_class / np.maximum(total_samples_class, np.finfo(np.float64).eps)
    print('Mean Recall: %.3f%%' % (acc.mean() * 100))
    acc = correct_samples_class / np.maximum(total_predicts_class, np.finfo(np.float64).eps)
    print('Mean Precision: %.3f%%' % (acc.mean() * 100))
    for idx in range(len(categories)):
        # True/False Positives (TP/FP) refer to the number of predicted positives that were correct/incorrect.
        # True/False Negatives (TN/FN) refer to the number of predicted negatives that were correct/incorrect.
        tp = cm[idx, idx]
        fp = sum(cm[:, idx]) - tp
        fn = sum(cm[idx, :]) - tp
        tn = sum(np.delete(sum(cm) - cm[idx, :], idx))
        # True Positive Rate: proportion of real positive cases that were correctly predicted as positive.
        recall = tp / np.maximum(tp+fn, np.finfo(np.float64).eps)
        # Precision: proportion of predicted positive cases that were truly real positives.
        precision = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
        # True Negative Rate: proportion of real negative cases that were correctly predicted as negative.
        specificity = tn / np.maximum(tn+fp, np.finfo(np.float64).eps)
        # Dice coefficient refers to two times the intersection of two sets divided by the sum of their areas.
        # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
        f1_score = 2 * ((precision * recall) / np.maximum(precision+recall, np.finfo(np.float64).eps))
        print('> %s: Recall: %.3f%% Precision: %.3f%% Specificity: %.3f%% Dice: %.3f%%' % (list(categories.values())[idx], recall*100, precision*100, specificity*100, f1_score*100))