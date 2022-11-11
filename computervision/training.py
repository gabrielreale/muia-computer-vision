import os
import math
import sys
import time
import shutil
import tensorflow as tf
from xml.dom import NotSupportedErr
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from computervision.base.callbacks import LogEpochTime
from computervision.data.xview_recognition_data import get_image_objects_list_from_file, get_categories, oversample_image_objects
from computervision.data.base_data import simple_image_transform
from computervision.ffnn.ffnn_trainer import FFNNModelTrainer
from computervision.cnn.resnet50_trainer import ResNet50ModelTrainer
from computervision.ffnn.ffnn_params_parser import FFNNParamsParser
from computervision.base.base_params_parser import BaseParamsParser

if __name__ == "__main__":
    input_args = sys.argv

    if len(input_args) <= 1 or input_args[1] is None:
        raise ValueError(r"Wrong app usage: python file.py 'params_json_file_path'")
    else:
        params_json_file = input_args[1]

    print("Parsing parameter file from ", params_json_file)
    params_parser = BaseParamsParser(params_json_file)
    model_type = params_parser.get_model_type()
    training_comment = params_parser.get_training_comment()
    categories = get_categories()
    
    if model_type == 'FFNN':
        params_parser = FFNNParamsParser(params_json_file)
        model_trainer = FFNNModelTrainer(params_parser, len(categories), training_comment)
    elif model_type == 'ResNet50':
        model_trainer = ResNet50ModelTrainer(params_parser, len(categories), training_comment)
    else:
        raise NotSupportedErr(f"Model type {model_type} not supported.")

    rand_seed = 11
    dataset_dirpath = 'datasets/xview_recognition'
    log_dir = 'log/tensorboard'
    models_dir = 'models/xview_recognition'
    oversample_flag = params_parser.get_oversample_data_flag()
    data_augmentation_flag = params_parser.get_data_augmentation_flag()
    max_number_of_samples_by_category = params_parser.get_max_number_of_samples_by_category()
    train_transform = None
    if data_augmentation_flag:
        train_transform = simple_image_transform 
    
    # Get training data
    categories = get_categories()
    train_database_json_file = os.path.join(dataset_dirpath, 'xview_ann_train.json')
    anns, _ = get_image_objects_list_from_file(train_database_json_file, dataset_dirpath, maximum_training_samples_per_category=max_number_of_samples_by_category)
    anns_train, anns_valid = train_test_split(anns, test_size=0.1, random_state=1, shuffle=True)
    
    if oversample_flag:
        categories_to_oversample_and_size_multiplier = params_parser.get_categories_to_oversample_and_size_multiplier()
        # Note: these changes should occur after splitting data into train and validation sets to ensure that no data in the validation set is included in the training set.
        anns_train, train_counts = oversample_image_objects(anns_train, categories_to_oversample_and_size_multiplier=categories_to_oversample_and_size_multiplier, seed_value=rand_seed)
        print("Training samples oversampled to: ", train_counts)
        # There is no need to apply oversampling to validation data
        # anns_valid, valid_counts = oversample_image_objects(anns_valid, categories_to_oversample_and_size_multiplier=categories_to_oversample_and_size_multiplier, seed_value=rand_seed) 

    objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
    objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]

    # Get training params
    optimizer_str = params_parser.get_optimizer_str()
    lr=params_parser.get_learning_rate()
    epochs = params_parser.get_epochs()
    batch_size=params_parser.get_batch_size()
    train_steps = math.ceil(len(anns_train)/batch_size)
    valid_steps = math.ceil(len(anns_train)/batch_size)
    loss = params_parser.get_loss_function()
    if optimizer_str == 'adam':
        opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.00, amsgrad=True, clipnorm=1.0, clipvalue=0.5)
    elif optimizer_str == 'sgd':
        momentum = params_parser.get_momentum()
        opt = SGD(learning_rate=lr, momentum=momentum)
    else:
        raise NotSupportedErr(f"Optimizer {optimizer_str} not supported.")

    # Model name
    model_name = model_trainer.model_name
    current_tb_dir = os.path.join(log_dir, model_name)
    os.makedirs(current_tb_dir, exist_ok=True)

    # Callbacks
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name+'.hdf5')
    shutil.copyfile(params_json_file, os.path.join(model_dir, "train_params.json")) # Copy json file to model path to save config
    
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
    early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
    terminate = TerminateOnNaN()
    tensorboard = TensorBoard(log_dir=current_tb_dir)
    log_times_on_epoch = LogEpochTime()
    callbacks = [model_checkpoint, reduce_lr, early_stop, terminate, tensorboard, log_times_on_epoch ]

    #Preprocess
    train_generator, valid_generator = model_trainer.preprocess_training_data(
        objs_train, objs_valid, batch_size=batch_size,
        transform_train_imgs_func=train_transform, transform_validation_imgs_func=None)
    #Train
    start_time = time.time()
    model_trainer.train(
        train_generator, valid_generator, 
        optimizer=opt, loss=loss, metrics=['accuracy'],
        train_steps=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks)
    print("--- %s seconds ---" % (time.time() - start_time))
