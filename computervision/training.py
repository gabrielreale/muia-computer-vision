from datetime import datetime
import os
import math
from xml.dom import NotSupportedErr
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from computervision.data.xview_recognition_data import get_image_objects_list_from_file, get_categories
from computervision.ffnn.ffnn_trainer import FFNNModelTrainer

if __name__ == "__main__":
    dataset_dirpath = 'datasets/xview_recognition'
    log_dir = 'log/tensorboard'
    models_dir = 'models/xview_recognition'
    training_comment = "Basic model provided by the professors of CV."
    
    # Get training data
    categories = get_categories()
    train_database_json_file = os.path.join(dataset_dirpath, 'xview_ann_train.json')
    anns, _ = get_image_objects_list_from_file(train_database_json_file, dataset_dirpath, maximum_training_samples_per_category=5000)
    anns_train, anns_valid = train_test_split(anns, test_size=0.1, random_state=1, shuffle=True)

    objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
    objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]

    # Get training params
    optimizer_str = 'adam'
    lr=1e-3
    epochs = 20
    batch_size=32
    train_steps = math.ceil(len(anns_train)/batch_size)
    valid_steps = math.ceil(len(anns_train)/batch_size)
    if optimizer_str == 'adam':
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.00, amsgrad=True, clipnorm=1.0, clipvalue=0.5)
    else:
        raise NotSupportedErr(f"Optimizer {optimizer_str} not supported.")
    
    # Model name
    model_name = f"FFNN_opt_{optimizer_str}_lr_{str(lr).replace('.', '')}_batch_size_{batch_size}_time_{datetime.now().strftime('%Y%m%d%H%M')}"
    current_tb_dir = os.path.join(log_dir, model_name)
    os.makedirs(current_tb_dir, exist_ok=True)

    # Load architecture
    print('Load model')
    model = Sequential(name=model_name)
    model.add(Flatten(input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))
    model.summary()
    
    # Callbacks
    model_path = os.path.join(models_dir, model_name+'.hdf5')
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
    early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
    terminate = TerminateOnNaN()
    tensorboard = TensorBoard(log_dir=current_tb_dir)
    callbacks = [model_checkpoint, reduce_lr, early_stop, terminate, tensorboard]

    ffnn_model_trainer = FFNNModelTrainer(model, training_comment)
    #Preprocess
    train_generator, valid_generator = ffnn_model_trainer.preprocess_training_data(
        objs_train, objs_valid, batch_size=batch_size,
        transform_train_imgs_func=None, transform_validation_imgs_func=None)
    #Train
    ffnn_model_trainer.train(
        train_generator, valid_generator, 
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'],
        train_steps=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks)
