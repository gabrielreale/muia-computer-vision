import os
import json
import numpy as np
import math
from generic_objects import GenericImage, GenericObject
from generators import generator_images
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from focal_loss import SparseCategoricalFocalLoss

# Hyper-parameters
batch_size = 16
epochs = 20
opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.00, amsgrad=True, clipnorm=1.0, clipvalue=0.5)
loss_funct = 'sparse_categorical_crossentropy'  # I hardcoded this as its not from keras
number_of_hidden_layers = 0
neurons_per_hidden_layer = [75264, 37632, 18816]
activation = 'relu'
dropout = 0.2


categories = {13: 'CARGO_PLANE', 15: 'HELICOPTER', 18: 'SMALL_CAR', 19: 'BUS', 23: 'TRUCK', 41: 'MOTORBOAT',
              47: 'FISHING_VESSEL', 60: 'DUMP_TRUCK', 64: 'EXCAVATOR', 73: 'BUILDING', 86: 'STORAGE_TANK',
              91: 'SHIPPING_CONTAINER'}

dataset_dirpath = 'datasets/xview_recognition'

# Load database
json_file = os.path.join(dataset_dirpath, 'xview_ann_train.json')
with open(json_file) as ifs:
    json_data = json.load(ifs)
ifs.close()

counts = dict.fromkeys(categories.values(), 0)
anns = []
for json_img, json_ann in zip(json_data['images'], json_data['annotations']):
    image = GenericImage(os.path.join(dataset_dirpath, json_img['file_name']))
    image.tile = np.array([0, 0, json_img['width'], json_img['height']])
    obj = GenericObject()
    obj.id = json_ann['id']
    obj.bb = (int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
    obj.category = list(categories.values())[json_ann['category_id']-1]
    # Resampling strategy to reduce training time
    if counts[obj.category] >= 5000:
        continue
    counts[obj.category] += 1
    image.add_object(obj)
    anns.append(image)
print(counts)

# Splitting data
anns_train, anns_valid = train_test_split(anns, test_size=0.1, random_state=1, shuffle=True)

# Model architecture
model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))
model.add(Activation(activation))

for i in range(number_of_hidden_layers):
    model.add(Dense(neurons_per_hidden_layer[i]))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

model.add(Dense(len(categories)))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=opt, loss=SparseCategoricalFocalLoss(gamma=2), metrics=['accuracy'])

# Callbacks
model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]

# Generate the list of objects from annotations
objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]
# Generators


train_generator = generator_images(objs_train, batch_size, do_shuffle=True)
valid_generator = generator_images(objs_valid, batch_size, do_shuffle=False)


# Training

train_steps = math.ceil(len(objs_train)/batch_size)
valid_steps = math.ceil(len(objs_valid)/batch_size)
h = model.fit(train_generator, steps_per_epoch=train_steps, validation_data=valid_generator, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks, verbose=1)
# Best validation model
best_idx = int(np.argmax(h.history['val_accuracy']))
best_value = np.max(h.history['val_accuracy'])

with open('out.txt', 'w') as f:
    print('Best validation model: epoch ' + str(best_idx+1), ' - val_accuracy ' + str(best_value), file=f)