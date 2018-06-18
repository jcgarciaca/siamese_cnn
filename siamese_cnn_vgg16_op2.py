# coding: utf-8

import numpy as np
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate, Flatten
from keras.optimizers import RMSprop
from keras import backend as K

import sys
import os

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from keras.applications import VGG16

from skimage.transform import rescale, resize
import skimage.io as io
from sklearn.metrics import accuracy_score


np.random.seed(1337)  # for reproducibility


# convolutional base

WIDTH = 200
HEIGHT = 200 

conv_base = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(WIDTH, HEIGHT, 3))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
conv_base.summary()


# create dataset
def create_data_siamese_from_images(path1, path2, data_size):
    set1 = []
    set2 = []
    labels = []
    
    cnt_img = 0      
    
    # true pairs
    images_in_path1 = os.listdir(path1)    
    images_in_path2 = os.listdir(path2)
    
    for i in range(data_size):
        
        # true data C1
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path1) - 1)        
        img1 = io.imread(os.path.join(path1, images_in_path1[index1]))
        img2 = io.imread(os.path.join(path1, images_in_path1[index2]))                
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(1.0)
        
        
        # true data C2
        index1 = np.random.randint(0, len(images_in_path2) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        img1 = io.imread(os.path.join(path2, images_in_path2[index1]))
        img2 = io.imread(os.path.join(path2, images_in_path2[index2]))                
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(1.0)
        
        
        
        # false data 1
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        
        if(i%2 == 0):
            img1 = io.imread(os.path.join(path1, images_in_path1[index1]))
            img2 = io.imread(os.path.join(path2, images_in_path2[index2]))
        else:
            img1 = io.imread(os.path.join(path2, images_in_path2[index2]))
            img2 = io.imread(os.path.join(path1, images_in_path1[index1]))            
            
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(0.0)
        
        
        # false data 2
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        
        if(i%2 == 0):
            img1 = io.imread(os.path.join(path1, images_in_path1[index1]))
            img2 = io.imread(os.path.join(path2, images_in_path2[index2]))
        else:
            img1 = io.imread(os.path.join(path2, images_in_path2[index2]))
            img2 = io.imread(os.path.join(path1, images_in_path1[index1]))            
            
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(0.0)
        
                
    return(np.array(set1), np.array(set2), np.array(labels))

def binarize_predictions(pred, threshold):
    bin_predictions = []
    for prediction in pred:
        if(prediction >= threshold):
            bin_predictions.append(1.0)
        else:
            bin_predictions.append(0.0)
    
    return(bin_predictions)


# the data split between train and test sets
original_dataset_dir = '/home/jcgarcia_mill/cnn_test/cats_dogs/train' #'/home/JulioCesar/cnn_test/cats_dogs/train'
base_dir = '/home/jcgarcia_mill/cnn_test/cats_dogs/cats_and_dogs_small' #'/home/JulioCesar/cnn_test/cats_dogs/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


data_size = 2000

tr_set1, tr_set2, tr_y = create_data_siamese_from_images(train_cats_dir, train_dogs_dir, int(data_size / 4))

print(tr_set1.shape)
print(tr_set2.shape)

te_set1, te_set2, te_y = create_data_siamese_from_images(test_cats_dir, test_dogs_dir, int(data_size / 10))

print(te_set1.shape)
print(te_set2.shape)


# In[5]:


# network definition

left_input = Input(shape=(WIDTH, HEIGHT, 3))
right_input = Input(shape=(WIDTH, HEIGHT, 3))

left_output = conv_base(left_input)#base_network(input_a)
right_output = conv_base(right_input)#base_network(input_b)

merged = concatenate([left_output, right_output], axis=-1)
merged = Flatten()(merged)

full_cnt = Dense(256, activation='relu')(merged)
predictions = Dense(1, activation='sigmoid')(full_cnt)

# model = Model(input=[input_a, input_b], output=distance)

model = Model([left_input, right_input], predictions)


# In[6]:


model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['acc'])
model.summary()


# In[7]:


nb_epochs = 2

history = model.fit([tr_set1, tr_set2], tr_y,
          validation_data=([te_set1, te_set2], te_y),
          batch_size=1,
          epochs=nb_epochs)

model.save('siamese_cnn_vgg16_2.h5')


# In[31]:


# compute final accuracy on training and test sets
pred = model.predict([tr_set1, tr_set2])
print("--------------------------")
# print("pred: ", pred)
print("tr_y: ", tr_y)
# print("pred.shape: ", pred.shape)

threshold = 0.7
pred_bin = binarize_predictions(pred, threshold)
print("pred_bin: ", pred_bin)

tr_acc = accuracy_score(pred_bin, tr_y)


threshold = 0.7

pred = model.predict([te_set1, te_set2])
# print("--------------------------")
# print("pred: ", pred)
print("te_y: ", te_y)

pred_bin = binarize_predictions(pred, threshold)
print("pred_bin: ", pred_bin)
te_acc = accuracy_score(pred_bin, te_y)
# print("pred.shape: ", pred.shape)
# print("--------------------------")

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#epochs = range(1, len(loss) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

