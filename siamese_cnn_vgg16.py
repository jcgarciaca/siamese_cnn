
# coding: utf-8

# In[1]:


import numpy as np
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

import sys
import os

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from keras.applications import VGG16

from skimage.transform import rescale, resize
import skimage.io as io


np.random.seed(1337)  # for reproducibility


# In[2]:


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


# In[3]:


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))



def create_data_siamese_from_images(path1, path2, data_size):
    set1 = []
    set2 = []
    labels = []
    
    cnt_img = 0      
    
    # true pairs
    images_in_path1 = os.listdir(path1)    
    images_in_path2 = os.listdir(path2)
        
    # print("len(images_in_path1): ", len(images_in_path1))
    
    # np.random.shuffle(images_in_path1)
    
    # print(np.random.randint(0, len(images_in_path1) - 1))
    
    
    for i in range(data_size):
        
        # true data C1
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path1) - 1)        
        # print("image1: ", images_in_path1[index1])
        # print("image2: ", images_in_path1[index2])
        img1 = io.imread(os.path.join(path1, images_in_path1[index1]))
        img2 = io.imread(os.path.join(path1, images_in_path1[index2]))                
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(1)
        
        
        # true data C2
        index1 = np.random.randint(0, len(images_in_path2) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        # print("image1: ", images_in_path2[index1])
        # print("image2: ", images_in_path2[index2])
        img1 = io.imread(os.path.join(path2, images_in_path2[index1]))
        img2 = io.imread(os.path.join(path2, images_in_path2[index2]))                
        img1 = resize(img1, (WIDTH, HEIGHT), anti_aliasing = True)
        img2 = resize(img2, (WIDTH, HEIGHT), anti_aliasing = True)
        set1.append(img1)
        set2.append(img2)
        labels.append(1)
        
        
        
        # false data 1
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        # print("image1: ", images_in_path1[index1])
        # print("image2: ", images_in_path2[index2])
        
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
        labels.append(0)
        
        
        # false data 2
        index1 = np.random.randint(0, len(images_in_path1) - 1)
        index2 = np.random.randint(0, len(images_in_path2) - 1)        
        # print("image1: ", images_in_path1[index1])
        # print("image2: ", images_in_path2[index2])
        
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
        labels.append(0)
        
                
    return(np.array(set1), np.array(set2), np.array(labels))


def create_data_siamese(x, digit_indices):
    set1 = []
    set2 = []
    labels = []
    # print(len(digit_indices[0]))
    # print(len(digit_indices[1]))
    n = min([len(digit_indices[d]) for d in range(2)])
    if(n > 50):
        n = 50
    
    print("min: ", n)
    
    # true pairs
    for i in range(n):
        for j in range(n):
            if(i != j):
                z1, z2 = digit_indices[0][i], digit_indices[0][j]
                set1.append(x[z1])
                set2.append(x[z2])
                labels.append(1)
                z1, z2 = digit_indices[1][i], digit_indices[1][j]
                set1.append(x[z1])
                set2.append(x[z2])
                labels.append(1)
    
    print("set1: ", len(set1))
    print("set2: ", len(set2))
    
    # false pairs
    for i in range(n):
        for j in range(2 * n):
            if(i != j):
                z1, z2 = digit_indices[0][i], digit_indices[1][j]
                set1.append(x[z1])
                set2.append(x[z2])
                labels.append(0)                
    
    print("set1: ", len(set1))
    print("set2: ", len(set2))
    
    return(np.array(set1), np.array(set2), np.array(labels))
    
    
    
          
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            #labels += [1, 0]
            labels += [0, 1]
    return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    #return labels[predictions.ravel() < 0.5].mean()
    return np.mean(labels == (predictions.ravel() > 0.5))


# In[4]:


'''
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# input_dim = 784

nb_epochs = 20

print("----------------------------")
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
print("----------------------------")
'''
'''
for index in range(5):
    plt.figure(index)
    imgplot = plt.imshow(X_train[index])
    print(y_train[index])
'''


# In[5]:


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


# In[6]:


'''
index = 0
for img in tr_set1:
    plt.figure(index)
    imgplot = plt.imshow(img)
    index += 1
'''


# In[7]:


'''
index = 0
for img in tr_set2:
    plt.figure(index)
    imgplot = plt.imshow(img)
    index += 1
'''


# In[8]:


'''
index = 0
for img in te_set1:
    plt.figure(index)
    imgplot = plt.imshow(img)
    index += 1
'''


# In[9]:


'''
index = 0
for img in te_set2:
    plt.figure(index)
    imgplot = plt.imshow(img)
    index += 1
'''


# In[10]:


# network definition

input_a = Input(shape=(WIDTH, HEIGHT, 3))
input_b = Input(shape=(WIDTH, HEIGHT, 3))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = conv_base(input_a)#base_network(input_a)
processed_b = conv_base(input_b)#base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)


# In[11]:


# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['acc'])
model.summary()


# In[ ]:


nb_epochs = 10

history = model.fit([tr_set1, tr_set2], tr_y,
          validation_data=([te_set1, te_set2], te_y),
          batch_size=1,
          epochs=nb_epochs)

model.save('siamese_cnn_vgg16_1.h5')


# In[ ]:


# compute final accuracy on training and test sets
pred = model.predict([tr_set1, tr_set2])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_set1, te_set2])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


# In[ ]:


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

