#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import glob


# In[ ]:


Images El=[cv2.imread(file) for file in glob.glob(“C:/Users/abc/Downloads/galaxy-image-classifier-tensorflow-master/xy_photos/elliptical/*..jpg”)]
Images Sp=[cv2.imread(file) for file in glob.glob(“C:/Users/abc/Downloads/galaxy-image-classifier-tensorflow-master/xy_photos/spiral/*.jpg”)]
Images El=[cv2.imread(file) for file in glob.glob(“C:/Users/abc/Downloads/galaxy-image-classifier-tensorflow-master/xy_photos/elliptical/*.jpg”)]


# In[3]:


i=ime[0].flatten()
for I in ime :
      ell.appent(i.flatten())


# In[ ]:


import numpy as np
import keras
import sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator , load_img, img_to_array
from keras.layers import Dens,Activation,Flatten,Dropout,BatchNormalization
from keras.layers import ConV2D,MaxPooling2D
from keras.datasets import cifar10
from keras import regulizers,optimizers
import os


# In[ ]:


x=np.ndarray(shape=(519,256,256,3), dtype=float, order=’F’)
y=np.ndarray(shape=(519,1), dtype=float, order=’F’)
classes=0
count=0


# In[ ]:





# In[ ]:


path=”../input/galaxyimageclassifier/galaxy-image-classifier-tensorflow-master/ galaxy-image-classifier-tensorflow-master”
for I in os.listdr(path):
flag=0
c=0
for j in os.listdr(path+1)
	img=load_img(path+1+’/ ’+j, target_size= (256,256,3))
	x=img_to_array(img)
	X[count]=x
	Y[count]=classes
    	count += 1
	classes += 1



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# In[ ]:



print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


num_classes = 2
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)


# In[ ]:



datagen = ImageDataGenerator(
	featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	rotation_range=15,
	width_shift_range=0.1,
	height-shift_range=0.1,
	horizontal_flip=True,
	vertical_flip=False


# In[ ]:


baseMapNum = 32
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_tra
model.add(Activation(‘relu’))
model.add(Conv2D(baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation(‘relu’))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation(‘relu’))
model.add(Conv2D(2*baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation(‘relu’))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_tra
model.add(Activation(‘relu’))
model.add(Conv2D(4*baseMapNum,  (3,3), padding=’same’, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation(‘relu’))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# In[ ]:


batch_size = 8
epochs=25
opt_rms =keras.optimizers.rmsprop(lr=0.002,decay=1e-6)
model.compile(loss=’ategorical_crossentropy’,
	optimizer=opt_rms,
	metrics=[‘accuracy’])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=X_train.shape[0] 

opt_rms =keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
model.compile(loss=’ategorical_crossentropy’,
	optimizer=opt_rms,
	metrics=[‘accuracy’])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=X_train.shape[0] 

opt_rms =keras.optimizers.rmsprop(lr=0.002,decay=1e-6)
model.compile(loss=’ategorical_crossentropy’,
	optimizer=opt_rms,
	metrics=[‘accuracy’])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=X_train.shape[0] 


# In[ ]:


scores= model.evaluate(X_test , y_test, batch_size=120, verbose=1)
print(‘\nTest result : %.3f loss : %.3f’ %(scores[1]*100,scores[0]))


# In[ ]:


TESTING

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)


# In[ ]:


Print(X_train.shape,X_test.shape)
Print(y_train.shape,y_test.shape)


# In[ ]:


Num_classes =2
y_train= np_utils.to_categorical(y_train,num_classes)
y_train= np_utils.to_categorical(y_test,num_classes

