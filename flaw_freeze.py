import os
import cv2
import random
import csv
import numpy as np
import timm
import matplotlib.pyplot as plt
from keras.models import Model, load_model
#from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet import EfficientNetBㄆㄆ4
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.layers import *
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = "train"
n = 404+505

test = "C:\\Users\\Buslab_GG\\Desktop\\T"
weight = "C:\\Users\\Buslab_GG\\Desktop\\2022-07-19\\AA_DL\\mobile_Au_weights-improvement-10.hdf5"
out = "C:\\Users\\Buslab_GG\\Desktop\\output2.csv"


labels = ["dot","hair","hole","noflaw"]
inputShape = (224,224,3)
epoch = 40
batchSize = 8
steps_per = 584 // batchSize
learningRate = 1e-5


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return



###############
net = MobileNet(weights='imagenet', include_top=False, input_shape=inputShape)
x = net.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
output = Dense(4, activation='softmax')(x)

model = Model(inputs=net.input, outputs=output)
model.compile(optimizer=Adam(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy',tfa.metrics.F1Score(num_classes=4, threshold=0.5)])



testDatagen = ImageDataGenerator()
testBatches = testDatagen.flow_from_directory(test,
        target_size=(inputShape[0], inputShape[1]),
        interpolation='bicubic',
        class_mode='categorical',
        shuffle=False,
        batch_size=1)

model.fit_generator(trainBatches, 
              steps_per_epoch=steps_per, 
              epochs=25, 
              verbose=1, 
              callbacks=callbacks_list, 
              validation_data=validBatches, 
              validation_steps=74, 
              #class_weight={0:0.3,1:1,2:10,3:0.3}, 
              max_queue_size=10, 
              workers=1, 
              use_multiprocessing=False, 
              shuffle=True, 
              initial_epoch=0)




y_ = model.predict(testBatches, verbose=0)
y_ = np.argmax(y_,axis=-1)
target_names = ["dot","hair","hole","noflaw"]


with open(out, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['照片名', '預測'])
  for i in range(168):
    writer.writerow([testBatches.filenames[i][5:],y_[i]])

