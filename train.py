import numpy as np
from funcs import *
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score
import tensorflow as tf
from keras import layers


def Indices2OneHot(class_indices):
    class_indices = class_indices.astype(int)
    max_i=np.max(class_indices)+1
    class_labels=np.zeros([np.size(class_indices,0),max_i])
    for i in range(np.size(class_indices,0)):
        class_labels[i][class_indices[i]]=1
    #class_indices = class_indices.astype(int)
    return class_labels

x_train = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_lift_aug.npy")
x_test = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_lift_aug.npy")
y_train = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_lift_aug.npy")
y_test = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_lift_aug.npy")

## change 7 to 4
import math
for i in range(len(y_train)):
    y_train[i] = math.ceil(y_train[i]/2.)
for i in range(len(y_test)):
    y_test[i] = math.ceil(y_test[i]/2.)

print(y_test)

y_train = Indices2OneHot(y_train - 1)
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
# Add another:
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
# Add another:
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
# Add another:
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
# Add a softmax layer with 10 output units:
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(x_train, y_train,
       nb_epoch=50, batch_size=10, validation_split=0.2)

## save loss
train_loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
np.savetxt("./result/vali_loss.txt", np.array(val_loss))
np.savetxt("./result/train_loss.txt", np.array(train_loss))

## display test results
probabilities = model.predict(x_test)

## one hot prediction
predictions = np.argmax(probabilities, axis=-1)+1
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
#savemat('result.mat', {'pred': predictions, 'gt': y_test})

from keras.models import model_from_json, load_model

# Option 1: Save Weights + Architecture
model.save_weights('./video/model_weights.h5')
with open('./video/model_architecture.json', 'w') as f:
    f.write(model.to_json())
# # Option 1: Load Weights + Architecture
# with open('model_architecture.json', 'r') as f:
#     new_model_1 = model_from_json(f.read())
# new_model_1.load_weights('model_weights.h5')