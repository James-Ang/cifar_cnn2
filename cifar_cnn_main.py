# Ref https://www.tensorflow.org/tutorials/images/cnn

import matplotlib.pyplot as plt
from load_cifar_10_alt import load_data
import numpy as np

cifar_dir = r'C:\Users\User\Documents\virtual\cifar_basic1\cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_data(cifar_dir)

print(x_train.shape)
print(y_train.shape)

# PREVIEW DATA
x_train[:5]
y_train[:10]

# SHOW IMAGES
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
#plt.show()

# CONVERT LABELS TO CATEGORICAL
# from keras.utils import to_categorical
# y_train_cat = to_categorical(y_train)
# y_test_cat = to_categorical(y_test)



# PREPROCESSING - NORMALISING

# Transforms from 0-255 --> 0-1
x_traindatascaled, x_testdatascaled = x_train / 255, x_test / 255
x_traindatascaled[0].shape, x_testdatascaled[0].shape
x_traindatascaled[0].max(), x_testdatascaled[0].max()


# BUILDING DEEP LEARNING MODELS
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation = 'relu')) # params (1024+1)*64
model.add(Dense(10))
# TO SEE MODEL SUMMARY
# Watch this to understand output Shape & number of params
# https://www.evernote.com/l/AgIU4zGykSRKf5cP4oBKT5EC9mnZXXdXrg0/
# Max Pooling has no params to learn
model.summary()
# COMPILE AND TRAIN THE MODELS
from keras.losses import SparseCategoricalCrossentropy

model.compile(optimizer ='adam',
              loss = SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#model.compile(optimizer='adam',
#              loss="SparseCategoricalCrossentropy",
#              metrics=['accuracy'])

history = model.fit(x=x_traindatascaled,
            y=y_train, # Notice here use the original labels
            validation_data = (x_testdatascaled,y_test),
            shuffle=True,
            batch_size = 100,
            epochs = 30)


scores = model.evaluate(x=x_traindatascaled,
               batch_size = 100,
               y=y_train)
scores =model.evaluate(x=x_testdatascaled,
                batch_size = 100,
               y=y_test)
x_testdatascaled.shape

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(x_testdatascaled, y_test, verbose=2)
history
history.history
