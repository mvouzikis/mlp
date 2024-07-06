
from keras.datasets import cifar10
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

def display_images(original, decoded, count = 10): 
    n = count
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # show original input image
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display decoded image
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(decoded[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

(x_train, _), (x_test, _) = cifar10.load_data()

print("train data", x_train.shape)
print("test data", x_test.shape)

batch_size = 50
img_width, img_height, img_channels = 32, 32, 3
no_classes = 10
no_epochs = 50

#define the input shape
input_img = Input(shape = (img_width, img_height, img_channels))

# convert to float32 format
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255


x = Conv2D(64, (3, 3), activation='relu', padding='same') (input_img)
x = MaxPooling2D((2, 2)) (x)

x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
x = MaxPooling2D((2, 2)) (x)

x = Conv2D(16, (3, 3), activation='relu', padding='same') (x)
#x = MaxPooling2D((2, 2)) (x)

x = Conv2D(8, (3, 3), activation='relu', padding='same') (x)
encoded = MaxPooling2D((2, 2)) (x)

x = Conv2D(8, (3, 3), activation='relu', padding='same') (encoded)
x = UpSampling2D((2, 2)) (x)

x = Conv2D(16, (3, 3), activation='relu', padding='same') (x)
x = UpSampling2D((2, 2)) (x)

x = Conv2D(32, (3, 3), activation='relu', padding='same') (x)
x = UpSampling2D((2, 2)) (x)

x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)
#x = UpSampling2D((2, 2)) (x)

decoded = Conv2D(3, (3, 3), padding='same') (x)

cae = Model(input_img,decoded)
cae.compile(optimizer = 'adam', loss ='mse', metrics=['accuracy'] )
cae.summary()


print(x_train.shape)
print(x_test.shape)


early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

np.random.seed(123)
history = cae.fit(x_train,x_train,
           epochs = 100,
           batch_size = 50,
           validation_data = (x_test, x_test),
           callbacks=[early_stopping_monitor])


#score = cae.evaluate(x_train, x_train, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
     
decoded_data = cae.predict(x_test)
display_images(x_test, decoded_data)