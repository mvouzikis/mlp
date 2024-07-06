import numpy as np
import pandas as pd 
import keras
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Μοντελο με 2 κρυφα επιπεδα, 256 και 128 νευρωνες με συναρτηση ενεργοποιησης relu
#2 dropout layers με χαμηλο rate για αποφυγη over fitting
#Softmax στο επιπεδο εξοδου
model1 = keras.Sequential()
model1.add(layers.Flatten(input_shape=(32, 32, 3)))
model1.add(layers.Dense(256, activation='relu'))  
model1.add(layers.Dropout(0.3))  
model1.add(layers.Dense(128, activation='relu'))  
model1.add(layers.Dropout(0.3))
model1.add(layers.Dense(num_classes, activation='softmax'))  

#Χρηση adam optimizer, συναρτηση κοστους 'categorical_crossentropy' και υπολογιζουμε ακριβεια
model1.compile(optimizer= 'adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#10 εποχές, με batch size 80
model1.fit(x_train, y_train, epochs = 10, batch_size=80, validation_data=(x_test, y_test))
model1_test_acc = model1.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {model1_test_acc}')
       
#Προβλεψη κλάσεων του μοντέλου
y_pred1 = model1.predict(x_test)
    
#Υπολογισμος 5 λανθασμενων και 5 σωστων προβλεψεων κλασεων του μοντελου
wrong_indices = np.nonzero(np.argmax(y_pred1, axis=1) != np.argmax(y_test, axis=1))[0]
correct_indices = np.nonzero(np.argmax(y_pred1, axis=1) == np.argmax(y_test, axis=1))[0]
    

incorrect_predictions_msg = "Incorrectly predicted examples"
correct_predictions_msg = "Correctly predicted examples"

#Εκτυπωση των λανθασμενων προβλεψεων του μοντελου
print("\n" + incorrect_predictions_msg)
for i in range(min(5, len(wrong_indices))):
    index = wrong_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred1[index])
    print(f"    Example {i+1}: Predicted={predicted_label}, True={true_label}")

#Εκτυπωση των σωστων προβλεψεων του μοντελου
print("\n" + correct_predictions_msg)
for i in range(min(5, len(correct_indices))):
    index = correct_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred1[index])
    print(f"    Example {i+1}: Predicted={predicted_label}, True={true_label}") 
    
#Μοντελο με ενα κρυφο επιπεδο, 256 νευρώνες, και συνάρτηση relu
#Ενα dropout layer για να αποφυγουμε υπερ εκπαιδευση
#Softmax στο επιπεδο εξοδου
model2 = keras.Sequential()
model2.add(layers.Flatten(input_shape=(32, 32, 3)))
model2.add(layers.Dense(256, activation='relu'))  
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(num_classes, activation='softplus'))  

#Optimizer Adagrad, με learning rate 0.01, συναρτηση κοστους categorical_crossentropy
model2.compile(optimizer= keras.optimizers.Adagrad(learning_rate = 0.01), 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#20 εποχες, με batch size 128
model2.fit(x_train, y_train, epochs = 20, batch_size=128, validation_data=(x_test, y_test))
model2_test_acc = model2.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {model2_test_acc}')
       
y_pred2 = model2.predict(x_test)

#Υπολογισμος λανθασμενων και σωστων προβλεψεων κλασεων
wrong_indices = np.nonzero(np.argmax(y_pred2, axis=1) != np.argmax(y_test, axis=1))[0]
correct_indices = np.nonzero(np.argmax(y_pred2, axis=1) == np.argmax(y_test, axis=1))[0]

#Print των λανθασμενων προβλεψεων
print("\n" + incorrect_predictions_msg)
for i in range(min(5, len(wrong_indices))):
    index = wrong_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred2[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")

#Print των σωστων προβλεψεων   
print("\n" + correct_predictions_msg)   
for i in range(min(5, len(correct_indices))):
    index = correct_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred2[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")
    
#Μοντελο με 3 κρυφα επιπεδα, και 3 dropout επιπεδα με διαφορα rates. Συναρτηση εξοδου softmax
model3 = keras.Sequential()
model3.add(layers.Flatten(input_shape=(32, 32, 3)))
model3.add(layers.Dense(256, activation='relu'))  
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(128, activation='relu')) 
model3.add(layers.Dropout(0.2))
model3.add(layers.Dense(512, activation='relu')) 
model3.add(layers.Dropout(0.7))
model3.add(layers.Dense(num_classes, activation='softmax'))  

#Stochastic Gradient Descent optimizer, learning rate 0.001, και sparse_categorical_crossentropy συναρτηση κοστους
model3.compile(optimizer= keras.optimizers.SGD(learning_rate = 0.001), 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#50 εποχες, batch size 256
model3.fit(x_train, y_train, epochs = 50, batch_size=256, validation_data=(x_test, y_test))
model3_test_acc = model3.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {model3_test_acc}')
       
#Προβλεψη του μοντελου για το test σετ
y_pred3 = model3.predict(x_test)
    
#Υπολογισμος λανθασμενων και σωστων προβλεψεων του μοντελου
wrong_indices = np.nonzero(np.argmax(y_pred3, axis=1) != np.argmax(y_test, axis=1))[0]
correct_indices = np.nonzero(np.argmax(y_pred3, axis=1) == np.argmax(y_test, axis=1))[0]
    
#Εκτυπωση των λανθασμενων προβλεψεων
print("\n" + incorrect_predictions_msg)
for i in range(min(5, len(wrong_indices))):
    index = wrong_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred3[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")
   
#Εκτυπωση των σωστων προβλεψεων του μοντελου 
print("\n" + correct_predictions_msg)   
for i in range(min(5, len(correct_indices))):
    index = correct_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred3[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")

#Δημιουργουμε εναν ImageDataGenerator για να μετασχηματισει τα δεδομενα που θα του δωθουν
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Υλοποιουμε τον μετασχηματισμο των δεδομενων
augmented_data = datagen.flow(x_train, y_train, batch_size=256)

#Μοντελο με 4 κρυφα επιπεδα, και 3 dropout επιπεδα με διαφορα rates. Συναρτηση εξοδου softplus
model4 = keras.Sequential()
model4.add(layers.Flatten(input_shape=(32, 32, 3)))
model4.add(layers.Dense(256, activation='relu'))  
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(128, activation='relu')) 
model4.add(layers.Dropout(0.2))
model4.add(layers.Dense(512, activation='relu')) 
model4.add(layers.Dropout(0.7))
model4.add(layers.Dense(256, activation='relu'))
model4.add(layers.Dense(num_classes, activation='softplus'))  

#Χρηση adam optimizer και categorical_crossentropy συναρτηση κοστους
model4.compile(optimizer= 'adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#70 εποχες, batch size 128 με χρηση των μετασχηματισμενων δεδομενων
model4.fit(augmented_data, epochs = 70, batch_size=128, validation_data=(x_test, y_test))
model4_test_acc = model4.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {model4_test_acc}')
       
#Προβλεψη του μοντελου για το test σετ
y_pred4 = model4.predict(x_test)
    
#Υπολογισμος λανθασμενων και σωστων προβλεψεων του μοντελου
wrong_indices = np.nonzero(np.argmax(y_pred4, axis=1) != np.argmax(y_test, axis=1))[0]
correct_indices = np.nonzero(np.argmax(y_pred4, axis=1) == np.argmax(y_test, axis=1))[0]
    
#Εκτυπωση των λανθασμενων προβλεψεων
print("\n" + incorrect_predictions_msg)
for i in range(min(5, len(wrong_indices))):
    index = wrong_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred4[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")
   
#Εκτυπωση των σωστων προβλεψεων του μοντελου 
print("\n" + correct_predictions_msg)   
for i in range(min(5, len(correct_indices))):
    index = correct_indices[i]
    true_label = np.argmax(y_test[index])
    predicted_label = np.argmax(y_pred4[index])
    print(f"Example {i+1}: Predicted={predicted_label}, True={true_label}")

#Εκτυπωση των accuracies των μοντελων
print("\n\nModel 1 Accuracy: " + str(model1_test_acc))
print("Model 2 Accuracy: " + str(model2_test_acc))
print("Model 3 Accuracy: " + str(model3_test_acc))
print("Model 4 Accuracy: " + str(model4_test_acc))