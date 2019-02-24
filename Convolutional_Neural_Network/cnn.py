from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import pickle

#Define a sequential model
model = Sequential()

#Adding Convolution layer 1
#32 , (2,2)) : No. of filters = 32 and stride is 2 X 2 dimenoional
#input_shape=(64, 64, 3)) : 64 by 64 input selected for faster processing, larger value means better predictions
# 3 is the number of channel. Its 3 for colored image and 1 for B&W images
#activation='relu' - Rectifier activation functions
model.add(Conv2D(32 , (2,2), input_shape=(64, 64, 3), activation='relu'))

#Pooling layer 1
model.add(MaxPool2D(pool_size=(2,2)))

#Convolution layer 2
model.add(Conv2D(32 , (2,2), activation='relu'))

#Pooling layer 2
model.add(MaxPool2D(pool_size=(2,2)))

#Flattening layer
model.add(Flatten())

#Full Connection
#First hidden layer (input)
model.add(Dense(units = 128, activation='relu'))

#Output layer
model.add(Dense(units = 5, activation='sigmoid'))

#loss='binary_crossentropy since its binary classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

#Image preprocessing
from keras.preprocessing.image import ImageDataGenerator
#For Image data augmentation.

#rescale = 1./255 keeps all pixel values between 0 and 1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#64,64 since we have selected this size in step 1
#class_mode = 'binary' since its binary classification
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

label_map = (training_set.class_indices)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                    epochs = 10,
                    validation_data = test_set,
                    steps_per_epoch=1500,
                    validation_steps = 250)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))
print('Labels are : ' + str(label_map))
print('CNN model succesfully saved as model.pkl ....')