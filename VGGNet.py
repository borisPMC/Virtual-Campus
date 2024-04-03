import DataUtilizer
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import cv2
import numpy as np

class VGGNet:
    def createModel(shape, n_class):
        model = Sequential()
        model.add(Input(shape))
        
        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(n_class, activation='softmax'))  # Output layer for ImageNet classification

        # Compiling
        opt = Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def trainModel(imageCol, labelCol):
        # Create and compile the model
        model = VGGNet.createModel(INPUT_SHAPE, num_classes)

        labelCol = DataUtilizer.getOneHot(labelCol)

        # Train the model with 5-fold cross-validation
        history = model.fit(imageCol, labelCol, epochs=10, batch_size=32, validation_split=0.3, shuffle=True)
        return history, model
    
    def resizeInput(row):
        image = row["image"]
        image = cv2.resize(image, (224, 224))
        return image
    
# Instantiate the model
num_classes = 4 # Real vs Fake
INPUT_SHAPE = (224, 224, 3) # VGGNet input

# Import Data
path = "images"
imgDf = DataUtilizer.getImageDataframe(path, 4)
imgDf["image"] = imgDf.apply(VGGNet.resizeInput, axis=1)

# 5-fold cross validation
imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)
history, model = VGGNet.trainModel(imageCol, labelCol)

# Access validation score from history object
validation_accuracy = history.history['val_accuracy']
average_validation_accuracy = np.mean(validation_accuracy)
print(f"Average Validation Accuracy: {average_validation_accuracy}")