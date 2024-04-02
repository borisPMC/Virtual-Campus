import DataUtilizer
import tensorflow as tf
from tensorflow.keras import layers, models

def createAlexNet(shape, nClass):
    model = models.Sequential([S
        # 1st Convolutional Layer
        layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        # 2nd Convolutional Layer
        layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        # 3rd Convolutional Layer
        layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        layers.BatchNormalization(),

        # 4th Convolutional Layer
        layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        layers.BatchNormalization(),

        # 5th Convolutional Layer
        layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model