import DataUtilizer
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import numpy as np
np.random.seed(1000)

# Define AlexNet model
class AlexNet:
    def createModel():
        model = Sequential()
        model.add(Input(INPUT_SHAPE))

        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # Flatten layer to convert 3D features to 1D vector
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # Output Layer: Fully connected layer with num_classes units for classification
        model.add(Dense(N_CLASS, activation='softmax'))
        
        model.summary()

        # Compiling
        opt = Adam(learning_rate=0.0002*2)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def trainModel(imageCol, labelCol):
        # Create and compile the model
        model = AlexNet.createModel()

        labelCol = DataUtilizer.getOneHot(labelCol)

        # Train the model with 5-fold cross-validation
        history = model.fit(imageCol, labelCol, epochs=10, batch_size=32, validation_split=0.3, shuffle=True)
        return history, model

# Instantiate the model
N_CLASS = 4 # Real vs Fake
INPUT_SHAPE = (227, 227, 3) # AlexNet input
LR_ITER = 5 # Config Adam Learning Rate

# # Import Data
# path = "images"
# imgDf = DataUtilizer.getImageDataframe(path, 4)
# imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)

# # 5-fold cross validation
# imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)
# history, model = AlexNet.trainModel(imageCol, labelCol)

# # Access validation score from history object
# DataUtilizer.showValidationResult(history)
# DataUtilizer.saveModel(model, "alex_a{}".format(LR_ITER))

# Import Test
path = "testset"
imgDf = DataUtilizer.getImageDataframe(path, 4)
imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)
DataUtilizer.testModel("alex_a5", imageCol, labelCol, AlexNet.createModel())