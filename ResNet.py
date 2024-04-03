import DataUtilizer
import cv2
from keras import layers, Model
from keras.optimizers import Adam

class ResNet:
    def createModel():
        def identity_block(X, filters, kernel_size):
            """
            Implementation of the identity block as defined in ResNet.
            """
            # Retrieve filters
            F1, F2, F3 = filters
            
            # Save the input value (shortcut)
            X_shortcut = X
            
            # First component of main path
            X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
            X = layers.BatchNormalization()(X)
            X = layers.Activation('relu')(X)
            
            # Second component of main path
            X = layers.Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
            X = layers.BatchNormalization()(X)
            X = layers.Activation('relu')(X)
            
            # Third component of main path
            X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
            X = layers.BatchNormalization()(X)
            
            # Add shortcut value to main path
            X = layers.Add()([X, X_shortcut])
            X = layers.Activation('relu')(X)
            
            return X

        def convolutional_block(X, filters, kernel_size, strides):
            """
            Implementation of the convolutional block as defined in ResNet.
            """
            # Retrieve filters
            F1, F2, F3 = filters
            
            # Save the input value (shortcut)
            X_shortcut = X
            
            # First component of main path
            X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=strides, padding='valid')(X)
            X = layers.BatchNormalization()(X)
            X = layers.Activation('relu')(X)
            
            # Second component of main path
            X = layers.Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
            X = layers.BatchNormalization()(X)
            X = layers.Activation('relu')(X)
            
            # Third component of main path
            X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
            X = layers.BatchNormalization()(X)
            
            # Shortcut path
            X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=strides, padding='valid')(X_shortcut)
            X_shortcut = layers.BatchNormalization()(X_shortcut)
            
            # Add shortcut value to main path
            X = layers.Add()([X, X_shortcut])
            X = layers.Activation('relu')(X)
            
            return X

        def ResNet50():
            """
            Implementation of the ResNet50 architecture.
            """
            # Define the input as a tensor with shape input_shape
            X_input = layers.Input(INPUT_SHAPE)
            
            # Zero-padding
            X = layers.ZeroPadding2D((3, 3))(X_input)
            
            # Stage 1
            X = layers.Conv2D(64, (7, 7), strides=(2, 2))(X)
            X = layers.BatchNormalization()(X)
            X = layers.Activation('relu')(X)
            X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
            
            # Stage 2
            X = convolutional_block(X, filters=[64, 64, 256], kernel_size=3, strides=(1, 1))
            X = identity_block(X, filters=[64, 64, 256], kernel_size=3)
            X = identity_block(X, filters=[64, 64, 256], kernel_size=3)
            
            # Stage 3
            X = convolutional_block(X, filters=[128, 128, 512], kernel_size=3, strides=(2, 2))
            X = identity_block(X, filters=[128, 128, 512], kernel_size=3)
            X = identity_block(X, filters=[128, 128, 512], kernel_size=3)
            X = identity_block(X, filters=[128, 128, 512], kernel_size=3)
            
            # Stage 4
            X = convolutional_block(X, filters=[256, 256, 1024], kernel_size=3, strides=(2, 2))
            X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
            X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
            X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
            X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
            X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
            
            # Stage 5
            X = convolutional_block(X, filters=[512, 512, 2048], kernel_size=3, strides=(2, 2))
            X = identity_block(X, filters=[512, 512, 2048], kernel_size=3)
            X = identity_block(X, filters=[512, 512, 2048], kernel_size=3)
            
            # Average pooling
            X = layers.AveragePooling2D((2, 2))(X)
            
            # Output layer
            X = layers.Flatten()(X)
            X = layers.Dropout(0.5)(X)
            X = layers.Dense(N_CLASS, activation='softmax')(X)
            
            # Create model
            model = Model(inputs=X_input, outputs=X)
            
            # Compiling
            opt = Adam(learning_rate=0.01)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        
        # Build the ResNet50 model
        resnet_model = ResNet50()

        # Summary of the model architecture
        resnet_model.summary()

        return resnet_model

    def trainModel(imageCol, labelCol):
        # Create and compile the model
        model = ResNet.createModel()

        labelCol = DataUtilizer.getOneHot(labelCol)

        # Train the model with 5-fold cross-validation
        history = model.fit(imageCol, labelCol, epochs=10, batch_size=32, validation_split=0.3, shuffle=True)
        return history, model
    
    def resizeInput(row):
        image = row["image"]
        image = cv2.resize(image, (224, 224))
        return image

INPUT_SHAPE = (224,224,3)
N_CLASS = 4

# # Import Data
# path = "images"
# imgDf = DataUtilizer.getImageDataframe(path, 4)
# imgDf["image"] = imgDf.apply(RegNet.resizeInput, axis=1)

# # 5-fold cross validation
# imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)
# history, model = RegNet.trainModel(imageCol, labelCol)

# # Access validation score from history object
# DataUtilizer.showValidationResult(history)
# DataUtilizer.saveModel(model, "regnet_1")

# Import Test
path = "testset"
imgDf = DataUtilizer.getImageDataframe(path, 4)
imgDf["image"] = imgDf.apply(ResNet.resizeInput, axis=1)
imageCol, labelCol = DataUtilizer.getImageAndLabel(imgDf)
DataUtilizer.testModel("regnet_2", imageCol, labelCol, ResNet.createModel())