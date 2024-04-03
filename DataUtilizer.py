import random
import cv2
import pandas as pd
import numpy as np
from keras import layers, Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class DataUtilizer:
    def getImageDataframe(folderPath: str, alter = 0) -> pd.DataFrame:
    
        FOLDER_PATH = folderPath + "\\"
        result = pd.DataFrame(columns=['id', 'annotation', 'extension'])

        def readImage(id, ext):
            imagePath = FOLDER_PATH + id
            image_path = f"{imagePath}.{ext}"
            image = cv2.imread(image_path)
            if image is not None:
                return image
            print(f"Failed to load image with ID: {id}")
            return None
                    
        def fillImage(row):
            id = row["id"]
            ext = row["extension"]
            image = readImage(id, ext)
            return image

        jsonPath = FOLDER_PATH + "annotation.jsonl"
        # First Iteration must be original
        imgDf = pd.read_json(jsonPath, lines=True)
        imgDf["image"] = imgDf.apply(fillImage, axis=1)
        # Perform to reduce computational cost on augmentation
        imgDf["image"] = imgDf.apply(DataUtilizer.preprocessImage, axis=1)
        result = pd.concat([result, imgDf])

        # Iteration to augment images
        for _ in range(alter):
            temp = imgDf.copy()
            temp["image"] = temp.apply(DataUtilizer.augmentImage, axis=1)
            result = pd.concat([result, temp])

        print("Imported {}".format(folderPath))
        return imgDf

    def preprocessImage(row, w=227, h=227): # Highest necessary size: (227,277) for AlexNet
        def standardizeImage(image):
            # Convert image to float32 for compatibility with standardization
            image = image.astype(np.float32)
            # Standardize image
            standardized_image = image / 255

            return standardized_image

        def resizeImage(image, width, height):
            # Resize image
            resized_image = cv2.resize(image, (width, height))
            return resized_image
        
        image = row["image"]
        image = standardizeImage(image)
        image = resizeImage(image, w, h)
        return image

    def getImageAndLabel(df: pd.DataFrame):

        img = np.stack(df["image"].values, axis=0).astype(np.float32)
        label = df["annotation"].values

        return img, label

    def getOneHot(list):
        y_one_hot = to_categorical(list, num_classes=4)
        return y_one_hot

    def showValidationResult(hist):
        validation_accuracy = hist.history['val_accuracy']
        average_validation_accuracy = np.mean(validation_accuracy)
        print(f"Average Validation Accuracy: {average_validation_accuracy}")

    def saveModel(model, name="output_model"):
        # Save the model weights to a HDF5 file
        model.save_weights(name+".weights.h5")

    def augmentImage(row, random_seed=0) -> np.ndarray:
        
        def rotate(image):
            # Rotate randomly between -45 and 45 degrees
            angle = random.uniform(-45, 45)
            image = ndimage.rotate(image, angle, reshape=False)
            return image
        
        def flip(image):
            # Flip horizontally or vertically
            axis = random.choice([0, 1])
            image = cv2.flip(image, axis)
            return image
        
        def scale(image):
            # Scale randomly between 0.8 and 1.2
            scale_factor = random.uniform(0.8, 1.2)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
            return image
        
        def translate(image):
            # Translate randomly between -20 and 20 pixels in both directions
            shift = (random.randint(-20, 20), random.randint(-20, 20))
            # For RGB images, the shape is (height, width, channels)
            height, width, channels = image.shape
            translation_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            # Apply the translation to each channel separately
            for i in range(channels):
                image[:, :, i] = cv2.warpAffine(image[:, :, i], translation_matrix, (width, height))
            return image

        image = row["image"]

        # Get the original size of the image
        original_size = image.shape

        if random_seed % 2 == 0:
            image = rotate(image)
        if random_seed % 3 == 0:
            image = flip(image)
        if random_seed % 5 == 0:
            image = scale(image)
        if random_seed % 7 == 0:
            image = translate(image)

        # Resize the augmented image to the original size
        image = cv2.resize(image, (original_size[1], original_size[0]))

        return image

    def testModel(modelFilename, imageCol, labelCol, model):

        modelPath = "{}.weights.h5".format(modelFilename)
        model.load_weights(modelPath)

        y_pred = np.argmax(model.predict(imageCol), axis=1)
        accuracy = accuracy_score(labelCol, y_pred)

        print("Test Accuracy:",accuracy)