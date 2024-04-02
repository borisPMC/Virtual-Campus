import cv2
import pandas as pd

def getImageDataframe(folderPath: str) -> pd.DataFrame:
    
    FOLDER_PATH = folderPath + "\\"

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
    imgDf = pd.read_json(jsonPath, lines=True)

    imgDf["image"] = imgDf.apply(fillImage, axis=1)
    print("Imported {}".format(folderPath))
    return imgDf


path = "images"
a = getImageDataframe(path)