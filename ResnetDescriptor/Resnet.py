from keras.applications import ResNet101
from keras.models import load_model
import numpy as np
import cv2

class ResnetDescruptor:
    @staticmethod
    def download_Resnet() -> None:
        model = ResNet101(weights='imagenet', include_top=False)
        model.save("resnet.h5")
        
    @staticmethod   
    def compute(image_path: str, index_process:int = -1) -> np.ndarray:
        model = load_model("resnet.h5")
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        features = model.predict(image_expanded)[0]
        return None, features.reshape(-1, features.shape[-1])
    
if __name__ == "__main__":
    resnet = ResnetDescruptor()
    resnet.download_Resnet()
    result = resnet.compute("book_in_scene.jpg")
    print(result[1].shape)
        