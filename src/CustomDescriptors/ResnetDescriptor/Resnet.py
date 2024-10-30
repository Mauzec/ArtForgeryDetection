from CustomDescriptors.abstract.abstract import ABSDescriptor
from tensorflow.keras.applications import ResNet101 # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from numpy.typing import NDArray
import numpy as np
import cv2
import os

class Resnet(ABSDescriptor):
    def __init__(self, resnet_path: str, onGPU: bool = False) -> None:
        if not onGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.resnet_path = resnet_path
        if not "resnet.keras" in os.listdir(self.resnet_path):
            model = ResNet101(weights='imagenet', include_top=False)
            model.save(f"{self.resnet_path}/resnet.keras")
        
    def compute(self, image_path: str, index_process:int = -1) -> tuple[NDArray, NDArray]:
        model = load_model(f"{self.resnet_path}/resnet.keras")
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        features = model.predict(image_expanded)[0]
        return None, features.reshape(-1, features.shape[-1])
    
    def __repr__(self) -> str:
        return "Resnet"
        