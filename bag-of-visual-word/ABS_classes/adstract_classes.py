from abc import ABC

class Descriptor(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def compute(self, image) -> tuple:
        keypoints = []
        descriptors = []
        return keypoints, descriptors