import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from imutils import face_utils
from CustomDescriptors.abstract.abstract import ABSDescriptor

class FACE(ABSDescriptor):
    def __init__(self,
                 predictor_path: str,
                 recognition_path: str
                 ) -> None:
        self.predictor_path = predictor_path
        self.recognition_path = recognition_path
        
    def compute(self, image_path: str, index_process = -1, drawkps: int = 0) -> tuple[NDArray, NDArray]:
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)
        recognition = dlib.face_recognition_model_v1(self.recognition_path)
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        shapes = list()
        descriptors = list()
        for rect in rects:
            shape = predictor(gray, rect)
            shapes.append(face_utils.shape_to_np(shape))
            face_desciptor = np.array(recognition.compute_face_descriptor(image, shape))
            descriptors.append(face_desciptor)
            
        kp = np.array(shapes).reshape(-1, 2)
        des = np.array(descriptors)
        
        print(f"found {kp.shape[0]} keypoints")
        if kp.shape[0] == 0:
            print(image_path)
        
        return kp, des
    
    def __repr__(self) -> str:
        return "FACE"
        