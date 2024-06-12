import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from imutils import face_utils
from CustomDescriptors.abstract.abstract import ABSDescriptor

class SSIM(ABSDescriptor):
    def compute(self, image_path: str, index_process = -1, drawkps: int = 0) -> tuple[NDArray, NDArray]:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        shapes = list()
        for rect in rects:
            shape = predictor(gray, rect)
            shapes.append(face_utils.shape_to_np(shape))
            
        kp = np.vstack(shapes)
        
        return kp, None
    
    def __repr__(self) -> str:
        return "SSIM"
        