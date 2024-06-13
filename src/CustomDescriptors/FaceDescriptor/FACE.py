import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from imutils import face_utils
from CustomDescriptors.abstract.abstract import ABSDescriptor

class FACE(ABSDescriptor):
    def __init__(self, size: int = 10**3) -> None:
        self.size = size
        
    def compute(self, image_path: str, index_process = -1, drawkps: int = 0) -> tuple[NDArray, NDArray]:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        shapes = list()
        edges = list()
        for rect in rects:
            shape = predictor(gray, rect)
            shapes.append(face_utils.shape_to_np(shape))
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                x, y = shape.part(i).x, shape.part(i).y
                w = abs(x - shape.part(j - 1).x)
                h = abs(y - shape.part(j - 1).y)
                
                roi = gray[y - h:y + h, x - w:x + w]
                edge = cv2.Canny(roi, threshold1=30, threshold2=100)
                
                edge_list = list()
                if edge.shape[1] < self.size:
                    
                
                
                edges.append(edge.tolist())
            
        kp = np.vstack(shapes)
        
        return kp, None
    
    def __repr__(self) -> str:
        return "SSIM"
        