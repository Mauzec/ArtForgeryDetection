import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from imutils import face_utils
from CustomDescriptors.abstract.abstract import ABSDescriptor

class FACE(ABSDescriptor):
    def __init__(self,
                 size: tuple,
                 predictor_path: str,
                 recognition_path: str
                 ) -> None:
        self.size = size
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.recognition = dlib.face_recognition_model_v1(recognition_path)
        
    def compute(self, image_path: str, index_process = -1, drawkps: int = 0) -> tuple[NDArray, NDArray]:
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        shapes = list()
        edges = list()
        for rect in rects:
            shape = self.predictor(gray, rect)
            shapes.append(face_utils.shape_to_np(shape))
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                x, y = shape.part(i).x, shape.part(i).y
                w = abs(x - shape.part(j - 1).x)
                h = abs(y - shape.part(j - 1).y)
                
                roi = gray[y - h:y + h, x - w:x + w]
                edge = cv2.Canny(roi, threshold1=30, threshold2=100)
                
                edge_resized = np.zeros(shape=self.size, dtype=int)
                
                for i in range(edge.shape[0]):
                    if i >= self.size[0]: break
                    for j in range(edge.shape[1]):
                        if j >= self.size[1]: break
                        edge_resized[i][j] = edge[i][j] // 255 
                
                edges.append(edge_resized)    
            
        kp = np.vstack(shapes)
        des = np.array(edges)
        
        return kp, des
    
    def __repr__(self) -> str:
        return "FACE"
        