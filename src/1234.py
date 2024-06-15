
import yaml
from CustomDescriptors.FaceDescriptor.FACE import FACE
import numpy as np


with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)

if __name__ == "__main__":
    face = FACE(
        predictor_path=cfg['Victor']['FACE']['PREDICTOR'],
        recognition_path=cfg['Victor']['FACE']['RECOGNITION']
    )
    
    img_path_first =  "C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\high_resolution\\mona_younger_1.jpg"
    img_path_second =  "C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\high_resolution\\mona_younger_2.jpg"
    
    face_1 = face.compute(image_path=img_path_first)[1]
    face_2 = face.compute(image_path=img_path_second)[1]
    print(face_1.shape, face_2.shape)
    
    distance = np.linalg.norm(face_1 - face_2)
    
    print(distance)