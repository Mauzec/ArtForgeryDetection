import cv2
import dlib
import numpy as np
from imutils import face_utils

import yaml
import os

with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cfg['Victor']['FACE']['PREDICTOR'])
face_rec_model = dlib.face_recognition_model_v1(cfg['Victor']['FACE']['RECOGNITION'])

imageA = cv2.imread("C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\high_resolution\\mona_original.png")
imageB = cv2.imread("C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\low_resolution\\mona_younger_1.jpg")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

imageB_resized = cv2.resize(imageB, (grayA.shape[1], grayA.shape[0]))
grayB_resized = cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображениях
rectsA = detector(grayA, 1)
rectsB = detector(grayB_resized, 1)

# Проверка, что обнаружено одинаковое количество лиц
if len(rectsA) != len(rectsB):
    print("Количество лиц в изображениях не совпадает.")
    exit()

# Обработка каждого обнаруженного лица
for (rectA, rectB) in zip(rectsA, rectsB):
    # Определение лицевых признаков для каждого лица
    shapeA = predictor(grayA, rectA)
    shapeB = predictor(grayB_resized, rectB)

    # Преобразование лицевых признаков в массив numpy
    arr1 = face_utils.shape_to_np(shapeA)
    arr2 = face_utils.shape_to_np(shapeB)
    
    # Создание лицевых дескрипторов
    face_descriptorA = np.array(face_rec_model.compute_face_descriptor(imageA, shapeA))
    face_descriptorB = np.array(face_rec_model.compute_face_descriptor(imageB_resized, shapeB))
    print(face_descriptorA.shape, face_descriptorB.shape)
    
    # Расчет евклидова расстояния между дескрипторами
    distance = np.linalg.norm(face_descriptorA - face_descriptorB)
    print(f"Расстояние между лицами: {distance}")

    if distance < 0.6:
        print("Лица на изображениях похожи.")
    else:
        print("Лица на изображениях различны.")
    
    # Рисование точек на первом изображении
    for (x, y) in arr1:
        cv2.circle(imageA, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Рисование точек на втором изображении
    for (x, y) in arr2:
        cv2.circle(imageB_resized, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Сохранение измененных изображений
    cv2.imwrite('face_points_imageA.jpg', imageA)
    cv2.imwrite('face_points_imageB.jpg', imageB_resized)