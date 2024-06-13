import cv2
import dlib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imutils import face_utils

# Инициализация детектора лица и предсказателя лицевых признаков
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("CustomDescriptors\\FaceDescriptor\\shape_predictor_68_face_landmarks.dat")

# Загрузка изображений и преобразование в градации серого
imageA = cv2.imread("C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\high_resolution\\mona_younger_1.jpg")
imageB = cv2.imread("C:\\home_screen\\programming\\algoritm and data structure\\Dataset\\Mona\\high_resolution\\mona_original.png")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Приведение размеров второго изображения к размеру первого
imageB_resized = cv2.resize(imageB, (grayA.shape[1], grayA.shape[0]))
grayB_resized = cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображениях
rectsA = detector(grayA, 1)
rectsB = detector(grayB_resized, 1)

# Проверка, что обнаружено одинаковое количество лиц
if len(rectsA)!= len(rectsB):
    print("Количество лиц в изображениях не совпадает.")
    exit()

# Инициализация общего счетчика для оценки схожести
total_similarity = 0

# Обработка каждого обнаруженного лица
for (rectA, rectB) in zip(rectsA, rectsB):
    # Определение лицевых признаков для каждого лица
    shapeA = predictor(grayA, rectA)
    shapeB = predictor(grayB_resized, rectB)
    arr1 = face_utils.shape_to_np(shapeA)
    arr2 = face_utils.shape_to_np(shapeB)
    
    # Рисование точек на первом изображении
    for (x, y) in arr1:
        cv2.circle(imageA, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Рисование точек на втором изображении
    for (x, y) in arr2:
        cv2.circle(imageB_resized, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Сохранение измененных изображений
    cv2.imwrite('face_points_imageA.jpg', imageA)
    cv2.imwrite('face_points_imageB.jpg', imageB_resized)

    # Остальная часть вашего кода для вычисления сходства...
