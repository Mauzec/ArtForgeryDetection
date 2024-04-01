import cv2
import dlib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imutils import face_utils

# Инициализация детектора лица и предсказателя лицевых признаков
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Загрузка изображений и преобразование в градации серого
imageA = cv2.imread("mona_younger.jpeg")
imageB = cv2.imread("mona_original.jpeg")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Приведение размеров второго изображения к размеру первого
imageB_resized = cv2.resize(imageB, (grayA.shape[1], grayA.shape[0]))
grayB_resized = cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображениях
rectsA = detector(grayA, 1)
rectsB = detector(grayB_resized, 1)

# Проверка, что обнаружено одинаковое количество лиц
if len(rectsA) != len(rectsB):
    print("Количество лиц в изображениях не совпадает.")
    exit()

# Инициализация общего счетчика для оценки схожести
total_similarity = 0

# Обработка каждого обнаруженного лица
for (rectA, rectB) in zip(rectsA, rectsB):
    # Определение лицевых признаков для каждого лица
    shapeA = predictor(grayA, rectA)
    shapeB = predictor(grayB_resized, rectB)

    # Инициализация суммы схожести для одного лица
    face_similarity = 0

    # Визуализация контуров лицевых признаков
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # Извлечение координат для каждого лицевого признака
        xA, yA = shapeA.part(i).x, shapeA.part(i).y
        xB, yB = shapeB.part(i).x, shapeB.part(i).y

        # Вычисление ширины и высоты прямоугольника вокруг лицевого признака
        w = abs(xA - shapeA.part(j - 1).x)
        h = abs(yA - shapeA.part(j - 1).y)

        # Извлечение ROI и контуров для каждого лицевого признака
        roiA = grayA[yA - h:yA + h, xA - w:xA + w]
        roiB = grayB_resized[yB - h:yB + h, xB - w:xB + w]
        edgesA = cv2.Canny(roiA, threshold1=30, threshold2=100)
        edgesB = cv2.Canny(roiB, threshold1=30, threshold2=100)

        # Вычисление размера окна
        min_height = min(edgesA.shape[0], edgesB.shape[0])
        min_width = min(edgesA.shape[1], edgesB.shape[1])
        win_size = min(min_height, min_width)

        # Убедимся, что размер окна нечетный
        if win_size % 2 == 0:
            win_size -= 1

        # Визуализация контуров
        cv2.imshow("Edges A", edgesA)
        cv2.imshow("Edges B", edgesB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Вычисление SSIM для сравнения контуров
        similarity, _ = ssim(edgesA, edgesB, win_size=win_size, full=True)
        face_similarity += similarity

    # Обновление общего счетчика схожести
    total_similarity += face_similarity / len(face_utils.FACIAL_LANDMARKS_IDXS)

# Вычисление средней схожести для всех лиц
average_similarity = total_similarity / len(rectsA)
print("Общая оценка схожести:", average_similarity)
