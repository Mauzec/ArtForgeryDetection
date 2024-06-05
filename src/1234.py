import cv2
import numpy as np

# Создаем экземпляр SIFT
sift = cv2.ORB.create()

# Загружаем изображение
img = cv2.imread('dataset/test/artist/mona_original.png', 0)

# Находим ключевые точки и дескрипторы
kp, des = sift.detectAndCompute(img, None)

print(kp, des)

# Рисуем ключевые точки на изображении
img_with_keypoints = cv2.drawKeypoints(img, kp, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Отображаем результат
cv2.imshow('Key Points', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()