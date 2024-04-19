import cv2
import numpy as np

def extract_brushstrokes(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение порогового значения для извлечения мазков
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Применение морфологической обработки для уточнения контуров
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Извлечение контуров мазков
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2 

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compare_images(image1, image2):
    # Извлечение мазков из изображений
    contours1 = extract_brushstrokes(image1)
    contours2 = extract_brushstrokes(image2)
    
    # Преобразование изображений в оттенки серого для вычисления SSIM
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Приведение изображений к одинаковым размерам
    target_size = (gray1.shape[1], gray1.shape[0])
    gray2_resized = cv2.resize(gray2, target_size, interpolation=cv2.INTER_AREA)
    
    # Вычисление SSIM между двумя изображениями
    score = ssim(gray1, gray2_resized)
    
    print("SSIM Score: {:.3f}".format(score))

# Загрузка изображений
image1 = cv2.imread('mona_younger.jpeg')
image2 = cv2.imread('mona_original.jpeg')

# Сравнение изображений
compare_images(image1, image2)
