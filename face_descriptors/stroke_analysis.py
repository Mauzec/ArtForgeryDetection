import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_seeds(image_patch):
    # Генерация начальных точек (семян)
    seeds = []
    for i in range(10):  # например, 10 семян
        y = np.random.randint(0, image_patch.shape[0])
        x = np.random.randint(0, image_patch.shape[1])
        seeds.append((x, y))
    return seeds

def segment_brushstroke(seed, threshold, image_patch):
    # Функция сегментации мазка
    x, y = seed
    mask = np.zeros(image_patch.shape[:2], np.uint8)
    cv2.floodFill(image_patch, mask, (x, y), 255, (threshold,)*3, (threshold,)*3, flags=cv2.FLOODFILL_MASK_ONLY)
    return mask

def check_area(candidate_brushstroke):
    # Проверка валидности области мазка
    area = np.sum(candidate_brushstroke) / 255
    return area > 50  # например, минимальная площадь 50 пикселей

def check_shape(candidate_brushstroke):
    # Проверка валидности формы мазка
    contours, _ = cv2.findContours(candidate_brushstroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        return width > 5 and height > 5  # например, минимальная ширина и высота 5 пикселей
    return False

def compute_orientation(candidate_brushstroke):
    # Вычисление ориентации мазка
    contours, _ = cv2.findContours(candidate_brushstroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        return angle
    return 0

def compute_length(candidate_brushstroke):
    # Вычисление длины мазка
    contours, _ = cv2.findContours(candidate_brushstroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        length = max(rect[1])
        return length
    return 0

def compute_width(candidate_brushstroke):
    # Вычисление ширины мазка
    contours, _ = cv2.findContours(candidate_brushstroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        width = min(rect[1])
        return width
    return 0

def extract_brushstrokes(image_patch):
    seeds = generate_seeds(image_patch)
    brushstrokes = []

    for seed in seeds:
        threshold = 20  # начальный порог
        valid_brushstroke = False
        
        while not valid_brushstroke:
            valid_area = False
            while not valid_area:
                candidate_brushstroke = segment_brushstroke(seed, threshold, image_patch)
                valid_area = check_area(candidate_brushstroke)
                if not valid_area:
                    threshold += 5  # обновление порога
            
            orientation = compute_orientation(candidate_brushstroke)
            length = compute_length(candidate_brushstroke)
            width = compute_width(candidate_brushstroke)
            
            valid_brushstroke = check_shape(candidate_brushstroke)
        
        brushstrokes.append({
            'brushstroke': candidate_brushstroke,
            'orientation': orientation,
            'length': length,
            'width': width
        })
    
    return brushstrokes

# Проверка наличия файла и его доступности
image_path = 'data/mona_original.png'
if not os.path.isfile(image_path):
    print(f"Error: File {image_path} does not exist or is not accessible")
else:
    image_patch = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_patch is None:
        print(f"Error: Unable to load image at {image_path}. Check if the file is a valid image.")
    else:
        brushstrokes = extract_brushstrokes(image_patch)
        
        # Визуализация мазков
        for idx, brushstroke in enumerate(brushstrokes):
            plt.subplot(1, len(brushstrokes), idx + 1)
            plt.imshow(brushstroke['brushstroke'], cmap='gray')
            plt.title(f'Stroke {idx + 1}')
            plt.axis('off')
        
        plt.show()
        
        # Дополнительно визуализируем мазки на исходном изображении
        for brushstroke in brushstrokes:
            contours, _ = cv2.findContours(brushstroke['brushstroke'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_patch, contours, -1, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
        plt.title('Brushstrokes on Image')
        plt.axis('off')
        plt.show()
