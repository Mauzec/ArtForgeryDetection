import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Инициализация ResNet50 без верхнего слоя и загрузка весов ImageNet
base_model = ResNet50(weights='imagenet', include_top=False)

# Сохранение модели ResNet50
base_model.save('resnet50_model.h5')


def load_and_preprocess_images(class_dir):
    images = []
    labels = []
    for filename in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224)) # Изменение размера изображения
            images.append(img)
            labels.append(0 if 'Leonardo' in class_dir else 1) # Пример метки
    return images, labels

class0_images, class0_labels = load_and_preprocess_images('Raphael')
class1_images, class1_labels = load_and_preprocess_images('Leonardo')

images = class0_images + class1_images
labels = class0_labels + class1_labels

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Загрузка модели ResNet50 для извлечения признаков
base_model = tf.keras.models.load_model('resnet50_model.h5')

X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

svm = SVC()
svm.fit(X_train_features, y_train)

y_pred = svm.predict(X_test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Применение t-SNE для визуализации признаков
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_features)

# Визуализация
plt.figure(figsize=(10, 10))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE visualization of image features')
plt.show()

# Загрузка модели ResNet50 и обученной SVM
base_model = tf.keras.models.load_model('resnet50_model.h5')
svm = SVC()
svm.load('path/to/svm_model.pkl') # Предполагается, что модель SVM сохраняется в формате pickle

test_features = base_model.predict(X_test)
predicted_classes = svm.predict(test_features)

def extract_edge_features(image):
    canny = cv2.Canny(image, 100, 200)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    scharr = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    
    combined = np.hstack((canny, sobelx, sobely, laplacian, scharr))
    return combined

test_image = cv2.imread('path/to/test_image.jpg')
test_edge_features = extract_edge_features(test_image)

# Сравнение краевых признаков с эталонным набором
is_likely_by_Raphael = compare_edge_features(test_edge_features, reference_set, delta=0.5, gamma=0.8)

if is_likely_by_Raphael:
    print("The painting is likely to be by Raphael.")
else:
    print("The painting is unlikely to be by Raphael.")
