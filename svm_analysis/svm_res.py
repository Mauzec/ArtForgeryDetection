from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Указываем пути к директориям с изображениями
leonardo_images_folder = "/Leonardo"
other_images_folder = "/Other"
test_folder = "path/to/test_folder"
reference_folder = "path/to/reference_folder"

# Процедура инициализации модели ResNet50
def model_setup():
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    return resnet_model

# Процедура организации изображений
def image_organisation(class0_folder, class1_folder):
    # Считываем изображения и выполняем аугментацию
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(directory=class0_folder,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='binary')

    X, _ = train_generator.next()

    # Загружаем модель ResNet50
    resnet_model = model_setup()

    # Извлекаем признаки для всех изображений
    features = resnet_model.predict(X)
    return features

# Процедура обучения SVM с настройкой гиперпараметров
def svm_training_with_hyperparameter_tuning(features, labels):
    # Разделяем данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Определяем набор гиперпараметров для поиска по сетке
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf', 'poly']}

    # Создаем экземпляр модели SVM
    svm_classifier = SVC()

    # Инициализируем GridSearchCV с моделью SVM и параметрами для поиска по сетке
    grid_search = GridSearchCV(svm_classifier, param_grid, refit=True, verbose=3, cv=5)

    # Запускаем поиск по сетке на обучающих данных
    grid_search.fit(X_train, y_train)

    # Выводим лучшие гиперпараметры, найденные поиском по сетке
    print("Best Parameters:", grid_search.best_params_)

    # Получаем лучшую модель SVM
    svm_model = grid_search.best_estimator_

    # Тестируем SVM
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return svm_model

# Процедура тестирования модели
def model_testing(resnet_model, svm_model, test_folder):
    # Извлекаем признаки для тестового набора данных
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(directory=test_folder,
                                                      target_size=(224, 224),
                                                      batch_size=32,
                                                      class_mode='binary')

    X_test, y_test = test_generator.next()

    test_features = resnet_model.predict(X_test)

    # Предсказываем классы тестовых изображений с помощью SVM модели
    y_pred = svm_model.predict(test_features)
    return y_pred

# Процедура тренировки окончательной модели
def train_final(svm_model, features, labels):
    # Обучаем SVM на всех данных
    svm_model.fit(features, labels)
    return svm_model

# Процедура извлечения граничных характеристик изображения
def edge_features(image_path):
    image = cv2.imread(image_path, 0)
    canny_edges = cv2.Canny(image, 100, 200)
    sobel_edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)
    scharr_edges = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    edge_features = np.concatenate((canny_edges.flatten(), sobel_edges_x.flatten(), sobel_edges_y.flatten(), laplacian_edges.flatten(), scharr_edges.flatten()))
    return edge_features

# Процедура проверки изображения
def verification_test(test_image_path, reference_set_folder, svm_model, threshold_delta, threshold_gamma):
    # Извлекаем граничные характеристики тестового изображения
    test_edge_features = edge_features(test_image_path)

    # Сравниваем граничные характеристики с референсным набором
    reference_edge_features = []
    for image_file in os.listdir(reference_set_folder):
        reference_image_path = os.path.join(reference_set_folder, image_file)
        reference_edge_features.append(edge_features(reference_image_path))

    # Сравниваем граничные характеристики тестового изображения с референсным набором
    for ref_edge_features in reference_edge_features:
        similarity = np.sum(np.abs(test_edge_features - ref_edge_features))
        if similarity < threshold_delta:
            
            test_feature = resnet_model.predict(np.expand_dims(cv2.imread(test_image_path), axis=0))
            test_prediction = svm_model.predict(test_feature)
            
            if test_prediction == 0 and svm_model.predict_proba(test_feature)[0][0] > threshold_gamma:
                return "Painting is likely to be by DaVinci"
            else:
                return "Painting is unlikely to be by DaVinci"
    return "Painting is unlikely to be by DaVinci"

# Инициализация модели ResNet50
resnet_model = model_setup()

# Организация изображений
leonardo_features = image_organisation(leonardo_images_folder, other_images_folder)

# Обучение SVM с настройкой гиперпараметров
svm_model = svm_training_with_hyperparameter_tuning(leonardo_features, np.zeros(leonardo_features.shape[0]))

# Тестирование модели
test_predictions = model_testing(resnet_model, svm_model, test_folder)

# Тренировка окончательной модели
final_svm_model = train_final(svm_model, leonardo_features, np.zeros(leonardo_features.shape[0]))

# Проверка изображения
result = verification_test("Test_image_path", reference_folder, final_svm_model, threshold_delta=100, threshold_gamma=0.7)
print(result)
