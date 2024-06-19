import itertools

# Предположим, что параметры заданы как словарь
parametres = {
    'n_clusters': [2, 3, 4],
    'n_init': [10, 20],
    'algorithm': ['kmeans', 'agglo'],
    'tol': [0.001, 0.01]
}

# Получаем список всех значений параметров
values = list(parametres.values())

# Генерируем все возможные комбинации параметров
for params in itertools.product(*values):
    # Делаем что-то с каждой комбинацией параметров
    print(params)