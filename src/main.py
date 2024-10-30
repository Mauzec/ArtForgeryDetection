from Helper import DataOperations
from BoVW.BoVW import BoVW
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from CustomDescriptors import *
import yaml
from copy import deepcopy

if __name__ == "__main__":
    cfg = None
    with open('config.yaml', 'r') as config:
        cfg = yaml.safe_load(config)["Victor"]
        
    bovw = BoVW(
        descriptor=FACE(
            predictor_path=cfg["FACE"]["PREDICTOR"],
            recognition_path=cfg["FACE"]["RECOGNITION"]
            ),
        number_words=200,
        clf=LinearSVC(max_iter = 8000),
        cluster=KMeans(n_clusters = 200),
        num_proceses=8
    )
    
    operations = DataOperations("Victor")
    operations.clearDirs()
    operations.getData("Bin")(getTest=False)
    operations.getData("Inference")()
    Data = operations.getData("FitPredict")(encoded=False)
    train, test = Data["Train"], Data["Test"]
    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]
    
    bovw.fit(deepcopy(X_train), deepcopy(y_train))
    print(bovw.predict(X_test), X_test)