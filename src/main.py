from Helper.DatasetOperations import DataOperations
from BoVW.BoVW import BoVW
from CustomDescriptors.OrbDescriptor.ORB import ORB
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
from CustomDescriptors.ResnetDescriptor.Resnet import Resnet
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import yaml

if __name__ == "__main__":
    cfg = None
    with open('config.yaml', 'r') as config:
        cfg = yaml.safe_load(config)["Victor"]
        
    bovw = BoVW(
        descriptor=ORB(),
        number_words=200,
        clf=LinearSVC(max_iter = 8000),
        cluster=KMeans(n_clusters = 200)
    )
    
    operations = DataOperations("Victor")
    operations.clearDirs()
    operations.getData("Split")(0.2)
    Data = operations.getData("FitPredict")(encoded=True)
    train, test = Data["Train"], Data["Test"]
    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]
    
    bovw.fit(X_train, y_train)
    print("train:", bovw.score(X_train, y_train))
    print("test:", bovw.score(X_test, y_test))
    
    # бля, я в ахуе с результатов, это пиздец какой-то