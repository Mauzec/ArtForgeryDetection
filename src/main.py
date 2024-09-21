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
        descriptor=SIFT(cfg["SIFT"]),
        number_words=50,
        clf=LinearSVC(max_iter = 800),
        cluster=KMeans(n_clusters = 50)
    )
    
    operations = DataOperations("Victor")
    operations.clearDirs()
    operations.getData("Split")(0.2)
    Data = operations.getData("FitPredict")()
    train, test = Data["Train"], Data["Test"]
    
    bovw.fit(*train)
    
    # бля, я в ахуе с результатов, это пиздец какой-то