import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import time
import yaml
import os
import random as rand

with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from CustomDescriptors.OrbDescriptor.ORB import ORB
from CustomDescriptors.ResnetDescriptor.Resnet import Resnet
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
from CustomDescriptors.FaceDescriptor.FACE import FACE
from helper.helper import Research, GridSearch
from BoVW.BoVW import BoVW

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, MiniBatchKMeans
    
clfs = {
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": LinearSVC(max_iter=8000),
    "LogisticRegression": LogisticRegression(),
}
clusters = {
    "KMeans": KMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
}

n_clusters = {
    "ORB": 200,
    "AKAZE": 200,
    "Resnet": 500,
    "SIFT" : 200,
    "FACE": 30,
}
descriptors = {
    "ORB": ORB(),
    "AKAZE": AKAZE(),
    "Resnet": Resnet(resnet_path=cfg['Victor']['Resnet']),
    "SIFT" : SIFT(cfg['Victor']['SIFT']),
    "FACE": FACE(
        predictor_path=cfg['Victor']['FACE']['PREDICTOR'],
        recognition_path=cfg['Victor']['FACE']['RECOGNITION']
    )
}

def testing():
    Research.add_train_dataset(percentage=40, scale=True, resolution="high")
    
    for name_clf, clf in clfs.items():
        for name_cluster, cluster in clusters.items():
            for name_descriptor, descriptor in descriptors.items():
                print(f"{name_clf}_{name_cluster}_{name_descriptor}")
                if not f"{name_clf}_{name_cluster}_{name_descriptor}.json" in os.listdir("results"):
                    Research.safe(
                        f"{name_clf}_{name_cluster}_{name_descriptor}",
                        Research.test(
                            Research.train(
                                clf=clf,
                                descriptor=descriptor,
                                cluster=cluster(n_clusters[name_descriptor]),
                                scale=False,
                                number_words=n_clusters[name_descriptor]
                            )
                        )

                    )


if __name__ == "__main__":
    parametres = {
        "n_clusters": [200],
        "tol": [10**(-4)],
        'algorithm': ["lloyd"],
        'n_init': ["auto"]
        
    }
    
    print(
        GridSearch.grid_search(
            cluster=KMeans,
            clf=GradientBoostingClassifier(),
            parametres=parametres
            )
        )
    