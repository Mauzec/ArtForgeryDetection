import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import yaml
import os

with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
from BoVW.BoVW import BoVW
from dataset.get_pictures import DatasetOperations
from CustomDescriptors.abstract.abstract import ABSDescriptor
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from CustomDescriptors.OrbDescriptor.ORB import ORB
# from CustomDescriptors.ResnetDescriptor.Resnet import Resnet
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
from CustomDescriptors.FaceDescriptor.FACE import FACE
from helper.helper import Research, Multiprocessor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralBiclustering, Birch


if __name__ == "__main__":

    Research.add_train_dataset(percentage=20, scale=True, resolution="low")
    
    clfs = {
        "RandomForest": RandomForestClassifier(),
        "Bagging": BaggingClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "LogisticRegression": LogisticRegression(),
    }
    
    clusters = {
        "KMeans": KMeans(n_clusters=20),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=20),
        "Birch": Birch(n_clusters=20),
        "SpectralBiclustering": SpectralBiclustering(n_clusters=20)
    }
    
    descriptors = {
        "ORB": ORB(),
        "AKAZE": AKAZE(),
        "Resnet": Resnet(),
        "SIFT" : SIFT(entry_path=cfg['Victor']['SIFT'])
    }
    
    for name_clf, clf in clfs.items():
        for name_cluster, cluster in clusters.items():
            for name_descriptor, descriptor in descriptors.items():
                print(f"{name_clf}+{name_cluster}+{name_descriptor}")
                if not f"{name_clf}_{name_cluster}_{name_descriptor}.json" in os.listdir("results"):
                    Research.safe(
                        f"{name_clf}_{name_cluster}_{name_descriptor}",
                        Research.test(
                            Research.train(
                                clf=clf,
                                descriptor=descriptor,
                                cluster=cluster,
                                scale=False,
                                number_words=200
                            )
                        )
                    )       
    
    