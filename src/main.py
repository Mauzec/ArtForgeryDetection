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

    Research.add_train_dataset(percentage=5, scale=True, resolution="low")
    
    Research.safe(
        "test",
        Research.test(
            Research.train(
                descriptor=FACE(
                    predictor_path=cfg['Victor']['FACE']['PREDICTOR'],
                    recognition_path=cfg['Victor']['FACE']['RECOGNITION'],
                )
            )
        )
    )
    
    