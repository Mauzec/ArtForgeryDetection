import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import time
import yaml
import os
import random as rand
import json

with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from CustomDescriptors.OrbDescriptor.ORB import ORB
from CustomDescriptors.ResnetDescriptor.Resnet import Resnet
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
from CustomDescriptors.FaceDescriptor.FACE import FACE
from helper.helper import Research, GridSearch, Multiprocessor
from BoVW.BoVW import BoVW
from dataset.get_pictures import DatasetOperations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, MiniBatchKMeans
    
clfs = {
    "GradientBoosting": GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=500
        ),
    "SVM": LinearSVC(
        max_iter=8000),
    "LogisticRegression": LogisticRegression(
        penalty='l2',
        solver='saga',
        max_iter=300
        ),
}
clusters = {
    "KMeans": KMeans(
        n_clusters=500,
        tol=10**(-4),
        algorithm='elkan',
        max_iter=300
    ),
    "MiniBatchKMeans": MiniBatchKMeans(
        n_clusters=500,
        batch_size=1024,
        tol=0.0,
        max_no_improvement=3
    )
}

n_clusters = {
    "FACE": 60
}
descriptors = {
    "FACE": FACE(
        predictor_path=cfg['Victor']['FACE']['PREDICTOR'],
        recognition_path=cfg['Victor']['FACE']['RECOGNITION']
    )
}


if __name__ == "__main__":
    
    Research.testing(
        clfs=clfs,
        clusters=clusters,
        descriptors=descriptors,
        n_clusters=n_clusters,
        quality_image="low"
    )
    