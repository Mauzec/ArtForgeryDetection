from Helper.DatasetOperations import DataOperations
from BoVW.BoVW import BoVW
from Helper.Research import Research
from CustomDescriptors.utilts import *
from classifier.utils import *
from cluster.utilts import *
import yaml
import json

cfg = None
with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)["Victor"]

OTHER_ARTISTS = ['Joshua_Reynolds', 'Raphael', 'Rembrandt', 'Thomas_Lawrence', 'Titian']
ARTISTS = ['dama_gorn', 'Mona', 'Saint-John-Baptist', 'woman-pearl-earring']

clfs = {
    "SVM": LinearSVC(
        max_iter=8000),
    
    "GradientBoosting": GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=500
        ),
    
    "LogisticRegression": LogisticRegression(
        penalty='l2',
        solver='saga',
        max_iter=300
        ),
 }

clusters = {
    "KMeans": KMeans(
        n_clusters=60,
        tol=10**(-4),
        algorithm='elkan',
        max_iter=300,
    ),
    
    "MiniBatchKMeans": MiniBatchKMeans(
        n_clusters=60,
        batch_size=1024,
        tol=0.0,
        max_no_improvement=3
    ),
    "BisectingKMeans": BisectingKMeans(
        n_clusters=60
    )
     
 }
n_clusters = {
    "ORB": 500,
    "AKAZE": 500,
    "Resnet": 500,
    "SIFT" : 500,
    "FACE": 60
 }

descriptors = {
    # "ORB": ORB(),
    # "AKAZE": AKAZE(),
    # "SIFT" : SIFT(),
     "FACE": FACE(
         predictor_path=cfg['FACE']['PREDICTOR'],
         recognition_path=cfg['FACE']['RECOGNITION']
     )
 }

if __name__ == "__main__":
    
    research = Research('Victor')
    operations = DataOperations('Victor')
    
    operations.clearDirs()
    results = []
    for clf_name, clf in clfs.items():
        for cluster_name, cluster in clusters.items():
            for descriptor_name, descriptor in descriptors.items():
                research.test_size(
                    descriptor=descriptor,
                    number_words=500,
                    clf=clf,
                    cluster=cluster,
                    artists=ARTISTS,
                    other_artists=OTHER_ARTISTS,
                    filename=f"{clf_name}_{cluster_name}_{descriptor_name}.json"
                )
    
    
    # бля, я в ахуе с результатов, это пиздец какой-то