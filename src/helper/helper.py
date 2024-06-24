from BoVW.BoVW import BoVW
from dataset.get_pictures import DatasetOperations
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Process, Queue
import json
import os
import itertools

import yaml
with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
PATH = cfg['Victor']['Dataset']

class Research:
    
    @staticmethod
    def add_train_dataset(percentage: int = 100, scale: bool = True, resolution: str = "high"):
        DatasetOperations.clear()
        
        for using in ["Train", "Test"]:
            for artist in ["artist", "other_artist"]:
                os.mkdir(f"{cfg['Victor'][using]}\\{artist}")
                
        DatasetOperations.get_mona_original(PATH=PATH)
        DatasetOperations.get_work_train_dataset(PATH=PATH, percentage_train=percentage)
        DatasetOperations.get_mona_test(PATH=PATH, resolution=resolution)
        if scale:
            DatasetOperations.scale_all()
        
    @staticmethod
    def train(
        descriptor = AKAZE(),
        cluster = KMeans(),
        number_words = 200,
        clf = GradientBoostingClassifier(),
        scale = False
        ) -> BoVW:
        
        bovw = BoVW(
            descriptor=descriptor,
            cluster=cluster,
            clf=clf,
            number_words=number_words,
            scale=scale
        )
        bovw.add_train_dataset("dataset\\train")

        bovw.model_training()

        return bovw
    
    @staticmethod
    def test(bovw: BoVW, only_testing: bool = False) -> tuple[list[str], str, float] | float:
        propotion_correctly_definded = bovw.testing("dataset\\test")
        
        if not only_testing:
            result = [
                    bovw.classification_image(f"{cfg['Victor']['Test']}\\{path}\\{image}")
                    for path in ["artist", "other_artist"]
                    for image in os.listdir(f"{cfg['Victor']['Test']}\\{path}")
                    ]

            return bovw.parametres, result, propotion_correctly_definded
        else:
            return propotion_correctly_definded['accurancy']
    
    @staticmethod
    def safe(name: str, results: tuple) -> None:
        with open(f"results/{name}.json", "w") as file:
            data = {
                "parametres": results[0],
                "result": results[1],
                "testing": results[2]
            }
            json.dump(data, file)
            
    @staticmethod
    def testing(
        clfs: dict = None,
        clusters: dict = None,
        descriptors: dict = None,
        n_clusters: dict = None
    ):
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
                                    cluster=cluster,
                                    scale=False,
                                    number_words=n_clusters[name_descriptor]
                                )
                            )

                        )

class Multiprocessor:

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets
            
    
class GridSearch:
    @staticmethod                  
    def grid_search(
        cluster = None,
        clf  = None,
        descriptor = AKAZE(),
        parametres: dict = None,
        type_grid: str = "cluster",
        algorithm: str = "Without Name"
        ) -> dict:
        
        results = list()
        
        param_names = parametres.keys()
        kwargs = dict()
        for params in itertools.product(
            *parametres.values()
        ):
        
            attributes = ""
            for name, value in zip(param_names, params):
                attributes += f"{name}={value};"
                kwargs[name] = value
                
            print(f"{algorithm}: {kwargs}")
                
            results.append(
                (
                    attributes,
                    Research.test(
                        Research.train(
                            clf=clf,
                            descriptor=descriptor,
                            cluster=cluster(
                                **kwargs
                            ),
                            number_words=kwargs["n_clusters"]
                        ) if type_grid == "cluster" else Research.train(
                            clf=clf(
                                **kwargs
                                ),
                            descriptor=descriptor,
                            cluster=cluster
                        ),
                        only_testing=True
                    )
                )
            )
                        
        return algorithm, max(results, key=lambda item: item[1]), results
                        
                        
                        
                    
    @staticmethod
    def get_accurancy(
        cluster: KMeans = None,
        clf  = GradientBoostingClassifier(),
        descriptor = AKAZE(),
        number_words: int = None
    ) -> float:
        
        return Research.test(
            Research.train(
                descriptor=descriptor,
                clf=clf,
                cluster=cluster,
                number_words=number_words,
                scale=False
            ),
            only_testing=True
        )
        
        
        
        
    
        
            
        
        


        