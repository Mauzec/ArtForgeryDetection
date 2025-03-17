import yaml
from BoVW.BoVW import BoVW
from Helper.DatasetOperations import DataOperations
import json
import os

class Research:
    
    def __init__(self, user: str) -> None:
        self.cfg = None
        with open('config.yaml', 'r') as config:
            self.cfg = yaml.safe_load(config)[user]
            
        self.operations = DataOperations(user)
            
    def test_size(self,
                  descriptor,
                  number_words,
                  clf,
                  cluster,
                  artists,
                  other_artists,
                  filename
                  ) -> None:
        
        for artist in artists:
            for train_size in [1.0]:
                if os.path.exists(f"results/size/{train_size}/{artist}/{filename}"):
                    print(f"results/size/{train_size}/{artist}/{filename} exists")
                    continue
                print(filename)
                bovw = BoVW(
                    descriptor=descriptor,
                    number_words=number_words,
                    clf=clf,
                    cluster=cluster
                )
                train_data = self.operations.getData("TrainTest")(other_artists, [artist], train_size)
                X_train, y_train = self.operations.encode(train_data['Train'], 'Train')
                self.operations.scaleImages(X_train)
                bovw.fit(X_train, y_train)
                
                test_data = self.operations.getData("Inference")([artist])
                X_test, y_test = self.operations.encode(test_data, 'Inference')
                self.operations.scaleImages(X_test)
                y_pred = [int(y) for y in bovw.predict(X_test)]
                
                score = bovw.score(X_test, y_test)
                results = {
                    "X": X_test,
                    "y": y_test,
                    "pred": y_pred,
                    "score": score
                }
                self.__safe(f"results/size/{train_size}/{artist}", filename, results)
                self.operations.clearDirs()
                    
    def test_descriptor(self,
                  descriptors: list,
                  number_words,
                  clf,
                  cluster,
                  artists,
                  other_artists
                  ) -> None:
        
        for artist in artists:
            
            for descriptor in descriptors:
                    bovw = BoVW(
                        descriptor=descriptor,
                        number_words=number_words,
                        clf=clf,
                        cluster=cluster
                    )
                    train_data = self.operations.getData("TrainTest")(other_artists, [artist], 1.0)
                    X_train, y_train = self.operations.encode(train_data['Train'], 'Train')
                    self.operations.scaleImages(X_train)
                    bovw.fit(X_train, y_train)
                    
                    test_data = self.operations.getData("Inference")([artist])
                    X_test, y_test = self.operations.encode(test_data, 'Inference')
                    self.operations.scaleImages(X_test)
                    y_pred = [int(y) for y in bovw.predict(X_test)]
                    
                    score = bovw.score(X_test, y_test)
                    results = {
                        "X": X_test,
                        "y": y_test,
                        "pred": y_pred,
                        "score": score
                    }
                    self.__safe(f"results/descriptor/{descriptor}/{artist}", f"{clf}_{cluster}.json", results)
                    self.operations.clearDirs()
                    
    def test_cluster_classifier(self,
                  descriptor,
                  clf,
                  clf_params: list,
                  cluster,
                  cluster_params: list,
                  artists,
                  other_artists) -> None:
        
        mean_score = {}
        for clf_param in clf_params:
            for cluster_param in cluster_params:
                mean_score[f"{clf_param}, {cluster_param}"] = 0.0
        n_artists = len(artists)
        
        for artist in artists:
            
            for clf_param in clf_params:
                for cluster_param in cluster_params:
                    bovw = BoVW(
                        descriptor=descriptor,
                        number_words=cluster_param['n_clusters'],
                        clf=clf(**clf_param),
                        cluster=cluster(**cluster_param)
                    )
                    train_data = self.operations.getData("TrainTest")(other_artists, [artist], 0.4)
                    X_train, y_train = self.operations.encode(train_data['Train'], 'Train')
                    self.operations.scaleImages(X_train)
                    bovw.fit(X_train, y_train)
                    
                    test_data = self.operations.getData("Inference")([artist])
                    X_test, y_test = self.operations.encode(test_data, 'Inference')
                    self.operations.scaleImages(X_test)
                    mean_score[f"{clf_param}, {cluster_param}"] += bovw.score(X_test, y_test)
                    self.operations.clearDirs()
             
        best = ['', 0.0]      
        for clf_param in clf_params:
            for cluster_param in cluster_params:
                if mean_score[f"{clf_param}, {cluster_param}"] > best[1]:
                    best = [f"{clf_param}, {cluster_param}", mean_score[f"{clf_param}, {cluster_param}"]] 
        print(best)   
        return best
        
        
                    
    def __safe(self, path, filename, results: dict) -> None:
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{filename}", 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)