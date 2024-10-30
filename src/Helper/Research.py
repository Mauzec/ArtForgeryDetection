import yaml

class Research:
    
    def __init__(self, user: str) -> None:
        self.cfg = None
        with open('config.yaml', 'r') as config:
            self.cfg = yaml.safe_load(config)[user]
            
    
    