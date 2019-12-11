import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from blackbox.models.base_model import AnomalyModel

class AnomalyDecisionTree(AnomalyModel):
    """
    Supervised anomaly detection method using Decision Tree. This models expects to
    receive data already labeled (1 means Anomaly, 0 means normal) for the training 
    process. The training process will be ran several times using the Grid Search Cross
    Validation technique in order to find the best tree.
    
    Args:
        verbose (boolean): verbose mode. Defaults to False. 

    """
    from sklearn.tree import DecisionTreeClassifier

    def __init__(self, verbose=False):
        pass

    def train(self, data):
        pass

    def predict(self, data):
        pass

    def flag_anomaly(self, data):
        pass
