from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from model.preprocessing.data_extraction import DataExtractor
from model.preprocessing.data_cleaning import DataCleaner



def generalPipeline():
    """
    Crea un pipeline de preprocesamiento y modelo.
    """
    steps = [
       
    ]
    
    pipeline = Pipeline(steps=steps)
    return pipeline


    

