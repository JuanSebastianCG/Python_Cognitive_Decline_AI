from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from preprocessing.data_cleaning import clean_data, balance_data
from preprocessing.data_extraction import load_data

def create_pipeline():
    """
    Crea un pipeline de preprocesamiento y modelo.
    """
    steps = [
        ('data_cleaning', clean_data),
        ('model', RandomForestClassifier(random_state=42))
    ]
    
    pipeline = Pipeline(steps=steps)
    return pipeline
