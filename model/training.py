import pandas as pd
from sklearn.model_selection import train_test_split
from pipelines import create_pipeline  
from utils import get_tensorboard_writer
import pickle

def train_model(file_path):
    writer = get_tensorboard_writer() 

    writer.add_scalar('Accuracy', accuracy, 1)
    writer.close()





    return pipeline

if __name__ == "__main__":
    file_path = r"C:\Users\juans\OneDrive\Documentos\iaProyect\Cognitive\data\raw\BD sin RM(Base de datos).csv"
    trained_model = train_model(file_path)
