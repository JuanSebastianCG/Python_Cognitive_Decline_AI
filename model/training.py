import pandas as pd
from utils.tensorboard import get_tensorboard_writer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def train_model(file_path):
    writer = get_tensorboard_writer() 

    writer.add_scalar('Accuracy', 2, 1)
    writer.close()

    print("Training model...")




    return 

if __name__ == "__main__":
    file_path = r"C:\Users\juans\OneDrive\Documentos\iaProyect\Cognitive\data\raw\BD sin RM(Base de datos).csv"
    trained_model = train_model(file_path)
