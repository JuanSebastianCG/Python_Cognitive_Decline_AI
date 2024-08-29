import pandas as pd
from sklearn.model_selection import train_test_split
from pipelines import create_pipeline  
from utils import get_tensorboard_writer
import pickle

def train_model(file_path):
    writer = get_tensorboard_writer() 

    # Cargar los datos
    data = load_data(file_path)

    # Separar características y objetivo
    X = data.drop(columns=['target'])
    y = data['target']

    # Balancear los datos
    X_balanced, y_balanced = balance_data(X, y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Crear el pipeline
    pipeline = create_pipeline()

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calcular y registrar la precisión
    accuracy = accuracy_score(y_test, y_pred)
    writer.add_scalar('Accuracy', accuracy, 1)
    writer.close()

    print("Model accuracy:", accuracy)

    # Guardar el modelo en un archivo pickle
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)

    return pipeline

if __name__ == "__main__":
    file_path = r"C:\Users\juans\OneDrive\Documentos\iaProyect\Cognitive\data\raw\BD sin RM(Base de datos).csv"
    trained_model = train_model(file_path)
