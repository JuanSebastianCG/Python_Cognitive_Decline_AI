import pandas as pd
from sklearn.model_selection import train_test_split
from pipelines import create_pipeline  
from sklearn.metrics import accuracy_score
from utils import get_tensorboard_writer

def train_model(data):
    writer = get_tensorboard_writer() 

    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    # Calcular y registrar la precisión
    accuracy = accuracy_score(y_test, y_pred)
    writer.add_scalar('Accuracy', accuracy, 1)
    writer.close()

    print("Model accuracy:", accuracy)
    return pipeline

if __name__ == "__main__":
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    trained_model = train_model(data)
