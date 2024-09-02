import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

class DataAnalyzer:
    def __init__(self):
        # Constructor vacío en este caso. Si no es necesario, se puede eliminar.
        pass

    @staticmethod
    def visualize_data(X, kind='boxplot', cols_per_plot=10):
        """
        Visualiza los datos de un DataFrame usando un tipo de gráfico especificado, dividiendo los datos en múltiples gráficos si hay demasiadas columnas.
        
        Parámetros:
        - X (pd.DataFrame): El DataFrame con los datos a visualizar.
        - kind (str): El tipo de gráfico. Valores admitidos son 'boxplot' o 'histogram'.
        - cols_per_plot (int): Número máximo de columnas a mostrar por gráfico.
        """
        # Número de gráficos necesarios
        num_plots = (len(X.columns) - 1) // cols_per_plot + 1
        
        for i in range(num_plots):
            # Selecciona un subconjunto de columnas para el gráfico actual
            start_col = i * cols_per_plot
            end_col = min((i + 1) * cols_per_plot, len(X.columns))
            subset = X.iloc[:, start_col:end_col]

            if kind == 'boxplot':
                ax = subset.plot(kind='box', figsize=(12, 8))
                plt.title(f"Boxplot de características: Gráfico {i + 1}")
                plt.xticks(rotation=90)  # Rotación de las etiquetas del eje X para mejor visualización
                plt.show()

            elif kind == 'histogram':
                subset.hist(figsize=(12, 8), bins=15, edgecolor='black', layout=(4, (end_col - start_col + 3) // 4))
                plt.suptitle(f"Histogramas de características: Gráfico {i + 1}")
                plt.show()

    @staticmethod
    def visualize_correlation(data):
        """
        Visualiza un mapa de calor de la matriz de correlación del DataFrame.

        Parámetros:
        - data (pd.DataFrame): El DataFrame cuya correlación se quiere visualizar.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Mapa de calor de correlación")
        plt.show()

    @staticmethod
    def describe_data(data):
        """
        Genera una descripción detallada del DataFrame incluyendo información general,
        conteo de valores nulos y frecuencias de valores únicos por columna.

        Parámetros:
        - data (pd.DataFrame): El DataFrame a describir.

        Retorna:
        - str: Descripción detallada del DataFrame.
        """
        output = io.StringIO()
        data.info(buf=output)  # Obtención de la información general del DataFrame
        info = output.getvalue()
        
        null_counts = data.isnull().sum().to_string()
        info += "\nValores nulos por columna:\n" + null_counts
    
        unique_counts = ""
        for col in data.columns:
            unique_counts += f"\n{col}:\n{data[col].value_counts().to_string()}\n"

        info += "\n\n"+data.dtypes.to_string()
        
        shape_info = "\nForma del DataFrame: " + str(data.shape)
        info += shape_info
        
        return info

    @staticmethod
    def scatter_plot(X, y, x_col, y_col):
        """
        Crea un gráfico de dispersión entre dos columnas.

        Parámetros:
        - X (pd.DataFrame): DataFrame de los datos.
        - y (pd.Series): Datos de la variable respuesta o una columna del DataFrame.
        - x_col (str): Nombre de la columna del eje X.
        - y_col (str): Nombre de la columna del eje Y o 'y' si se usa la variable y.
        """
        plt.scatter(X[x_col], y if y_col == 'y' else X[y_col])
        plt.title(f"Gráfico de dispersión entre {x_col} y {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col if y_col != 'y' else y.name)
        plt.show()

    @staticmethod
    def pair_plot(data, columns=None):
        """
        Visualiza gráficos de pares para las columnas especificadas del DataFrame.

        Parámetros:
        - data (pd.DataFrame): El DataFrame de los datos.
        - columns (list): Lista de columnas a incluir en el pair plot. Si es None, usa todas las columnas.
        """
        sns.pairplot(data[columns])
        plt.show()

    @staticmethod
    def plot_distribution(data, column):
        """
        Visualiza la distribución de una columna específica usando un KDE plot.

        Parámetros:
        - data (pd.DataFrame): DataFrame con los datos.
        - column (str): Nombre de la columna cuya distribución se quiere visualizar.
        """
        sns.kdeplot(data=data, x=column)
        plt.title(f"Distribución de {column}")
        plt.show()
