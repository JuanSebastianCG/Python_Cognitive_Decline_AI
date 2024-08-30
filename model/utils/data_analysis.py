import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

class DataAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_data(X, kind='boxplot'):

        if kind == 'boxplot':
            X.plot(kind='box', figsize=(12, 8))
            plt.title("Boxplot of Features")
            plt.xticks(rotation=90)
            
        elif kind == 'histogram':
            X.hist(figsize=(12, 8), bins=15, edgecolor='black')
            plt.suptitle("Histograms of Features")
        plt.show()

    @staticmethod
    def visualize_correlation(data: pd.DataFrame):

        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()

    @staticmethod
    def describe_data(data: pd.DataFrame):
        output = io.StringIO()
        data.info(buf=output)  #
        info = output.getvalue() 
        
        null_counts = data.isnull().sum().to_string()
        info += "\nMissing values by column:\n" + null_counts
    
        unique_counts = ""
        for col in data.columns:
            unique_counts += f"\n{col}:\n{data[col].value_counts().to_string()}\n"
        
        shape_info = "\nShape of DataFrame: " + str(np.shape(data))
        info += shape_info
        
        return info

    @staticmethod
    def scatter_plot(X, y, x_col, y_col):

        plt.scatter(X[x_col], y if y_col == 'y' else X[y_col])
        plt.title(f"Scatter Plot between {x_col} and {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col if y_col != 'y' else y.name)
        plt.show()

    @staticmethod
    def pair_plot(data, columns=None):

        sns.pairplot(data[columns])
        plt.show()

    @staticmethod
    def plot_distribution(data, column):

        sns.kdeplot(data=data, x=column)
        plt.title(f"Distribution of {column}")
        plt.show()
