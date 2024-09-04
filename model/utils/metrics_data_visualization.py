import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

class DataAnalyzer:
    def __init__(self):
        # Empty constructor for this class. If not necessary, it can be removed.
        pass

    @staticmethod
    def visualize_data(X, kind='boxplot', cols_per_plot=10):
        """
        Visualizes data from a DataFrame using a specified plot type, dividing the data into multiple plots if there are too many columns.
        
        Parameters:
        - X (pd.DataFrame): The DataFrame containing the data to visualize.
        - kind (str): The type of plot. Supported values are 'boxplot' or 'histogram'.
        - cols_per_plot (int): Maximum number of columns to display per plot.
        """
        # Number of plots needed
        num_plots = (len(X.columns) - 1) // cols_per_plot + 1
        
        for i in range(num_plots):
            # Select a subset of columns for the current plot
            start_col = i * cols_per_plot
            end_col = min((i + 1) * cols_per_plot, len(X.columns))
            subset = X.iloc[:, start_col:end_col]

            if kind == 'boxplot':
                ax = subset.plot(kind='box', figsize=(12, 8))
                plt.title(f"Boxplot of Features: Chart {i + 1}")
                plt.xticks(rotation=90)  # Rotate X-axis labels for better visualization
                plt.show()

            elif kind == 'histogram':
                subset.hist(figsize=(12, 8), bins=15, edgecolor='black', layout=(4, (end_col - start_col + 3) // 4))
                plt.suptitle(f"Histograms of Features: Chart {i + 1}")
                plt.show()

    @staticmethod
    def visualize_correlation(data):
        """
        Visualizes a heatmap of the correlation matrix of the DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame whose correlation is to be visualized.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Heatmap of Correlation")
        plt.show()

    @staticmethod
    def describe_data(data):
        """
        Generates a detailed description of the DataFrame including general information,
        count of null values, and frequencies of unique values per column.

        Parameters:
        - data (pd.DataFrame): The DataFrame to describe.

        Returns:
        - str: Detailed description of the DataFrame.
        """
        output = io.StringIO()
        data.info(buf=output)  # Retrieve general information about the DataFrame
        info = output.getvalue()
        
        null_counts = data.isnull().sum().to_string()
        info += "\nNull values per column:\n" + null_counts
    
        unique_counts = ""
        for col in data.columns:
            unique_counts += f"\n{col}:\n{data[col].value_counts().to_string()}\n"

        info += "\n\n"+data.dtypes.to_string()
        
        shape_info = "\nDataFrame Shape: " + str(data.shape)
        info += shape_info
        
        return info

    @staticmethod
    def scatter_plot(X, y, x_col, y_col):
        """
        Creates a scatter plot between two columns.

        Parameters:
        - X (pd.DataFrame): DataFrame of the data.
        - y (pd.Series): Response variable data or a column from the DataFrame.
        - x_col (str): Name of the X-axis column.
        - y_col (str): Name of the Y-axis column or 'y' if using the y variable.
        """
        plt.scatter(X[x_col], y if y_col == 'y' else X[y_col])
        plt.title(f"Scatter Plot between {x_col} and {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col if y_col != 'y' else y.name)
        plt.show()

    @staticmethod
    def pair_plot(data, columns=None):
        """
        Visualizes pair plots for the specified columns of the DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame of the data.
        - columns (list): List of columns to include in the pair plot. If None, uses all columns.
        """
        sns.pairplot(data[columns])
        plt.show()

    @staticmethod
    def plot_distribution(data, column):
        """
        Visualizes the distribution of a specific column using a KDE plot.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.
        - column (str): Name of the column whose distribution is to be visualized.
        """
        sns.kdeplot(data=data, x=column)
        plt.title(f"Distribution of {column}")
        plt.show()
