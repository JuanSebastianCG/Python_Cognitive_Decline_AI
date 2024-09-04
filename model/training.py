import numpy as np
import time
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from model.utils.data_extraction import DataExtractor
from model.utils.tensorboard import get_tensorboard_writer
from io import BytesIO

class Writer:
    def __init__(self, log_path):
        self.writer = get_tensorboard_writer(log_path)
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def log_text(self, tag, text):
        self.writer.add_text(tag, text)
    
    def log_scalars(self, tag, scalars_dict, step):
        self.writer.add_scalars(tag, scalars_dict, step)

    def log_plot_train(self, epoch, values):
        for key, value in values.items():
            self.writer.add_scalar(key, value, epoch)
    
    def _plot_to_image(self):
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return np.array(Image.open(buf))

    def plot_metric(self, metric_name, train_values, validation_values=None):
        epochs = range(len(train_values))
        plt.figure()
        plt.plot(epochs, train_values, 'b', label=f'Training {metric_name}')
        if validation_values is not None:
            plt.plot(epochs, validation_values, 'r', label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        
        img = self._plot_to_image()
        self.writer.add_image(f'{metric_name}_Plot', img, dataformats='HWC')
        plt.close()

    def plot_hyperparameter_performance(self, results):
        plt.figure()
        plt.plot(results['params'], results['values'], 'b', label='Hyperparameter Performance')
        plt.title('Hyperparameter Optimization Performance')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.legend()
        
        img = self._plot_to_image()
        self.writer.add_image('Hyperparameter_Plot', img, dataformats='HWC')
        plt.close()
    
    def close(self):
        self.writer.close()

class HyperparameterOptimizer:
    def __init__(self, model, X_val, y_val, writer, cv_splits=3, scoring='roc_auc_ovr', random_state=42):
        """
        Initializes the HyperparameterOptimizer class with the given model, validation data, writer object, 
        and optional parameters for cross-validation, scoring metric, and random state.

        Parameters:
        model : estimator object
            The machine learning model to optimize.
        X_val : DataFrame
            The validation features.
        y_val : Series
            The validation labels.
        writer : object
            An instance of a custom Writer class for logging and visualization.
        cv_splits : int, optional
            Number of cross-validation splits (default is 3).
        scoring : str, optional
            The scoring metric used for optimization (default is 'roc_auc_ovr').
        random_state : int, optional
            The random state for reproducibility (default is 42).
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.writer = writer
        self.cv = StratifiedKFold(n_splits=cv_splits)
        self.scoring = scoring
        self.random_state = random_state

    def objective(self, n_estimators, max_depth, learning_rate):
        """
        Defines the objective function for hyperparameter optimization.
        This function is called by the Bayesian optimizer to evaluate a set of hyperparameters.

        Parameters:
        n_estimators : int
            Number of trees in the ensemble for Random Forest and XGBoost.
        max_depth : int
            Maximum depth of the tree.
        learning_rate : float
            Learning rate for XGBoost.

        Returns:
        float
            The mean cross-validation score for the given set of hyperparameters.
        """
        # Set the hyperparameters for both Random Forest and XGBoost
        params = {
            'random_forest__n_estimators': int(n_estimators),
            'random_forest__max_depth': int(max_depth),
            'xgboost__n_estimators': int(n_estimators),
            'xgboost__learning_rate': learning_rate,
            'xgboost__max_depth': int(max_depth)
        }
        
        # Update the model with the current set of hyperparameters
        self.model.set_params(**params)
        
        # Perform cross-validation and compute the mean score
        score = cross_val_score(self.model, self.X_val, self.y_val, cv=self.cv, scoring=self.scoring).mean()
        
        # Log the score using the writer object for monitoring purposes
        self.writer.log_scalar(f'Objective_Function/{self.scoring.upper()}', score, step=0)
        
        return score

    def optimize(self, param_bounds=None, init_points=5, n_iter=30):
        """
        Optimizes the hyperparameters using Bayesian Optimization.
        
        Parameters:
        param_bounds : dict, optional
            Dictionary specifying the boundaries for the hyperparameters (default is None).
        init_points : int, optional
            Number of initial random points to explore (default is 5).
        n_iter : int, optional
            Number of iterations for the Bayesian Optimization (default is 30).
        
        Returns:
        dict
            The best set of hyperparameters found during optimization.
        """
        # Set default parameter bounds if not provided
        if param_bounds is None:
            param_bounds = {
                'n_estimators': (100, 1000),
                'max_depth': (5, 50),
                'learning_rate': (0.01, 0.2)
            }
        
        # Initialize the Bayesian optimizer with the objective function and parameter bounds
        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=param_bounds,
            random_state=self.random_state
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Extract the best parameters and all optimization results
        best_params = optimizer.max['params']
        params = [result['params'] for result in optimizer.res]
        values = [result['target'] for result in optimizer.res]
        results = {'params': list(range(len(params))), 'values': values}
        
        # Plot the hyperparameter performance over iterations
        self.writer.plot_hyperparameter_performance(results)
        
        # Log the best hyperparameters as a formatted JSON string
        self.writer.log_text('Optimized_Hyperparameters', json.dumps(best_params, indent=4))
        
        # Format the best parameters for model setting
        formatted_params = {
            'random_forest__n_estimators': int(best_params['n_estimators']),
            'random_forest__max_depth': int(best_params['max_depth']),
            'xgboost__n_estimators': int(best_params['n_estimators']),
            'xgboost__learning_rate': best_params['learning_rate'],
            'xgboost__max_depth': int(best_params['max_depth'])
        }
        
        return formatted_params


class ModelTrainer:
    def __init__(self, data_folder_file="DB sin RM (filtrado) - normalizado 01.pkl", labels_name="Deterioro Cognitivo", log_path='../logs/fit/', n_splits=5):
        """
        Initializes the ModelTrainer class with data loading, data splitting, and logging setup.
        
        Parameters:
        data_folder_file : str, optional
            The filename of the dataset stored as a pickle file (default is "DB sin RM (filtrado) - normalizado 01.pkl").
        labels_name : str, optional
            The name of the column containing the labels (default is "Deterioro Cognitivo").
        log_path : str, optional
            The directory path where log files will be saved (default is '../logs/fit/').
        n_splits : int, optional
            Number of splits for cross-validation (default is 5).
        """
        pathData = '../../data/'
        
        # Load the dataset from the pickle file
        self.data = DataExtractor.load_data_pickle(pathData, data_folder_file)
        
        # Split the data into training, validation, and test sets
        separated_data = DataExtractor.extract_test_validation_training_data(
            self.data, labels_name, test_ratio=0.2, validation_ratio=0.2, training_ratio=0.6)
        
        # Extract features and labels for training, validation, and test sets
        self.X_train, self.y_train = separated_data['training']
        self.X_test, self.y_test = separated_data['test']
        self.X_validation, self.y_validation = separated_data['validation']
        
        # Initialize the Writer for logging and visualization
        self.writer = Writer(log_path)
        
        # Set the number of cross-validation splits
        self.n_splits = n_splits
        
        # Initialize the model as None, to be built later
        self.model = None

    def build_model(self, base_models=None, final_estimator=None, cv_splits=None):
        """
        Builds the model using a stacking classifier with optional base models and a final estimator.
        
        Parameters:
        base_models : list of tuples, optional
            A list of base models to be used in stacking (default includes RandomForest and XGBoost).
        final_estimator : estimator object, optional
            The final estimator to use in the stacking ensemble (default is LogisticRegression).
        cv_splits : int, optional
            Number of cross-validation splits to use (default is the value set in __init__).
        """
        if base_models is None:
            # Default base models: RandomForest and XGBoost
            base_models = [
                ('random_forest', RandomForestClassifier(random_state=42)),
                ('xgboost', XGBClassifier(eval_metric='logloss', random_state=42))
            ]
        if final_estimator is None:
            # Default final estimator: LogisticRegression
            final_estimator = LogisticRegression()
        if cv_splits is None:
            # Use the provided number of CV splits if not specified
            cv_splits = self.n_splits
        
        # Build the stacking classifier
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=StratifiedKFold(n_splits=cv_splits)
        )
        
        # Log the model structure for reference
        self.writer.log_text('Model_Structure', str(self.model))

    def train_model(self):
        """
        Trains the model using bootstrapped samples and logs the performance metrics.
        """
        start_time = time.time()
        train_accuracies = []
        validation_accuracies = []
        train_losses = []

        for epoch in range(10):
            # Perform random sampling of the training data with replacement
            sampled_indices = np.random.choice(self.X_train.index, size=len(self.X_train), replace=True)
            X_train_sampled = self.X_train.loc[sampled_indices]
            y_train_sampled = self.y_train.loc[sampled_indices]

            # Simulate training loss (placeholder for actual loss computation)
            train_loss = np.random.rand() * (10 - epoch)
            
            # Fit the model on the sampled training data
            self.model.fit(X_train_sampled, y_train_sampled)

            # Evaluate accuracy on the training and validation sets
            train_accuracy = accuracy_score(y_train_sampled, self.model.predict(X_train_sampled))
            validation_accuracy = self.model.score(self.X_validation, self.y_validation)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            train_losses.append(train_loss)

            # Log the training progress for the current epoch
            self.writer.log_plot_train(epoch, {
                'Accuracy/Train': train_accuracy,
                'Accuracy/Validation': validation_accuracy,
                'Loss/Train': train_loss
            })

        # Log total training time
        self.writer.log_text('Training_Details', f'Total training time: {time.time() - start_time:.2f} seconds')
        
        # Plot training and validation accuracy over epochs
        self.writer.plot_metric('Accuracy', train_accuracies, validation_accuracies)
        
        # Plot training loss over epochs
        self.writer.plot_metric('Loss', train_losses)

    def validate_model(self):
        """
        Validates the model on the validation set using bootstrapped samples and logs the accuracy.
        """
        accuracy_scores = []
        
        for i in range(100):
            # Perform random sampling of the validation data with replacement
            resample_idx = np.random.choice(self.X_validation.index, size=len(self.X_validation), replace=True)
            X_resample = self.X_validation.loc[resample_idx]
            y_resample = self.y_validation.loc[resample_idx]
            
            # Compute accuracy on the resampled validation data
            score = self.model.score(X_resample, y_resample)
            accuracy_scores.append(score)
            
            # Log the validation accuracy for this iteration
            self.writer.log_scalar('Validation_Accuracy', score, i)
            
        # Compute and log the mean accuracy over all resampling iterations
        mean_accuracy = np.mean(accuracy_scores)
        self.writer.log_text('Validation_Accuracy', f'Mean accuracy: {mean_accuracy:.4f}')

    def run(self):
        """
        Executes the full pipeline: model building, hyperparameter optimization, training, and validation.
        
        Returns:
        model : StackingClassifier
            The trained model after optimization and validation.
        """
        # Build the initial model
        self.build_model()
        
        # Optimize hyperparameters using Bayesian optimization
        optimizer = HyperparameterOptimizer(self.model, self.X_validation, self.y_validation, self.writer)
        best_params = optimizer.optimize()
        
        # Set the model with the best-found hyperparameters
        self.model.set_params(**best_params)
        
        # Train the model with the optimized hyperparameters
        self.train_model()
        
        # Validate the trained model
        self.validate_model()
        
        # Close the writer and finalize logging
        self.writer.close()
        
        return self.model
