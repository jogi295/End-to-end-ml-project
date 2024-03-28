import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
from lib.utils import save_pkl_object
import sys
import logging
from lib.exceptions import CustomException

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics import classification_report


from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.neural_network import MLPClassifier, MLPRegressor


class ModelTrainer:
    def __init__(self):
        self.trained_model_path=os.path.join("models","model.pkl")

    def test_classification_algorithms(self, X_train, y_train, X_test, y_test, classifiers_to_run=None):
        if classifiers_to_run is None:
            classifiers_to_run = ['RandomForest', 'SVM', 'KNN', 'LogisticRegression', 'DecisionTree', 'NaiveBayes']

        results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', 'Train_Accuracy','Test_Accuracy', 'Test_F1_Score'])

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define classification algorithms and their parameter grids
        classifiers = {
            'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10]}),
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [10, 50, 100]}),
            'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
            'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
            'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
            'NaiveBayes': (GaussianNB(), {})
        }

        # Iterate through classifiers, perform grid search, and store results
        for name, (classifier, param_grid) in classifiers.items():
            if name in classifiers_to_run:
                grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
                grid_search.fit(X_train, y_train)

                # Get the best parameters
                best_params = grid_search.best_params_
                
                train_accuracy = accuracy_score(y_train, grid_search.best_estimator_.predict(X_train))

                # Evaluate the model on the test set
                y_pred = grid_search.best_estimator_.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1_score = f1_score(y_test, y_pred, average='weighted')

                # Store the results in the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Algorithm': [name],
                    'Parameters': [best_params],
                    'Train_Accuracy': [train_accuracy],
                    'Test_Accuracy': [test_accuracy],
                    'Test_F1_Score': [test_f1_score]
                })], ignore_index=True)
        
        results_df.to_csv('models/results.csv', index=False)
        best_model = grid_search.best_estimator_
        save_pkl_object('models/best_model.pkl', best_model)
        return results_df

    
    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            my_classifiers = ['RandomForest', 'SVM', 'DecisionTree']
            classification_results = self.test_classification_algorithms(X_train, y_train, X_test, y_test, classifiers_to_run=my_classifiers)
            return classification_results

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    print("Training started::")
    trainer = ModelTrainer()
    X_train = pd.read_csv('./data/processed/X_train_normalized.csv')
    y_train = pd.read_csv('./data/processed/y_train.csv')
    X_test =  pd.read_csv('./data/processed/X_test_normalized.csv')
    y_test =  pd.read_csv('./data/processed/y_test.csv')
    trainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
    print("Training completed::")

