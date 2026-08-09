# Jonathan Dargakis
# PA2 - Pipeline/GridSearch
# The purpose of this file is to use SVM and KNN pipelines with GridSearchCV
# to find the best model to determine the severity of an accident.

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('testing_data.csv')

# seperate target label
X_train = train_data.drop('Severity', axis=1)
y_train = train_data['Severity']
X_test = test_data.drop('Severity', axis=1)
y_test = test_data['Severity']

# Support Vector Classifier with class balancing
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svc', SVC(class_weight='balanced'))
])

# K-Nearest Neighbors Classifier
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])


# SVM grid (with linear, rbf, and polynomial kernels)
svm_param_grid = [
    {'svc__C': [0.01, 0.1, 1, 10], 'pca__n_components': [0.90, 0.95], 'svc__kernel': ['linear']},
    {'svc__C': [0.1, 1, 10], 'svc__gamma': [0.01, 0.1, 1], 'pca__n_components': [0.90, 0.95], 'svc__kernel': ['rbf']},
    {'svc__C': [0.1, 1], 'svc__gamma': [0.01, 0.1], 'svc__degree': [2, 3], 'pca__n_components': [0.90, 0.95], 'svc__kernel': ['poly']}
]

# KNN grid (different neighbors, weights, distance metrics)
knn_param_grid = {
    'knn__n_neighbors': [5, 10, 15, 20],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
    'pca__n_components': [0.90, 0.95]
}

# Stratified K-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# GridSearchCV
grid_search_svm = GridSearchCV(pipe_svm, svm_param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=2)
grid_search_knn = GridSearchCV(pipe_knn, knn_param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=2)

# Fit models
grid_search_svm.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)

# Compare best scores
print(f"SVM Best CV Score: {grid_search_svm.best_score_:.4f}")
print(f"KNN Best CV Score: {grid_search_knn.best_score_:.4f}")

# select best model based on score
if grid_search_svm.best_score_ > grid_search_knn.best_score_:
    print("\nSelected Model: SVM")
    best_model = grid_search_svm.best_estimator_
    best_params = grid_search_svm.best_params_
else:
    print("\nSelected Model: KNN")
    best_model = grid_search_knn.best_estimator_
    best_params = grid_search_knn.best_params_

# Evaluate best model on test data
y_pred = best_model.predict(X_test)

print("\nBest Parameters:", best_params)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Classification Report
print(classification_report(y_test, y_pred))
