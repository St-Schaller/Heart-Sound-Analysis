import glob
from pathlib import Path
import pandas as pd
import librosa
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from hypopt import GridSearch
from sklearn.metrics import classification_report
import seaborn as sns



def grid_search_kNN(inputpath):
    df_train = pd.read_csv(inputpath)
    df_train.rename(columns={'0': "filename"}, inplace=True)


    '''validation_list = []
    for file in sorted(glob.glob("validationpath/*.wav")):
        validation_list.append(Path(file).stem)
        print(f"Validation List: {validation_list}")

    df_test = df_train[df_train['filename'].str.startswith(tuple(validation_list))]
    df_train = df_train[~df_train['filename'].str.startswith(tuple(validation_list))]'''

    df_test = df_train[df_train['filename'].str.startswith('d')]
    df_train = df_train[~df_train['filename'].str.startswith('d')]

    print(df_train)
    print(df_test)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    pipelines = [
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('classifier',  KNeighborsClassifier())
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier',  KNeighborsClassifier())
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier',  KNeighborsClassifier(n_neighbors=3))
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        {
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
    ]
    scores_best_estimators = []
    params_best_estimators = []
    best_estimators = []
    best_scores = []
    for i in range(len(pipelines)):
        gs = GridSearchCV(pipelines[i], parameter_grids[i], scoring='recall_macro', cv=8, verbose=3)
        gs.fit(X_train, y_train.values.ravel())
        best_scores.append(gs.best_score_)
        scores_best_estimators.append(gs.best_estimator_.score(X_test, y_test))
        params_best_estimators.append(gs.best_params_)
        best_estimators.append(gs.best_estimator_)

    print(best_scores)
    print(scores_best_estimators)
    print(best_estimators)


def grid_search_svm(inputpath):
    df_train = pd.read_csv(inputpath)
    df_train.rename(columns={'0': "filename"}, inplace=True)


    '''validation_list = []
    for file in sorted(glob.glob("validationpath/*.wav")):
        validation_list.append(Path(file).stem)

    df_test = df_train[df_train['filename'].str.startswith(tuple(validation_list))]
    df_train = df_train[~df_train['filename'].str.startswith(tuple(validation_list))]'''


    df_test = df_train[df_train['filename'].str.startswith('d')]
    df_train = df_train[~df_train['filename'].str.startswith('d')]

    print(df_train)
    print(df_test)
    df_train = df_train.dropna()
    print(df_train)
    df_test = df_test.dropna()

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    pipelines = [
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('classifier', SVC(kernel= 'rbf'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', SVC(kernel= 'rbf'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', SVC(kernel= 'rbf'))
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': [1, 0.1, 0.01, 0.001],
        },
        {
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': [1, 0.1, 0.01, 0.001],
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        }
    ]
    scores_best_estimators = []
    params_best_estimators = []
    best_estimators = []
    best_scores = []
    for i in range(len(pipelines)):
        gs = GridSearchCV(pipelines[i], parameter_grids[i], scoring='recall_macro', cv=2, verbose=3)
        gs.fit(X_train, y_train.values.ravel())
        best_scores.append(gs.best_score_)
        scores_best_estimators.append(gs.best_estimator_.score(X_test, y_test))
        params_best_estimators.append(gs.best_params_)
        best_estimators.append(gs.best_estimator_)

    print(best_scores)
    print(scores_best_estimators)
    print(best_estimators)

grid_search_svm()



