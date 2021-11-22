import glob
from pathlib import Path
import pandas as pd
import librosa
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from hypopt import GridSearch
from sklearn.metrics import classification_report
import seaborn as sns



def grid_search():
    df_train = pd.read_csv(
        '/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_training_05hop_meanvector_6144_openl3.csv')
    df_test = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_test_05hop_meanvector_6144_openl3.csv')

    df_train = pd.concat([df_train, df_test], ignore_index=True)

    '''
    validation_list = []
    for file in sorted(glob.glob("/home/local/Dokumente/HeartApp/physionet_challenge/validation/*.wav")):
        validation_list.append(Path(file).stem)
        print(f"Validation List: {validation_list}")

    df_test = df_train[df_train.filename.isin(validation_list)]
    df_train = df_train[~df_train.filename.isin(validation_list)]
    
    '''
    df_test = df_train[df_train['filename'].str.startswith('c')]
    df_train = df_train[~df_train['filename'].str.startswith('c')]
    



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
                ('classifier',  KNeighborsClassifier())
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__n_neighbors': [3, 5, 11, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        {
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__n_neighbors': [3, 5, 11, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__n_neighbors': [3,5,11,9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
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
    print(params_best_estimators)
    print(best_estimators)


grid_search()


