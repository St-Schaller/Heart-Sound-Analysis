import glob
import pandas as pd
import librosa
from matplotlib import pyplot as plt
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
        '/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_allwav_with_labels_05hop_meanvector_6144_openl3.csv')
    df_test = pd.read_csv(
        '/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_validation_05hop_meanvector_6144_openl3.csv')
    print(df_train)
    print(df_test)
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_train = df_train[~df_train.file_name.isin(df_test['file_name'])]
    print(df_train)
    print(df_test)

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
                ('classifier', LinearSVC(max_iter=10000, class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000, class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000, class_weight='balanced'))
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        {
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1.0]
        }
    ]
    scores_best_estimators = []
    params_best_estimators = []
    best_estimators = []
    best_scores = []
    for i in range(len(pipelines)):
        gs = GridSearchCV(pipelines[i], parameter_grids[i], scoring='recall_macro', n_jobs=-1, cv=8, verbose=2)
        gs.fit(X_train, y_train.values.ravel())
        best_scores.append(gs.best_score_)
        scores_best_estimators.append(gs.best_estimator_.score(X_test, y_test))
        params_best_estimators.append(gs.best_params_)
        best_estimators.append(gs.best_estimator_)

    print(best_scores)
    print(scores_best_estimators)
    print(params_best_estimators)
    print(best_estimators)



def grid_search_with_val():
    df_train = pd.read_csv('Snore_Train_merged_with_labels_05hop_meanvector_6144_openl3.csv')
    df_devel = pd.read_csv('Snore_Devel_merged_with_labels_05hop_meanvector_6144_openl3.csv')
    df_test = pd.read_csv('Snore_Test_merged_with_labels_05hop_meanvector_6144_openl3.csv')
    df_train = df_train.dropna()
    df_devel = df_devel.dropna()
    df_test = df_test.dropna()
    print(df_train)

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_devel = df_devel.iloc[:, 1:-1]
    print(X_devel)
    y_devel = df_devel.iloc[:, -1:]
    print(y_devel)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    param_grid = {
        'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [200],
        'classifier__C': [1.0]
        # 'scaler__feature_range': [(0, 1), (-1, 1)],
        # 'dimensionality_reduction__n_components': [8, 16, 32, 100, 200],
        # 'classifier__C': np.logspace(0, -8, num=9)
    }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', LinearSVC(max_iter=10000))])
    # Grid-search all parameter combinations using a validation set.
    opt = GridSearch(model=pipeline, param_grid=param_grid)
    opt.fit(X_train, y_train.values.ravel(), X_devel, y_devel)
    print(f'Best Parameters: {opt.best_params}')
    print(f'Best Score on Devel: {opt.best_score}')
    print('Test Score for Optimized Parameters:', opt.score(X_test, y_test))
    preds = opt.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(confusion_matrix(y_test, preds, labels=["E", "O", "T", "V"]))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True)
    plt.show()

    def get_mean_wav_duration():
        sum = 0
        count = 0
        for audio_path in sorted(glob.iglob('/home/local/Dokumente/CI_14/Snore_dist/wav/Train_wav/*.wav')):
            duration = librosa.get_duration(filename=audio_path)
            print(duration)
            sum += duration
            count += 1
        print(f"Mean Wav Duration: {sum / count}")

