import glob
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import resample


def train_model_kNN():
    #df_train = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_training_05hop_meanvector_6144_openl3.csv')
    #df_test = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_test_05hop_meanvector_6144_openl3.csv')
    #df_train = pd.concat([df_train, df_test], ignore_index=True)
    df_train = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/Manual_Feature_Extraction_All_Data_Splitted_with_MFCC_Spectroids.csv')
    df_train.rename(columns={'0': "filename"}, inplace=True)
    df_train.replace('...', 0, inplace=True)
    df_train.rename(columns={'273': "labels"}, inplace=True)
    df_train = df_train.fillna(0)

    '''validation_list = []
    for file in sorted(glob.glob("/home/local/Dokumente/HeartApp/physionet_challenge/validation/*.wav")):
        validation_list.append(Path(file).stem)

    df_test = df_train[df_train['filename'].str.startswith(tuple(validation_list))]
    df_train = df_train[~df_train['filename'].str.startswith(tuple(validation_list))]'''

    df_test = df_train[df_train['filename'].str.startswith('d')]
    df_train = df_train[~df_train['filename'].str.startswith('d')]



    print(df_train['labels'].value_counts())
    print(df_test['labels'].value_counts())

    max = df_train['labels'].value_counts().max()
    abnormal_class = df_train[df_train.labels == 1]
    print(abnormal_class)
    normal_class = df_train[df_train.labels == -1]
    
    abnormal_class = resample(abnormal_class, replace=True, n_samples=max, random_state=120)
    normal_class = resample(normal_class, replace=True, n_samples=max, random_state=120)

    df_train = pd.concat([abnormal_class, normal_class], ignore_index=True)
    print(df_train['labels'].value_counts())
    print(df_train)


    X_train = df_train.iloc[:, 1:-37]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-37]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    param_grid = {

        'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [32],
        'classifier__weights': ['distance'],
        'classifier__metric': ['manhattan']
    }

    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('dimensionality_reduction', PCA()),
        ('classifier', KNeighborsClassifier(n_neighbors=3))])

    gs = GridSearchCV(pipeline, cv=10, scoring="recall_macro",refit=True, param_grid=param_grid,
                      verbose=2)
    gs.fit(X_train, y_train.values.ravel())
    print(f'Best Train score: {gs.best_score_}')
    preds_test = gs.predict(X_test)
    print(f"Test Score: {recall_score(y_test, preds_test, average='macro')}")
    #print(f"Best Test Score: {gs.best_estimator_.score(X_test, y_test)}")
    print(f'Best Parameters: {gs.best_params_}')
    print(X_test.shape, y_test.shape)
    print(classification_report(y_test, preds_test))
    disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["Normal", "Abnormal"], cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Manually extracted No Segmentation Resampled Test D')
    print(disp.confusion_matrix)
    plt.savefig('conf_matrix_manual_no_seg_resampled_D.png')
    plt.show()


def train_model_svm():
    #df_train = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_training_05hop_meanvector_512_openl3.csv')
    #df_test = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_test_05hop_meanvector_512_openl3.csv')
    #df_train = pd.concat([df_train, df_test], ignore_index=True)
    df_train = pd.read_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/Manual_Feature_Extraction_All_Data_with_MFCC_Spectroids.csv')
    df_train.rename(columns={'0': "filename"}, inplace=True)
    df_train.rename(columns={'273': "labels"}, inplace=True)

    df_train.replace('...',0, inplace=True)
    df_train = df_train.fillna(0)

    validation_list = []
    for file in sorted(glob.glob("/home/local/Dokumente/HeartApp/physionet_challenge/validation/*.wav")):
        validation_list.append(Path(file).stem)

    df_test = df_train[df_train['filename'].str.startswith(tuple(validation_list))]
    df_train = df_train[~df_train['filename'].str.startswith(tuple(validation_list))]

    #df_test = df_train[df_train['filename'].str.startswith('d')]
    #df_train = df_train[~df_train['filename'].str.startswith('d')]

    '''print(df_train.shape)
    max = df_train['labels'].value_counts().max()
    abnormal_class = df_train[df_train.labels == 1]
    print(abnormal_class)
    normal_class = df_train[df_train.labels == -1]

    abnormal_class = resample(abnormal_class, replace=True, n_samples=max, random_state=120)
    df_train = pd.concat([abnormal_class, normal_class], ignore_index=True)
    print(df_train['labels'].value_counts())
    print(df_train)'''

    X_train = df_train.iloc[:, 1:-37]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-37]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)


    param_grid = {

        #'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [200],
        'classifier__C': [10],
        'classifier__gamma': [0.1]
    }

    pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', SVC(kernel='rbf'))])

    gs = GridSearchCV(pipeline, cv=4, scoring="recall_macro", refit=True, param_grid=param_grid,
                      verbose=2)
    gs.fit(X_train, y_train.values.ravel())
    print(f'Best Train score: {gs.best_score_}')
    preds_test = gs.predict(X_test)
    print(f"Test Score: {recall_score(y_test, preds_test, average='macro')}")
    # print(f"Best Test Score: {gs.best_estimator_.score(X_test, y_test)}")
    print(f'Best Parameters: {gs.best_params_}')

    print(classification_report(y_test, preds_test))
    disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["Normal", "Abnormal"], cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title('Manually extracted no Segmentation Original Test D')
    print(disp.confusion_matrix)
    plt.savefig('conf_matrix_manual_no_sg_original_svm_D.png')
    plt.show()



train_model_kNN()


