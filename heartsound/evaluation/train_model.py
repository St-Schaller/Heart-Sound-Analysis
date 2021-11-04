import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def train_model():
    df_train = pd.read_csv('Physionet_allwav_with_labels_05hop_meanvector_6144_openl3.csv')
    df_test = pd.read_csv('Physionet_allwav_validation_panns_CNN14.csv')
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    print(df_train)

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    '''
    print(df_train['labels'].value_counts())
    max = df_train['labels'].value_counts().max()
    abnormal_class = df_train[df_train.labels == '1']
    print(abnormal_class)
    normal_class = df_train[df_train.Snore == '-1']

    abnormal_class = resample(abnormal_class, replace=True, n_samples=max, random_state=120)
    normal_class = resample(normal_class, replace=True, n_samples=max, random_state=120)


    df_train = pd.concat([abnormal_class, normal_class], ignore_index=True)
    print(df_train['labels'].value_counts())
    print(df_train)
    '''

    param_grid = {

        'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [200],
        'classifier__C': [1.0]
    }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))])

    gs = GridSearchCV(pipeline, cv=2, scoring="recall_macro",refit=True, param_grid=param_grid,
                      verbose=2)
    gs.fit(X_train, y_train.values.ravel())
    print(f'Best Train score: {gs.best_score_}')
    #preds_devel = gs.predict(X_devel)
    preds_test = gs.predict(X_test)
    #print(f"Devel Score: { recall_score(y_devel, preds_devel, average='macro')}")
    print(f"Test Score: {recall_score(y_test, preds_test, average='macro')}")
    #print(f"Best Test Score: {gs.best_estimator_.score(X_test, y_test)}")
    print(f'Best Parameters: {gs.best_params_}')

    print(classification_report(y_test, preds_test))
    #disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["E", "O", "T", "V"], cmap=plt.cm.Blues, normalize=None)
    disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["Normal", "Abnormal"], cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Physionet Openl3 6144 Upsampled')
    print(disp.confusion_matrix)
    plt.savefig('conf_matrix_cold_openl3_6144_upsampled.png')
    plt.show()
