import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def run_feature_selection(inputfolder):
    #import and prepare data
    data = pd.read_csv(inputfolder)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1:]

    #Define Sequential Forward Selection (sfs)
    sfs = SFS(LinearRegression(),
              k_features=13,
              forward=True,
              floating=False,
              scoring='neg_mean_squared_error',
              cv=5)

    sfs = sfs.fit(X, y)
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

    plt.title('Sequential Forward Selection')
    plt.grid()
    plt.show()