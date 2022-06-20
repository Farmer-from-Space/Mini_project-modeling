import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def pre_M(X, test_merge):
    
    
    X_M = X.loc[:,X.columns.str.contains('M')]
    X = X.drop(X.loc[:,X.columns.str.contains('M')].columns, axis=1)

    test_M = test_merge.loc[:,test_merge.columns.str.contains('M')]
    test_merge = test_merge.drop(test_merge.loc[:,test_merge.columns.str.contains('M')].columns, axis=1)


    nan_counts = list(X_M.isnull().sum().value_counts().index)
    X2 = X
    for i in nan_counts:
        T = X_M.loc[:,X_M.isnull().sum() == i].dropna(axis=0)
        temp_index = T.index
        for f in T.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(T[f].values))
            T[f] = lbl.transform(list(T[f].values))
        pca = PCA(n_components=1)
        ratio =  pd.DataFrame(pca.fit_transform(T), columns=[f'M{i}'],index = temp_index)
        X2 = X2.merge(ratio, how="left", on='TransactionID')
        
    nan_counts = list(test_M.isnull().sum().value_counts().index)
    test_merge2 = test_merge
    for i in nan_counts:
        T = test_M.loc[:,test_M.isnull().sum() == i].dropna(axis=0)
        temp_index = T.index
        for f in T.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(T[f].values))
            T[f] = lbl.transform(list(T[f].values))
        pca = PCA(n_components=1)
        ratio =  pd.DataFrame(pca.fit_transform(T), columns=[f'M{i}'],index = temp_index)
        test_merge2 = test_merge2.merge(ratio, how="left", on='TransactionID')
    



    return(X2, test_merge2)