import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import argparse
from random import randrange
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from lpproj import LocalityPreservingProjection as lpp

sys.path.append("../PaperLinCFA")
from methods import preprocess, single_experiment_realData_nDim, train_sup_PCA, compute_PCA, compute_LPP, compute_kernelPCA, compute_isomap, compute_LLE


### main run ###

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='../PaperLinCFA/dataset/Life Expectancy Data.csv')
    parser.add_argument("--dataset", default='Life_Expectancy')
    parser.add_argument("--n_samples", default=20000, type=int)

    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.dataset_path)

    if args.dataset == 'Life_Expectancy':
        
        df = df.dropna()

        X = df.iloc[:,4:]
        Y = df.iloc[:,3]

        X_train, X_test, y_train, y_test = preprocess(X,Y)
    
    if args.dataset == 'finance':
        
        df = df.select_dtypes(['number'])
        cols_to_delete = df.columns[df.isnull().sum()/len(df) > .50]
        df.drop(cols_to_delete, axis = 1, inplace = True)
        df = df.dropna()
        X = df.drop(['Unnamed: 0','Cash Ratio'],axis=1)
        y=df['Cash Ratio']

        print("Dataset shape: {}".format(X.shape))

        X_train, X_test, y_train, y_test = preprocess(X,y)

    if args.dataset == 'climate1':
        y=df.iloc[:,1]
        X = df.iloc[:,2:]

        X_train, X_test, y_train, y_test = preprocess(X,y,shuffle=False)

    if args.dataset == 'climate2':
        X_train = np.array(df.iloc[:-392,:-1])
        y_train = np.array(df.iloc[:-392,-1]).reshape(-1,1)
        X_test = np.array(df.iloc[-392:,:-1])
        y_test = np.array(df.iloc[-392:,-1]).reshape(-1,1)

    if args.dataset == 'Housing':
        
        df = pd.read_csv(args.dataset_path, header=None)
        df = df.dropna()

        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]

        X_train, X_test, y_train, y_test = preprocess(X,Y)

    if args.dataset == 'superconductivity':
        
        df = pd.read_csv(args.dataset_path)
        df = df.dropna()
        print(args.n_samples)
        X = df.iloc[:args.n_samples,:-1]
        X = X[np.random.default_rng(seed=42).permutation(X.columns.values)]
        Y = df.iloc[:args.n_samples,-1]

        X_train, X_test, y_train, y_test = preprocess(X,Y)

    if args.dataset == 'cifar':
        
        df = pd.read_csv(args.dataset_path, header=None)
        df = df.dropna()

        X = df.iloc[:6000,:-1]
        Y = df.iloc[:6000,-1]

        X_train, X_test, y_train, y_test = preprocess(X,Y)


    print("Dataset shape: {}".format(df.shape))


    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)

    print('LinCFA number of reduced dimensions: ')
    print(len(cluster))
    
    pca_res = []
    mse_res = []
    i=0

    #train_sup_PCA(X_train,y_train,X_test,y_test) 

    compute_PCA(0.8, X_train, X_test, y_train, y_test)

    #compute_LPP(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_kernelPCA(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_isomap(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
    
    #compute_LLE(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

"""
    ################### Finance ###################

    print('\n### Finance ###')

    df = pd.read_csv('../PaperLinCFA/dataset/fundamentals.csv')
    df = df.select_dtypes(['number'])
    cols_to_delete = df.columns[df.isnull().sum()/len(df) > .50]
    df.drop(cols_to_delete, axis = 1, inplace = True)
    df = df.dropna()
    X = df.drop(['Unnamed: 0','Cash Ratio'],axis=1)
    y=df['Cash Ratio']

    print("Dataset shape: {}".format(X.shape))

    X_train, X_test, y_train, y_test = preprocess(X,y)

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)
    print('LinCFA number of reduced dimensions: ')
    print(len(cluster))

    train_sup_PCA(X_train,y_train,X_test,y_test) 

    compute_PCA(0.95, X_train, X_test, y_train, y_test)

    compute_LPP(X_train, X_test, y_train, y_test)

    ################### Climate 1 ###################

    print('\n### Climate ###')

    df = pd.read_csv('../PaperLinCFA/dataset/NDVI_anomalies.csv')
    y=df.iloc[:,1]
    #df = df.iloc[:,4:]
    X = df.iloc[:,2:]
    #y=df[df.columns[1]]

    print(X.shape)

    X_train, X_test, y_train, y_test = preprocess(X,y,shuffle=False)

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)
    print('LinCFA number of reduced dimensions: ')
    print(len(cluster))

    train_sup_PCA(X_train,y_train,X_test,y_test) 

    compute_PCA(0.95, X_train, X_test, y_train, y_test)

    compute_LPP(X_train, X_test, y_train, y_test)

    ################### Climate 2 ###################

    print('\n### Climate extended ###')

    df = pd.read_csv('../PaperLinCFA/dataset/droughts_extended.csv')

    X_train = np.array(df.iloc[:-392,:-1])
    y_train = np.array(df.iloc[:-392,-1]).reshape(-1,1)
    X_test = np.array(df.iloc[-392:,:-1])
    y_test = np.array(df.iloc[-392:,-1]).reshape(-1,1)

    cluster,score_full,score_aggr,mse_full,mse_aggr = single_experiment_realData_nDim(X_train, X_test, y_train, y_test)
    print('LinCFA number of reduced dimensions: ')
    print(len(cluster))

    train_sup_PCA(X_train,y_train,X_test,y_test) 

    compute_PCA(0.95, X_train, X_test, y_train, y_test)

    compute_LPP(X_train, X_test, y_train, y_test)
"""