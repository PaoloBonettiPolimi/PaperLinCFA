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
from sklearn.utils import resample

sys.path.append("../PaperLinCFA")
from methods import preprocess, single_experiment_realData_nDim, train_sup_PCA, compute_PCA, compute_LPP, compute_kernelPCA, compute_isomap, compute_LLE, compute_RRelieFF

def compute_CI(list,n):
    print(f'{np.mean(list)} +- {1.96*np.std(list)/np.sqrt(n)}')

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

    scores_aggr = []
    mses_aggr = []
    red_dims = []
    aggr_RRMSEs = []
    r2s_svr_aggr = []
    mses_svr_aggr = []
    RRMSEs_svr_aggr = []
    r2s_xgboost_aggr = []
    mses_xgboost_aggr = []
    RRMSEs_xgboost_aggr = []
    r2s_mlp_aggr = []
    mses_mlp_aggr = []
    RRMSEs_mlp_aggr = []

    for curr_seed in range(5):
        curr_xy_train = resample(np.concatenate((X_train,y_train),axis=1), random_state=curr_seed, replace=True)#,n_samples=1000)
        curr_X_train = curr_xy_train[:,:-1]
        curr_y_train = curr_xy_train[:,-1].reshape(-1,1)

        print(curr_xy_train.shape,curr_X_train.shape,curr_y_train.shape,X_test.shape,y_test.shape)

        #best_dimension,best_r2,best_mse,best_RRMSE,r2_svr,mse_svr,RRMSE_svr,r2_xgboost,mse_xgboost,RRMSE_xgboost,r2_mlp,mse_mlp,RRMSE_mlp = compute_isomap(curr_X_train, X_test, curr_y_train, y_test, min(50,curr_X_train.shape[1]-1))
        best_dimension,best_r2,best_mse,best_RRMSE,r2_svr,mse_svr,RRMSE_svr,r2_xgboost,mse_xgboost,RRMSE_xgboost,r2_mlp,mse_mlp,RRMSE_mlp = compute_RRelieFF(curr_X_train, X_test, curr_y_train, y_test, min(50,curr_X_train.shape[1]-1))

        scores_aggr.append(best_r2)
        mses_aggr.append(best_mse)
        aggr_RRMSEs.append(best_RRMSE)

        r2s_svr_aggr.append(r2_svr)
        mses_svr_aggr.append(mse_svr)
        RRMSEs_svr_aggr.append(RRMSE_svr)

        r2s_xgboost_aggr.append(r2_xgboost)
        mses_xgboost_aggr.append(mse_xgboost)
        RRMSEs_xgboost_aggr.append(RRMSE_xgboost)

        r2s_mlp_aggr.append(r2_mlp)
        mses_mlp_aggr.append(mse_mlp)
        RRMSEs_mlp_aggr.append(RRMSE_mlp)

        red_dims.append(best_dimension)

        #best_dimension_isomap,best_r2_isomap,best_mse_isomap = compute_isomap(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
        #best_dimensions_isomap.append(best_dimension_isomap)
        #best_r2s_isomap.append(best_r2_isomap)
        #best_mses_isomap.append(best_mse_isomap)

    print(f'\nScores aggr: ')
    compute_CI(scores_aggr,5)
    print(f'\nMSEs aggr: ')
    compute_CI(mses_aggr,5)
    print(f'\nRRMSE aggr: ')
    compute_CI(aggr_RRMSEs,5)
    print(f'\nReduced dimensions: ')
    compute_CI(red_dims,5)

    print(f'\nScores aggr SVR: ')
    compute_CI(r2s_svr_aggr,5)
    print(f'\nMSEs aggr SVR: ')
    compute_CI(mses_svr_aggr,5)
    print(f'\nRRMSE aggr SVR: ')
    compute_CI(RRMSEs_svr_aggr,5)

    print(f'\nScores aggr xgboost: ')
    compute_CI(r2s_xgboost_aggr,5)
    print(f'\nMSEs aggr xgboost: ')
    compute_CI(mses_xgboost_aggr,5)
    print(f'\nRRMSE aggr xgboost: ')
    compute_CI(RRMSEs_xgboost_aggr,5)
    
    print(f'\nScores aggr mlp: ')
    compute_CI(r2s_mlp_aggr,5)
    print(f'\nMSEs aggr mlp: ')
    compute_CI(mses_mlp_aggr,5)
    print(f'\nRRMSE aggr mlp: ')
    compute_CI(RRMSEs_mlp_aggr,5)

    #train_sup_PCA(X_train,y_train,X_test,y_test) 

    #compute_PCA(0.8, X_train, X_test, y_train, y_test)

    #compute_LPP(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_kernelPCA(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
    
    #compute_LLE(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_RRelieFF(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
