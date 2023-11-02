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

    if args.dataset == 'gene':
        
        df = df.dropna()
        df = df.select_dtypes(include=[float])
        df = df.drop(df.std()[df.std() <= 0.1].index.values, axis=1)
        normalized_df=(df-df.mean())/df.std()
        normalized_df = normalized_df.drop(normalized_df.std()[normalized_df.std() <= 0.1].index.values, axis=1)
        normalized_df = normalized_df.iloc[:,:-1]
        normalized_df["mean_std"] = normalized_df["gene_305"]
        normalized_df = normalized_df.drop("gene_305",axis=1)
        print(normalized_df.std())
        X_train = normalized_df.iloc[:-265,:-1].values
        X_test = normalized_df.iloc[-265:,:-1].values
        y_train = normalized_df.iloc[:-265,-1].values.reshape(-1,1)
        y_test = normalized_df.iloc[-265:,-1].values.reshape(-1,1)

    print("Dataset shape: {}".format(df.shape))

    scores_full = []
    scores_aggr = []
    mses_full = []
    mses_aggr = []
    red_dims = []
    best_dimensions_isomap = []
    best_r2s_isomap = []
    best_mses_isomap = []
    aggr_RRMSEs = []
    full_RRMSEs = []
    scores_ridge = []
    MSEs_ridge = []
    RRMSEs_ridge = []
    scores_lasso = []
    MSEs_lasso = []
    RRMSEs_lasso = []
    r2s_svr_full = []
    mses_svr_full = []
    RRMSEs_svr_full = []
    r2s_xgboost_full = []
    mses_xgboost_full = []
    RRMSEs_xgboost_full = []
    r2s_mlp_full = []
    mses_mlp_full = []
    RRMSEs_mlp_full = []
    r2s_svr_aggr = []
    mses_svr_aggr = []
    RRMSEs_svr_aggr = []
    r2s_xgboost_aggr = []
    mses_xgboost_aggr = []
    RRMSEs_xgboost_aggr = []
    r2s_mlp_aggr = []
    mses_mlp_aggr = []
    RRMSEs_mlp_aggr = []
    for curr_seed in [0]:#range(5):
        curr_xy_train = resample(np.concatenate((X_train,y_train),axis=1), random_state=curr_seed, replace=True)#,n_samples=1000)
        curr_X_train = curr_xy_train[:,:-1]
        curr_y_train = curr_xy_train[:,-1].reshape(-1,1)

        print(curr_xy_train.shape,curr_X_train.shape,curr_y_train.shape,X_test.shape,y_test.shape)
        cluster,score_full,score_aggr,mse_full,mse_aggr,aggr_RRMSE,full_RRMSE,score_ridge,MSE_ridge,RRMSE_ridge,score_lasso,MSE_lasso,RRMSE_lasso,r2_svr_full,mse_svr_full,RRMSE_svr_full,r2_xgboost_full,mse_xgboost_full,RRMSE_xgboost_full,r2_mlp_full,mse_mlp_full,RRMSE_mlp_full,r2_svr_aggr,mse_svr_aggr,RRMSE_svr_aggr,r2_xgboost_aggr,mse_xgboost_aggr,RRMSE_xgboost_aggr,r2_mlp_aggr,mse_mlp_aggr,RRMSE_mlp_aggr = single_experiment_realData_nDim(curr_X_train, X_test, curr_y_train, y_test)
        scores_full.append(score_full)
        scores_aggr.append(score_aggr)
        mses_full.append(mse_full)
        mses_aggr.append(mse_aggr)
        aggr_RRMSEs.append(aggr_RRMSE)
        full_RRMSEs.append(full_RRMSE)
        scores_ridge.append(score_ridge)
        MSEs_ridge.append(MSE_ridge)
        RRMSEs_ridge.append(RRMSE_ridge)
        scores_lasso.append(score_lasso)
        MSEs_lasso.append(MSE_lasso)
        RRMSEs_lasso.append(RRMSE_lasso)

        r2s_svr_full.append(r2_svr_full)
        mses_svr_full.append(mse_svr_full)
        RRMSEs_svr_full.append(RRMSE_svr_full)
        r2s_xgboost_full.append(r2_xgboost_full)
        mses_xgboost_full.append(mse_xgboost_full)
        RRMSEs_xgboost_full.append(RRMSE_xgboost_full)
        r2s_mlp_full.append(r2_mlp_full)
        mses_mlp_full.append(mse_mlp_full)
        RRMSEs_mlp_full.append(RRMSE_mlp_full)
        r2s_svr_aggr.append(r2_svr_aggr)
        mses_svr_aggr.append(mse_svr_aggr)
        RRMSEs_svr_aggr.append(RRMSE_svr_aggr)
        r2s_xgboost_aggr.append(r2_xgboost_aggr)
        mses_xgboost_aggr.append(mse_xgboost_aggr)
        RRMSEs_xgboost_aggr.append(RRMSE_xgboost_aggr)
        r2s_mlp_aggr.append(r2_mlp_aggr)
        mses_mlp_aggr.append(mse_mlp_aggr)
        RRMSEs_mlp_aggr.append(RRMSE_mlp_aggr)
        print('LinCFA number of reduced dimensions: ')
        print(len(cluster))
        red_dims.append(len(cluster))

        #best_dimension_isomap,best_r2_isomap,best_mse_isomap = compute_isomap(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
        #best_dimensions_isomap.append(best_dimension_isomap)
        #best_r2s_isomap.append(best_r2_isomap)
        #best_mses_isomap.append(best_mse_isomap)

    print(f'Scores full: ')
    compute_CI(scores_full,5)
    print(f'\nScores aggr: ')
    compute_CI(scores_aggr,5)
    print(f'\nMSEs full: ')
    compute_CI(mses_full,5)
    print(f'\nMSEs aggr: ')
    compute_CI(mses_aggr,5)
    print(f'\nRRMSE full: ')
    compute_CI(full_RRMSEs,5)
    print(f'\nRRMSE aggr: ')
    compute_CI(aggr_RRMSEs,5)
    print(f'\nFull dimensions: ')
    compute_CI([curr_X_train.shape[1],curr_X_train.shape[1]],2)
    print(f'\nReduced dimensions: ')
    compute_CI(red_dims,5)

    print(f'\nScores ridge: ')
    compute_CI(scores_ridge,5)
    print(f'\nMSEs ridge: ')
    compute_CI(MSEs_ridge,5)
    print(f'\nRRMSE ridge: ')
    compute_CI(RRMSEs_ridge,5)

    print(f'\nScores lasso: ')
    compute_CI(scores_lasso,5)
    print(f'\nMSEs lasso: ')
    compute_CI(MSEs_lasso,5)
    print(f'\nRRMSE lasso: ')
    compute_CI(RRMSEs_lasso,5)

    print(f'Scores full SVR: ')
    compute_CI(r2s_svr_full,5)
    print(f'\nScores aggr SVR: ')
    compute_CI(r2s_svr_aggr,5)
    print(f'\nMSEs full SVR: ')
    compute_CI(mses_svr_full,5)
    print(f'\nMSEs aggr SVR: ')
    compute_CI(mses_svr_aggr,5)
    print(f'\nRRMSE full SVR: ')
    compute_CI(RRMSEs_svr_full,5)
    print(f'\nRRMSE aggr SVR: ')
    compute_CI(RRMSEs_svr_aggr,5)

    print(f'Scores full xgboost: ')
    compute_CI(r2s_xgboost_full,5)
    print(f'\nScores aggr xgboost: ')
    compute_CI(r2s_xgboost_aggr,5)
    print(f'\nMSEs full xgboost: ')
    compute_CI(mses_xgboost_full,5)
    print(f'\nMSEs aggr xgboost: ')
    compute_CI(mses_xgboost_aggr,5)
    print(f'\nRRMSE full xgboost: ')
    compute_CI(RRMSEs_xgboost_full,5)
    print(f'\nRRMSE aggr xgboost: ')
    compute_CI(RRMSEs_xgboost_aggr,5)
    
    print(f'Scores full mlp: ')
    compute_CI(r2s_mlp_full,5)
    print(f'\nScores aggr mlp: ')
    compute_CI(r2s_mlp_aggr,5)
    print(f'\nMSEs full mlp: ')
    compute_CI(mses_mlp_full,5)
    print(f'\nMSEs aggr mlp: ')
    compute_CI(mses_mlp_aggr,5)
    print(f'\nRRMSE full mlp: ')
    compute_CI(RRMSEs_mlp_full,5)
    print(f'\nRRMSE aggr mlp: ')
    compute_CI(RRMSEs_mlp_aggr,5)
 
    #print(f'Isomap Reduced dimensions: \n')
    #compute_CI(best_dimensions_isomap,5)
    #print(f'Scores Isomap: \n')
    #compute_CI(best_r2s_isomap,5)
    #print(f'MSEs Isomap: \n')
    #compute_CI(best_mses_isomap,5)

    pca_res = []
    mse_res = []
    i=0

    #train_sup_PCA(X_train,y_train,X_test,y_test) 

    #compute_PCA(0.8, X_train, X_test, y_train, y_test)

    #compute_LPP(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_kernelPCA(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))
    
    #compute_LLE(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))

    #compute_RRelieFF(X_train, X_test, y_train, y_test, min(50,X_train.shape[1]-1))


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