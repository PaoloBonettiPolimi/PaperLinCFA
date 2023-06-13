import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import column_or_1d
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

### compute correlation between two random variables
def compute_corr(x1,x2):
    return pearsonr(x1,x2)[0]

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

### compute the correlation threshold with empirical data
def compute_empirical_bound(x1,x2,y):

    x = np.zeros((len(x1),2))
    x[:,0] = x1
    x[:,1] = x2
    
    regr = LinearRegression()
    regr.fit(x,y)
    w1 = regr.coef_[0][0]
    w2 = regr.coef_[0][1]
    preds = regr.predict(x)
    residuals = y - preds
    
    n=len(x1)
    
    s_squared = np.dot(residuals.reshape(1,n),residuals)/(n-3)
    bound = 1 - (2*s_squared/((n-1)*(w1-w2)**2))
    corr = compute_corr(x1.reshape(n),x2.reshape(n))

    return bound

### aggregate each group of elements with their mean
def aggregate_clusters(cluster,x):
    a = np.zeros((len(x),len(cluster)))
    
    k=0
    for i in cluster:
        a[:,k] = np.mean(x[:,i],axis=1)
        k += 1
    return a

### D-dimensional experiment on the dataset
def single_experiment_realData_nDim(X_train, X_test, y_train, y_test):

    n_variables = X_train.shape[1]
    n_data = X_train.shape[0]
    
    score_full = []
    score_aggr = []
    mse_full = []
    mse_aggr = []
    predictions_tot = np.zeros((int(n_data),1))
    predictions_aggr = np.zeros((int(n_data),1))
    n_clust = []
    emp_score_full = []
    emp_score_aggr = []
    emp_mse_full = []
    emp_mse_aggr = []
    emp_predictions_tot = np.zeros((int(n_data),1))
    emp_predictions_aggr = np.zeros((int(n_data),1))
    emp_n_clust = []
        
    features_df = pd.DataFrame(X_train)
    target_df = pd.DataFrame(y_train)
    x = np.array(X_train)
    y = np.array(y_train)
    
    cluster=[]
    used_indices = []
    
    for j in range(n_variables):
        
        if j in used_indices: continue
        curr_list = [j]
        used_indices.append(j)
        for i in range(n_variables-j-1):
            corr = compute_corr(x[:,j],x[:,i+j+1])
            
            real_bound = compute_empirical_bound(x[:,j],x[:,i+j+1],y) 
                
            if ((real_bound<= corr)):
                curr_list.append(i+j+1)
                used_indices.append(i+j+1)
        cluster.append(curr_list)
    
    emp_n_clust = len(cluster)
    
    x_aggr=aggregate_clusters(cluster,x)
    x_test_aggr = aggregate_clusters(cluster,X_test)
    aggregate_df = pd.DataFrame(x_aggr)
        
    regr_full = LinearRegression().fit(features_df, target_df)
    regr_aggr = LinearRegression().fit(aggregate_df, target_df)
    
    print("full regression score: {0}".format(regr_full.score(X_test, y_test)))
    full_RRMSE = relative_root_mean_squared_error(y_test, regr_full.predict(X_test))
    print(f'RRMSE: {full_RRMSE}')
    compute_r2_nonLin(features_df, target_df, X_test, y_test)

    print("aggr regression score: {0}".format(regr_aggr.score(x_test_aggr, y_test)))
    aggr_RRMSE = relative_root_mean_squared_error(y_test, regr_aggr.predict(x_test_aggr))
    print(f'RRMSE: {aggr_RRMSE}')
    score_full.append(regr_full.score(X_test, y_test))
    score_aggr.append(regr_aggr.score(x_test_aggr, y_test))
    compute_r2_nonLin(aggregate_df, target_df, x_test_aggr, y_test)
    
    print("full regression MSE: {0}".format(mean_squared_error(y_test,regr_full.predict(X_test))))
    print("aggr regression MSE: {0}".format(mean_squared_error(y_test,regr_aggr.predict(x_test_aggr))))
    
    mse_full=mean_squared_error(y_test,regr_full.predict(X_test))
    mse_aggr=mean_squared_error(y_test,regr_aggr.predict(x_test_aggr))

    print ('Ridge Regression:')
    regr = Ridge().fit(features_df, target_df)
    print("R2: {0}".format(regr.score(X_test, y_test)))
    print(f'MSE: {mean_squared_error(y_test,regr.predict(X_test))}')
    print(f'RRMSE: {relative_root_mean_squared_error(y_test, regr.predict(X_test))}')
    print(regr.predict(X_test))

    print ('Lasso Regression:')
    regr = Lasso(0.1).fit(features_df, target_df)
    print("R2: {0}".format(regr.score(X_test, y_test)))
    print(f'MSE: {mean_squared_error(y_test,regr.predict(X_test))}')
    print(f'RRMSE: {relative_root_mean_squared_error(y_test, regr.predict(X_test).reshape(-1,1))}')
    print(regr.predict(X_test))

    return cluster,score_full,score_aggr,mse_full,mse_aggr


### supervised PCA
class spca(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_components, kernel="linear", eigen_solver='auto', 
                 max_iterations=None, gamma=0, degree=3, coef0=1, alpha=1.0, 
                 tolerance=0, fit_inverse_transform=False):
        
        self._num_components = num_components
        self._gamma = gamma
        self._tolerance = tolerance
        self._fit_inverse_transform = fit_inverse_transform
        self._max_iterations = max_iterations
        self._degree = degree
        self._kernel = kernel
        self._eigen_solver = eigen_solver
        self._coef0 = coef0
        self._centerer = KernelCenterer()
        self._alpha = alpha
        self._alphas = []
        self._lambdas = []
        
        
    def _get_kernel(self, X, Y=None):
        # Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors 
        # of the given matrix X, if Y is None. 
        
        # If Y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.
        
        # valid kernels are 'linear, rbf, poly, sigmoid, precomputed'
        
        args = {"gamma": self._gamma, "degree": self._degree, "coef0": self._coef0}
        
        return pairwise_kernels(X, Y, metric=self._kernel, n_jobs=-1, filter_params=True, **args)
    
    
    
    def _fit(self, X, Y):
        
        # calculate kernel matrix of the labels Y and centre it and call it K (=H.L.H)
        K = self._centerer.fit_transform(self._get_kernel(Y))
        
        # deciding on the number of components to use
        if self._num_components is not None:
            num_components = min(K.shape[0], self._num_components)
        else:
            num_components = self.K.shape[0]
        
        # Scale X
        # scaled_X = scale(X)
        
        # calculate the eigen values and eigen vectors for X^T.K.X
        Q = (X.T).dot(K).dot(X)
        
        # If n_components is much less than the number of training samples, 
        # arpack may be more efficient than the dense eigensolver.
        if (self._eigen_solver=='auto'):
            if (Q.shape[0]/num_components) > 20:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self._eigen_solver
        
        if eigen_solver == 'dense':
            # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
            self._lambdas, self._alphas = linalg.eigh(Q, eigvals=(Q.shape[0] - num_components, Q.shape[0] - 1))
            # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
        
        elif eigen_solver == 'arpack':
            # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, num_components, which="LA", tol=self._tolerance)
            self._lambdas, self._alphas = ssl_eigsh(A=Q, k=num_components, which="LA", tol=self._tolerance)
            
        indices = self._lambdas.argsort()[::-1]
        
        self._lambdas = self._lambdas[indices]
        self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self._alphas = self._alphas[:, indices]
        #return self._alphas
        self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self.X_fit = X

        
    def _transform(self):
        return self.X_fit.dot(self._alphas)
        
        
    def transform(self, X):
        return X.dot(self._alphas)
        
        
    def fit(self, X, Y):
        self._fit(X,Y)
        return
        
        
    def fit_and_transform(self, X, Y):
        self.fit(X, Y)
        return self._transform()

### R2 computation
def compute_r2(x_train, y_train, x_val, y_val):
    regr = LinearRegression().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    mse = mean_squared_error(y_val,y_pred)
    return r2_score(y_val, y_pred), mse

def compute_r2_nonLin(x_train, y_train, x_val, y_val):
    y_train = np.asarray(y_train).reshape(-1,)
    y_val = np.asarray(y_val).reshape(-1,)
    regr = SVR().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    mse = mean_squared_error(y_val,y_pred)
    print(f'SVM for regression: {r2_score(y_val, y_pred)}, {mse}\n')
    RRMSE = relative_root_mean_squared_error(y_val, y_pred)
    print(f'RRMSE: {RRMSE}')
    regr = XGBRegressor().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    mse = mean_squared_error(y_val,y_pred)
    print(f'XGBoost for regression: {r2_score(y_val, y_pred)}, {mse}\n')
    RRMSE = relative_root_mean_squared_error(y_val, y_pred)
    print(f'RRMSE: {RRMSE}')
    regr = MLPRegressor().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    mse = mean_squared_error(y_val,y_pred)
    print(f'MLP for regression: {r2_score(y_val, y_pred)}, {mse}\n')
    RRMSE = relative_root_mean_squared_error(y_val, y_pred)
    print(f'RRMSE: {RRMSE}')

### unsupervised PCA 
def compute_PCA(max_components, train_df, val_df, train_target, val_target):
    pca = PCA(n_components=max_components)
    actual_train = pca.fit_transform(train_df)
    actual_val = pca.transform(val_df)
    actual_r2, mse = compute_r2(actual_train, train_target, actual_val, val_target)
    print(f'95% Components: {actual_train.shape[1]},\n Linear regression R2: {actual_r2}, MSE: {mse}\n')
    regr = LinearRegression().fit(actual_train, train_target)
    print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
    compute_r2_nonLin(actual_train, train_target, actual_val, val_target)

    pca = PCA(n_components=73)
    actual_train = pca.fit_transform(train_df)
    actual_val = pca.transform(val_df)
    actual_r2, mse = compute_r2(actual_train, train_target, actual_val, val_target)
    print(f'73 Components: {actual_train.shape[1]},\n Linear regression R2: {actual_r2}, MSE: {mse}\n')
    regr = LinearRegression().fit(actual_train, train_target)
    print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
    return actual_r2

### LPP 
def compute_LPP(train_df, val_df, train_target, val_target, max_dim=50):
    best_r2 = 0
    best_mse = 0
    best_dimension = 1
    for i in range(max_dim):
        dimRedMethod = lpp(n_components=i+1)
        actual_train = dimRedMethod.fit_transform(train_df)
        actual_val = dimRedMethod.transform(val_df)
        actual_r2, actual_mse = compute_r2(actual_train, train_target, actual_val, val_target)
        print(actual_r2,actual_mse)
        if actual_r2>best_r2:
            best_r2 = actual_r2
            best_mse = actual_mse
            best_dimension = i+1 
    print(f'LPP --> Components: {best_dimension}, R2: {best_r2}, MSE: {best_mse}\n')
    dimRedMethod = lpp(n_components=best_dimension)
    actual_train = dimRedMethod.fit_transform(train_df)
    actual_val = dimRedMethod.transform(val_df)
    regr = LinearRegression().fit(actual_train, train_target)
    print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
    compute_r2_nonLin(actual_train, train_target, actual_val, val_target)
    return best_dimension,best_r2,best_mse


### preprocessing
def preprocess(X,Y, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=shuffle)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = preprocessing.StandardScaler().fit(np.array(X_train))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))

    return X_train, X_test, y_train, y_test

### train supervised PCA
def train_sup_PCA(X_train,y_train,X_test,y_test):
    r2_tot = {'linear':[],'poly':[],'sigmoid':[]}
    for kernel in ['linear', 'poly', 'sigmoid']:
            i=0
            pca_res = []
            mse_res = []
            while (i<=50):
                try:
                
                    trial = spca(num_components=i+1, kernel=kernel, degree=3, gamma=None, coef0=1)
                    X_train_proj = trial.fit_and_transform(X_train,y_train)
                    X_test_proj = trial.transform(X_test)
                    
                    if X_train_proj.shape[1]==0: continue
                
                    regr = LinearRegression().fit(X_train_proj, y_train)
                    pca_res.append(regr.score(X_test_proj, y_test))
                    mse_res.append(mean_squared_error(y_test,regr.predict(X_test_proj)))
                    if i==6:
                        print('6 features: ')
                        print(regr.score(X_test_proj, y_test))
                        print(mean_squared_error(y_test,regr.predict(X_test_proj)))
                    i += 1

                except: break 
            
            print("Supervised PCA ({1} kernel) best number of components, R2 score, MSE:\n {0}".format(np.argmax(pca_res)+1,kernel))
            print(pca_res[np.argmax(pca_res)])
            print(mse_res[np.argmax(pca_res)])
            
            trial = spca(num_components=np.argmax(pca_res)+1, kernel=kernel, degree=3, gamma=None, coef0=1)
            X_train_proj = trial.fit_and_transform(X_train,y_train)
            X_test_proj = trial.transform(X_test)
            regr = LinearRegression().fit(X_train_proj, y_train)
            print(relative_root_mean_squared_error(y_test, regr.predict(X_test_proj)))
            compute_r2_nonLin(X_train_proj, y_train, X_test_proj, y_test)
            
            r2_tot[kernel] = pca_res
    return r2_tot

### kernel PCA
def compute_kernelPCA(train_df, val_df, train_target, val_target, max_dim=50):
    best_r2 = 0
    best_mse = 0
    best_dimension = 1
    for curr_kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for i in range(max_dim):
            dimRedMethod = KernelPCA(n_components=i+1, kernel=curr_kernel)
            actual_train = dimRedMethod.fit_transform(train_df)
            actual_val = dimRedMethod.transform(val_df)
            actual_r2, actual_mse = compute_r2(actual_train, train_target, actual_val, val_target)
            #print(actual_r2, actual_mse) 
            if actual_r2>best_r2:
                best_r2 = actual_r2
                best_mse = actual_mse
                best_dimension = i+1 
        print(f'kernelPCA ({curr_kernel}) --> Components: {best_dimension}, R2: {best_r2}, MSE: {best_mse}\n')
        dimRedMethod = KernelPCA(n_components=best_dimension, kernel=curr_kernel)
        actual_train = dimRedMethod.fit_transform(train_df)
        actual_val = dimRedMethod.transform(val_df)
        regr = LinearRegression().fit(actual_train, train_target)
        print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
        compute_r2_nonLin(actual_train, train_target, actual_val, val_target)
    return best_dimension,best_r2,best_mse

### Isomap
def compute_isomap(train_df, val_df, train_target, val_target, max_dim=50):
    best_r2 = 0
    best_mse = 0
    best_dimension = 1
    for i in range(max_dim):
        dimRedMethod = Isomap(n_components=i+1,n_neighbors=25)
        actual_train = dimRedMethod.fit_transform(train_df)
        actual_val = dimRedMethod.transform(val_df)
        actual_r2, actual_mse = compute_r2(actual_train, train_target, actual_val, val_target)
        #print(actual_r2, actual_mse) 
        if actual_r2>best_r2:
            best_r2 = actual_r2
            best_mse = actual_mse
            best_dimension = i+1 
    print(f'Isomap --> Components: {best_dimension}, R2: {best_r2}, MSE: {best_mse}\n')
    dimRedMethod = Isomap(n_components=best_dimension,n_neighbors=10)
    actual_train = dimRedMethod.fit_transform(train_df)
    actual_val = dimRedMethod.transform(val_df)
    regr = LinearRegression().fit(actual_train, train_target)
    print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
    compute_r2_nonLin(actual_train, train_target, actual_val, val_target)
    return best_dimension,best_r2,best_mse

### LLE
def compute_LLE(train_df, val_df, train_target, val_target, max_dim=50):
    best_r2 = 0
    best_mse = 0
    best_dimension = 1
    for i in range(max_dim):
        dimRedMethod = LLE(n_components=i+1,n_neighbors=10)
        actual_train = dimRedMethod.fit_transform(train_df)
        actual_val = dimRedMethod.transform(val_df)
        actual_r2, actual_mse = compute_r2(actual_train, train_target, actual_val, val_target)
        #print(actual_r2, actual_mse) 
        if actual_r2>best_r2:
            best_r2 = actual_r2
            best_mse = actual_mse
            best_dimension = i+1 
    print(f'LLE --> Components: {best_dimension}, R2: {best_r2}, MSE: {best_mse}\n')
    dimRedMethod = LLE(n_components=best_dimension,n_neighbors=10)
    actual_train = dimRedMethod.fit_transform(train_df)
    actual_val = dimRedMethod.transform(val_df)
    regr = LinearRegression().fit(actual_train, train_target)
    print(relative_root_mean_squared_error(val_target, regr.predict(actual_val)))
    compute_r2_nonLin(actual_train, train_target, actual_val, val_target)
    return best_dimension,best_r2,best_mse

